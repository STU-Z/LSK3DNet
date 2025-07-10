import os
import random
import time
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel 

from tqdm import tqdm
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, loss_builder, optim_builder
from network.largekernel_model import get_model_class
from easydict import EasyDict
import shutil

from utils.load_util import load_yaml
from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask,load_checkpoint_model_mask_optimizer_scheduler_scaler_epoch
from utils.erk_sparse_core import Masking, CosineDecay

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_path', default='./config/lk-semantickitti_erk_finetune.yaml')
parser.add_argument('--ip', default='127.0.0.1', type=str)
parser.add_argument('--port', default='3020', type=str)
args = parser.parse_args()
config_path = args.config_path
configs = load_yaml(config_path)
configs.update(vars(args))  # override the configuration using the value in args
configs = EasyDict(configs)

exp_dir_root = configs['model_params']['model_save_path'].split('/')
exp_dir_root = exp_dir_root[0] if len(exp_dir_root) > 1 else ''
exp_dir = './'+exp_dir_root+'/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
shutil.copy('train_skitti.py', str(exp_dir))
shutil.copy('dataloader/dataset2.py', str(exp_dir))
shutil.copy('dataloader/pc_dataset.py', str(exp_dir))
shutil.copy('dataloader/utils.py', str(exp_dir))
shutil.copy('builder/data_builder.py', str(exp_dir))
shutil.copy('network/largekernel_model.py', str(exp_dir))
shutil.copy('utils/erk_sparse_core.py', str(exp_dir))
shutil.copy('config/lk-semantickitti_erk_finetune.yaml', str(exp_dir))


def main(configs):
    configs.nprocs = torch.cuda.device_count()
    configs.train_params.distributed = True if configs.nprocs > 1 else False
    if configs.train_params.distributed:
        mp.spawn(main_worker, nprocs=configs.nprocs, args=(configs.nprocs, configs))
    else:
        main_worker(0, 1, configs)

def main_worker(local_rank, nprocs, configs):
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(log_dir='runs/exp1')
    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    sparse_config = configs['sparse_params']
    train_hypers.local_rank = local_rank
    train_hypers.world_size = nprocs
    configs.train_params.world_size = nprocs
    
    if train_hypers['distributed']:
        init_method = 'tcp://' + args.ip + ':' + args.port
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=nprocs, rank=local_rank)
        dataset_config.train_data_loader.batch_size = dataset_config.train_data_loader.batch_size // nprocs

    pytorch_device = torch.device('cuda:' + str(local_rank))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)

    print('dataset_config.train_data_loader.num_works: ', dataset_config.train_data_loader.num_workers)

    # seed
    if ('seed' not in train_hypers) or (train_hypers.seed is None):
        train_hypers.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = train_hypers.seed + local_rank * dataset_config.train_data_loader.num_workers * train_hypers['max_num_epochs']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = get_model_class(model_config['model_architecture'])(configs)

    if train_hypers['distributed']:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
    print("model_config['model_load_path']: ",model_config['model_load_path'])
    if os.path.exists(model_config['model_load_path']):
        print('pre-train')
        try:
            print("load model and pre_weight")
            # my_model, pre_weight = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
            my_model, pre_weight,optimizer_state,scheduler_state,scaler_state,epoch_state = load_checkpoint_model_mask_optimizer_scheduler_scaler_epoch(model_config['model_load_path'], my_model,train_hypers,pytorch_device)
        except:
            print("load model only")
            my_model = load_checkpoint_old(model_config['model_load_path'], my_model)

    my_model.to(pytorch_device)

    if train_hypers['distributed']:
        train_hypers.local_rank = train_hypers.local_rank % torch.cuda.device_count()
        my_model= DistributedDataParallel(my_model,device_ids=[train_hypers.local_rank],find_unused_parameters=False)

    print(' --------------------------------------- data_builder --------------------------------------- ')
    train_dataset_loader, val_dataset_loader, train_sampler = data_builder.build(dataset_config, train_hypers)

    configs.train_params.total_steps = train_hypers['max_num_epochs'] * len(train_dataset_loader) 
    print(len(train_dataset_loader))
    sparse_config['stop_sparse_epoch'] = sparse_config['stop_sparse_epoch'] * len(train_dataset_loader) 
    optimizer, scheduler = optim_builder.build(configs, my_model)
    criterion = loss_builder.criterion(configs, pytorch_device)
    scaler = amp.GradScaler(enabled=train_hypers['amp_enabled']) 
    
       # 恢复优化器、调度器、scaler等状态
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    if scaler_state is not None and train_hypers['amp_enabled']:
        scaler.load_state_dict(scaler_state)

    if sparse_config['use_sparse']:
        print(' --- use_sparse: --- ')
        decay = CosineDecay(sparse_config['prune_rate'], int(configs.train_params.total_steps))
        mask = Masking(optimizer, scaler, 
                       spatial_partition = model_config['spatial_group_partition'],
                       prune_mode=sparse_config['prune'], prune_rate_decay=decay, 
                       growth_mode=sparse_config['growth'], redistribution_mode=sparse_config['redistribution'], 
                       fp16=train_hypers['amp_enabled'], update_frequency=sparse_config['update_frequency'], 
                       sparsity=sparse_config['sparsity'], sparse_init=sparse_config['sparse_init'], 
                       device=train_hypers.local_rank, distributed=train_hypers['distributed'], stop_iter = sparse_config['stop_sparse_epoch'])
        try:
            print(' ------ mask add_module pre_weight ------ ')
            mask.add_module(my_model, pre_weight)
        except:
            print(' ------ mask add_module only ------ ')
            mask.add_module(my_model)

    print("train_hypers['distributed']: ",train_hypers['distributed'])
    # training
    print(' training start ')
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    
    if os.path.exists(model_config['model_load_path']):
        epoch = epoch_state
    train_sampler.set_epoch(epoch)
    sche_epoch_update = True
    print('train hypers:', train_hypers)
    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        train_hist_list=[]
        torch.cuda.empty_cache()
        my_model.train()
        if train_hypers.local_rank == 0:
            pbar = tqdm(total=len(train_dataset_loader), ncols=80)
            pbar.set_description('Epoch %i' % epoch)
        else:
            pbar = None
        train_sampler.set_epoch(epoch)
        time.sleep(10)
        
        for i_iter, (train_data_dict) in enumerate(train_dataset_loader):
            # if i_iter % 100 == 0:
            # print(f"[DEBUG] Epoch {epoch}, Iter {i_iter}: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB", flush=True)
            # for device_id in range(torch.cuda.device_count()):
            #     print('device_id:', device_id)
            #     allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            #     reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            #     print(f"[DEBUG] Epoch {epoch}, Iter {i_iter}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB", flush=True)
            torch.cuda.empty_cache()
            if global_iter % check_iter == 0 and global_iter != 0: 
                torch.cuda.empty_cache()
                my_model.eval()
                hist_list = []
                val_loss_list = []
                total_time = 0
                print("validation start ... ...")
                with torch.no_grad(): 
                    valid_start_time=time.time()
                    for i_iter_val, (val_data_dict) in enumerate(
                            val_dataset_loader):

                        val_data_dict['points'] = val_data_dict['points'].to(pytorch_device)
                        val_data_dict['normal'] = val_data_dict['normal'].to(pytorch_device)
                        val_data_dict['batch_idx'] = val_data_dict['batch_idx'].to(pytorch_device)
                        val_data_dict['labels'] = val_data_dict['labels'].to(pytorch_device)
                    
                        torch.cuda.synchronize()
                        start = time.time()
                        val_data_dict = my_model(val_data_dict)
                        torch.cuda.synchronize()
                        end = time.time()
                        total_time += (end-start)
                        # ====== 这里补充loss计算 ======
                        val_loss = criterion(val_data_dict)
                        val_loss_list.append(val_loss.item())
                        # ===========================
                        predict_labels = torch.argmax(val_data_dict['logits'], dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        val_pt_labs = val_data_dict['labels'].cpu().detach().numpy()
                        hist_list.append(fast_hist_crop(predict_labels, val_pt_labs, unique_label))
                    valid_time= time.time() - valid_start_time
                    print('validation time:', valid_time)
                if train_hypers.local_rank == 0:
                    print('inference speed:', total_time / 4071)
                    iou = per_class_iu(sum(hist_list))
                    print('Validation per class iou: ')
                    for class_name, class_iou in zip(unique_label_str, iou):
                        print('%s : %.2f%%' % (class_name, class_iou * 100))
                    val_miou = np.nanmean(iou) * 100
                    # 记录训练loss、验证loss、mIoU、学习率等
                    train_log = {
                           'inference speed': total_time / 4071,
                           'epoch': epoch,
                           'iter': i_iter,
                           # 'train_loss': np.mean(loss_list),
                           'val_loss': np.mean(val_loss_list) if len(val_loss_list) > 0 else None,
                           'val_miou': val_miou,
                           'lr': optimizer.param_groups[0]['lr'],
                           # 可加更多指标
                       }
                    # 记录每个类别的IoU
                    for class_name, class_iou in zip(unique_label_str, iou):
                        train_log[f'iou_{class_name}'] = class_iou * 100
                    with open('output_skitti/train_log.csv', 'a') as f:
                        f.write(','.join([str(train_log[k]) for k in train_log]) + '\n')
                           
                    if best_val_miou < val_miou:
                        best_val_miou = val_miou
                        

                        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
                            if sparse_config['use_sparse']:
                                save_dict = {
                                    'epoch': epoch,
                                            'checkpoint':my_model.module.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'mask':mask.masks,
                                            'scale_state_dict': scaler.state_dict() if train_hypers['amp_enabled'] else None
                                            }
                            else:
                                save_dict = {
                                    'epoch': epoch,
                                            'checkpoint':my_model.module.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'scale_state_dict': scaler.state_dict() if train_hypers['amp_enabled'] else None
                                    }
                        except:
                            if sparse_config['use_sparse']:
                                save_dict = {
                                            'epoch': epoch,
                                            'checkpoint':my_model.module.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'mask':mask.masks,
                                            'scale_state_dict': scaler.state_dict() if train_hypers['amp_enabled'] else None
                                            }
                            else:
                                save_dict = {'epoch': epoch,
                                            'checkpoint':my_model.module.state_dict(),
                                            'optimizer_state_dict': optimizer.state_dict(),
                                            'scheduler_state_dict': scheduler.state_dict(),
                                            'scale_state_dict': scaler.state_dict() if train_hypers['amp_enabled'] else None
                                            }

                        torch.save(save_dict, model_config['model_save_path'][:-3] + str(train_hypers.local_rank)+ model_config['model_save_path'][-3:],
                                _use_new_zipfile_serialization=False)
                        '''
                        # # 保存示例
                        # torch.save({
                        #     'epoch': epoch,
                        #     'model_state_dict': model.state_dict(),
                        #     'optimizer_state_dict': optimizer.state_dict(),
                        #     'scheduler_state_dict': scheduler.state_dict(),
                        #     # 还可以加scaler、mask等
                        # }, PATH)

                        # # 加载
                        # checkpoint = torch.load(PATH)
                        # model.load_state_dict(checkpoint['model_state_dict'])
                        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        # epoch = checkpoint['epoch']
                        # if train_hypers['amp_enabled'] and 'scale_state_dict' in checkpoint and checkpoint['scale_state_dict'] is not None:
                        # scaler.load_state_dict(checkpoint['scale_state_dict'
                        '''
                        
                        print('Saved: ' + model_config['model_save_path'][:-3] + str(train_hypers.local_rank)+ model_config['model_save_path'][-3:])

                    print('Current val miou is %.3f while the best val miou is %.3f' %
                        (val_miou, best_val_miou))                

                my_model.train()
                torch.cuda.empty_cache()
                time.sleep(10)
                if train_hypers['distributed']:
                    dist.barrier()
                loss_list = []

            train_data_dict['points'] = train_data_dict['points'].to(pytorch_device)
            train_data_dict['normal'] = train_data_dict['normal'].to(pytorch_device)
            train_data_dict['batch_idx'] = train_data_dict['batch_idx'].to(pytorch_device)
            train_data_dict['labels'] = train_data_dict['labels'].to(pytorch_device)
                
            with amp.autocast(enabled=train_hypers['amp_enabled']):
                # forward + backward + optimize
                train_data_dict = my_model(train_data_dict)
                loss = criterion(train_data_dict)

            loss_list.append(loss.item())

            if sparse_config['use_sparse']:
                if train_hypers['amp_enabled']:
                    mask.optimizer.zero_grad()
                    with torch.autograd.detect_anomaly():
                        mask.scaler.scale(loss).backward()
                    mask.scaler.unscale_(mask.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=my_model.parameters(), max_norm=0.1)
                    mask.scaler.step(mask.optimizer)
                    mask.step()
                    mask.scaler.update()
                    scale = mask.scaler.get_scale()
                    skip_lr_sched = (scale != mask.scaler.get_scale())
                    if not skip_lr_sched:
                        scheduler.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=my_model.parameters(), max_norm=0.25)
                    mask.step()
                    if not sche_epoch_update:
                        scheduler.step()
            else:
                if train_hypers['amp_enabled']:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=my_model.parameters(), max_norm=0.25)
                    scaler.step(optimizer)
                    scaler.update()
                    scale = scaler.get_scale()
                    skip_lr_sched = (scale != scaler.get_scale())
                    if not skip_lr_sched:
                        scheduler.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=my_model.parameters(), max_norm=0.25)
                    optimizer.step()    
                    scheduler.step()

            # if torch.isnan(loss).any():
            #     # continue
            #     for name, param in my_model.named_parameters():
            #         if param.grad is not None and torch.isnan(param.grad).any():
            #             print("nan gradient found")
            #             print("name:", name)
            #     quit()

            if torch.isnan(loss).any():
                '''
                在训练过程中，实时检测 loss 和参数梯度是否出现 NaN，
                一旦发现数值异常，立即打印出有问题的参数并终止训练，方便及时发现和定位问题，保证训练过程的数值稳定性
                '''
                for name, param in my_model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print("nan gradient found")
                        print("name:", name)
                print("NaN detected in loss! Exiting all processes.")
                if dist.is_initialized():
                    dist.destroy_process_group()
                raise RuntimeError("NaN detected in loss!")
            
            # # ===========================
            train_predict_labels = torch.argmax(train_data_dict['logits'], dim=1)
            train_predict_labels = train_predict_labels.cpu().detach().numpy()
            train_pt_labs = train_data_dict['labels'].cpu().detach().numpy()
            train_hist_list.append(fast_hist_crop(train_predict_labels, train_pt_labs, unique_label))


            if train_hypers.local_rank == 0:
                pbar.set_postfix({'loss':'{0:1.2f}'.format(loss.item()), 'lr':'{0:1.8f}'.format(optimizer.param_groups[0]['lr'])})
                pbar.update(1)
                writer.add_scalar('Loss/train', loss.item(), global_iter)  # tensorboard --logdir runs
                # for device_id in range(torch.cuda.device_count()):
                #     print('device_id:', device_id)
                #     allocated = torch.cuda.memory_allocated(device_id) / 1024**3
                #     reserved = torch.cuda.memory_reserved(device_id) / 1024**3
                #     print(f"[DEBUG] Epoch {epoch}, Iter {i_iter}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB", flush=True)


            global_iter += 1
            if train_hypers.local_rank == 0:
                if global_iter % check_iter == 0 and global_iter != 0:
                    train_iou = per_class_iu(sum(train_hist_list))
                    print('------Train per class iou: ')
                    for class_name, class_iou in zip(unique_label_str, train_iou):
                        print('%s : %.2f%%' % (class_name, class_iou * 100))
                    train_miou = np.nanmean(train_iou) * 100
                    print(f"epoch: {epoch}, global_iter: {global_iter} , Train loss: {np.mean(loss_list):.3f} , train_miou: {train_miou}")
                    train_hist_list=[]
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                            (epoch, i_iter, np.mean(loss_list)))
                         # 记录本epoch的train_loss
                        only_train_log = {
                            'epoch': epoch,
                            'train_loss': np.mean(loss_list) if len(loss_list) > 0 else None,
                            'lr': optimizer.param_groups[0]['lr'],
                            'train_miou': train_miou
                            # 其他指标可选
                        }     
                        for class_name, class_iou in zip(unique_label_str, train_iou):
                            only_train_log[f'iou_{class_name}'] = class_iou * 100                
                        with open('output_skitti/only_train_log.csv', 'a') as f:
                            f.write(','.join([str(only_train_log[k]) for k in only_train_log]) + '\n')
                    else:
                        print('loss error')

            if global_iter % check_iter == 0:
                loss_list = []

        # if sche_epoch_update:
        #     scheduler.step()  # 按 epoch 更新
        if sche_epoch_update:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # 取验证集loss均值
                val_loss = np.mean(val_loss_list) if len(val_loss_list) > 0 else 0
                scheduler.step(val_loss)
            else:
                scheduler.step()

        torch.cuda.empty_cache()
        if train_hypers.local_rank == 0:
            pbar.close()
        epoch += 1
    

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print(configs)
    main(configs)