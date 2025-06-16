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
from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask
from utils.erk_sparse_core import Masking, CosineDecay

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--config_path', default='./config/lk-semantickitti_erk_finetune.yaml')
parser.add_argument('--ip', default='127.0.0.1', type=str)
parser.add_argument('--port', default='3020', type=str)
args = parser.parse_args()
config_path = args.config_path
print("================================")
print("config_path: ", config_path)
print("================================")
configs = load_yaml(config_path)

# override the configuration using the value in args
configs.update(vars(args))
configs = EasyDict(configs)  # 使用 EasyDict 使配置可以通过点符号访问（如configs.param_name）

# 创建实验目录用于保存模型和代码备份
# 将关键代码文件复制到实验目录，确保实验可复现
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
    configs.nprocs = torch.cuda.device_count()#表示当前机器上可用的GPU数量
    configs.train_params.distributed = True if configs.nprocs > 1 else False
    if configs.train_params.distributed:
        mp.spawn(main_worker, nprocs=configs.nprocs,
                 args=(configs.nprocs, configs))
    else:
        main_worker(0, 1, configs)


def main_worker(local_rank, nprocs, configs):
    print(f"Process local_rank: {local_rank}, GPU: {torch.cuda.current_device()}")
    torch.autograd.set_detect_anomaly(True)
    writer = SummaryWriter(log_dir='output_skitti/log')
    
    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    sparse_config = configs['sparse_params']
    train_hypers.local_rank = local_rank
    train_hypers.world_size = nprocs
    configs.train_params.world_size = nprocs

    # 初始化分布式训练环境
    if train_hypers['distributed']:
        init_method = 'tcp://' + args.ip + ':' + args.port
        dist.init_process_group(
            backend='nccl', init_method=init_method, world_size=nprocs, rank=local_rank)
        dataset_config.train_data_loader.batch_size = dataset_config.train_data_loader.batch_size // nprocs

    pytorch_device = torch.device('cuda:' + str(local_rank))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)

    # seed
    if ('seed' not in train_hypers) or (train_hypers.seed is None):
        train_hypers.seed = torch.initial_seed() % (2 ** 32 - 1)

     # 设置随机种子确保可复现性
    seed = train_hypers.seed + local_rank * \
        dataset_config.train_data_loader.num_workers * \
        train_hypers['max_num_epochs']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    SemKITTI_label_name = get_SemKITTI_label_name(
        dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = get_model_class(model_config['model_architecture'])(configs)

    if train_hypers['distributed']:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    # 加载预训练权重
    if os.path.exists(model_config['model_load_path']):
        print('pre-train')
        try:
            my_model, pre_weight = load_checkpoint_model_mask(
                model_config['model_load_path'], my_model, pytorch_device)
        except:
            my_model = load_checkpoint_old(
                model_config['model_load_path'], my_model)

    my_model.to(pytorch_device)

    # 设置分布式训练和混合精度
    if train_hypers['distributed']:
        train_hypers.local_rank = train_hypers.local_rank % torch.cuda.device_count()
        my_model = DistributedDataParallel(
            my_model, device_ids=[train_hypers.local_rank], find_unused_parameters=False)

    # 构建数据加载器
    train_dataset_loader, val_dataset_loader, train_sampler = data_builder.build(
        dataset_config, train_hypers)

    configs.train_params.total_steps = train_hypers['max_num_epochs'] * len(
        train_dataset_loader)
    print(len(train_dataset_loader))
    sparse_config['stop_sparse_epoch'] = sparse_config['stop_sparse_epoch'] * \
        len(train_dataset_loader)

    # 构建优化器和学习率调度器
    optimizer, scheduler = optim_builder.build(configs, my_model)
    # 构建损失函数
    criterion = loss_builder.criterion(configs, pytorch_device)
    # 设置混合精度训练
    scaler = amp.GradScaler(enabled=train_hypers['amp_enabled'])

    # 设置模型稀疏化
    if sparse_config['use_sparse']:
        decay = CosineDecay(sparse_config['prune_rate'], int(
            configs.train_params.total_steps))
        mask = Masking(optimizer, scaler,
                       spatial_partition=model_config['spatial_group_partition'],
                       prune_mode=sparse_config['prune'], prune_rate_decay=decay,
                       growth_mode=sparse_config['growth'], redistribution_mode=sparse_config['redistribution'],
                       fp16=train_hypers['amp_enabled'], update_frequency=sparse_config['update_frequency'],
                       sparsity=sparse_config['sparsity'], sparse_init=sparse_config['sparse_init'],
                       device=train_hypers.local_rank, distributed=train_hypers['distributed'], stop_iter=sparse_config['stop_sparse_epoch'])
        try:
            mask.add_module(my_model, pre_weight)
        except:
            mask.add_module(my_model)

    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    train_sampler.set_epoch(0)
    sche_epoch_update = True

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        torch.cuda.empty_cache() # 会释放 PyTorch 进程自己“暂时不用但还没归还操作系统”的那部分缓存
        my_model.train()
        if train_hypers.local_rank == 0:
            pbar = tqdm(total=len(train_dataset_loader), ncols=80)
            pbar.set_description('Epoch %i' % epoch)
        else:
            pbar = None
        train_sampler.set_epoch(epoch)
        time.sleep(10)
        # for i in range(5):
        #for i_iter in range(len(train_dataset_loader)):
        for i_iter, (train_data_dict) in enumerate(train_dataset_loader):
            # for device_id in range(torch.cuda.device_count()):
            #     allocated = torch.cuda.memory_allocated(device_id) / 1024**3
            #     reserved = torch.cuda.memory_reserved(device_id) / 1024**3
            #     print(f"[DEBUG] Epoch {epoch}, Iter {i_iter}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB", flush=True)
            # 每n步进行一次验证
            torch.cuda.empty_cache()
            if global_iter % check_iter == 0 and global_iter != 0:  # 判断是否到验证步
                torch.cuda.empty_cache()
                my_model.eval()  # 模型切换到 eval 模式
                hist_list = []
                val_loss_list = []
                total_time = 0
                with torch.no_grad(): # 遍历验证集
                    for i_iter_val, (val_data_dict) in enumerate(
                            val_dataset_loader):
                        # 不计算梯度，节省显存和加速。
                        # 把验证数据搬到 GPU，前向推理，统计预测标签和真实标签
                        val_data_dict['points'] = val_data_dict['points'].to(
                            pytorch_device)
                        val_data_dict['normal'] = val_data_dict['normal'].to(
                            pytorch_device)
                        val_data_dict['batch_idx'] = val_data_dict['batch_idx'].to(
                            pytorch_device)
                        val_data_dict['labels'] = val_data_dict['labels'].to(
                            pytorch_device)

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

                        predict_labels = torch.argmax(
                            val_data_dict['logits'], dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        val_pt_labs = val_data_dict['labels'].cpu(
                        ).detach().numpy()
                        hist_list.append(fast_hist_crop(
                            predict_labels, val_pt_labs, unique_label))

                if train_hypers.local_rank == 0:
                    print('inference speed:', total_time / 4071)
                    # 计算每类 IoU 和 mean IoU。
                    # 打印每类 IoU 和当前/最佳 mean IoU
                    iou = per_class_iu(sum(hist_list))
                    print('Validation per class iou: ')
                    for class_name, class_iou in zip(unique_label_str, iou):
                        print('%s : %.2f%%' % (class_name, class_iou * 100))
                    val_miou = np.nanmean(iou) * 100 

                    if best_val_miou < val_miou:
                        best_val_miou = val_miou  # 如果当前 mean IoU 超过历史最佳，则保存模型参数和稀疏掩码

                        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                            if sparse_config['use_sparse']:
                                save_dict = {
                                    'checkpoint': my_model.module.state_dict(), 'mask': mask.masks}
                            else:
                                save_dict = {
                                    'checkpoint': my_model.module.state_dict()}
                        except:
                            if sparse_config['use_sparse']:
                                save_dict = {
                                    'checkpoint': my_model.state_dict(), 'mask': mask.masks}
                            else:
                                save_dict = {
                                    'checkpoint': my_model.state_dict()}

                        torch.save(save_dict, model_config['model_save_path'][:-3] + str(train_hypers.local_rank) + model_config['model_save_path'][-3:],
                                   _use_new_zipfile_serialization=False)

                        print('Saved: ' + model_config['model_save_path'][:-3] + str(
                            train_hypers.local_rank) + model_config['model_save_path'][-3:])

                    print('Current val miou is %.3f while the best val miou is %.3f' %
                          (val_miou, best_val_miou))

                my_model.train()  # 恢复为训练模式，继续后续训练
                torch.cuda.empty_cache()
                time.sleep(10)
                if train_hypers['distributed']:
                    dist.barrier()
                loss_list = []
                

            train_data_dict['points'] = train_data_dict['points'].to(
                pytorch_device)
            train_data_dict['normal'] = train_data_dict['normal'].to(
                pytorch_device)
            train_data_dict['batch_idx'] = train_data_dict['batch_idx'].to(
                pytorch_device)
            train_data_dict['labels'] = train_data_dict['labels'].to(
                pytorch_device)

            with amp.autocast(enabled=train_hypers['amp_enabled']):
                # forward + backward + optimize
                train_data_dict = my_model(train_data_dict)
                loss = criterion(train_data_dict)

            loss_list.append(loss.item())

            if sparse_config['use_sparse']:
                if train_hypers['amp_enabled']:
                    mask.optimizer.zero_grad()  # 清空优化器管理的所有参数的梯度，为新一轮反向传播做准备
                    with torch.autograd.detect_anomaly(): #  PyTorch 的异常检测工具，用于在反向传播时自动检测和定位梯度计算中的异常（如梯度为 NaN 或 Inf）
                        mask.scaler.scale(loss).backward() # 在混合精度反向传播时自动检测和定位梯度异常（如 NaN/Inf），帮助快速发现和修复训练中的数值问题
                    '''
                    混合精度训练时，loss.backward() 前会对 loss 进行缩放，这样可以防止 float16 下的梯度下溢。
                    反向传播后，参数的梯度是被缩放过的。
                    调用 scaler.unscale_(optimizer)，会把优化器管理的所有参数的梯度除以缩放因子，还原为正常梯度。
                    这样后续的梯度裁剪、检查等操作就是在“真实梯度”上进行的。
                    '''
                    mask.scaler.unscale_(mask.optimizer) # 在混合精度训练中，把缩放过的梯度还原为正常尺度，为后续的梯度裁剪和优化器 step 做准备
                    
                    '''
                    这句代码的作用是对模型参数的梯度进行裁剪（clip），防止梯度爆炸。
                    my_model.parameters()：传入所有需要优化的参数。
                    max_norm=0.1：如果所有参数的梯度范数超过0.1，就按比例缩放，使总范数不超过0.1。
                    为什么要用梯度裁剪？
                    在深度学习训练中，尤其是大模型或不稳定任务，梯度有时会突然变得很大（梯度爆炸），导致训练不稳定甚至发散。
                    梯度裁剪可以有效防止这种情况，保证训练过程的稳定性
                    '''
                    torch.nn.utils.clip_grad_norm_(
                        parameters=my_model.parameters(), max_norm=0.1)
                    
                    '''
                    mask.scaler.step(mask.optimizer) 会在混合精度训练下安全地更新参数，
                    只有当梯度有效时才会执行参数更新，否则自动跳过，防止数值异常影响训练
                    '''
                    '''
                    典型流程
                    1.反向传播时用 scaler.scale(loss).backward() 计算缩放后的梯度。
                    2.用 scaler.unscale_(optimizer) 还原梯度到正常尺度。
                    3.用 scaler.step(optimizer) 安全地更新参数。
                    4.用 scaler.update() 更新缩放因子
                    '''
                    mask.scaler.step(mask.optimizer)
                    mask.step()
                    mask.scaler.update()
                    scale = mask.scaler.get_scale()
                    skip_lr_sched = (scale != mask.scaler.get_scale())
                    if not skip_lr_sched:
                        #  这段代码确保只有在参数真正被更新时才执行学习率调度，防止因 AMP 跳过参数更新导致学习率和参数步数不同步
                        scheduler.step()  # 按 step 更新
                else:
                    optimizer.zero_grad() # 清空优化器管理的所有参数的梯度，为新一轮反向传播做准备；这一步不会影响当前的前向传播和反向传播，只是把上一次累积的梯度清掉
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=my_model.parameters(), max_norm=0.25)
                    mask.step()
                    if not sche_epoch_update:
                        scheduler.step()
            else:
                if train_hypers['amp_enabled']:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        parameters=my_model.parameters(), max_norm=0.25)
                    scaler.step(optimizer)
                    scaler.update()
                    scale = scaler.get_scale()
                    skip_lr_sched = (scale != scaler.get_scale())
                    if not skip_lr_sched:
                        scheduler.step()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=my_model.parameters(), max_norm=0.25)
                    optimizer.step()
                    scheduler.step()

            # if torch.isnan(loss).any():
            #     '''
            #     在训练过程中，实时检测 loss 和参数梯度是否出现 NaN，
            #     一旦发现数值异常，立即打印出有问题的参数并终止训练，方便及时发现和定位问题，保证训练过程的数值稳定性
            #     '''
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

            if train_hypers.local_rank == 0:
                pbar.set_postfix({'loss': '{0:1.2f}'.format(
                    loss.item()), 'lr': '{0:1.8f}'.format(optimizer.param_groups[0]['lr'])})
                pbar.update(1)
                tqdm.write(f"[DEBUG] Epoch {epoch}, Iter {i_iter}: Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB", flush=True)

            global_iter += 1
            if train_hypers.local_rank == 0:
                 #tensorboard记录
                writer.add_scalar('Loss/train', np.mean(loss_list), global_iter)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_iter)
                if global_iter % check_iter == 0:
                    writer.add_scalar('mIoU/val', val_miou, global_iter)
                    writer.add_scalar('Loss/val', np.mean(val_loss_list), global_iter)
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                              (epoch, i_iter, np.mean(loss_list)))
                    else:
                        print('loss error')

            if global_iter % check_iter == 0:
                loss_list = []

        # if sche_epoch_update:
        #     scheduler.step() # 按 epoch 更新
        if sche_epoch_update:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if len(val_loss_list) > 0:
                    val_loss = np.mean(val_loss_list)
                    scheduler.step(val_loss)
                else:
                    print("Warning: No val_loss_list for ReduceLROnPlateau this epoch, skip scheduler.step()")
            else:
                scheduler.step()

        torch.cuda.empty_cache()
        if train_hypers.local_rank == 0:
            pbar.close()
            # 记录训练loss、验证loss、mIoU、学习率等
            train_log = {
                'epoch': epoch,
                'iter': i_iter,
                'train_loss': np.mean(loss_list),
                'val_loss': np.mean(val_loss_list) if len(val_loss_list) > 0 else None,
                'val_miou': val_miou,
                'lr': optimizer.param_groups[0]['lr'],
                # 可加更多指标
            }
            with open('output_skitti/train_log.csv', 'a') as f:
                f.write(','.join([str(train_log[k]) for k in train_log]) + '\n')
           
        epoch += 1


if __name__ == '__main__':
    print(' '.join(sys.argv))
    print(configs)
    main(configs)
