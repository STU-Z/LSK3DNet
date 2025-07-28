# -*- coding:utf-8 -*-
# author: Xinge
# @file: load_save_util.py 

import torch


def load_checkpoint(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))
            match_size += 1
            part_load[k] = value
        else:
            print(k in my_model_dict)
            print(my_model_dict[k].shape, value.shape)
            assert len(value.shape) == 1 or len(value.shape) == 5
            if len(value.shape) == 1:
                c = value.shape[0]
                cc = my_model_dict[k].shape[0] - c #int(c*0.5)
                if cc <= c:
                    value = torch.cat([value, value[:cc]], dim=0)
                else:
                    value = torch.cat([value, value, value[:(cc-c)]], dim=0)
            else:
                _, _, _, c1, c2 = value.shape
                cc1 = my_model_dict[k].shape[3] - c1 #int(c1*0.5)
                cc2 = my_model_dict[k].shape[4] - c2 #int(c2*0.5)
                if cc1 > 0 and cc1 <= c1:
                    value1 = torch.cat([value, value[:, :, :, :cc1, :]], dim=3) 
                elif cc1 > c1:
                    value1 = torch.cat([value, value, value[:, :, :, :(cc1-c1), :]], dim=3) 
                else:
                    value1 = value
                if cc2 > 0 and cc2 <= c2:
                    value = torch.cat([value1, value1[:, :, :, :, :cc2]], dim=4) 
                elif cc2 > c2:
                    value = torch.cat([value1, value1, value1[:, :, :, :, :(cc2-c2)]], dim=4) 
                else:
                    value = value1
            nomatch_size += 1
            part_load[k] = value
            assert my_model_dict[k].shape == value.shape
            #print("model shape:{}, pre shape:{}".format(str(my_model_dict[k].shape), str(value.shape)))

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model

def load_checkpoint_old(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)['checkpoint']

    part_load = {}
    match_size = 0
    nomatch_size = 0
    for k in pre_weight.keys():
        value = pre_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # print("loading ", k)
            match_size += 1
            part_load[k] = value
        else:
            # import pdb
            # pdb.set_trace()
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model

def load_checkpoint_model_mask(model_load_path, model, device):
    my_model_dict = model.state_dict()  # 当前模型的参数字典
    pre_weight = torch.load(model_load_path,map_location=device)  # 加载预训练权重
    model_weight = pre_weight['checkpoint']  # 取出参数部分
    part_load = {}  # 用于存放可以加载的参数
    match_size = 0  # 匹配的参数数量
    nomatch_size = 0  # 不匹配的参数数量
    for k in model_weight.keys():
        value = model_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # 如果参数名和shape都匹配，就加载
            match_size += 1
            part_load[k] = value
        else:
            # 否则打印出来，不加载
            print("not matched key", k)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)  # 用匹配的参数更新当前模型参数
    model.load_state_dict(my_model_dict)  # 加载到模型
    # model.load_state_dict(my_model_dict, strict=False)  # 也可以用strict=False方式
    return model, pre_weight['mask']  # 返回模型和mask（稀疏训练用）

def load_checkpoint_model_mask_optimizer_scheduler_scaler_epoch(model_load_path, model,train_hypers, device):
    my_model_dict = model.state_dict()  # 当前模型的参数字典
    pre_weight = torch.load(model_load_path,map_location=device)  # 加载预训练权重
    model_weight = pre_weight['checkpoint']  # 取出参数部分
    part_load = {}  # 用于存放可以加载的参数
    match_size = 0  # 匹配的参数数量
    nomatch_size = 0  # 不匹配的参数数量
    for k in model_weight.keys():
        value = model_weight[k]
        if k in my_model_dict and my_model_dict[k].shape == value.shape:
            # 如果参数名和shape都匹配，就加载
            match_size += 1
            part_load[k] = value
        else:
            # 否则打印出来，不加载
            print("not matched key", k)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)  # 用匹配的参数更新当前模型参数
    model.load_state_dict(my_model_dict)  # 加载到模型
    # model.load_state_dict(my_model_dict, strict=False)  # 也可以用strict=False方式
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state = pre_weight['optimizer_state_dict']
    scheduler_state=pre_weight['scheduler_state_dict']
    epoch = pre_weight['epoch']
    scaler_state=None
    if train_hypers['amp_enabled'] and 'scale_state_dict' in pre_weight and pre_weight['scale_state_dict'] is not None:
        scaler_state= pre_weight['scaler_state_dict']
    return model, pre_weight['mask'], optimizer_state, scheduler_state, scaler_state, epoch # 返回模型和mask（稀疏训练用）
    

def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model
