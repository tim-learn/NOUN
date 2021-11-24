import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import pdb
import math
from sklearn.metrics import confusion_matrix
from scipy.linalg import qr
import copy

def image_classification_test(loader, model, test_10crop=False, visda=False):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test_10crop'][i]) for i in range(10)]
            for i in range(len(loader['test_10crop'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                #labels = labels.cuda()
                if model.disjoint:
                    outputs = model.classifier(model(inputs))
                else:
                    _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    if model.disjoint:
        softmax_out = nn.Softmax(dim=1)(all_output)
        p_cls = softmax_out / (1e-5 + 1 - softmax_out[:, -1]).unsqueeze(1)
        mean_ent = torch.mean(loss.Entropy(p_cls[:, :-1])).cpu().data.item()
        all_output = all_output[:, :-1]
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        # mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    else:
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    all_label = all_label.long()
    tp = torch.gather(all_output, 1, all_label.view(-1,1)).repeat(1, all_output.size(1))
    co = (all_output > tp).sum(dim=1)
    xt = torch.stack([(co == x).sum() for x in range(all_output.size(1))]).float()
    xt = xt / xt.sum()
    log_str = "1: {:.4f}, 2: {:.4f}, 3: {:.4f}, sum: {:.4f}\n".format(xt[0], xt[1], xt[2], xt[0:3].sum())
    # print(log_str)

    if visda:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1)
        cls_acc_str = ' '.join(['{:.4f}'.format(x) for x in acc.tolist()])
        cls_str_format = "Per-class accuracy is: "+cls_acc_str+"; mean acc is "+str(np.mean(acc))+".\n"
        log_str += cls_str_format

    return accuracy, mean_ent, log_str

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def test(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                            shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()
    base_network.load_state_dict(torch.load(osp.join(config["output_path"], "final_model.pt")))

    base_network.train(False)
    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders["test"])
        for i in range(len(dset_loaders['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            names = data[2]
            inputs = inputs.cuda()
            if base_network.disjoint:
                outputs = base_network.classifier(base_network(inputs))
            else:
                _, outputs = base_network(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                names_lst = list(names)
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                names_lst += list(names)

    if base_network.disjoint:
        softmax_out = nn.Softmax(dim=1)(all_output)
        p_cls = softmax_out / (1e-5 + 1 - softmax_out[:, -1]).unsqueeze(1)
        mean_ent = torch.mean(loss.Entropy(p_cls[:, :-1])).cpu().data.item()
        all_output = all_output[:, :-1]
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        
    else:
        _, predict = torch.max(all_output, 1)
        with open(osp.join(config["output_path"], 'tgt_pred.txt'), 'w') as f:
            for i_img in range(len(names_lst)):
                f.write("%s %d\n" % (names_lst[i_img], predict[i_img].item()))
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()    
    
    all_label = all_label.long()
    tp = torch.gather(all_output, 1, all_label.view(-1,1)).repeat(1, all_output.size(1))
    co = (all_output > tp).sum(dim=1)
    xt = torch.stack([(co == x).sum() for x in range(65)]).float()
    xt = xt / xt.sum()
    log_str = "1: {:.4f}, 2: {:.4f}, 3: {:.4f}, sum: {:.4f}\n".format(xt[0], xt[1], xt[2], xt[0:3].sum())
    if config["dataset"] == "visda":
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1)
        print(acc, np.mean(acc))
    print(log_str)

def train(config):

    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    eps = config["eps"]

    if config['visda']:
        prep_dict["source"] = prep.image_visda(**config["prep"]['params'])
        prep_dict["target"] = prep.image_visda(**config["prep"]['params'])
    else:
        prep_dict["source"] = prep.image_train(**config["prep"]['params'])
        prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])
    prep_dict["test_10crop"] = prep.image_test_10crop(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=4, drop_last=True)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                            shuffle=False, num_workers=4)

    for i in range(10):
        dsets["test_10crop"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                            transform=prep_dict["test_10crop"][i]) for i in range(10)]
        dset_loaders["test_10crop"] = [DataLoader(dset, batch_size=test_bs, \
                            shuffle=False, num_workers=4) for dset in dsets['test_10crop']]

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net1 = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
        ad_net1 = ad_net1.cuda() 
        parameter_list = base_network.get_parameters() + ad_net1.get_parameters()
    else:
        random_layer = None
        if config['method'] == 'DANN':
            ad_net1 = network.AdversarialNetwork(base_network.output_num(), 1024, max_iter=config['num_iterations'])
            ad_net1 = ad_net1.cuda()
            parameter_list = base_network.get_parameters() + ad_net1.get_parameters()
        elif config['method'] == 'NOUN':
            # dann-[f,p]
            ad_net1 = network.AdversarialNetwork(base_network.output_num() + class_num, 1024, max_iter=config['num_iterations'])
            ad_net1 = ad_net1.cuda()
            # noun
            ad_net2 = network.AdversarialNetwork(base_network.output_num() * 2, 1024, max_iter=config['num_iterations'])
            ad_net2 = ad_net2.cuda()
            parameter_list = base_network.get_parameters() + ad_net1.get_parameters() + ad_net2.get_parameters()
        elif config['method'][:4] == 'CDAN':
            ad_net1 = network.AdversarialNetwork(base_network.output_num() * class_num, 1024, max_iter=config['num_iterations'])
            ad_net1 = ad_net1.cuda()
            parameter_list = base_network.get_parameters() + ad_net1.get_parameters()
        elif config['method'] == 'Srconly':
            parameter_list = base_network.get_parameters()
        elif config['method'] == 'MADA':
            ad_net_lst = nn.ModuleList()
            parameter_list = base_network.get_parameters()
            for j in range(class_num):
                ad_net_lst.append(network.AdversarialNetwork(base_network.output_num(), 1024, max_iter=config['num_iterations']))
            for j in range(class_num):
                ad_net_lst[j] = ad_net_lst[j].cuda()
                parameter_list += ad_net_lst[j].get_parameters() 
        elif config['method'] == 'IDDA':
            ad_net1 = network.AdversarialNetwork_k1(class_num, base_network.output_num(), 1024, max_iter=config['num_iterations'])
            ad_net1 = ad_net1.cuda()
            parameter_list = base_network.get_parameters() + ad_net1.get_parameters()
        elif config['method'] == 'DANN_CA':
            parameter_list = base_network.get_parameters()
        elif config['method'] == 'RCA':
            ad_net1 = network.AdversarialNetwork_2k(class_num, base_network.output_num(), 1024, max_iter=config['num_iterations'])
            ad_net1 = ad_net1.cuda()
            parameter_list = base_network.get_parameters() + ad_net1.get_parameters()
        else:
            raise ValueError('Method cannot be recognized.')
    if config["loss"]["random"]:
        random_layer.cuda()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        if config['method'] == 'MADA':
            for j in range(class_num):
                ad_net_lst[j] = nn.DataParallel(ad_net_lst[j], device_ids=[int(i) for i in gpus])
        elif config['method'] == 'DANN_CA' or config['method'] == 'Srconly':
            pass
        else:
            ad_net1 = nn.DataParallel(ad_net1, device_ids=[int(i) for i in gpus])
            if config['method'] == 'NOUN':
                ad_net2 = nn.DataParallel(ad_net2, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])
        
    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_ent = 100

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc, temp_ent, log = image_classification_test(dset_loaders, base_network, test_10crop=False, visda=config['visda'])

            temp_model = nn.Sequential(base_network)
            if temp_ent < best_ent:
                best_ent = temp_ent
                best_model = copy.deepcopy(base_network)
                best_log_str = "Best entropy occurs in task: {}, iter: {:05d}, precision: {:.4f}, entropy: {:.4f}\n".format(config['name'], i, temp_acc, temp_ent)
                best_log_str += log
            log_str = "Task: {}, iter: {:05d}, precision: {:.4f}, entropy: {:.4f}\n".format(config['name'], i, temp_acc, temp_ent)
            log_str += log
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
            
        loss_params = config["loss"] 

        ## train one iter
        base_network.train(True)
        if config['method'] == 'MADA':
            for j in range(class_num):
                ad_net_lst[j].train(True)
        elif config['method'] == 'DANN_CA' or config['method'] == 'Srconly':
            pass
        else:
            ad_net1.train(True)
            if config['method'] == 'NOUN':
                ad_net2.train(True)

        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source, _ = iter_source.next()
        inputs_target, labels_target, _ = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        
        if config['method'] == 'DANN_CA':
            features_source = base_network(inputs_source)
            features_target = base_network(inputs_target)

            for param in base_network.classifier.parameters():
                param.requires_grad = False
            outputs_source = base_network.classifier(features_source)
            outputs_target = base_network.classifier(features_target)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            p_t = softmax_out[train_bs:, -1]
            p_s_t = softmax_out[:train_bs, -1]
            p_bar_s = softmax_out[:train_bs, :] / (eps + 1 - softmax_out[:train_bs, -1]).unsqueeze(1) 
            lbd = calc_coeff(i)
            f_loss = nn.NLLLoss()(torch.log(eps+p_bar_s[:, :-1]), labels_source) - lbd*(torch.mean(torch.log(eps+p_s_t)) + torch.mean(torch.log(eps+1-p_t)))
            f_loss.backward()

            for param in base_network.classifier.parameters():
                param.requires_grad = True
            outputs_source = base_network.classifier(features_source.detach())
            outputs_target = base_network.classifier(features_target.detach())
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            label_t = (class_num-1)*torch.ones(train_bs)
            label_all = torch.cat((labels_source, label_t.long().cuda()), dim=0)
            classifier_loss = nn.CrossEntropyLoss()(outputs, label_all) 
            classifier_loss.backward()
            optimizer.step()

        elif config['method']  == 'RCA':
            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)
            _, idx_t = torch.max(outputs_target, dim=1)

            for param in ad_net1.parameters():
                param.requires_grad = False
            g_out = ad_net1(features)
            g_sfmx = nn.Softmax(dim=1)(g_out)
            p_t_g = g_sfmx[train_bs:, :class_num]
            p_s_g = g_sfmx[:train_bs, class_num:]
            lbd = calc_coeff(i)
            f_loss = nn.CrossEntropyLoss()(outputs_source, labels_source) + lbd*(nn.NLLLoss()(torch.log(eps+p_t_g), idx_t) + nn.NLLLoss()(torch.log(eps+p_s_g), labels_source))
            f_loss.backward()

            for param in ad_net1.parameters():
                param.requires_grad = True
            d_out = ad_net1(features.detach())
            d_sfmx = nn.Softmax(dim=1)(d_out)
            p_t_d = d_sfmx[train_bs:, class_num:]
            p_s_d = d_sfmx[:train_bs, :class_num]
            d_loss = nn.NLLLoss()(torch.log(eps+p_t_d), idx_t) + nn.NLLLoss()(torch.log(eps+p_s_d), labels_source)
            d_loss.backward()
            optimizer.step()

        elif config['method'] == 'Srconly':
            features_source, outputs_source = base_network(inputs_source)
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            classifier_loss.backward()
            optimizer.step()
             
        elif config['method'] in ['CDAN', 'CDAN_E', 'DANN', 'NOUN', 'MADA', 'IDDA']:
            features_source, outputs_source = base_network(inputs_source)
            features_target, outputs_target = base_network(inputs_target)
            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)
            softmax_out = nn.Softmax(dim=1)(outputs)

            if config['method'] == 'CDAN_E':           
                entropy = loss.Entropy(softmax_out)
                transfer_loss = loss.CDAN([features, softmax_out], ad_net1, entropy, network.calc_coeff(i), random_layer)

            elif config['method']  == 'CDAN':
                transfer_loss = loss.CDAN([features, softmax_out], ad_net1, None, None, random_layer)

            elif config['method']  == 'DANN':
                entropy = loss.Entropy(softmax_out) 
                transfer_loss = loss.DANN(features, ad_net1)        

            elif config['method']  == 'MADA':
                entropy = loss.Entropy(softmax_out)
                transfer_loss = 0
                for j in range(class_num):
                    transfer_loss += loss.DANN(features*(softmax_out[:, j].unsqueeze(1).detach()), ad_net_lst[j]) 

            elif config['method']  == 'IDDA':
                entropy = loss.Entropy(softmax_out) 
                transfer_loss = loss.IDDA(features, labels_source, ad_net1)  

            elif config['method']  == 'NOUN':
                entropy = loss.Entropy(softmax_out)

                softmax_s = nn.Softmax(dim=1)(outputs_source)
                softmax_t = nn.Softmax(dim=1)(outputs_target)

                if config["cond_type"] == 'ema_ctr_norm':
                    gt_list = labels_source.tolist()
                    gt_list = np.unique(gt_list)

                    if i == 0:
                        center_feat = torch.randn(softmax_s.shape[1], features_source.shape[1]).cuda().detach()
                    else:
                        for c in range(len(gt_list)):
                            c_idx = (labels_source==gt_list[c]).nonzero().squeeze()
                            c_feat = torch.index_select(features_source, 0, c_idx)
                            c_ctr = torch.mean(c_feat, dim=0)
                            center_feat[gt_list[c], :] = config['ema']*center_feat[gt_list[c], :].detach() + (1-config['ema'])*c_ctr.squeeze().detach()
                    center = center_feat.detach()
                    cond_p_s = torch.mm(softmax_s, center).detach()
                    cond_p_t = torch.mm(softmax_t, center).detach()
                    cond_p_de = torch.cat((cond_p_s, cond_p_t), dim=0)
                    if config["dnmc"]:
                        norm_factor = torch.norm(features)/torch.norm(cond_p_de)
                        norm_factor = norm_factor.detach()
                    else:
                        norm_factor = 1
                    feat1 = torch.cat((features, norm_factor*config["norm_factor"]*cond_p_de), dim=1)

                    if config["ent_cond"]:
                        transfer_loss = loss.NOUN(feat1, ad_net2, entropy, network.calc_coeff(i))
                    else:
                        transfer_loss = loss.NOUN(feat1, ad_net2, None) 

    
                elif config["cond_type"] == 'p':  
                    feat1 = torch.cat((features, softmax_out.detach()), dim=1)
                    transfer_loss = loss.NOUN(feat1, ad_net1, None)
    
                elif config["cond_type"] == 'p_norm':
                    if config["dnmc"]:
                        norm_factor = torch.norm(features)/torch.norm(softmax_out)
                        norm_factor = norm_factor.detach()
                    else:
                        norm_factor = 1
                    feat1 = torch.cat((features, norm_factor*config["norm_factor"]*softmax_out.detach()), dim=1)

                    if config["ent_cond"]:
                        transfer_loss = loss.NOUN(feat1, ad_net1, entropy, network.calc_coeff(i))
                    else:
                        transfer_loss = loss.NOUN(feat1, ad_net1, None)
    
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
            total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
            total_loss.backward()
            optimizer.step()

        else:
            raise ValueError('Method cannot be recognized.')


    torch.save(best_model.state_dict(), osp.join(config["output_path"], "final_model.pt"))
    config["out_file"].write(best_log_str+"\n")
    config["out_file"].flush()
    print(best_log_str)
    return best_ent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NOUN')
    parser.add_argument('--method', type=str, default='NOUN', choices=['Srconly', 'CDAN', 'CDAN_E', 'DANN', 'NOUN', 'MADA', 'IDDA', 'DANN_CA', 'RCA'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet50", "ResNet101"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--output_dir', type=str, default='noun', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--bs', type=int, default=36, help='batch size')
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--eval', type=bool, default=False, help="evaluate the ckpt")

    parser.add_argument('--ema', type=float, default=0.5)
    parser.add_argument('--norm_factor', type=float, default=1)
    parser.add_argument('--e', type=bool, default=False, help="whether condition on the entropy")
    parser.add_argument('--cond_feat', type=str, default='ema_ctr_norm', choices=['p', 'p_norm', 'ema_ctr_norm'], help="The type of conditional feature.")
    parser.add_argument('--dnmc_norm', type=bool, default=True, help="whether to dynamically normalize features")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    SEED = args.seed
    s = args.s
    t = args.t
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list.txt'

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list.txt'

    if args.dset == 'image-clef':
        names = ['c', 'i', 'p']
        args.s_dset_path = './data/' + args.dset + '/' + names[s] + '_list.txt'
        args.t_dset_path = './data/' + args.dset + '/' + names[t] + '_list.txt'


    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.dset == 'visda':
        names = ['training', 'validation']
        args.s_dset_path = './data/visda17/train_list.txt'
        args.t_dset_path = './data/visda17/validation_list.txt'

    # train config
    config = {}
    config['visda'] = (args.dset == 'visda')
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config['name'] = args.dset + '/' + names[s][0].upper() + names[t][0].upper()
    config["output_for_test"] = True
    config["output_path"] = args.output_dir  + args.dset + '/' + names[s][0].upper() + names[t][0].upper()
    config["ema"] = args.ema
    config["norm_factor"] = args.norm_factor
    config["ent_cond"] = args.e
    config["cond_type"] = args.cond_feat
    config["eps"] = 1e-5
    config["dnmc"] = args.dnmc_norm

    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])

    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
        
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    
    config["loss"] = {"trade_off":1.0}

    # whether to use bottleneck
    if "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    if config['method'] == 'DANN_CA':
        config["network"]["params"]["disjoint"] = True

    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    # parameter from orignial papers or released codes
    if config['method'] == 'MADA':
        config["optimizer"]["lr_param"]["gamma"] = 0.0003
    if config['method'] == 'DANN_CA':
        config["optimizer"]["optim_params"]["weight_decay"] = 0.0001

    config["dataset"] = args.dset
    
    # 36 100 96
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":args.bs}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":args.bs}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":100}}

    if config["dataset"] == "office":      
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "image-clef":
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
        config['loss']["trade_off"] = 1
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    if config['method'] == 'DANN_CA':
        config["network"]["params"]["class_num"] += 1 
        
    if not args.eval:
        config["out_file"].write(str(config))
        config["out_file"].flush()

    print(config)

    if args.eval:
        test(config)
    else:
        train(config)
