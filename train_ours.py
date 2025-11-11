import logging
import torch
import numpy as np
import random
import argparse
import os

from utils import set_for_logger, compute_accuracy, set_requires_grad
from dataset import partition_data, get_dataloader
from models import init_model
from pathlib import Path
import torch.nn.functional as F  

import copy

def get_teacher_prototype(c, dataloader, net_g, device):
    outputs = None
    targets = None
    for batch_idx, (x, target) in enumerate(dataloader):
        x = x.to(device)
        target = target.to(device)
        out_g = net_g(x)
        if batch_idx == 0:
            outputs = out_g
            targets = target
        else:
            outputs = torch.cat((outputs, out_g), dim=0)
            targets = torch.cat((targets, target), dim=0)
    Prototype = torch.zeros((c, c)).to(device)

    for i in range(c):
        index = (targets == i)
        nums = sum(index)
        if nums.item() == 0:
            continue
        index = index.unsqueeze(1)
        index = index.expand_as(outputs)
        temp_prototype = outputs * index
        temp_prototype = torch.sum(temp_prototype, dim=0)
        temp_prototype /= nums
        Prototype[i] = temp_prototype

    return Prototype

def cal_cosine_similarity(v1, v2):
     _v1 = F.normalize(v1, dim=-1, p=2)
     _v2 = F.normalize(v2, dim=-1, p=2)
     inner = torch.matmul(_v1, _v2.transpose(1, 0))
     s = F.softmax(inner, dim=-1)
     return s
    
    
            

def train_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, weight_decay, device):
    logging.info('client training %s' % str(net_id))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logging.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logging.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay,
                               amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epochs+1):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logging.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logging.info('>> Training accuracy: %f' % train_acc)
    logging.info('>> Test accuracy: %f' % test_acc)
    logging.info(' ** Training complete **')
    return net.state_dict()


def train_epoch(net_id, net, g_mt, l_mt,  train_dataloader, test_dataloader, epochs, lr,optimizer, weight_decay, device, T, mu, kd_mode, use_mask, use_normalize, use_sub, lam1, lam2, num_class):

    logging.info('Training network %s' % str(net_id))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logging.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logging.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                     weight_decay=weight_decay)
    elif optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                                     weight_decay=weight_decay,
                                     amsgrad=True)
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                                    weight_decay=weight_decay)

    set_requires_grad(g_mt, False)
    #set_requires_grad(l_mt, False)

    Prototype_t = get_teacher_prototype (num_class, train_dataloader, g_mt, device)

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_gloss_collector = []
        epoch_lloss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()

            optimizer.zero_grad()

            out = net(x)
            out_g = g_mt(x)

            out_g = out_g.detach()

            s_v = cal_cosine_similarity(out, Prototype_t)

            out_g = out_g * s_v

            b, c = out.shape

            out_softmax = F.softmax(out_g, dim=1)
            out_real = torch.zeros(b)
            out_real = out_real.to(device)
            for i in range(b):
                out_real[i] = out_softmax[i, target[i]]

            mask = out_real > (1/c)
            mask = mask.float()

            loss_ce = F.cross_entropy(out, target)
            loss_kd = torch.nn.KLDivLoss(reduction='none')(F.log_softmax(out/T, dim=1), F.softmax(out_g/T, dim=1)) * (T * T)
            loss_kd = torch.sum(loss_kd, dim=1)
            loss_kd *= mask
            loss_kd = loss_kd.mean()
            
            loss = (1-mu) * loss_ce + mu * loss_kd

            
            loss.backward()
            optimizer.step()
            epoch_loss_collector.append(loss.item())
            epoch_gloss_collector.append(loss_ce.item())
            epoch_lloss_collector.append(loss_kd.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_gloss = sum(epoch_gloss_collector) / len(epoch_gloss_collector)
        epoch_lloss = sum(epoch_lloss_collector) / len(epoch_lloss_collector)

        logging.info('Epoch: %d Loss: %f CELOSS: %f KDLOSS: %f'% (epoch, epoch_loss, epoch_gloss, epoch_lloss))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    set_requires_grad(g_mt, True)
    #set_requires_grad(l_mt, True)

    logging.info('>> Training accuracy: %f' % train_acc)
    logging.info('>> Test accuracy: %f' % test_acc)
    logging.info(' ** Training complete **')
    return net.state_dict()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset used for training')
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--partition', type=str, default='iid', help='the data partitioning strategy')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:3', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='resnet50', help='neural network used in training')
    parser.add_argument('--fl_method', type=str, default='fedavg')
    parser.add_argument('--T', type=float, default=4.0)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--alf', type=float, default=0.9)
    parser.add_argument('--kd_mode', type=str, default='cse')
    parser.add_argument('--use_mask', action='store_true', default=False)
    parser.add_argument('--use_normalize', action='store_true', default=False)
    parser.add_argument('--use_sub', action='store_true', default=False)
    parser.add_argument('--lam1', type=float, default=1)
    parser.add_argument('--lam2', type=float, default=1)
    args = parser.parse_args()
    return args

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta)

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    _, global_model = init_model(args.model, args.dataset, args.n_parties, device)

    g_mt_model = copy.deepcopy(global_model)
    l_mt_models = []
    for i in range(args.n_parties):
        l_mt_models.append(copy.deepcopy(global_model))
    
    g_mt_weight = g_mt_model.state_dict()
    l_mt_weights = []
    for i in range(args.n_parties):
        l_mt_weights.append(l_mt_models[i].state_dict())

    best_accuarcy = 0
    best_round = 0

    weight_save_dir = os.path.join(args.save_dir, args.partition, args.fl_method, str(os.getpid()))
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    if args.dataset == 'cifar100':
        num_class = 100
    elif args.dataset == 'cifar10':
        num_class = 10

    for round in range(1, args.comm_round):

        logging.info('----Communication Round: %d -----' % round)

        party_list_this_round = party_list_rounds[round-1]
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        local_params = {}

        for party in party_list_this_round:
            dataidxs = net_dataidx_map[party]
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs)
            if round > 0:
                param = train_epoch(party, copy.deepcopy(global_model), g_mt_model, l_mt_models[party], train_dl_local, test_dl_local, args.epochs, args.lr, args.optimizer, args.weight_decay, device, args.T, args.mu, args.kd_mode, args.use_mask, args.use_normalize,
                args.use_sub, args.lam1, args.lam2, num_class)
            else:
                param = train_fedavg(party, copy.deepcopy(global_model), train_dl_local, test_dl_local, args.epochs, args.lr, args.optimizer, args.weight_decay, device)

            local_params[party] = param

        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        global_w = global_model.state_dict()
        for id, party in enumerate(party_list_this_round):
            model_param = local_params[party]
            for key in model_param:
                if id == 0:
                    global_w[key] = model_param[key] * fed_avg_freqs[id]
                else:
                    global_w[key] += model_param[key] * fed_avg_freqs[id]

        global_model.load_state_dict(global_w)
        train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
        test_acc, conf_matrix, test_loss = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device)

        for k in g_mt_weight.keys():
            g_mt_weight[k] = g_mt_weight[k] * args.alf  + (1 - args.alf) * global_w[k]
        g_mt_model.load_state_dict(g_mt_weight)

        for id, party in enumerate(party_list_this_round):
            for key in l_mt_weights[party].keys():
                l_mt_weights[party][key] = l_mt_weights[party][key]* args.alf + (1 - args.alf) * local_params[party][key]
            l_mt_models[party].load_state_dict(l_mt_weights[party])

        if test_acc > best_accuarcy:
            best_accuarcy = test_acc
            best_round = round

        logging.info('>> Global Model Train accuracy: %f' % train_acc)
        logging.info('>> Global Model Test accuracy: %f' % test_acc)
        logging.info('>> Global Model Train loss: %f' % train_loss)
        logging.info('>> Global Model Test loss: %f' % train_loss)

        weight_save_path = os.path.join(weight_save_dir, 'checkpoint_{}.pth'.format(round))
        torch.save(global_model.state_dict(), weight_save_path)

    logging.info(' %d epoch get the best acc %f' % (best_round, best_accuarcy))

if __name__ == '__main__':
    args = get_args()
    main(args)