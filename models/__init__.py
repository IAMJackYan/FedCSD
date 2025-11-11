import copy
import logging

from .resnet import ResNet18, ResNet50
from .vggnet import vgg11
from .resnet_prototype import ResNet50_pro
from .resnet_proto import ResNet50_2stage

model_factory = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'vgg11': vgg11,
    'resnet50_pro':ResNet50_pro,
    'resnet50_2stage': ResNet50_2stage
}

def init_model(net_name, dataset_name, client_nums, device):
    assert net_name in model_factory.keys(), logging.info('please check your net name')
    in_chans = 3
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    elif dataset_name == 'tinyimagenet':
        num_classes = 200
    elif dataset_name == 'FEMNIST':
        num_classes = 10
        in_chans = 1
    elif dataset_name == 'emnist':
        num_classes = 62
        in_chans = 1

    if net_name.startswith('resnet'):
        global_model = model_factory[net_name](num_classes=num_classes, in_chans=in_chans)
    elif net_name.startswith('vgg'):
        global_model = model_factory[net_name](num_classes=num_classes)
    global_model = global_model.to(device)

    #global_model = ResNet50_clip(num_classes, device=device)

    local_models = []
    for _ in range(client_nums):
        if net_name.startswith('resnet'):
            model = model_factory[net_name](num_classes=num_classes, in_chans=in_chans)
        elif net_name.startswith('vgg'):
            model = model_factory[net_name](num_classes=num_classes)
        #model = ResNet50_clip(num_classes, device=device)
        local_models.append(model.to(device))

    return local_models, global_model