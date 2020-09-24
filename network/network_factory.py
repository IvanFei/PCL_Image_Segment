
import importlib
import torch

from config import cfg


def get_net(args, criterion, cuda=False):
    """
    Get Network Architecture based on args.
        e.g. args.arch = deeplabv3.DeepV3Plus
    """
    network = "network." + args.arch
    net = get_model(network, num_classes=cfg.DATASET.NUM_CLASSES, criterion=criterion)
    num_params = sum([param.nelement() for param in net.parameters()])
    print("[*] Model {} params = {:2.1f}M".format(network, num_params / 1000000))

    if cuda:
        net.cuda()

    return net


def get_model(network, num_classes, criterion=None):
    """
    Fetch Network Function Pointer
        e.g. network = "network.deeplabv3.DeepV3Plus"
    """
    module = network[:network.rfind(".")]
    model = network[network.rfind(".") + 1: ]
    mod = importlib.import_module(module)
    net_func = getattr(mod, model)
    net = net_func(num_classes=num_classes, criterion=criterion)

    return net


if __name__ == '__main__':
    import easydict as edict
    args = {"arch": "deeplabv3.DeepV3PlusX71", "num_classes": 80}
    args = edict.EasyDict(args)

    print(args.arch, args.num_classes)
    net = get_net(args, criterion=None)
    print(net.state_dict().keys())


