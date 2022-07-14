import torch
import torch.nn as nn
import torch.nn.init as init


# Use the parameters of the pre-trained Resnet-18, keeping only the conv layers
# Note the difference in loading model when training by dataparallel
def model_parameters(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)

    return _structure




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (256 * 1) -> (1 * 256)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)




def _initialize_weights(self):

    for m in self.modules():

        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
