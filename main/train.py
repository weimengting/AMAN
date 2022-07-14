import os
import sys
sys.path.append('../')
import torch.backends.cudnn as cudnn

from main.config import parse_option
from main.network import AMAN
from dataset import load_dataset
from utils.util import *

opt = parse_option()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_number

def main():

    for i in range(27):
        train_loader, val_loader, num_test = load_dataset.generateIntoBatch(opt.batchsize, i)
        set_train(train_loader, val_loader, num_test, i)


def set_train(train_loader, val_loader, len_val, s_index):

    ''' Load model '''

    criterion = nn.CrossEntropyLoss()
    _structure = AMAN(opt.num_classes)
    model = _structure

    if torch.cuda.is_available():
        _structure = _structure.cuda()
        model = torch.nn.DataParallel(_structure).cuda()
        criterion = criterion.cuda()

    ''' Loss & Optimizer '''
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), opt.learning_rate,
                                    eps=1e-08, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)



    cudnn.enabled = True
    cudnn.benchmark = True
    ''' Train & Eval '''
    best_acc = 0
    temp_num = 0

    for epoch in range(opt.epochs):
        train(train_loader, model, optimizer, epoch, criterion)
        acc_number, len_val, pred, target = val(val_loader, model, len_val)
        if (acc_number / len_val) > best_acc:
            best_acc = (acc_number / len_val)
            temp_num = acc_number
            pp = []
            df = []
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for k in pred:
                pp.append(str(k[0]))
            for m in target:
                df.append(str(m))

        if best_acc == 1:
            break
        if best_acc == 0:
            pp = []
            df = []
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for k in pred:
                pp.append(str(k[0]))
            for m in target:
                df.append(str(m))

        lr_scheduler.step()

    with open(opt.record_path, 'a') as fp:
        fp.write(str(s_index) + ':' + '   num_of_acc:' + str(temp_num) +
                 '   num_of_samples:' + str(len_val) + '\n')
        fp.write('pred: ' + str(pp) + '   ')
        fp.write('target: ' + str(df) + '\n')

    fp.close()


def train(train_loader, model, optimizer, epoch, criterion):
    losses = AverageMeter()
    topframe = AverageMeter()

    # switch to train mode
    output_store_fc = []
    target_store = []

    model.train()
    for i, (feature, target) in enumerate(train_loader):

        target_var = target.cuda()
        input_var = feature.cuda()
        # compute output
        ''' model & full_model'''
        pred_score = model(input_var)
        loss = criterion(pred_score, target_var)
        loss = loss.sum()
        #
        output_store_fc.append(pred_score)
        target_store.append(target_var)

        # measure accuracy and record loss
        acc_iter = accuracy(pred_score.data, target_var, topk=(1,))
        losses.update(loss.item(), input_var.size(0))
        topframe.update(acc_iter[0], input_var.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1 == 0:
            print('Epoch: [{:3d}][{:3d}/{:3d}]\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc {topframe.val:.3f} ({topframe.avg:.3f})\t'
                .format(
                epoch, i, len(train_loader), loss=losses, topframe=topframe))



def val(val_loader, model, len_val):
    acc_number = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (feature, target) in enumerate(val_loader):
            # compute output
            target = target.to(DEVICE)
            input_var = feature.to(DEVICE)
            ''' model & full_model'''
            pred_score = model(input_var)
            _, pred = pred_score.topk(1, 1, True, True)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            for k in correct:
                if k:
                    acc_number += 1

    return acc_number, len_val, pred, target



if __name__ == '__main__':
    main()