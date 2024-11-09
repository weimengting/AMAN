import os
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

import network
from utils import denorm

"""
    magnify an example image here, we provide an example of magnifying with many amplification factors
"""


def load_network():
    load_ckpt = './learning_based_mag.pth.tar'
    # create model
    model = network.MagNet().cuda()
    # model  = torch.nn.DataParallel(model).cuda()

    # load checkpoint
    if os.path.isfile(load_ckpt):
        print("=> loading checkpoint '{}'".format(load_ckpt))
        checkpoint = torch.load(load_ckpt)

        # to load state_dict trained with DataParallel to model without DataParallel
        new_state_dict = OrderedDict()
        state_dict = checkpoint['state_dict']
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_ckpt, checkpoint['time']))
    else:
        print("=> no checkpoint found at '{}'".format(load_ckpt))
        assert (False)

    # cudnn enable
    cudnn.benchmark = True

    return model

def magnify_one_image():
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    x = np.linspace(2, 12, 32)
    model = load_network()
    model.eval()
    img1_path = './test_imgs/reg_img46.jpg'
    img2_path = './test_imgs/reg_img59.jpg'
    tt = transforms.Compose([transforms.Resize((384, 384)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    img1 = tt(img1)
    img2 = tt(img2)
    for amp_factor in x:
        mag = torch.from_numpy(np.array([amp_factor])).float()
        mag = mag.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        imga = img1.view(1, img1.shape[0], img1.shape[1], img1.shape[2]).cuda()
        imgb = img2.view(1, img2.shape[0], img2.shape[1], img2.shape[2]).cuda()
        mag = mag.cuda()

        y_hat, _, _ = model(imga, imgb, mag)
        y_hat = denorm(y_hat)
        tmp_factor = round(amp_factor, 2)
        save_image(y_hat.data, os.path.join(output_dir, str(tmp_factor) + '.jpg'))


if __name__ == '__main__':
    magnify_one_image()




