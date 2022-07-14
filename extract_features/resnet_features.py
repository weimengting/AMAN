"""
    In order to prevent over-fitting and to reduce computational cost, we extract the features of magnified
    frames, so during training, the actual input is features
"""

import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from utils import util

sys.path.append('../')

from extract_features.resnet import resnet18_at


transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def load_model():
    _structure = resnet18_at()
    _parameterDir = 'F:/trained_models/Resnet18_FER+_pytorch.pth.tar'
    model = util.model_parameters(_structure, _parameterDir)
    model = torch.nn.DataParallel(model).cuda()

    return model



def produce_features():
    img_path = 'F:\datasets\micro_expression\SAMM\some_changes\samm_magnified'
    dst_path = 'F:\datasets\micro_expression\SAMM\some_changes\samm_magnified_features2'
    model = load_model()
    subjects = os.listdir(img_path)
    for subject in subjects:
        expressions_path = os.path.join(img_path, subject)
        expressions = os.listdir(expressions_path)
        for expression in expressions:
            imgs_path = os.path.join(expressions_path, expression)
            imgs = sorted(os.listdir(imgs_path))
            for img in imgs:
                cur_img_path = os.path.join(imgs_path, img)
                factors = sorted(os.listdir(cur_img_path))
                vs = []
                for factor in factors:
                    cur_factor = Image.open(os.path.join(cur_img_path, factor))
                    cur_factor = transform(cur_factor)
                    vs.append(cur_factor)
                vs_stack = torch.stack(vs, dim=0)
                rep = model(vs_stack)
                temp_path = os.path.join(dst_path, subject, expression, img)
                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                temp_features = rep.detach().cpu().numpy()
                print(temp_features.shape)
                np.save(os.path.join(temp_path, 'feature512.npy'), temp_features)
            # vs = []
            # for img in imgs:
            #     cur_img = Image.open(os.path.join(imgs_path, img))
            #     cur_img = transform(cur_img)
            #     vs.append(cur_img)
            # vs_stack = torch.stack(vs, dim=0)
            # rep = model(vs_stack)
            # temp_path = os.path.join(dst_path, subject, expression)
            # if not os.path.exists(temp_path):
            #     os.makedirs(temp_path)
            # temp_features = rep.detach().cpu().numpy()
            # print(temp_features.shape)
            # np.save(os.path.join(temp_path, 'feature512.npy'), temp_features)

if __name__ == '__main__':
    produce_features()
    # btz, frames, feature_dim
    # load_model()