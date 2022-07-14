import argparse
import math

def parse_option():
    parser = argparse.ArgumentParser('arguments for training the AMAN')

    parser.add_argument('--batchsize', type=int, default=4,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='total training epochs')
    parser.add_argument('--dataset', type=str, default='SAMM',
                        help='training on which dataset, chosen from SMIC-HS, SAMM and CASME II')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='number of classes on the training dataset')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='learning rate')
    parser.add_argument('--dataset_path', type=str, default='..',
                        help='the root directory of the dataset')
    parser.add_argument('--record_path', type=str, default='../record',
                        help='path to save the record')
    parser.add_argument('--CUDA_number', type=str, default='0, 1, 2, 3',
                        help='select the number of graphics card for training')
    parser.add_argument('--model_directory', type=str, default='../Resnet18_FER+_pytorch.pth.tar',
                        help='the path to the pre-trained Resnet-18 model')
    parser.add_argument('--images_directory', type=str, default='..',
                        help='the path to the magnified images')
    parser.add_argument('--feature_directory', type=str, default='../samm_magnified_features',
                        help='the path to the extracted features')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_option()
    print(opt)