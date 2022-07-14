import torch
import torch.utils.data as data
import os
import numpy as np
import sys
sys.path.append("../")
from main.config import parse_option

opt = parse_option()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_number



def leave_one_subject(s_index=0):
    '''
        :param s_index: index of the subject for testing under the LOSO protocol
        :param num: number of frames selected, integrated into features
        :return: subjects list for training, testing; number of samples for testing
    '''
    root = opt.feature_directory
    subjects = sorted(os.listdir(root))
    test = subjects[s_index]
    train = []
    for i in range(len(subjects)):
        if i != s_index:
            train.append(subjects[i])
    num_test = len(os.listdir(os.path.join(root, test)))
    return train, test, num_test


class GetFeatures(data.Dataset):
    def __init__(self, root=opt.feature_directory,
                       train_list=None, test_subject=None, phase='train'):

        self.img_list = []
        self.root = root
        self.train_list = train_list
        self.test_subject = test_subject
        self.phase = phase

        if self.phase == 'train':
            for subject in sorted(os.listdir(self.root)):
                if subject in self.train_list:
                    subject_path = os.path.join(self.root, subject)
                    files = sorted(os.listdir(subject_path))
                    for file in files:
                        video_path = os.path.join(subject_path, file)
                        self.img_list.append(video_path)

        elif self.phase == 'val':
            for subject in sorted(os.listdir(self.root)):
                if subject == self.test_subject:
                    subject_path = os.path.join(self.root, subject)
                    files = sorted(os.listdir(subject_path))
                    for file in files:
                        video_path = os.path.join(subject_path, file)
                        self.img_list.append(video_path)


    def __getitem__(self, item):
        imgs = os.listdir(self.img_list[item])
        vs = []
        for integrated_imgs in imgs:
            cur_img_path = os.path.join(self.img_list[item], integrated_imgs)
            feature = np.load(os.path.join(cur_img_path, os.listdir(cur_img_path)[0]))
            feature = torch.from_numpy(feature)
            vs.append(feature)
        vs = torch.stack(vs, dim=0)

        target = int(self.img_list[item].split('\\')[-1][-1])
        return vs, target

    def __len__(self):
        return len(self.img_list)

def generateIntoBatch(batchsize_train, s_index):
    '''
        :param batchsize_train: batchsize for training
        :param s_index: index of the subject for testing under the LOSO protocol
        :param num: number of frames selected, integrated into features
        :return: train_loader, test_loader, number of samples for testing
    '''
    train, test, num_test = leave_one_subject(s_index)
    print('the left subject for testing is ' + test)
    train_dataset = GetFeatures(train_list=train, test_subject=test,
                                  phase='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize_train, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=False)

    val_dataset = GetFeatures(train_list=train, test_subject=test,
                                phase='val')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=num_test, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False)

    return train_loader, val_loader, num_test


if __name__ == '__main__':
    # (btz, 10, 9, 512)
    train, val, num_sample = leave_one_subject(0)
    features = GetFeatures(train_list=train, test_subject=val)
    train_loader, val_loader, num_test = generateIntoBatch(4, 0)
    for i, (features, target) in enumerate(train_loader):
        print(features.shape)
        print(features)
        print(target)
