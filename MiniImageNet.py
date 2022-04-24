import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from autoaugment import ImageNetPolicy
from sklearn.model_selection import train_test_split

class MiniImageNet(Dataset):
    def __init__(self, images_path, labels_path, mode=False, setname='train', way=5, shot=1, query=15, augmentation=False, augment_rate=0.5):
        assert os.path.exists(images_path), "threre is no directory {}".format(images_path)
        assert os.path.exists(labels_path), "there is no directory {}".format(labels_path)

        self.mode = mode
        self.way = way
        self.shot = shot
        self.query = query
        self.augmentation = augmentation
        self.augment_rate = augment_rate

        # static settings
        self.channel = 3
        self.size = 84

        self.datas = []
        self.labels = np.array([])
        self.num_classes = -1
        with open(os.path.join(labels_path, setname + ".csv")) as f:
            # remove first head
            lines = f.readlines()[1:]
            temp = []
            for line in lines:
                filename, label = line.strip().split(',')
                self.datas.append(os.path.join(images_path, setname, label, filename))
                if label not in temp:
                    temp.append(label)
                    self.num_classes += 1
                self.labels = np.append(self.labels, self.num_classes)
            self.labels = torch.from_numpy(self.labels)
        self.num_classes += 1

        # default transform
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # autoaugmentation transform
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.mode:
            datas = torch.zeros(len(index) * (self.shot + self.query), self.channel, self.size, self.size)
            labels = torch.zeros(len(index) * (self.shot + self.query))
            # ways
            for p1 in range(len(index)):
                # shots
                for p2, s in enumerate(index[p1]):
                    if self.augmentation and self.augment_rate > torch.rand(1):
                        datas[p1 * (self.shot + self.query) + p2] = self.transform_autoaugment(Image.open(self.datas[s]).convert('RGB'))
                    else:
                        datas[p1 * (self.shot + self.query) + p2] = self.transform(Image.open(self.datas[s]).convert('RGB'))
                    labels[p1 * (self.shot + self.query) + p2] = self.labels[s]
        else:
            if self.augmentation and self.augment_rate > torch.rand(1):
                datas = self.transform_autoaugment(Image.open(self.datas[index]).convert('RGB'))
            else:
                datas = self.transform(Image.open(self.datas[index]).convert('RGB'))
            labels = self.labels[index]
        return datas, labels

class CategoriesSampler():
    def __init__(self, dataset, iter_size, batch_size, repeat = False):
        self.iter_size = iter_size
        self.batch_size = batch_size
        self.repeat = repeat
        
        self.way = dataset.way
        self.shot = dataset.shot
        self.query = dataset.query
        self.labels = dataset.labels

        # compute episode_size (ex. iter: 100, batch: 4 => total 400 episodes)
        self.episode_size = self.iter_size * self.batch_size
        
        # reconstruct labels
        temp = []
        for i in range(int(max(self.labels)) + 1):
            temp.append(torch.where(self.labels == i)[0])
        self.labels = temp

    def __len__(self):
        return self.iter_size

    def __iter__(self):
        for i in range(self.iter_size):
            batchs = []
            for b in range(self.batch_size):
                way = torch.randperm(len(self.labels))[:self.way]
                ways = []
                for w in way:
                    shot = self.labels[w]
                    shot = shot[torch.randperm(len(shot))]
                    ways.append(shot[:self.shot + self.query])
                # repeat datas (ex. [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3],...])
                # no repeat datas (ex. [[1, 2, 3,...], [1, 2, 3,...], [1, 2, 3,...],...])
                if self.repeat:
                    batchs.append(ways)
                else:
                    ways_stacked = torch.stack(ways)
                    ways_stacked_shape = ways_stacked.shape
                    batchs.append(ways_stacked.t().reshape(ways_stacked_shape))
            yield batchs
