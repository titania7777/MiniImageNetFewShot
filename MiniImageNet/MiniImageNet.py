import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from autoaugment import ImageNetPolicy
from torch.utils.data import DataLoader

class MiniImageNet(Dataset):
    def __init__(self, images_path, labels_path, setname='train', way=5, shot=1, query=15, augmentation=False, augment_rate=0.5):
        csv_path = os.path.join(labels_path, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        
        self.way = way
        self.shot = shot
        self.query = query
        self.augmentation = augmentation
        self.augment_rate = augment_rate

        self.datas = []
        self.labels = np.array([], dtype=int)
        counter = -1

        temp = []
        for line in lines:
            filename, label = line.split(',')
            self.datas.append(os.path.join(images_path, filename))
            if label not in temp:
                temp.append(label)
                counter += 1
            self.labels = np.append(self.labels, counter)
        
        self.indices = []
        for i in range(1, max(self.labels) + 1):
            self.indices.append(torch.from_numpy(np.argwhere(self.labels == i).reshape(-1)))

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # index sampling
        indices = []
        classes = torch.randperm(len(self.indices))[:self.way]
        for c in classes:
            l = self.indices[c]
            pos = torch.randperm(len(l))[:self.shot + self.query]
            indices.append(l[pos])
        indices = torch.stack(indices).t().reshape(-1)
        
        # image transform
        batchs = torch.zeros(self.way*self.shot + self.way*self.query, 3, 84, 84)
        # support transform
        for i, s in enumerate(indices[:self.way*self.shot]):
            if self.augment_rate > np.random.rand() and self.augmentation:
                batchs[i] = self.transform_autoaugment(Image.open(self.datas[s]).convert('RGB'))
            else:
                batchs[i] = self.transform(Image.open(self.datas[s]).convert('RGB'))
        
        # query transform
        for i, q in enumerate(indices[self.way*self.shot:]):
            batchs[i+self.way*self.shot] = self.transform(Image.open(self.datas[s]).convert('RGB'))

        return batchs, self.labels[indices]

trainset = MiniImageNet(
    images_path="../../datasets/mini_imagenet/images/", 
    labels_path="../../datasets/mini_imagenet/",
    setname='train',
    way=5,
    shot=5,
    query=15,
    augmentation=True,
    augment_rate=0.5
)
train_loader = DataLoader(dataset=trainset, batch_size=1, num_workers=8, pin_memory=True)

for i, (datas, labels) in enumerate(train_loader):
    print(labels.size())