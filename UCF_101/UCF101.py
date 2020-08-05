import os
import time
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from autoaugment import ImageNetPolicy


class UCF101(Dataset):
    def __init__(self, frames_path, labels_path, list_number, frame_size, sequence_length, train=True, random_sample_frame=True, interval=True):
        self.sequence_length = sequence_length
        self.random_sample_frame = random_sample_frame
        self.interval = interval
        self.train = train
        # make path list of data
        self.data_paths = []
        assert list_number in [1, 2, 3], "list_number need to be one of 1, 2, 3"
        listfilename = "trainlist0" + str(list_number) + ".txt" if train else "testlist0" + str(list_number) + ".txt"
        with open(os.path.join(labels_path, listfilename)) as f:
            lines = f.readlines()
        for line in lines:
            frame_name = line.split("/")[1].split(".avi")[0]
            self.data_paths.append(os.path.join(frames_path, frame_name))

        # make labels dictionary
        self.labels = {}
        with open(os.path.join(labels_path, "classInd.txt")) as f:
            lines = f.readlines()
        for line in lines:
            number, action = line.split()
            self.labels[action] = int(number) - 1
        
        self.num_classes = len(self.labels)

        # transformer default
        self.transform = transforms.Compose([
            transforms.Resize([frame_size, frame_size], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # transformer autoaugment for insufficient frames
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize([frame_size, frame_size], Image.BICUBIC),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        data_path = self.data_paths[index % len(self)]
        sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("/")[-1]))
        sorted_frames_length = len(sorted_frames_path)
        assert sorted_frames_length is not 0, "please ensure that '{}' path is exist or not".format(data_path)
        datas = []
        if self.train and sorted_frames_length < self.sequence_length:
            sequence = np.arange(sorted_frames_length)
            add_sequence = np.random.choice(sequence, self.sequence_length - len(sequence))
            sequence = sorted(np.append(sequence, add_sequence, axis=0))

            temp = None
            for s in sequence:
                if temp == s:
                    datas.append(self.transform_autoaugment(Image.open(sorted_frames_path[s])))
                else:
                    datas.append(self.transform(Image.open(sorted_frames_path[s])))
                temp = s
        elif self.random_sample_frame:
            if self.interval:
                interval = np.random.randint(1, (sorted_frames_length // self.sequence_length) + 1)
                start = np.random.randint(0, sorted_frames_length - (interval * self.sequence_length + 1))
                for i, s in enumerate(range(start, sorted_frames_length, interval)):
                    datas.append(self.transform(Image.open(sorted_frames_path[s])))
                    if i >= self.sequence_length-1:
                        break
            else:
                sequence = sorted(np.random.permutation(np.arange(sorted_frames_length))[:self.sequence_length])
                for s in sequence:
                    datas.append(self.transform(Image.open(sorted_frames_path[s])))
        else:
            interval = sorted_frames_length // self.sequence_length
            start = 0
            for i, s in enumerate(range(start, sorted_frames_length, interval)):
                datas.append(self.transform(Image.open(sorted_frames_path[s])))
                if i >= self.sequence_length - 1:
                    break

        datas = torch.stack(datas)
        labels = self.labels[data_path.split("_")[-3]]
        return datas, labels