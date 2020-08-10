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
    def __init__(self, 
        frames_path, labels_path, list_number, frame_size, sequence_length, train=True,
        random_pad_sample=True, pad_option='default', 
        uniform_frame_sample=True, random_start_position=True, max_interval=7, random_interval=True):

        self.sequence_length = sequence_length

        # pad option
        self.random_pad_sample = random_pad_sample
        self.pad_option = pad_option

        # frame sampler option
        self.uniform_frame_sample = uniform_frame_sample
        self.random_start_position = random_start_position
        self.max_interval = max_interval
        self.random_interval = random_interval

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
            transforms.Resize((frame_size, frame_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # transformer autoaugment for insufficient frames
        self.transform_autoaugment = transforms.Compose([
            transforms.Resize((frame_size, frame_size), Image.BICUBIC),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path, random_pad_sample, pad_option):
        # random pad sample
        sequence = np.arange(len(sorted_frames_path))
        if random_pad_sample:
            add_sequence = np.random.choice(sequence, self.sequence_length - len(sequence))
        else:
            add_sequence = np.repeat(sequence[0], self.sequence_length - len(sequence))
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        # attach the pads
        datas = []
        temp = None
        for s in sequence:
            if temp == s:
                # you can add another options in here
                if pad_option == 'default':
                    datas.append(self.transform(Image.open(sorted_frames_path[s])))
                elif pad_option == 'autoaugment':
                    datas.append(self.transform_autoaugment(Image.open(sorted_frames_path[s])))
                else:
                    raise ValueError("{} option is invalid".format(pad_option))
            else:
                datas.append(self.transform(Image.open(sorted_frames_path[s])))
            temp = s
        return datas

    def _frame_sampler(self, sorted_frames_path, uniform_frame_sample, random_start_position, max_interval, random_interval):
        sorted_frames_length = len(sorted_frames_path)

        interval = sorted_frames_length // self.sequence_length
        interval = 1 if interval is 0 else interval
        
        # max interval 
        interval = max_interval if interval >= max_interval else interval

        # random interval
        if random_interval:
            interval = np.random.permutation(np.arange(start=1, stop=interval + 1))[0]

        # random start position
        if random_start_position:
            start_position = np.random.randint(0, sorted_frames_length - (interval * self.sequence_length) + 1)
        else:
            start_position = 0

        # sample frames
        datas = []
        if uniform_frame_sample:
            for i, s in enumerate(range(start_position, sorted_frames_length, interval)):
                datas.append(self.transform(Image.open(sorted_frames_path[s])))
                if i >= self.sequence_length - 1:
                    break
        else:
            sequence = sorted(np.random.permutation(np.arange(sorted_frames_length))[:self.sequence_length])
            for s in sequence:
                datas.append(self.transform(Image.open(sorted_frames_path[s])))
        return datas
    
    def __getitem__(self, index):
        data_path = self.data_paths[index % len(self)]
        sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
        sorted_frames_length = len(sorted_frames_path)
        assert sorted_frames_length is not 0, "please ensure that '{}' path is exist or not".format(data_path)
        
        if sorted_frames_length < self.sequence_length:
            datas = self._add_pads(sorted_frames_path, random_pad_sample=self.random_pad_sample, pad_option=self.pad_option)
        else:
            datas = self._frame_sampler(sorted_frames_path, uniform_frame_sample=self.uniform_frame_sample, 
                    random_start_position=self.random_start_position, max_interval=self.max_interval, random_interval=self.random_interval)
        
        datas = torch.stack(datas)
        labels = self.labels[data_path.split("_")[-3]]
        return datas, labels
