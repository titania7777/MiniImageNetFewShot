# Pytorch_Sampler

## UCF101
-------------
options
1. frames_path = extracted frames folder path(you need to use our frame extractor !!)
2. labels_path = labels folder path
3. list_number = splited task numbers(trainlist01 ~ 03 = 9537, 9586, 9624 | testlist01 ~ 03 = 3783, 3734, 3696)
4. frame_size = frame size(width and height is same)
5. train = train mode, if this mode is True then the sampler read a trainlist text file to load dataset
6. random_sample = (when interval mode is True)set interval and start position randomly, (when interval mode is False)sampled point without interval
7. interval = choose uniform sample or not

if you want use this UCF101 sampler then please follow this commands.
download UCF101 dataset.
```
wget http://hcir.iptime.org/UCF101.tar
```
extract frames from UCF101 videos.
```
python frame_extractor.py
```
example(default)
```
from UCF101 import UCF101
train_dataset = UCF101(
    frames_path='./datas/UCF_frames/',
    labels_path='./datas/UCF_labels/',
    list_number=1,
    frame_size=224,
    sequence_length=80,
    train=True,
    random_sample_frame=True,
    interval=False,
)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
```
## references
-------------
[1] matchingnet...
[2] autoaugment...