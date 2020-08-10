## UCF101
-------------
this UCF101 sampler utilize autoaugment[1] when scarcity of frame in dataset, and additionally you can check another options.

* this sampler is working for video action recognition !!
if you want few-shot action recognition(not traditional action recognition) then [here](https://github.com/titania7777/video_few_shot/tree/master/UCF101)

### common options
1. frames_path: extracted frames folder path(you need to use our frame extractor !!)
2. labels_path: labels folder path
3. list_number: splited task numbers(trainlist01 ~ 03 = 9537, 9586, 9624 | testlist01 ~ 03 = 3783, 3734, 3696)
4. frame_size: frame size(width and height is same)
5. sequence_length: number of video frames
6. train(default: True): train mode, if this mode is True then the sampler read a 'rainlist0x.txt' file to load dataset(False: test)
7. random_pad_sample(default: True): randomly sample from existing frames when frames are insufficient(False: only use first frame)
### pad options
8. pad_option(default: 'default'): augment option, there is two option('default', 'autoaugment')
9. uniform_frame_sample(default: True): uniformly sampled the frame(False: random normally)
### frame sampler options
10. random_start_position(default: True): randomly decides the starting point by considering the interval(False: 0)
11. max_interval(default: 7): maximum frame interval, this value is high then you may miss the sequence of video
12. random_interval(default: True): randomly decides the interval value(False: use maximum interval)


download UCF101 dataset.
```
wget http://hcir.iptime.org/UCF101.tar
```
extract frames from UCF101 videos.
```
python frame_extractor.py
```
train(example)
```
python train.py
```
## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 113-123