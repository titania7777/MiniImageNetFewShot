## MiniImageNet
-------------
this MiniImageNet sampler is support to autoaugment[1] when training phase.

## MiniImageNet Options
1. images_path: raw images path
2. labels_path: labels path
3. mode: false is for general classification mode(this will be need to train feature extractor[2])and true is for episodic training strategy[3], (default: False)
4. setname: csv file name, (default: 'train')
5. way: number of way(number of class), (default: 5)
6. shot: number of shot(number of shot data), (default: 1)
7. query: number of query(number of query data), (default: 15)
8. augmentation: autoaugment mode, (default: False)
9. augment_rate: autoaugment rate, (default: 0.5)
## CategoriesSampler Options
1. dataset: implemented MiniImageNet class is need
2. iter_size: batch_size per iteration
3. batch_size: batch_size per episode
4. repeat: you can set the order of listing data, (default: False)
*repeat is true (ex. [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3],...])
*repeat is false (ex. [[1, 2, 3,...], [1, 2, 3,...], [1, 2, 3,...],...])

download MiniImageNet dataset.
```
wget http://hcir.iptime.org/mini_imagenet_raw.tar
```
train(example)
```
python train.py
```
## references
-------------
[1] Spyros Gidaris and Nikos Komodakis, "Dynamic few-shot visual learning without forgetting", Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4367-4375  
[2] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  
[3] Vinyals, Oriol and Blundell, Charles and Lillicrap, Timothy and kavukcuoglu, koray and Wierstra, Daan, "Matching Networks for One Shot Learning", Neural Information Processing Systems(NIPS), 2016, pp. 3630-3638

