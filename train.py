import torch
import os
import sys
import argparse
import torch.nn.functional as F
from models import Model
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from MiniImageNet import MiniImageNet, CategoriesSampler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-path", type=str, default="../Data/mini_imagenet/images/")
    parser.add_argument("--labels-path", type=str, default="./labels/")
    parser.add_argument("--mode", type=bool, default=False)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--augmentation", type=bool, default=False)
    parser.add_argument("--augment-rate", type=float, default=0.5)
    parser.add_argument("--num-epochs-1", type=int, default=40)
    parser.add_argument("--num-epochs-2", type=int, default=200)
    parser.add_argument("--batch-size-1", type=int, default=128)
    parser.add_argument("--batch-size-2", type=int, default=1)
    parser.add_argument("--learning-rate-1", type=float, default=1e-3)
    parser.add_argument("--learning-rate-2", type=float, default=1e-3)
    parser.add_argument("--scheduler-step-size-1", type=int, default=20)
    parser.add_argument("--scheduler-step-size-2", type=int, default=20)
    parser.add_argument("--scheduler-gamma-1", type=float, default=0.9)
    parser.add_argument("--scheduler-gamma-2", type=float, default=0.9)
    args = parser.parse_args()

    print("=================================================")
    [print("{}:{}".format(arg, getattr(args, arg))) for arg in vars(args)]
    print("=================================================")
    
    # for train backbone with linear classifier or few-shot manners
    if not args.mode:
        train_dataset = MiniImageNet(
            images_path=args.images_path,
            labels_path=args.labels_path,
            mode=args.mode,
            setname='train',
            augmentation=args.augmentation,
            augment_rate=args.augment_rate,
        )

        val_dataset = MiniImageNet(
            images_path=args.images_path,
            labels_path=args.labels_path,
            mode=args.mode,
            setname='train',
            augmentation=False,
        )

        # split dataset
        train_dataset.datas, val_dataset.datas, train_dataset.labels, val_dataset.labels = train_test_split(train_dataset.datas, train_dataset.labels, train_size=0.7)

        # data loader for train backbone
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size_1, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size_1, shuffle=False, num_workers=4, pin_memory=True)

    # for fine-tune or train from the scracth with few-shot manners
    few_shot_train_dataset = MiniImageNet(
        images_path=args.images_path,
        labels_path=args.labels_path,
        mode=True,
        setname='train',
        way=args.way,
        shot=args.shot,
        query=args.query,
        augmentation=args.augmentation,
        augment_rate=args.augment_rate,
    )

    few_shot_val_dataset = MiniImageNet(
        images_path=args.images_path,
        labels_path=args.labels_path,
        mode=True,
        setname='val',
        way=args.way,
        shot=args.shot,
        query=args.query,
        augmentation=False,
    )

    few_shot_train_sampler = CategoriesSampler(few_shot_train_dataset, 100, args.batch_size_2, repeat=False)
    few_shot_val_sampler = CategoriesSampler(few_shot_val_dataset, 200, args.batch_size_2, repeat=False)

    # data loader for fine-ture or train from the scratch with few-shot manners
    few_shot_train_loader = DataLoader(dataset=few_shot_train_dataset, batch_sampler=few_shot_train_sampler, num_workers=4, pin_memory=True)
    few_shot_val_loader = DataLoader(dataset=few_shot_val_dataset, batch_sampler=few_shot_val_sampler, num_workers=4, pin_memory=True)

    model = Model(
        mode=args.mode,
        num_classes=None if args.mode else train_dataset.num_classes,
        way=args.way,
        shot=args.shot,
        query=args.query,
        attention=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not args.mode:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_1)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_1, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size_1, gamma=args.scheduler_gamma_1)
        
        best = 0
        for e in range(1, args.num_epochs_1+1):
            train_acc = []
            train_loss = []

            model.train()
            for i, (datas, labels) in enumerate(train_loader):
                datas, labels = datas.to(device), labels.to(device).type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
                
                pred = model(datas, linear=True)

                loss = F.cross_entropy(pred, labels)
                train_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
                train_acc.append(acc)
                
                printer("train classifier", e, args.num_epochs_1, i+1, len(train_loader), loss.item(), sum(train_loss)/len(train_loss), acc, sum(train_acc)/len(train_acc))

            print("")
            val_acc = []
            val_loss = []
            model.eval()
            for i, (datas, labels) in enumerate(val_loader):
                datas, labels = datas.to(device), labels.to(device).type(torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor)
                
                pred = model(datas, linear=True)
                
                loss = F.cross_entropy(pred, labels)
                val_loss.append(loss.item())
                
                acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
                val_acc.append(acc)
                
                printer("val classifier", e, args.num_epochs_1, i+1, len(val_loader), loss.item(), sum(val_loss)/len(val_loss), acc, sum(val_acc)/len(val_acc))
            if sum(val_acc)/len(val_acc) > best:
                best = sum(val_acc)/len(val_acc)
            print(" Best: {:.2f}%".format(best))
            lr_scheduler.step()
    
    # few-shot
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_2, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size_2, gamma=args.scheduler_gamma_2)
    
    best = 0
    for e in range(1, args.num_epochs_2+1):
        few_shot_train_acc = []
        few_shot_train_loss = []
        model.train()
        for i, (datas, _) in enumerate(few_shot_train_loader):
            datas = datas.to(device)
            labels = torch.arange(args.way).repeat(args.query*args.batch_size_2).to(device)

            pred = model(datas, linear=False)

            loss = F.cross_entropy(pred, labels)
            few_shot_train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            few_shot_train_acc.append(acc)

            printer("train few-shot", e, args.num_epochs_2, i+1, len(few_shot_train_loader), loss.item(), sum(few_shot_train_loss)/len(few_shot_train_loss), acc, sum(few_shot_train_acc)/len(few_shot_train_acc))

        print("")
        few_shot_val_acc = []
        few_shot_val_loss = []
        model.eval()
        for i, (datas, _) in enumerate(few_shot_val_loader):
            datas = datas.to(device)
            labels = torch.arange(args.way).repeat(args.query*args.batch_size_2).to(device)

            pred = model(datas, linear=False)

            loss = F.cross_entropy(pred, labels)
            few_shot_val_loss.append(loss.item())

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            few_shot_val_acc.append(acc)

            printer("val few-shot", e, args.num_epochs_2, i+1, len(few_shot_val_loader), loss.item(), sum(few_shot_val_loss)/len(few_shot_val_loss), acc, sum(few_shot_val_acc)/len(few_shot_val_acc))
        
        if sum(few_shot_val_acc)/len(few_shot_val_acc) > best:
            best = sum(few_shot_val_acc)/len(few_shot_val_acc)
            torch.save(model.state_dict(), "./save/best.pth")
        print(" Best: {:.2f}%".format(best))

        lr_scheduler.step()
