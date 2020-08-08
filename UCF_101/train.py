import torch
import sys
import argparse
import torch.nn.functional as F
from models import Model
from torch.utils.data import DataLoader
from torch.autograd import Variable
from UCF101 import UCF101

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_path", type=str, default="../datas/UCF101_frames/")
    parser.add_argument("--labels_path", type=str, default="../datas/UCF101_labels/")
    parser.add_argument("--list_number", type=int, default=1)
    parser.add_argument("--frame_size", type=str, default=224)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=40)
    args = parser.parse_args()

    train_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        list_number=args.list_number,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        train=True,
        # pad option
        random_pad_sample=True,
        pad_option='autoaugment',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=True,
        max_interval=7,
        random_interval=True,
    )

    test_dataset = UCF101(
        frames_path=args.frames_path,
        labels_path=args.labels_path,
        list_number=args.list_number,
        frame_size=args.frame_size,
        sequence_length=args.sequence_length,
        train=False,
        # pad option
        random_pad_sample=False,
        pad_option='default',
        # frame sampler option
        uniform_frame_sample=True,
        random_start_position=False,
        max_interval=7,
        random_interval=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    model = Model(
        num_classes=train_dataset.num_classes,
        num_layers=1,
        hidden_size=1024,
        bidirectional=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    best = 0
    for e in range(1, args.num_epochs+1):
        train_acc = []
        train_loss = []
        for i, (datas, labels) in enumerate(train_loader):
            model.train()
            datas, labels = datas.to(device), labels.to(device)
            
            model.init_hidden()
            pred = model(datas)

            loss = F.cross_entropy(pred, labels)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            train_acc.append(acc)

            sys.stdout.write(
                "\rtrain-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.2f} (mean: {:.2f}), Acc: {:.2f}% (mean: {:.2f}%)]".format
                (
                    e,
                    args.num_epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                    sum(train_loss)/len(train_loss),
                    acc,
                    sum(train_acc)/len(train_acc),
                )
            )
        
        test_acc = []
        test_loss = []
        for i, (datas, labels) in enumerate(test_loader):
            model.eval()
            datas, labels = datas.to(device), labels.to(device)
            
            model.init_hidden()
            pred = model(datas)

            loss = F.cross_entropy(pred, labels)
            test_loss.append(loss.item())

            acc = 100 * (pred.argmax(1) == labels).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor).mean().item()
            test_acc.append(acc)

            sys.stdout.write(
                "\rtest-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.2f} (mean: {:.2f}), Acc: {:.2f}% (mean: {:.2f}%), Best: {:.2f}%]".format
                (
                    e,
                    args.num_epochs,
                    i,
                    len(test_loader),
                    loss.item(),
                    sum(test_loss)/len(test_loss),
                    acc,
                    sum(test_acc)/len(test_acc),
                    best
                )
            )
        
        if sum(test_acc)/len(test_acc) > best:
            best = sum(test_acc)/len(test_acc)
        
        lr_scheduler.step()
