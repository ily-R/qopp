import argparse
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from qutils import *
np.random.seed(seed=1)

PRINT_INTERVAL = 20

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def epoch(model, data, criterion, optimizer=None):
    model.eval() if optimizer is None else model.train()
    avg_loss = AverageMeter()
    avg_batch_time = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc2 = AverageMeter()

    tic = time.time()
    for i, (imgs, labels) in enumerate(data):
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(imgs)
            loss = criterion(pred, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc1, acc2 =accuracy(pred, labels, (1, 2))
        avg_acc1.update(acc1.item())
        avg_acc2.update(acc2.item())
        batch_time = time.time() - tic
        tic = time.time()

        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0:
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc1 {acc1.val:5.1f} ({acc1.avg:5.1f})\t'
                  'acc2 {acc2.val:5.1f} ({acc2.avg:5.1f})'.format(
                "EVAL" if optimizer is None else "TRAIN", i, len(data), batch_time=avg_batch_time, loss=avg_loss,
                acc1=avg_acc1, acc2=avg_acc2))

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg_acc1 {acc1.avg:5.2f} \t'
          'Avg_acc2 {acc2.avg:5.2f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        acc1=avg_acc1, acc2=avg_acc2))

    return avg_acc1.avg, avg_acc2.avg, avg_loss.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../Images', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=1e-6, type=float, metavar='LR', help='learning rate')
    parser.add_argument("--augment", help="perform data augmentation", action="store_true")

    args = parser.parse_args()
    mymodel = transferedModel(120)
    mymodel.to(device)

    optim = torch.optim.Adam(mymodel.parameters(), args.lr)

    if args.augment:
        img_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(17),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        img_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train, val, test = getDataloaders(args.batch_size, args.root, img_transforms=img_transforms)

    os.makedirs(os.path.join("Checkpoints"), exist_ok=True)
    os.makedirs(os.path.join("model_weight"), exist_ok=True)
    tb = SummaryWriter(os.path.join("runs"), flush_secs=20)

    starting_epoch = 0
    best_loss = 100000
    if os.path.exists(os.path.join("Checkpoints", 'training_state.pt')):
        checkpoint = torch.load(os.path.join("Checkpoints", 'training_state.pt'),  map_location = device)
        mymodel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']

    count_early_stopping = 0
    for e in range(starting_epoch, args.epochs):

        print("=================\n=== EPOCH " + str(e + 1) + " =====\n=================\n")

        avg_acc1, avg_acc2, loss = epoch(mymodel, train, myloss, optim)

        torch.save({
            'epoch': e,
            'model_state_dict': mymodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints", 'training_state.pt'))

        avg_acc1_val, avg_acc2_val, loss_val = epoch(mymodel, val, myloss)
        count_early_stopping += 1
        if loss_val < best_loss:
            count_early_stopping = 0
            best_loss = loss_val
            torch.save({
                'model_state_dict': mymodel.state_dict(),
                'loss_val': loss_val, 'acc1_val': avg_acc1_val,
                'acc2_val': avg_acc2_val,
                'loss': loss, 'acc1': avg_acc1, 'acc2': avg_acc2
            }, os.path.join("model_weight", 'best_weight.pt'))

        avg_acc1_test, avg_acc2_test, loss_test = epoch(mymodel, test, myloss)

        tb.add_scalars('Loss', {"train": loss, "val": loss_val, "test": loss_test}, e)
        tb.add_scalars('Accuracy Top-1', {"train": avg_acc1, "val": avg_acc1_val, "test": avg_acc1_test}, e)
        tb.add_scalars('Accuracy Top-2', {"train": avg_acc2, "val": avg_acc2_val, "test": avg_acc2_test}, e)

        if count_early_stopping >= 30:
            print("===========Early stopping==========\n No improvement in validation "
                  "loss after {} epochs".format(count_early_stopping))

        # if e < 100:
        #     optim.param_groups[0]['lr'] += (1e-4 - args.lr) / 100

    tb.close()
