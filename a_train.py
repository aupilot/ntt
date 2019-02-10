import sys
import time
import os.path
import argparse
from multiprocessing import freeze_support
import torch.utils.data as data
from torchvision.models import resnet18

from metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchsummary import summary
from dataloader import NttDataset, NttDataset2, NttDataset3
from logger import Logger
from net_resnet import SuperNet740
from net_resnet_light import resnet_light
from net_simple import CNN1

# export CUDA_VISIBLE_DEVICES=0; python3 a_train.py -f0 -t3 -e45

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-f', '--fold', type=int, default=0)
# parser.add_argument('-v', '--val_fold', type=int, default=1)
parser.add_argument('-t', '--total_folds', type=int, default=3)
parser.add_argument('-e', '--epochs', type=int, default=40)
parser.add_argument('-d', '--dataset', type=int, default=2)
args = parser.parse_args()


# === Parameters ===
resume_from = None
learning_rate_sgd = 0.002
learning_rate_adam = 2e-4
input_depth = 1
validation_size = 512


data_dir = "/Volumes/KProSSD/Datasets/ntt/"
# data_dir = "./data"
if not os.path.isdir(data_dir):
    # windows
    data_dir = "D:/Datasets/ntt/"

params_train = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}
params_valid = {'batch_size': 128,
          'shuffle': False,
          'num_workers': 4}

if args.dataset == 2:
    MyDataSet = NttDataset2
elif args.dataset == 3:
    MyDataSet = NttDataset3
else:
    raise NotImplementedError

training_set = MyDataSet(folds_total=args.total_folds,
                          root_dir=data_dir,
                          chunk_exclude=args.fold,
                          validation=False,
                          )
training_generator = data.DataLoader(training_set, **params_train)

validation_set = Subset(MyDataSet(folds_total=args.total_folds,
                                   root_dir=data_dir,
                                   chunk_exclude=args.fold,
                                   validation=True,),
                        range(validation_size))
validation_generator = data.DataLoader(validation_set, **params_valid)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if resume_from is None:
    if args.dataset == 2:
        # cnn = CNN1(input_depth=input_depth)
        cnn = SuperNet740(input_depth=input_depth)
    elif args.dataset == 3:
        # cnn = resnet18(pretrained=False)
        # cnn.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 1024, kernel_size=2, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(inplace=True)
        # )
        # num_ftrs = 1024 # cnn.fc.in_features
        # cnn.fc = nn.Sequential(
        #     nn.Linear(num_ftrs, 6),
        #     nn.LogSoftmax(dim=1)
        # )
        cnn = resnet_light()

    cnn.to(device)
    resume_from = 0
else:
    # resume training from saved checkpoint
    cnn = torch.load(resume_from, map_location=device)

log_prefix = time.strftime("%m%d-%H%M", time.localtime())
if args.dataset == 2:
    log_prefix += cnn.name
else:
    log_prefix += '2D'
log_prefix += f'_fold_{args.fold}'
logger = Logger('./logs/{}'.format(log_prefix))

# optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate_sgd, momentum=0.9, weight_decay=0.00005, nesterov=True)
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate_adam, weight_decay=0.00005)
scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.2)
criterion = nn.NLLLoss()
# print("WARNING: make sure that the NN model has softmax!!!")

# create a dir for new model checkpoints
save_dir = './save/{}/'.format(log_prefix)
os.makedirs(save_dir, exist_ok=True)

# write model description in plain text
with open(save_dir+'_summary.txt', 'w') as sys.stdout:
    if args.dataset == 2:
        print('CNN Type: {}'.format(cnn.name))
        print(cnn)
        print(optimizer)
        summary(cnn, (input_depth, 4000))
    else:
        print('2D Standard ResNet')
        print(cnn)
        print(optimizer)
        # summary(cnn, (3,128,128))   # uncomment for a proper ResNet with 3 channel input
        summary(cnn, (1, 128, 128))  # uncomment for a proper ResNet with 3 channel input
sys.stdout = sys.__stdout__

def train():
    # === Train the Model ===
    prev_val_loss = 9999.0
    tick = time.time()
    # Loop over epochs
    for epoch in range(resume_from, args.epochs):

        # === Training
        cnn.train()  # Change model to train mode
        train_total = 0
        i = 0
        for local_batch, local_labels in training_generator:

            # Transfer to GPU
            batch, labels = local_batch.to(device).type(torch.cuda.FloatTensor), local_labels.to(device)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(batch)
            loss = criterion(outputs, labels)    # label class must be of type long (on Windows) - add  .long()
            loss_np = loss.item()

            train_total = train_total+loss_np
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, args.epochs, i + 1, len(training_generator), loss_np))

            i = i+1

        train_total = train_total / len(training_generator)
        tock = time.time()
        print('One Epoch time: {:.1f} seconds.'.format(tock - tick))
        tick = time.time()

        # === Validation
        cnn.eval()  # Change model to 'eval' mode
        total = 0.0
        totf1 = 0.0
        one_hot_converter = torch.eye(6, device=device)
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                batch, labels = local_batch.to(device).type(torch.cuda.FloatTensor), local_labels.to(device)

                # Model computations
                outputs = cnn(batch)
                loss_val = criterion(outputs, labels)
                f1_val = f1_score(one_hot_converter[labels], torch.exp(outputs))
                total += loss_val.item()
                totf1 += f1_val.item()

        total = total / len(validation_generator)
        totf1 = totf1 / len(validation_generator)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total)
        else:
            scheduler.step(epoch=epoch)

        print(f'Validation Loss: {total} F1: {totf1}')

        # === TensorBoard logging ===
        info = {
            'training loss': train_total,
            'validation loss': total,
            'f1': totf1,
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)
        logger.writer.flush()

        try:
            # depends on the net structure!!!
            images1 = cnn.conv1.weight  #
            images2 = cnn.conv2.weight  #
            info = {
                'images1': images1[:24].data.cpu().numpy(),
                'images2': images2[:24].data.cpu().numpy(),
            }
        except:
            pass

        for tag, images in info.items():
            logger.image_summary(tag, images, epoch + 1)
        # === save the whole model only if perform better OR every 8th epochs anyway
        if (total < prev_val_loss) or ((epoch + 1) % 8 == 0):
            prev_val_loss = total
            # torch.save(cnn.state_dict(), './save/cnn-%.3d.pkl' % epoch)
            torch.save(cnn, save_dir + 'net-{0:03d}-{1:.3f}.pkl'.format(epoch+1, total))


if __name__ == '__main__':
    freeze_support()
    train()
