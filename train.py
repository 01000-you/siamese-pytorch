import torch
import pickle
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import Siamese
import time
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import argparse

writer = SummaryWriter()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Siamese Network')
    parser.add_argument("--exp_name", type=str, default="unnamed_exp", help="Experiment name")
    parser.add_argument("--train_path", type=str, default="datasets/omniglot/python/images_background",
                        help="Training folder path")
    parser.add_argument("--test_path", type=str, default="datasets/omniglot/python/images_evaluation",
                        help="Testing folder path")
    parser.add_argument("--way", type=int, default=20, help="How much way one-shot learning")
    parser.add_argument("--times", type=int, default=400, help="Number of samples to test accuracy")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.00006, help="Learning rate")
    parser.add_argument("--show_every", type=int, default=10, help="Show result after each show_every iter")
    parser.add_argument("--save_every", type=int, default=100, help="Save model after each save_every iter")
    parser.add_argument("--test_every", type=int, default=100, help="Test model after each test_every iter")
    parser.add_argument("--max_iter", type=int, default=50000, help="Number of iterations before stopping")
    parser.add_argument("--log_path", type=str, default="./tensorboard", help="Path to save logs")
    parser.add_argument("--model_path", type=str, default="models", help="Path to store model")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation (cuda, cpu, mps)")

    args = parser.parse_args()

    writer = SummaryWriter(args.log_path)

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    if args.device == 'mps':
        device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    elif args.device == 'cuda':
        # get gpu ids
        gpu_count = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'use gpu : {device}')  # todo : change logger

    # train_dataset = dset.ImageFolder(root=args.train_path)
    # test_dataset = dset.ImageFolder(root=args.test_path)

    trainSet = OmniglotTrain(args.train_path, transform=data_transforms)
    testSet = OmniglotTest(args.test_path, transform=transforms.ToTensor(),
                           times=args.times, way=args.way)
    testLoader = DataLoader(testSet, batch_size=args.way,
                            shuffle=False, num_workers=gpu_count if device.type == 'cuda' else 1)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size,
                             shuffle=False, num_workers=gpu_count if device.type == 'cuda' else 1)

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)

    net = Siamese()
    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > args.max_iter:
            break
        img1, img2, label = Variable(img1.to(device)), Variable(img2.to(device)), Variable(label.to(device))
        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % args.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/args.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % args.save_every == 0:
            torch.save(net.state_dict(), args.model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % args.test_every == 0:
            right, error = 0, 0
            for _, (test1, test2) in enumerate(testLoader, 1):
                test1, test2 = Variable(test1.to(device)), Variable(test2.to(device))
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else: error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            queue.append(right*1.0/(right+error))
        train_loss.append(loss_val)
        writer.add_scalar("Loss/train", loss_val / args.show_every, batch_id)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)
    writer.close()
