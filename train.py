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
import logging

writer = SummaryWriter()

# 로그 파일 경로 설정
log_file = 'experiment.log'

# 로그 포매터 설정
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 로거 생성
logger = logging.getLogger('experiment_logger')
logger.setLevel(logging.DEBUG)

# 파일 핸들러 생성
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 생성 (터미널 출력용)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 로그 기록
logger.info('Experiment started.')

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
    parser.add_argument("--dr", type=float, default=0.2, help="dropout rate")
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
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if args.device == 'mps':
        device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
    elif args.device == 'cuda':
        # get gpu ids
        gpu_count = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'use gpu : {device}')  # todo : change logger

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

    net = Siamese(dropout_rate=args.dr)
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
            logger.info('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/args.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % args.save_every == 0:
            torch.save(net.state_dict(), args.model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % args.test_every == 0:
            right, error = 0, 0
            right_top3, error_top3 = 0, 0
            right_top5, error_top5 = 0, 0
            net.eval()
            for _, (test1, test2) in enumerate(testLoader, 1):
                test1, test2 = Variable(test1.to(device)), Variable(test2.to(device))
                # output = net.forward(test1, test2).data.cpu().numpy()
                # pred = np.argmin(output)
                output = net.forward(test1, test2).data
                pred = torch.topk(output[:, 0], 1, largest=False).indices
                pred_top3 = torch.topk(output[:, 0], 3, largest=False).indices
                pred_top5 = torch.topk(output[:, 0], 5, largest=False).indices

                right, error = (right + 1, error) if 0 in pred else (right, error + 1)
                right_top3, error_top3 = (right_top3 + 1, error_top3) if 0 in pred_top3 else \
                    (right_top3, error_top3 + 1)
                right_top5, error_top5 = (right_top5 + 1, error_top5) if 0 in pred_top5 else \
                    (right_top5, error_top5 + 1)
                #
                # if 0 in pred:
                #     right += 1
                # else: error += 1
                #
                # if 0 in pred_top3:
                #     right_top3 += 1
                # else: error_top3 += 1
                #
                # if 0 in pred_top5:
                #     right_top5 += 1
                # else: error_top5 += 1

            logger.info('*'*70)
            logger.info('[%d]\tTest set Top1\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'
                        %(batch_id, right, error, right*1.0/(right+error)))
            logger.info('[%d]\tTest set Top3\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'
                        %(batch_id, right_top3, error_top3, right_top3*1.0/(right_top3+error_top3)))
            logger.info('[%d]\tTest set Top5\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'
                        %(batch_id, right_top5, error_top5, right_top5*1.0/(right_top5+error_top5)))

            logger.info('*'*70)
            queue.append(right*1.0/(right+error))
            net.train()
        train_loss.append(loss_val)
        writer.add_scalar("Loss/train", loss_val / args.show_every, batch_id)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    logger.info("#"*70)
    logger.info("final accuracy: ", acc/20)
    writer.close()
