import argparse
import torch
from torch.utils.data import DataLoader

# Custom imports
import sys
sys.path.insert(0, "../")
from datasets import get_dataset
from logger import Logger
from datetime import datetime
from utils import import_module

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest="model", help = 'path to the module with model definition', required=True)
parser.add_argument('-t', dest="task", help = 'type of ML task. One of: end, gen, cls, cls_at', required=True)
parser.add_argument('-d', dest="dataset", help = 'name of dataset', required=True)
parser.add_argument('-o', dest="output", help = 'path to output folders', default='runs')
parser.add_argument('--tag', dest="tag", help = 'name of experiment', default='experiment')
parser.add_argument('--device', dest="device", help = 'training device', default='cuda')
parser.add_argument('--epochs', dest="epochs", help = 'number of training epochs', type=int, default=50)
parser.add_argument('--batch_size', dest="batch_size", help = 'batch size', type=int, default=128)
parser.add_argument('--seed', dest="seed", help = 'random seed', type=int, default=999)
args = parser.parse_args()

# Initialize dataset and model
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
dataset_train = get_dataset(dataset=args.dataset, train=True)
loader = DataLoader(dataset_train, args.batch_size, shuffle=True)
model = getattr(import_module(args.model), 'model')

# Initialize Logger and start logging
dataset_test = get_dataset(dataset=args.dataset, train=False)
logdir = f"{args.output}/{args.tag}_{datetime.now().strftime('%b%d_%H-%M-%S')}"
logger = Logger(logdir=logdir, dataset=dataset_test, task=args.task, model=model, seed=args.seed)
logger.save_source(model=model)
logger.save_dictionary(args.__dict__, filename='args.json')
logger.save_file(file=args.model)

# Start training
logger.log_summary(step=0, model=model)
for i in range(args.epochs):
    for j, (batch, label) in enumerate(loader):
        step = i*(round(len(dataset_train) / args.batch_size))+j
        batch = batch.to(args.device)

        if args.task in ['end', 'gen']:
            loss = model.train_step(batch)
        else:
            label = label.to(args.device)
            loss = model.train_step(batch, label)

        if (step % 100) == 0:
            print(f'epoch: {i} \t step: {step}')
            for name, value in loss.items():
                logger.writer.add_scalar(f'loss/{name}', value, global_step=step)

    model.epoch_end()
    logger.log_summary(step=i+1, model=model)
logger.save_model(model=model)