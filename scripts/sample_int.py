
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# Custom imports
import sys
sys.path.insert(0, "../")
from utils import class_sampler, load_model
from models.classifier import ChainedClassifier
from attacks import InterpolationValue, bisection

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest="classifier", help = 'path to trained classifier', required=True)
parser.add_argument('-g', dest="generator", help = 'path to trained generator', required=True)
parser.add_argument('-ratio', dest="goal_ratio", help = 'goal ratio of origin class', type=float, required=True)
parser.add_argument('-samples', dest="no_samples", help = 'number of samples per class pair', type=int, required=True)
parser.add_argument('-classes', dest="no_classes", help = 'number of samples per class', type=int, default=10)

# Optional
parser.add_argument('-name', dest="name", help = 'experiment name', default='int_sample')
parser.add_argument('-output', dest="path_output", help = 'path to output folder', default='.')
parser.add_argument('-device', dest="device", default='cuda')
parser.add_argument('-seed', dest="seed", type=int, default=1)
parser.add_argument('-sampler_batch_size', dest="s_batch_size", type=int, default=32)
parser.add_argument('-sampler_max_steps', dest="s_max_steps", type=int, default=200)
parser.add_argument('-sampler_threshold', dest="s_threshold",  type=float, default=0.99)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

params_sampler = {
    'samples': args.no_samples, 
    'threshold': args.s_threshold,
    'max_steps': args.s_max_steps,
    'batch_size': args.s_batch_size,
    'device': args.device,
}


classifier = load_model(args.classifier, device=args.device)
generator = load_model(args.generator, device=args.device)
combined = ChainedClassifier(generator, classifier)
pairs = [(i, j) for i in range(args.no_classes) for j in range(args.no_classes) if j!=i]

data = defaultdict(list)
for label_origin, label_target in pairs:
    print(f'{label_origin}-{label_target}')

    # Generate new samples per each class
    z_origin = class_sampler(classifier, generator, label_origin, **params_sampler)
    z_target = class_sampler(classifier, generator, label_target, **params_sampler)

    # Create samples via interpolation
    samples = []
    for i in range(args.no_samples):
        eval_function = InterpolationValue(combined, z_origin[i], z_target[i], label_origin)
        c = bisection(eval_function, args.goal_ratio, a=0, b=1, threshold=0.001)
        samples.append(eval_function.evaluate(c))
    samples = torch.cat(samples)

    # Store results
    data['z_origin'].append(z_origin.cpu().numpy())
    data['z_target'].append(z_target.cpu().numpy())
    data['z_perturbed'].append(samples.cpu().numpy())
    data['labels_origin'].append(np.full((args.no_samples,), label_origin, dtype=np.long))
    data['labels_target'].append(np.full((args.no_samples,), label_target, dtype=np.long))
    data['preds'].append(F.softmax(combined(samples), dim=1).detach().cpu().numpy())


# Save output
data = {key: np.concatenate(value) for key, value in data.items()}
meta = {
    'name': args.name,
    'args': vars(args),
    'classifier': str(classifier),
    'generator': str(generator),
    }

file_data = {'data': data, 'meta': meta}
gen = args.generator.split('\\')[-1]
file_name = args.path_output + '/' + f'{args.name}_{args.goal_ratio}_{gen}_{args.seed}'.replace('.', '-') + '.pickle'
with open(file_name, 'wb') as handle:
    pickle.dump(file_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
