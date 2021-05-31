import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Custom imports
import sys
sys.path.insert(0, "../")
from models.classifier import ChainedClassifier
from attacks import GradientPerturbation, cross_entropy_loss
from attacks import SchedulerExponential, mix_classes
from utils import load_model, class_sampler

parser = argparse.ArgumentParser()
parser.add_argument('-c', dest="classifier", help = 'path to trained classifier', required=True)
parser.add_argument('-g', dest="generator", help = 'path to trained generator', required=True)
parser.add_argument('-ratio', dest="goal_ratio", help = 'goal ratio of origin class', type=float, required=True)
parser.add_argument('-samples', dest="no_samples", help = 'number of samples per class', type=int, required=True)
parser.add_argument('-classes', dest="no_classes", help = 'number of samples per class', type=int, default=10)

# Optional
parser.add_argument('-name', dest="name", help = 'experiment name', default='grad_sample')
parser.add_argument('-output', dest="path_output", help = 'path to output folder', default='.')
parser.add_argument('-device', dest="device", default='cuda')
parser.add_argument('-seed', dest="seed", type=int, default=1)
parser.add_argument('-max_size', dest="max_size", type=int, default=100)
parser.add_argument('-sampler_batch_size', dest="s_batch_size", type=int, default=64)
parser.add_argument('-sampler_max_steps', dest="s_max_steps", type=int, default=1000)
parser.add_argument('-sampler_threshold', dest="s_threshold",  type=float, default=0.95)

# Optional attack params
parser.add_argument('-suggest_initial', dest="suggest_initial", type=float, default=0.1)
parser.add_argument('-suggest_gamma', dest="suggest_gamma", type=float, default=0.01)
parser.add_argument('-suggest_steps', dest="suggest_steps",  type=int, default=100)
parser.add_argument('-attack_initial', dest="attack_initial", type=float, default=0.1)
parser.add_argument('-attack_gamma', dest="attack_gamma", type=float, default=0.05)
parser.add_argument('-attack_steps', dest="attack_steps",  type=int, default=200)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

params_sampler = {
    'threshold': args.s_threshold,
    'max_steps': args.s_max_steps,
    'batch_size': args.s_batch_size,
    'device': args.device,
}

epsilons_suggest = SchedulerExponential(args.suggest_initial, args.suggest_gamma, args.suggest_steps)
epsilons_attack = SchedulerExponential(args.attack_initial, args.attack_gamma, args.attack_steps)

classifier = load_model(args.classifier, device=args.device)
generator = load_model(args.generator, device=args.device)
combined = ChainedClassifier(generator, classifier)

# Split samples
sample_sizes = [args.max_size for i in range(args.no_samples // args.max_size)]
if (args.no_samples % args.max_size) != 0 :
    sample_sizes.append(args.no_samples % args.max_size)

data = defaultdict(list)
for class_id in range(args.no_classes):
    for samples in sample_sizes:
        print(f"class: {class_id} - {samples}")

        # Sample new latent vectors
        z_origin = class_sampler(classifier, generator, class_id, samples=samples, **params_sampler)
        labels_origin = torch.tensor(np.repeat(class_id, samples), device=args.device, dtype=torch.long)

        # Suggest targets
        attack = GradientPerturbation(nn.CrossEntropyLoss(), norm='l2', steps=epsilons_suggest, targeted=False)
        z_untargeted = attack.run(z_origin, labels_origin, combined)
        labels_target = F.softmax(combined(z_untargeted), dim=1).argmax(dim=1)

        # Attack targets
        mixed = mix_classes(labels_origin, labels_target, args.goal_ratio).to(args.device)
        attack = GradientPerturbation(cross_entropy_loss, norm='l2', steps=epsilons_attack, targeted=True)
        z_perturbed = attack.run(z_origin, mixed, combined)

        # Store results
        data['z_origin'].append(z_origin.cpu().numpy())
        data['z_perturbed'].append(z_perturbed.cpu().numpy())
        data['labels_origin'].append(labels_origin.cpu().numpy())
        data['labels_target'].append(labels_target.cpu().numpy())
        data['preds'].append(F.softmax(combined(z_perturbed), dim=1).detach().cpu().numpy())

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
