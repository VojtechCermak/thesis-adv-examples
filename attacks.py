import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fgsm(batch, labels, model, epsilon, targeted=False):
    '''
    FGSM Fast Gradient Sign Method (L-infinity bounded attack)
    '''
    loss = nn.CrossEntropyLoss()
    attack = GradientPerturbation(loss=loss, norm='sign', steps=[epsilon], targeted=targeted)
    return attack.run(batch, labels, model)


def ifgsm(batch, labels, model, epsilon, steps=50,  targeted=False):
    '''
    Iterative FGSM (L-infinity bounded attack)
    '''
    
    alpha = epsilon / steps
    for i in range(steps):
        batch = fgsm(batch, labels, model, epsilon=alpha, targeted=targeted)
    return batch


class GradientPerturbation():
    def __init__(self, loss, norm, steps, targeted):
        self.targeted = targeted
        self.norm = norm
        self.steps = steps
        self.loss = loss

    def grad_calculate(self, batch, labels, model):
        batch = batch.detach()
        batch.requires_grad = True
        output = model(batch)
        loss = self.loss(output, labels)
        loss.backward()
        return batch.grad

    def grad_normalize(self, grad):
        if callable(self.norm):
            return self.norm(grad)
        elif self.norm == 'sign':
            return grad.sign()
        elif self.norm == 'l1':
            return grad / torch.norm(grad, p=1, dim=(1, 2, 3), keepdim=True)
        elif self.norm == 'l2':
            return grad / torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        else:
            raise Exception('Unknown normalization method')

    def run_step(self, batch, labels, model, epsilon):
        grad = self.grad_calculate(batch, labels, model)
        grad = self.grad_normalize(grad)

        if self.targeted:
            perturbed_batch = batch - epsilon*grad
        else:
            perturbed_batch = batch + epsilon*grad
        return perturbed_batch

    def run(self, batch, labels, model):
        for epsilon in self.steps:
            batch = self.run_step(batch, labels, model, epsilon)
        return batch


class Value():
    def __call__(self, value):
        batch = self.evaluate(value)
        softmax = F.softmax(self.classifier(batch), dim=1)[0]
        return softmax[self.source_label].item()


class InterpolationValue(Value):
    '''
    Given interpolation ratio, return softmax of source label
    '''
    def __init__(self, classifier, source, target, source_label):
        self.classifier = classifier
        self.source = source
        self.source_label = source_label
        self.target = target

    def evaluate(self, ratio):
        interpolated = ratio*self.source + (1-ratio)*self.target
        return interpolated.unsqueeze(0)


class LatentGradientValue(Value):
    '''
    Given alpha, return softmax of source label
    '''
    def __init__(self, classifier, source, source_label):
        self.classifier = classifier
        self.source = source
        self.source_label = source_label

    def evaluate(self, epsilon):
        source = self.source.unsqueeze(0)
        source_label = torch.tensor([self.source_label], device=source.device)
        attack = GradientPerturbation(loss=nn.CrossEntropyLoss(), norm='l2', steps=[epsilon], targeted=False)
        return attack.run(source, source_label, self.classifier)


def bisection(evaluate, target, a, b, threshold=0.05, max_steps=100):
    '''
    Bisection search for parameter with value near target value.
    
    Inputs
        evaluate - function that returns value given parameter
        target - target value
        a - initial upper value of parameter
        b - initial lower value of parameter
        threshold - stop if value is within target +/- threshold
    
    '''
    for i in range(max_steps):
        c = (a + b) / 2
        c_value = evaluate(c)
        if abs(c_value - target) < threshold:
            break

        if c_value < target:
            a = c
        else:
            b = c
    return c


class BaseScheduler():
    '''
    Base class for scheduler iterators. Override the step method.
    '''
    i = 0
    def __init__(self, steps):
        self.steps = steps

    def __repr__(self):
        from inspect import signature
        fields = tuple(f'{k}={v}' for k,v  in self.__dict__.items() if k in signature(self.__class__).parameters)
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i <= self.steps:
            self.i += 1
            return self.step(self.i)
        else:
            self.i = 0
            raise StopIteration()

    def step(self, i):
        pass

class SchedulerPower(BaseScheduler):
    '''
    Iterator for scheduling steps according to: init*(1+i)^power
    '''

    def __init__(self, initial, power, steps):
        self.initial = initial
        self.power = power
        self.steps = steps

    def step(self, i):
        return self.initial * (1 + i)**(self.power)

class SchedulerExponential(BaseScheduler):
    '''
    Iterator for scheduling steps according to: init*e^(-p*gamma)
    '''
    def __init__(self, initial, gamma, steps):
        self.initial = initial
        self.gamma  = gamma
        self.steps = steps

    def step(self, i):
        return self.initial * np.exp(-self.gamma*i)

def validate(pred, labels, targets, ratio, tolerance):
    '''
    Validate if adversarial images are valid
    - Check if target class if within tolerance
    - Check if target labels != original labels ( = untargeted step failed)
    '''
    pred_original = np.zeros(len(pred))
    pred_target = np.zeros(len(pred))

    for i, p in enumerate(pred):
        pred_original[i] = p[labels[i]] # Original predictions
        pred_target[i] = p[targets[i]]  # Target prediction

    # Conditions
    valid_target = (pred_target < (1 - ratio) + tolerance) & (pred_target > (1 - ratio) - tolerance)
    valid_original = (pred_original < (ratio + tolerance))  & (pred_original > (ratio - tolerance))
    valid_untargeted = (targets != labels)
    return (valid_original & valid_target & valid_untargeted)

def mix_classes(original, target, original_ratio=0.5, no_classes=10, noise=0.0):
    '''
    Mix two tensors of labels to tensor of mixed labels.

    Input:
        original labels (1D tensor of size N)
        target labels (1D tensor of size N)
        ratio between labels (1 = 100% of original label)
        no_classes
    Output: N x no_classes shaped tensor of one-hot encoded mixed labels
    '''
    if noise*no_classes > 1:
        raise ValueError('Too large noise')

    data = []
    samples = len(original)
    labels = torch.full((samples, no_classes), noise)
    remaining = 1 - (noise * no_classes)

    for i in range(samples):
        labels[i, original[i]] = remaining * original_ratio
        labels[i, target[i]] = remaining * (1 - original_ratio)
    return labels
