import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import importlib

def grid_plot(img_batch, nrow=6, already_grid=False):
    '''
    Plot tenosr batch of images. Input dimension format: (B,C,W,H)
    '''
    if already_grid:
        grid = img_batch
    else:
        grid = make_grid(img_batch, nrow=nrow, normalize=True)
    grid = grid.cpu().detach().numpy()
    grid = np.transpose(grid, (1,2,0))

    fig = plt.figure(figsize=(nrow, nrow))
    ax = fig.add_subplot(111)
    ax.imshow(grid)
    plt.show()

def import_module(path, name='module'):
    '''
    Imports module from file given its path
    '''
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_args(path):
    with open(path + '\\' + 'args.json') as f:
        args = json.load(f)
    return args

def load_model(path, model_file=None, state_dict=None, device='cuda'):
    if model_file is None:
        args = load_args(path)
        _, model_file = os.path.split(args['model'])
    if state_dict is None:
        state_dict = 'model_state_dict.pth'

    model = import_module(path + '\\' + model_file).model
    model.load_state_dict(torch.load(path + '\\' + state_dict))
    model = model.to(device)
    model.eval()
    return model

def hide_code():
    from IPython.display import HTML
    return HTML('''<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()"><input type="submit" value="Toggle on/off the raw code."></form>''')

def class_sampler(classifier, generator, class_target, samples=100, threshold=0.99, max_steps=200, batch_size=32, device='cuda'):
    '''
    Sample data from single class, labeled by auxiliary classifier.
        classifier: Auxiliary classifier used for classification
        generator: Generator used for sampling
        class_target: Class id which will be sampled
        threshold: Threshold of softmax needed for sample to be considered as given class
    '''
    filled = 0
    data = torch.zeros((samples, generator.size_z, 1, 1), device=device)

    for i in range(max_steps):
        # Make predictions
        with torch.no_grad():
            z = torch.randn(batch_size, generator.size_z, 1, 1, device=device)
            imgs = generator.decode(z)
            output = F.softmax(classifier(imgs), 1).data.cpu()
            softmax, class_id = output.max(dim=1)

        # Collect the predictions of given class
        mask = (class_id == class_target) & (softmax > threshold)
        for tensor in z[mask]:
            data[filled] = tensor
            filled = filled + 1
            if filled >= samples:
                return data

    raise Exception('Not enough samples found. Decrease threshold! ')