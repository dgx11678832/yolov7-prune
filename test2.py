from common import load_dataset
from pathlib import Path
import torch
from models.experimental import attempt_load
import torch.nn.utils.prune as prune
from models.yolo import make_model
# model = torch.hub.load('ultralytics/yolov5','yolov5s')

DATA_DIR = Path('data')
DATASETS_DIR = Path('./datasets')
MODELS_DIR = Path('./exact_models')
PRUNED_DIR = Path('./pruned_models')
METRICS_DIR = Path('./model_metrics')
UNTRAINED_DIR = Path('./untrained_models')

pruning_methods = {
    'random_u': prune.RandomUnstructured,
    'random_s': prune.RandomStructured,
    'magnitude': prune.L1Unstructured,
    'filter': prune.LnStructured
}


def prune_model(model, untrained_model, device, method: str, sparsity_level):
    if method.startswith('filter') or method.startswith('kernel'):
        # The weights and biases of the conv layers excluding the ones in the HEAD block
        parameters_to_prune = [
            (val, 'weight') for key, val in model.named_modules() if
            not key.startswith('model.model.24') and isinstance(val, torch.nn.Conv2d)
        ]
    else:
        # The weights and biases of the conv layers will be pruneable
        parameters_to_prune = [
                                  (val, 'weight') for key, val in model.named_modules() if
                                  isinstance(val, torch.nn.Conv2d)
                              ] + [(val, 'bias') for key, val in model.named_modules() if
                                   isinstance(val, torch.nn.Conv2d) and val.bias is not None
                                   ]

    # Get the same prune-able parameters from the untrained model
    untrained_parameters = [
                               (val, 'weight') for key, val in untrained_model.named_modules() if
                               isinstance(val, torch.nn.Conv2d)
                           ] + [(val, 'bias') for key, val in untrained_model.named_modules() if
                                isinstance(val, torch.nn.Conv2d) and val.bias is not None
                                ]

    # Apply the pruning method globally
    # method = pruning_methods[method]
    # prune.global_unstructured(parameters_to_prune, pruning_method=method, amount=sparsity_level)

    # TODO: Figure out a tidier way to switch between pruning methods
    # Apply the proper pruning method
    if method == 'magnitude':
        # Apply the magnitude pruning globally
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=sparsity_level)
        for val, name in parameters_to_prune:
            prune.remove(val, name)
    elif method == 'random_u':
        # Apply the random pruning globally
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=sparsity_level)
        for val, name in parameters_to_prune:
            prune.remove(val, name)
    elif method == 'gradient':
        # Create the importance scores
        importance_scores = {}
        for (module, name), (old_module, _) in zip(parameters_to_prune, untrained_parameters):
            importance_scores[(module, name)] = getattr(module, name) - getattr(old_module, name)

        # Apply the gradient pruning globally
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,
                                  importance_scores=importance_scores, amount=sparsity_level)
        for val, name in parameters_to_prune:
            prune.remove(val, name)
    elif method == 'gradient_magnitude':
        # Create the importance scores
        importance_scores = {}
        for (module, name), (old_module, _) in zip(parameters_to_prune, untrained_parameters):
            importance_scores[(module, name)] = (getattr(module, name) - getattr(old_module, name)) * getattr(module,
                                                                                                              name)

        # Apply the gradient magnitude pruning globally
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,
                                  importance_scores=importance_scores, amount=sparsity_level)
        for val, name in parameters_to_prune:
            prune.remove(val, name)
    elif method == 'filter':
        # Apply the filter pruning globally
        global_smallest_filter(parameters_to_prune, amount=sparsity_level)
    elif method == 'filter_norm':
        # Apply the normalized filter pruning globally
        global_smallest_filter_normalized(parameters_to_prune, amount=sparsity_level)
    elif method == 'kernel':
        # Apply the kernel pruning globally
        global_smallest_kernel(parameters_to_prune, amount=sparsity_level)
    elif method == 'kernel_norm':
        # Apply the kernel pruning globally
        global_smallest_kernel_norm(parameters_to_prune, amount=sparsity_level)



def sparsity(model):
    nparams = 0
    pruned = 0
    for k, v in dict(model.named_modules()).items():
        if ((len(list(v.children())) == 0) and (k.endswith('conv'))):
            nparams += v.weight.nelement()
            pruned += torch.sum(v.weight == 0)
            if v.bias is None:
                continue
            nparams += v.bias.nelement()
            pruned += torch.sum(v.bias == 0)
    print('Global sparsity across the pruned layers: {:.2f}%'.format( 100. * pruned / float(nparams)))
    return 100. * pruned / float(nparams)
cfg = '/home/deepblue/PycharmProjects/yolov7-prune/yolov7-main/cfg/deploy/yolov7.yaml'
device = torch.device('cpu')
path = 'yolov7.pt'
# model = attempt_load(path, map_location='cpu')
# model.eval()
untrained_model = make_model(cfg, device)
untrained_model.eval()
img = torch.rand(1, 3, 640, 640).to(device)
# y = model(img)
y2 = untrained_model(img)
# model.model[0].conv.weight.data[0,0,0]
# untrained_model.model[0].conv.weight.data[0,0,0]
# data = load_dataset(DATA_DIR / 'coco128.yaml')
pruning_mthd = 'magnitude'
sparsity = [0.1 ,0.3,0.5]
all_scores = torch.zeros((6, len(sparsity)))

for i, sparsity_level in enumerate(sparsity):
    prune_model(model, untrained_model, device, pruning_mthd, sparsity_level)
    sparsity()