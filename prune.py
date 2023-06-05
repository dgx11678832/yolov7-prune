import argparse

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.utils.prune as prune

from common import *

DATA_DIR = Path('./data')
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
            (val, 'weight') for key, val in model.named_modules() if not key.startswith('model.model.24') and isinstance(val, torch.nn.Conv2d)
        ]
    else:
        # The weights and biases of the conv layers will be pruneable
        parameters_to_prune = [
            (val, 'weight') for key, val in model.named_modules() if isinstance(val, torch.nn.Conv2d)
            ] + [(val, 'bias') for key, val in model.named_modules() if isinstance(val, torch.nn.Conv2d) and val.bias is not None
        ]
    
    # Get the same prune-able parameters from the untrained model
    untrained_parameters = [
        (val, 'weight') for key, val in untrained_model.named_modules() if isinstance(val, torch.nn.Conv2d)
        ] + [(val, 'bias') for key, val in untrained_model.named_modules() if isinstance(val, torch.nn.Conv2d) and val.bias is not None
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
        for (module, name),(old_module, _) in zip(parameters_to_prune, untrained_parameters):
            importance_scores[(module, name)] = getattr(module, name) - getattr(old_module, name)
            
        # Apply the gradient pruning globally
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, importance_scores=importance_scores, amount=sparsity_level)
        for val, name in parameters_to_prune:
            prune.remove(val, name)
    elif method == 'gradient_magnitude':
        # Create the importance scores
        importance_scores = {}
        for (module, name),(old_module, _) in zip(parameters_to_prune, untrained_parameters):
            importance_scores[(module, name)] = (getattr(module, name) - getattr(old_module, name)) * getattr(module, name)
            
        # Apply the gradient magnitude pruning globally
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, importance_scores=importance_scores, amount=sparsity_level)
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+',type=str, default=MODELS_DIR / 'yolov7.pt', help='model path(s)')
    parser.add_argument('--pruning-method', nargs='+', type=str, default='random', help='pruning method names')
    parser.add_argument('--sparsity', nargs='+',type=float, default=[ 0.1 ,0.5 ,0.8], help='sparsity level after pruning')
    parser.add_argument('--scope', type=str, default='global', help='scope of pruning: layerwise/global')
    parser.add_argument('--data', type=str, default=DATA_DIR / 'coco128.yaml', help='dataset used for validation of pruned model')
    parser.add_argument('--device', type=str, default='cpu')
    opt = parser.parse_args()
    print(vars(opt))
    return opt
    
def main(opt):
    device = torch.device(opt.device)
    model = load_model(str(opt.weights), pretrained=True)
    # Model(str(opt.weights), device)
    data = load_dataset(opt.data)
    untrained_model = load_model(str(opt.weights), pretrained=False)
    pruning_mthd = opt.pruning_method
    all_scores = torch.zeros((6, len(opt.sparsity)))
    for i, sparsity_level in enumerate(opt.sparsity):
        # Load model
        # model = load_model(MODELS_DIR / w, pretrained=True)

        # Prune the and make changes permanent
        prune_model(model, untrained_model, device, pruning_mthd, sparsity_level)

        # Validate the pruned model and store the result
        images, instances, p, r, mAP50, mAP50_95, t = validate(model, data, device, 640, 32)

        all_scores[0, i] = model.sparsity()
        all_scores[1, i] = mAP50_95
        all_scores[2, i] = mAP50
        all_scores[3, i] = r
        all_scores[4, i] = p
        all_scores[5, i] = t[1]

        # Save the pruned model
        # torch.save(model, SAVE_DIR + '_'.join([w[:-3], pruning_mthd, f'{sparsity_level}.pt']))

    # Save the metrics
    torch.save(all_scores, 'yolov5_prune.pt')
    # for w in opt.weights:
    #     # Load the model to get the baseline performance
    #     # model = Model(MODELS_DIR / w, device)
    #     model = load_model(MODELS_DIR / w, pretrained=True)
    #     data = load_dataset(opt.data)
    #     # images, instances, p, r, mAP50, mAP50_95, t = validate(model, data, device, 640, 32)
    #
    #     # Load untrained model, used by some pruning methods
    #     # untrained_model = Model(UNTRAINED_DIR / f'{w[:-3]}_untrained.pt', device)
    #     untrained_model = load_model(UNTRAINED_DIR / w, pretrained=False)
    #     # untrained_model = None
    #
    #
    #     for pruning_mthd in opt.pruning_method:
    #
    #         # Create a results tensor for each level
    #         all_scores = torch.zeros((6, len(opt.sparsity)))
    #         all_inference_times = torch.zeros(len(opt.sparsity))
    #
    #         for i, sparsity_level in enumerate(opt.sparsity):
    #
    #             # Load model
    #             model = load_model(MODELS_DIR / w, pretrained=True)
    #
    #             # Prune the and make changes permanent
    #             prune_model(model, untrained_model, device, pruning_mthd, sparsity_level)
    #
    #             # Validate the pruned model and store the result
    #             images, instances, p, r, mAP50, mAP50_95, t = validate(model, data, device, 640, 32)
    #
    #             all_scores[0, i] = model.sparsity()
    #             all_scores[1, i] = mAP50_95
    #             all_scores[2, i] = mAP50
    #             all_scores[3, i] = r
    #             all_scores[4, i] = p
    #             all_scores[5, i] = t[1]
    #
    #             # Save the pruned model
    #             # torch.save(model, SAVE_DIR + '_'.join([w[:-3], pruning_mthd, f'{sparsity_level}.pt']))
    #
    #         # Save the metrics
    #         torch.save(all_scores, METRICS_DIR / f'{w[:-3]}_{pruning_mthd}.pt')
                
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 