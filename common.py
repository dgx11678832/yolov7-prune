import torch 
from torch import nn
import torch.nn.utils.prune as prune

import numpy as np

from tqdm import tqdm

from pathlib import Path

# from utils.callbacks import Callbacks
# from utils.dataloaders import create_dataloader
from utils.general import ( check_dataset, check_img_size, check_requirements,
                            coco80_to_coco91_class, colorstr, increment_path, non_max_suppression,
                           xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images
# from utils.torch_utils import select_device, smart_inference_mode

GITHUB = 'Didanny/perception-models'
DATA_DIR = Path('./data')
DATASETS_DIR = Path('./datasets')
MODELS_DIR = Path('./exact_models')
PRUNED_DIR = Path('./pruned_models')
METRICS_DIR = Path('./model_metrics')
UNTRAINED_DIR = Path('./untrained_models')

class Model(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights, device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        
        super().__init__()
        w = weights
        pt = True
        fp16 &= pt  # FP16
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA

        if pt:  # PyTorch
            model = torch.load(weights, map_location='cpu')
            model = model['model'].to(device).float()

            # Model compatibility updates
            if not hasattr(model, 'stride'):
                model.stride = torch.tensor([32.])
            if hasattr(model, 'names') and isinstance(model.names, (list, tuple)):
                model.names = dict(enumerate(model.names))  # convert to dict
                
            # Module compatibility updates
            for m in model.modules():
                t = type(m)
                if t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                    m.recompute_scale_factor = None  # torch 1.11.0 compatibility

            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        for _ in range(1):
            self.forward(im)  # warmup
            
    def sparsity(self):
        nparams = 0
        pruned = 0
        for k, v in dict(self.named_modules()).items():
            if ((len(list(v.children())) == 0) and (k.endswith('conv'))):
                nparams += v.weight.nelement()
                pruned += torch.sum(v.weight == 0)
                if v.bias is None:
                    continue
                nparams += v.bias.nelement()
                pruned += torch.sum(v.bias == 0)
        print('Global sparsity across the pruned layers: {:.2f}%'.format( 100. * pruned / float(nparams)))
        return 100. * pruned / float(nparams)
    
def load_model(weights: Path, pretrained: bool, device=torch.device('cpu')):
    print('weights',weights)
    # print('weights.stem',weights.stem)
    if pretrained:
        # if weights.is_file():
        if True:
            return Model(weights, device)
        else:
            model = torch.hub.load(GITHUB, weights.stem, pretrained=pretrained, force_reload=True)
            source_file = Path(weights.name)
            destination_file = Path(MODELS_DIR / weights.name)
            source_file.rename(destination_file)
            return Model(weights, device)
    else:
        # if weights.is_file():
        if True:

            return torch.load(weights)['model'].to(device)
        else:
            model = torch.hub.load(GITHUB, weights.stem, pretrained=pretrained, force_reload=True)
            torch.save(model, UNTRAINED_DIR / weights.name)
            # source_file = Path(weights.name)
            # destination_file = Path(UNTRAINED_DIR / weights.name)
            # source_file.rename(destination_file)
            return torch.load(weights).to(device=device)
                
# class SmallestFilter(prune.BasePruningMethod):
    
#     PRUNING_TYPE = 'global'
    
#     def __init__(self):
#         super().__init__()

# def _get_filter_importances(module, param):
#     pass

def global_smallest_filter_normalized(parameters, amount, **kwargs):
    # Ensure parameters is a list or generator of tuples
    if not isinstance(parameters, prune.Iterable):
        raise TypeError('global_smallest_filter(): parameters is not an iterable')
    
    # Get the total number of parameters
    num_params = 0
    for module, name in parameters:
        num_params += module.weight.numel()
        prune.identity(module, name)
        
    # Determine the number of params to prune
    num_to_prune = int(num_params * amount)
    
    # TODO: Replace the hard-coded 'module.weight' and use getattr on param instead with some checks
    # Create importance scores
    importance_scores = [
        torch.vstack([
            torch.norm(module.weight, 1, (1, 2, 3)).to(module.weight.device) / (module.kernel_size[0]*module.kernel_size[1]*module.in_channels),
            torch.range(0, module.weight.shape[0] - 1).to(module.weight.device),
            (torch.ones(module.weight.shape[0]) * i).to(module.weight.device),
            torch.ones(module.weight.shape[0]).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1]*module.in_channels)
        ])
        for i, (module, param) in enumerate(parameters)
    ]
    importance_scores = torch.hstack(importance_scores)
    
    # Sort the importance score matrix along the L1 norm row
    sorted_indices = importance_scores[0,:].sort()[1]
    importance_scores = importance_scores[:, sorted_indices]
    
    # Get the cumulative sum of the filter size row to determine when to stop pruning
    importance_scores[3,:] = torch.cumsum(importance_scores[3,:], -1)
    
    # Find the index of the last filter to be pruned
    importance_scores[3,:] > num_to_prune
    last_prune_idx = torch.argmax((importance_scores[3,:] > num_to_prune).to(dtype=torch.int))
    importance_scores = importance_scores[:, :last_prune_idx]
    
    # Iterate over the importance scores and prune the corresponding filter
    for i in range(last_prune_idx):        
        target = importance_scores[:, i]
        filter_idx = target[1].to(dtype=torch.int)
        module_idx = target[2].to(dtype=torch.int)
        
        module, name = parameters[module_idx]
        
        # Determine the filter size
        filter_size = module.in_channels*module.kernel_size[0]*module.kernel_size[1]
        
        # Get the default mask of the module
        mask = getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
        
        # Compute the new module mask
        mask = mask.view(-1)
        mask[filter_size * filter_idx : filter_size * filter_idx + filter_size] = 0
        mask = mask.view(module.weight.shape)
        
        # Apply the mask
        prune.custom_from_mask(module, name, mask=mask)
        prune.remove(module, name)

        
def global_smallest_filter(parameters, amount,  **kwargs):
    # Ensure parameters is a list or generator of tuples
    if not isinstance(parameters, prune.Iterable):
        raise TypeError('global_smallest_filter(): parameters is not an iterable')
    
    # Get the total number of parameters
    num_params = 0
    for module, name in parameters:
        num_params += module.weight.numel()
        prune.identity(module, name)
        
    # Determine the number of params to prune
    num_to_prune = int(num_params * amount)
    
    # TODO: Replace the hard-coded 'module.weight' and use getattr on param instead with some checks
    # Create importance scores
    importance_scores = [
        torch.vstack([
            torch.norm(module.weight, 1, (1, 2, 3)).to(module.weight.device),
            torch.range(0, module.weight.shape[0] - 1).to(module.weight.device),
            (torch.ones(module.weight.shape[0]) * i).to(module.weight.device),
            torch.ones(module.weight.shape[0]).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1]*module.in_channels)
        ])
        for i, (module, param) in enumerate(parameters)
    ]
    importance_scores = torch.hstack(importance_scores)
    
    # Sort the importance score matrix along the L1 norm row
    sorted_indices = importance_scores[0,:].sort()[1]
    importance_scores = importance_scores[:, sorted_indices]
    
    # Get the cumulative sum of the filter size row to determine when to stop pruning
    importance_scores[3,:] = torch.cumsum(importance_scores[3,:], -1)
    
    # Find the index of the last filter to be pruned
    importance_scores[3,:] > num_to_prune
    last_prune_idx = torch.argmax((importance_scores[3,:] > num_to_prune).to(dtype=torch.int))
    importance_scores = importance_scores[:, :last_prune_idx]
    
    # Iterate over the importance scores and prune the corresponding filter
    for i in range(last_prune_idx):
        target = importance_scores[:, i]
        filter_idx = target[1].to(dtype=torch.int)
        module_idx = target[2].to(dtype=torch.int)
        
        module, name = parameters[module_idx]
        
        # Determine the filter size
        filter_size = module.in_channels*module.kernel_size[0]*module.kernel_size[1]
        
        # Get the default mask of the module
        mask = getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
        
        # Compute the new module mask
        mask = mask.view(-1)
        mask[filter_size * filter_idx : filter_size * filter_idx + filter_size] = 0
        mask = mask.view(module.weight.shape)
        
        # Apply the mask
        prune.custom_from_mask(module, name, mask=mask)
        prune.remove(module, name)
        
def global_smallest_kernel(parameters, amount,  **kwargs):
    # Ensure parameters is a list or generator of tuples
    if not isinstance(parameters, prune.Iterable):
        raise TypeError('global_smallest_filter(): parameters is not an iterable')
    
    # Get the total number of parameters
    num_params = 0
    for module, name in parameters:
        num_params += module.weight.numel()
        prune.identity(module, name)
        
    # Determine the number of params to prune
    num_to_prune = int(num_params * amount)
    
    # TODO: Replace the hard-coded 'module.weight' and use getattr on param instead with some checks
    # Create importance scores
    importance_scores = [
        torch.vstack([
            torch.norm(module.weight, 1, (2, 3)).flatten().to(module.weight.device),
            torch.range(0, (module.in_channels * module.out_channels) - 1).to(module.weight.device),
            (torch.ones(module.in_channels * module.out_channels) * i).to(module.weight.device),
            torch.ones(module.in_channels * module.out_channels).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1])
        ])
        for i, (module, param) in enumerate(parameters)
    ]
    importance_scores = torch.hstack(importance_scores)
    
    # Sort the importance score matrix along the L1 norm row
    sorted_indices = importance_scores[0,:].sort()[1]
    importance_scores = importance_scores[:, sorted_indices]
    
    # Get the cumulative sum of the filter size row to determine when to stop pruning
    importance_scores[3,:] = torch.cumsum(importance_scores[3,:], -1)
    
    # Find the index of the last kernel to be pruned
    importance_scores[3,:] > num_to_prune
    last_prune_idx = torch.argmax((importance_scores[3,:] > num_to_prune).to(dtype=torch.int))
    importance_scores = importance_scores[:, :last_prune_idx]
    
    # Get the unique module indices
    module_indices = torch.unique(importance_scores[2,:]).to(dtype=torch.int)
    
    # Iterate over the module indices and prune all the kernels at once
    for module_idx in tqdm(module_indices):
        # Get the importance scores corresponding to this layer
        module_scores = importance_scores[:,(importance_scores[2,:] == module_idx).nonzero().squeeze(1)]
        
        # Get the kernel indices of all kernels in this layer
        kernel_indices = module_scores[1,:].to(dtype=torch.int)
        
        # Get the kernel size 
        module, name = parameters[module_idx]
        kernel_size = module.kernel_size[0] * module.kernel_size[1]
        
        # Get the indices of the weights corresponding to these kernels
        kernel_indices = kernel_indices * kernel_size
        all_indices = torch.clone(kernel_indices)        
        size_to_cat = kernel_indices.shape[0]
        for i in range(1, kernel_size):
            all_indices = torch.cat((all_indices, kernel_indices + torch.ones(size_to_cat) * i), 0)
        all_indices = all_indices.to(dtype=torch.long)
            
        # Get the default mask of the module
        mask = getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
        
        # Compute the new module mask
        mask.view(-1)[all_indices] = 0
        
        # Apply the mask
        prune.custom_from_mask(module, name, mask=mask)
        prune.remove(module, name)
        
def global_smallest_kernel_norm(parameters, amount,  **kwargs):
    # Ensure parameters is a list or generator of tuples
    if not isinstance(parameters, prune.Iterable):
        raise TypeError('global_smallest_filter(): parameters is not an iterable')
    
    # Get the total number of parameters
    num_params = 0
    for module, name in parameters:
        num_params += module.weight.numel()
        prune.identity(module, name)
        
    # Determine the number of params to prune
    num_to_prune = int(num_params * amount)
    
    # TODO: Replace the hard-coded 'module.weight' and use getattr on param instead with some checks
    # Create importance scores
    importance_scores = [
        torch.vstack([
            (torch.norm(module.weight, 1, (2, 3)) / (module.kernel_size[0]*module.kernel_size[1])).flatten().to(module.weight.device),
            torch.range(0, (module.in_channels * module.out_channels) - 1).to(module.weight.device),
            (torch.ones(module.in_channels * module.out_channels) * i).to(module.weight.device),
            torch.ones(module.in_channels * module.out_channels).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1])
        ])
        for i, (module, param) in enumerate(parameters)
    ]
    importance_scores = torch.hstack(importance_scores)
    
    # Sort the importance score matrix along the L1 norm row
    sorted_indices = importance_scores[0,:].sort()[1]
    importance_scores = importance_scores[:, sorted_indices]
    
    # Get the cumulative sum of the filter size row to determine when to stop pruning
    importance_scores[3,:] = torch.cumsum(importance_scores[3,:], -1)
    
    # Find the index of the last kernel to be pruned
    importance_scores[3,:] > num_to_prune
    last_prune_idx = torch.argmax((importance_scores[3,:] > num_to_prune).to(dtype=torch.int))
    importance_scores = importance_scores[:, :last_prune_idx]
    
    # Get the unique module indices
    module_indices = torch.unique(importance_scores[2,:]).to(dtype=torch.int)
    
    # Iterate over the module indices and prune all the kernels at once
    for module_idx in tqdm(module_indices):
        # Get the importance scores corresponding to this layer
        module_scores = importance_scores[:,(importance_scores[2,:] == module_idx).nonzero().squeeze(1)]
        
        # Get the kernel indices of all kernels in this layer
        kernel_indices = module_scores[1,:].to(dtype=torch.int)
        
        # Get the kernel size 
        module, name = parameters[module_idx]
        kernel_size = module.kernel_size[0] * module.kernel_size[1]
        
        # Get the indices of the weights corresponding to these kernels
        kernel_indices = kernel_indices * kernel_size
        all_indices = torch.clone(kernel_indices)        
        size_to_cat = kernel_indices.shape[0]
        for i in range(1, kernel_size):
            all_indices = torch.cat((all_indices, kernel_indices + torch.ones(size_to_cat) * i), 0)
        all_indices = all_indices.to(dtype=torch.long)
            
        # Get the default mask of the module
        mask = getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
        
        # Compute the new module mask
        mask.view(-1)[all_indices] = 0
        
        # Apply the mask
        prune.custom_from_mask(module, name, mask=mask)
        prune.remove(module, name)
        
def load_dataset(data):
    return check_dataset(data)
            
def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def validate(model, data, device, image_size, batch_size):
    # TODO: Make these customizable knobs
    num_classes = int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    cuda = device.type != 'cpu'
    
    # TODO: remove dataloader from the validation function
    imgsz = image_size
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    pad, rect = (0.5, model.pt)  # square inference for benchmarks
    task = 'val'  # path to train/val/test images
    stride = 32
    single_cls = False
    dataloader = create_dataloader(data[task],
                                        imgsz,
                                        batch_size,
                                        stride,
                                        single_cls,
                                        pad=pad,
                                        rect=rect,
                                        workers=8,
                                        prefix=colorstr(f'{task}: '))[0]
    
    # TODO: Remove unecessary variables
    callbacks = Callbacks()
    half = False
    compute_loss = None
    augment = False
    save_hybrid = False
    conf_thres = 0.001
    iou_thres = 0.6
    max_det = 300
    plots = False
    save_txt= False
    save_hybrid = False
    save_conf = False
    save_json = False
    save_dir = Path('')
    verbose = False
    training = False

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=num_classes)
    names = model.names if hasattr(model, 'names') else model.model.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
        
    # TODO: Modify this to load more datasets
    class_map = coco80_to_coco91_class() #if is_coco else list(range(1000))
    
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=augment), None)

        # Loss
        # if compute_loss:
        #     loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            # if save_txt:
            #     save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            # if save_json:
            #     save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(preds), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        # callbacks.run('on_val_batch_end', batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=num_classes)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    # if (verbose or (num_classes < 50 and not training)) and num_classes > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
        
    return seen, nt.sum(), mp, mr, map50, map, t


