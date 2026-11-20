

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataset import DataSet
import copy
import warnings
import csv

warnings.filterwarnings('ignore')


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)  
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed) 
random.seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description='PyTorch Relationship')


parser.add_argument('--mode', dest="mode", default="PISC_Coarse", type=str,
                    help='PISC_Fine, PISC_Coarse or PIPA')


parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')


parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1)')


parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')


parser.add_argument('--scale-size', default=256, type=int,
                    help='input size')

parser.add_argument('--crop-size', default=224, type=int,
                    help='crop size')


parser.add_argument('-PMs', '--pretrained_models', default=[], type=list,
                    help='List of pretrained models.')

parser.add_argument('--result_path', default='xxx/trained_models',
                    type=str, metavar='PATH', help='Path for saving result.')

parser.add_argument('--epochs', dest="epochs", default=100, type=int,
                    help='path for saving result (default: none)')

parser.add_argument('--lr', dest="lr", default=0.0001, type=float,
                    help='Learning rate.')


parser.add_argument('--start_epoch', dest="start_epoch", default=1, type=int,
                    help='')


parser.add_argument('--lr_step', dest='lr_step', default=20, type=int, metavar='N',
                    help='Step to adjust learning rate.')



args = parser.parse_args()


for k, v in sorted(vars(args).items()):
    print(k, ': ', v)


if args.mode == "PISC_Fine":
    data_dir = r"xxx/inputFile/image"
    num_class = 6
    train_list = r"xxx/PISCF_train.pkl"
    test_list = r"xxx/PISCF_test.pkl"
elif args.mode == "PISC_Coarse":
    data_dir = r"xxx/inputFile/image"
    num_class = 3
    train_list = r"xxx/PISCC_train.pkl"
    test_list = r"xxx/PISCC_test.pkl"
else:
    data_dir = r"xxx/inputFile/pipa"
    num_class = 16
    train_list = r"xxx/PIPA_train.pkl"
    test_list = r"xxx/PIPA_test.pkl"


def vg_collate(data):
    """
    Batch organization for pair-conditioned evidence reasoning.

    Image-level tensors:
        Full_im: (B, 3, 224, 224)

    Relation-level tensors:
        Union:            (R, 3, 224, 224)
        Obj1, Obj2:       (R, 3, 224, 224)
        Pair_mask:        (R, 1, 224, 224)
        Interaction_mask: (R, 1, 224, 224)
        Bpos:             (R, 6)
        Target:           (R,)
        Pair_indices:     (R, 2)

    R is the total number of person pairs in the batch.
    """

    Name = []
    Full_im = []

    Union = []
    Obj1 = []
    Obj2 = []

    Pair_mask = []
    Interaction_mask = []

    Bpos = []
    Target = []
    Pair_indices = []
    Img_rel_num = []

    for d in data:
        (
            name,
            full_im,
            union,
            obj1,
            obj2,
            pair_mask,
            interaction_mask,
            bpos,
            target,
            pair_indices,
            img_rel_num,
        ) = d

        Name.append(name)
        Full_im.append(full_im)

        Union.append(union)
        Obj1.append(obj1)
        Obj2.append(obj2)

        Pair_mask.append(pair_mask)
        Interaction_mask.append(interaction_mask)

        Bpos.append(bpos)
        Target.append(target)
        Pair_indices.append(pair_indices)
        Img_rel_num.append(img_rel_num)

    # Image-level data: one full image for each dataset item
    Full_im = torch.stack(Full_im, dim=0)

    # Relation-level data: concatenate all person pairs
    Union = torch.cat(Union, dim=0)
    Obj1 = torch.cat(Obj1, dim=0)
    Obj2 = torch.cat(Obj2, dim=0)

    Pair_mask = torch.cat(Pair_mask, dim=0)
    Interaction_mask = torch.cat(
        Interaction_mask,
        dim=0,
    )

    Bpos = torch.cat(Bpos, dim=0)
    Target = torch.cat(Target, dim=0)
    Pair_indices = torch.cat(Pair_indices, dim=0)
    Img_rel_num = torch.cat(Img_rel_num, dim=0)

    return (
        Name,
        Full_im,
        Union,
        Obj1,
        Obj2,
        Pair_mask,
        Interaction_mask,
        Bpos,
        Target,
        Pair_indices,
        Img_rel_num,
    )


def get_train_set(data_dir, train_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
   
    scale_size = args.scale_size
  
    crop_size = args.crop_size
    
    train_data_transform = transforms.Compose([
     
        transforms.Resize((scale_size, scale_size)),
       
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(crop_size),
       
        transforms.ToTensor(),
        normalize])
  
    train_full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
   
    train_set = DataSet(data_dir, train_list, train_data_transform, train_full_transform)
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers, collate_fn=vg_collate,
                              batch_size=args.batch_size, shuffle=True)
    return train_loader, train_set



def get_test_set(data_dir, test_list):
  
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

  
    crop_size = args.crop_size
  
    test_data_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize])
  
    test_full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
  
    test_set = DataSet(data_dir, test_list, test_data_transform, test_full_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, collate_fn=vg_collate,
                             batch_size=args.batch_size, shuffle=False)
    return test_loader



def validate(val_loader, model):
    model.eval()  
    tp, p, r = {}, {}, {}  
    out_add = torch.zeros([0, num_class]).cuda()
    tar_add = torch.zeros([0]).cuda()
    for batch_data in tqdm(val_loader):  

        name, full_im, union, obj1, obj2, pair_mask, interaction_mask, bpos, target, pair_indices, img_rel_num = batch_data
 
        target = target.cuda()

        full_im_var = torch.autograd.Variable(full_im).cuda()
        union_var = torch.autograd.Variable(union).cuda()
        obj1_var = torch.autograd.Variable(obj1).cuda()
        obj2_var = torch.autograd.Variable(obj2).cuda()
        pair_mask = torch.autograd.Variable(pair_mask).cuda()
        interaction_mask = torch.autograd.Variable(interaction_mask).cuda()
        pair_indices = torch.autograd.Variable(pair_indices).cuda()
        bpos_var = torch.autograd.Variable(bpos).cuda()

        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            logits, _ = model(full_im_var, union_var, obj1_var, obj2_var, pair_mask, interaction_mask, bpos_var, img_rel_num)

        output_f = F.softmax(logits, dim=1)
        out_add = torch.cat([out_add, output_f], dim=0)
        tar_add = torch.cat([tar_add, target_var], dim=0)
        output_np = output_f.data.cpu().numpy()

        pre = np.argmax(output_np, 1)
        t = target_var.data.cpu().numpy()

        for i, item in enumerate(t):
            if item in r:
                r[item] += 1
            else:
                r[item] = 1
            if pre[i] in p:
                p[pre[i]] += 1
            else:
                p[pre[i]] = 1
            if pre[i] == item:
                if item in tp:
                    tp[item] += 1
                else:
                    tp[item] = 1

    precision = {}
    recall = {}
    tp_total = 0

    for k in tp.keys():
        precision[k] = float(tp[k]) / float(p[k])
        recall[k] = float(tp[k]) / float(r[k])
        tp_total += tp[k]

    p_total = 0

    for k in p.keys():
        p_total += p[k]
    precision_total = float(tp_total) / float(p_total)

    output_np = out_add.data.cpu().numpy()
    label_np = tar_add.data.cpu().numpy()
    mAP = mean_average_precision(output_np, label_np)

    return precision_total, mAP, recall


def make_one_hot(input_array, num_class):
    out_array = np.eye(num_class)[input_array]
    return out_array


def mean_average_precision(all_probs, all_labels):
    '''
    Args:
        all_probs: 2D numpy.ndarray. The first dimension is number of samples, and the second one is the number of class
        all_labels: 1D numpy.ndarray. Save the index of labels.
    '''
    n_sample, num_class = all_probs.shape

    all_labels = make_one_hot(all_labels.astype(int), num_class)
    n_precision = []
    for each_class in range(num_class):
        probs = all_probs[:, each_class]
        labels = all_labels[:, each_class]
        order = np.argsort(-probs)  # Sort by confidence from largest to smallest
        probs = probs[order]
        labels = labels[order]
        precision = []
        recall = []
        for i in range(n_sample):
            pos_pred_label = labels[0:i + 1]
            neg_pred_label = labels[i + 1:]
            tp = np.sum(pos_pred_label)
            fp = len(pos_pred_label) - tp
            fn = np.sum(neg_pred_label)
            P = tp / (tp + fp + 1e-10)
            R = tp / (tp + fn + 1e-10)
            precision.append(P)
            recall.append(R)
        recall_change_index_0 = []  # The same recall value may correspond multiple precision values. So we take the largest precision value.
        for i in range(n_sample - 1):
            if recall[i] != recall[i + 1]:
                recall_change_index_0.append(i + 1)
        recall_change_index_1 = recall_change_index_0[0:]
        recall_change_index_0.insert(0, 0)
        recall_change_index_1.append(n_sample)
        precision = np.array(precision)
        recall = np.array(recall)
        for i in range(len(recall_change_index_1)):
            index_0 = recall_change_index_0[i]
            index_1 = recall_change_index_1[i]
            precision[index_0:index_1] = np.max(precision[index_0:])
        unique_precision = []
        unique_precision.append(precision[0])
        for i in range(n_sample - 1):
            if recall[i] != recall[i + 1]:  # Only take precision when recall changes
                unique_precision.append(precision[i + 1])
        n_precision.append(np.mean(unique_precision))

    mAP = np.mean(np.array(n_precision))
    return mAP


def compute_attention_intervention_losses(
    logits,
    logits_keep,
    logits_drop,
    logits_low,
    target_var,
    criterion,
    margin=0.1,
):
    """
    Compute classification and attention intervention losses.
    """
    target_var = target_var.long()

    # 1. Original classification loss
    loss_cls = criterion(logits, target_var)

    # 2. Retaining high-attention patches should preserve recognition
    loss_keep = criterion(logits_keep, target_var)

    # 3. Removing high-attention patches should reduce the target logit
    target_logit = logits.gather(
        1,
        target_var.unsqueeze(1),
    ).squeeze(1)

    dropped_target_logit = logits_drop.gather(
        1,
        target_var.unsqueeze(1),
    ).squeeze(1)

    loss_drop = F.relu(
        margin - (target_logit - dropped_target_logit)
    ).mean()

    # 4. Removing low-attention patches should preserve the original output
    loss_low = F.kl_div(
        F.log_softmax(logits_low, dim=1),
        F.softmax(logits.detach(), dim=1),
        reduction="batchmean",
    )

    return (
        loss_cls,
        loss_keep,
        loss_drop,
        loss_low,
    )


def train(train_loader, test_loader, model, criterion, optimizer):
    best_mAP = 0
    best_prec = 0
    for i in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        total_losses = AverageMeter()
  
        loss_keep_meter = AverageMeter()
        loss_cls_meter = AverageMeter()
        loss_drop_meter = AverageMeter()
        loss_low_meter = AverageMeter()
        top1 = AverageMeter()
        adjust_learning_rate(optimizer, i)
        for j, batch_data in enumerate(tqdm(train_loader)):

            name, full_im, union, obj1, obj2, pair_mask, interaction_mask, bpos, target, pair_indices, img_rel_num = batch_data
            target = target.cuda()

            full_im_var = torch.autograd.Variable(full_im).cuda()
            union_var = torch.autograd.Variable(union).cuda()
            obj1_var = torch.autograd.Variable(obj1).cuda()
            obj2_var = torch.autograd.Variable(obj2).cuda()
            pair_mask = torch.autograd.Variable(pair_mask).cuda()
            interaction_mask = torch.autograd.Variable(interaction_mask).cuda()
            pair_indices = torch.autograd.Variable(pair_indices).cuda()
            bpos_var = torch.autograd.Variable(bpos).cuda()

            target_var = torch.autograd.Variable(target)
            optimizer.zero_grad()

            (
                logits,
                visualization_info,
                logits_keep,
                logits_drop,
                logits_low,
            ) = model.forward_with_interventions(
                full_image=full_im_var,
                union_crop=union_var,
                person1_crop=obj1_var,
                person2_crop=obj2_var,
                pair_mask=pair_mask,
                interaction_mask=interaction_mask,
                spatial_config=bpos_var,
                image_relation_counts=img_rel_num,
                top_k=10,
            )

            # -----------------------------------------------------
            # Attention intervention losses
            # -----------------------------------------------------
            loss_cls, loss_keep, loss_drop, loss_low = (
                compute_attention_intervention_losses(
                    logits=logits,
                    logits_keep=logits_keep,
                    logits_drop=logits_drop,
                    logits_low=logits_low,
                    target_var=target_var,
                    criterion=criterion,
                    margin=0.1,
                )
            )

            total_loss = (
                    loss_cls
                    + 0.2 * loss_keep
                    + 0.2 * loss_drop
                    + 0.1 * loss_low
            )

            prec1 = accuracy(
                logits.detach(),
                target_var,
            )

            top1.update(
                prec1[0],
                target_var.size(0),
            )
            loss_cls_meter.update(
                loss_cls.item(),
                target_var.size(0),
            )
            loss_keep_meter.update(
                loss_keep.item(),
                target_var.size(0),
            )

            loss_drop_meter.update(
                loss_drop.item(),
                target_var.size(0),
            )

            loss_low_meter.update(
                loss_low.item(),
                target_var.size(0),
            )

            total_losses.update(
                total_loss.item(),
                target_var.size(0),
            )

            total_loss.backward()
            optimizer.step()

        print(
            "Train: {}/{}:\tPrec@1 {} ({})".format(
                i,
                len(train_loader),
                top1.val,
                top1.avg,
            )
        )

        print(
            "epoch {}\t"
            "total_loss {:.4f}\t"
            "loss_cls {:.4f}\t"
            "loss_keep {:.4f}\t"
            "loss_drop {:.4f}\t"
            "loss_low {:.4f}".format(
                i,
                total_losses.avg,
                loss_cls_meter.avg,
                loss_keep_meter.avg,
                loss_drop_meter.avg,
                loss_low_meter.avg,
            )
        )

        prec, mAP, recall = validate(test_loader, model)
        if best_prec < prec:
            best_prec = prec
            torch.save({"epoch": i,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       "{}/{}_{}_{:.2%}_{:.2%}.pth.tar".format(args.result_path, args.mode, i, prec, mAP))

        total_losses.reset()
        print(recall)
        print("validate ====>> precision: {:.2%} mAP: {:.2%}".format(prec, mAP))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_graph(rel_num):
    numNode = rel_num
    edge_index = []
    if numNode != 1:
        for i in range(numNode):
            for j in range(numNode):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = [[], []]
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.contiguous()
    return edge_index


def init_network(num_classes):
    from network import network

    model = network(num_classes)

  
    for parameter in model.parameters():
        parameter.requires_grad = False

  
    trainable_modules = [
        model.visual_encoder.projection,
        model.visual_encoder.norm,

  
        model.spatial_encoder,


        model.pair_query_encoder,

      
        model.multi_scale_encoder,


        model.cross_attention,


        model.classifier,
    ]

    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    params = [
        {
            "params": model.visual_encoder.projection.parameters(),
        },
        {
            "params": model.visual_encoder.norm.parameters(),
        },
        {
            "params": model.spatial_encoder.parameters(),
        },
        {
            "params": model.pair_query_encoder.parameters(),
        },
        {
            "params": model.multi_scale_encoder.parameters(),
        },
        {
            "params": model.cross_attention.parameters(),
        },
        {
            "params": model.classifier.parameters(),
        },
    ]

    return model, params


if __name__ == '__main__':
    # Create dataloader
    print('====> Creating dataloader...')
    train_loader, train_set = get_train_set(data_dir, train_list)
    test_loader = get_test_set(data_dir, test_list)

    # load network
    print('====> Loading the network...')
    model, params = init_network(num_class)

    optimizer = torch.optim.Adam(params, weight_decay=0.0005, lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    model.cuda()
    train(train_loader, test_loader, model, criterion, optimizer)

