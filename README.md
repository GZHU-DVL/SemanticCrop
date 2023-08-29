# SemanticCrop
## Code for SemanticCrop: Boosting Contrastive Learning via Semantic-cropped Views
Abstract:Siamese-structure-based contrastive learning has shown excellent performance in learning visual representations due to its ability to minimize the distance between positive pairs and increase the distance between negative pairs. Existing works mostly employ RandomCrop or ContrastiveCrop to obtain positive pairs of an image. However, RandomCrop causes the cropped views to contain many useless backgrounds, while ContrastiveCrop produces positive pairs that are too similar. In  this paper, we propose a novel SemanticCrop to yield cropped views containing as much semantic information as possible. Specifically, SemanticCrop first computes a heatmap of an image. Then, an empirical threshold is tuned to box out a semantic region whose heatmap values are over this threshold. Finally, we design a center-suppressed probabilistic sampling to avoid excessive similarity between positive pairs, making the cropped view contain more parts of an object. As a plug-and-play module, the MoCo, SimCLR, SimSiam, and BYOL models equipped with our SemanticCrop module achieve an accuracy improvement from 0.5% to 2.34% on the CIFAR10, CIFAR100, IN-200, and IN-1K datasets.

# Usage

## Requirement

- `pytorch >= 1.8.1`.
- `pip install -r requirements.txt`

#  Dataset Preparation

Please  organize the datasets in this structure:

```
├── data/
    ├── ImageNet/
    │   ├── train/ 
    │   ├── val/
    ├── cifar-10-batches-py/
    ├── cifar-100-python/
    ├── tiny-imagenet-200/
    │   ├── train/
    │   ├── val/
```

# Training and Evaluating

## Pre-train

```
# MoCo, CIFAR-10, CCrop
python DDP_moco_ccrop.py configs/small/cifar10/moco_ccrop.py

# SimSiam, CIFAR-100, CCrop
python DDP_simsiam_ccrop.py configs/small/cifar100/simsiam_ccrop.py

# MoCo V2, IN-200, CCrop
python DDP_moco_ccrop.py configs/IN200/mocov2_ccrop.py

# MoCo V2, IN-1K, CCrop
python DDP_moco_ccrop.py configs/IN1K/mocov2_ccrop.py
```

If you want to change the path of the dataset, please locate the corresponding parameter in the "configs" folder and make the replacement.

## Linear Evaluation

```
# CIFAR-10
python DDP_linear.py configs/linear/cifar10_res18.py --load ./checkpoints/small/cifar10/moco_ccrop/last.pth

# CIFAR-100
python DDP_linear.py configs/linear/cifar100_res18.py --load ./checkpoints/small/cifar100/simsiam_ccrop/last.pth

# IN-200 
python DDP_linear.py configs/linear/IN200_res50.py --load ./checkpoints/IN200/mocov2_ccrop/last.pth

# IN-1K
python DDP_linear.py configs/linear/IN1K_res50.py --load ./checkpoints/IN1K/mocov2_ccrop/last.pth
```
