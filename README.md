# RED

**Visual Social Relationship Recognition via Relationship-relevant Regions of Interest**

Relationship Evidence Discovery (RED) predicts the social relationship between two annotated people in a still image. It also localizes Relationship-relevant Regions of Interest (R<sup>2</sup>oIs), visual regions that provide evidence for the target pair's relationship. These regions can span individual appearance, interpersonal interactions, and the surrounding scene.

## Method

- **Dyad-conditioned Evidence Query (DEQ).** A frozen ViT-L/16 encodes the two person crops, their union crop, and the full image. DEQ combines features from the two people and their union region with relative bounding-box geometry.
- **Relationship Evidence Attention (REA).** The query attends to 196 full-image tokens at the 1×1 scale, 169 at 2×2, and 121 at 4×4. Person and interaction masks provide spatial priors for evidence retrieval. The model projects attention back onto the image grid to produce three scale-specific maps and one fused map per pair.
- **Evidence Intervention Loss (EI-Loss).** Training retains the highest-scoring patches in one intervention, masks them in a second, and masks the lowest-scoring patches in a third.

The supplied implementation uses a single fusion MLP and recomputes evidence attention on masked patch features during training. Sections 3.6.1 and 3.6.2 of the paper describe the dual-branch classifier and attention-reweighting formulation.

## Environment

Training requires a CUDA-capable NVIDIA GPU. Use Python 3.10 for the PyTorch 1.12.1 and TorchVision 0.13.1 CUDA 11.3 builds recorded in `environment.txt`. The environment snapshot contains machine-specific build paths. Install the packages required by the training scripts in a clean environment using the commands below. See the [PyTorch installation archive](https://pytorch.org/get-started/previous-versions/) for other CUDA builds.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install timm==1.0.27 numpy==1.25.2 pandas==2.3.3 Pillow==9.5.0 tqdm==4.67.1
```

The entry point `train_vit.py` imports the model from `network_pced.py` and the data loader from `dataset_llm.py`. Keep the three scripts in the same directory under these names.

## Datasets

- **People in Social Context (PISC).** Download the images, person annotations, relationship labels, and splits from the [official PISC release](https://zenodo.org/records/1059155).
- **People in Photo Album (PIPA).** Use the PIPA images and PIPA-relation annotations and splits. Access instructions are available on the [official PIPA-relation project page](https://www.mpi-inf.mpg.de/social-relation).

| Training mode | Benchmark | Relationship labels | Paper metric |
| --- | --- | --- | --- |
| `PISC_Coarse` | PISC-C | 3 classes (Intimate, Non-intimate, No Relation) | mAP |
| `PISC_Fine` | PISC-F | 6 classes (Friend, Family, Couple, Professional, Commercial, No Relation) | mAP |
| `PIPA` | PIPA-relation | 16 relationship classes across 5 social domains | Top-1 accuracy |

## Data Preparation

Prepare a directory of RGB images and two annotation pickle files, one for the training split and one for the test split. Each pickle contains a list of image records with the following keys.

| Key | Expected value |
| --- | --- |
| `img` | Image filename or relative path under the image directory. |
| `Pbboxs` | Person bounding boxes in original image pixels, with coordinates ordered as `[x_min, y_min, x_max, y_max]`. |
| `person_pair` | Integer NumPy array of shape `(R, 2)`. Each row contains two zero-based indices into `Pbboxs`. |
| `rel` | A sequence of `R` zero-based relationship class IDs in the same order as `person_pair`. |

Use the official splits and keep the class mapping consistent in both files. The loader generates the two person crops, the union crop, spatial masks, and the six-dimensional geometry descriptor from these annotations.

## Training

Before running a training command, configure these paths and the GPU index.

| Location | Setting |
| --- | --- |
| `train_vit.py` | Set `data_dir`, `train_list`, and `test_list` in the block for your `--mode`. They point to the image directory, training pickle, and test pickle, respectively. |
| `train_vit.py` | Set `CUDA_VISIBLE_DEVICES` to an available GPU index. Use `"0"` on a machine with a single visible GPU. |
| `network_pced.py` | Set `DEFAULT_VIT_CHECKPOINT` to your local copy of [`jx_vit_large_p16_224-4ee7a4dc.pth`](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth). |

Run the command for your dataset from the directory containing the scripts.

```bash
mkdir -p checkpoints/PISC_Coarse checkpoints/PISC_Fine checkpoints/PIPA

python train_vit.py --mode PISC_Coarse --result_path checkpoints/PISC_Coarse
python train_vit.py --mode PISC_Fine --result_path checkpoints/PISC_Fine
python train_vit.py --mode PIPA --result_path checkpoints/PIPA
```

### Command-line options

| Argument | Default | Description |
| --- | --- | --- |
| `--mode` | `PISC_Coarse` | Select `PISC_Coarse`, `PISC_Fine`, or `PIPA`. |
| `-b`, `--batch_size` | `64` | Number of images per batch. Each image can contain multiple annotated pairs. |
| `-j`, `--workers` | `4` | Number of data-loading workers. |
| `--epochs` | `100` | Number of training epochs. |
| `--lr` | `0.0001` | Initial learning rate for Adam. |
| `--lr_step` | `20` | Learning-rate decay interval in epochs. Each step multiplies the rate by 0.1. |
| `--start_epoch` | `1` | First epoch index. |
| `--scale-size` | `256` | Resize size for person and union crops before random cropping during training. |
| `--crop-size` | `224` | Person and union crop size. Use 224 for the supplied loader and ViT-L/16 backbone. |
| `--result_path` | Local path in `train_vit.py` | Existing directory for model checkpoints. |

Training uses Adam with a weight decay of `0.0005`. The intervention settings in `train_vit.py` use 10 patches per ranking group, a removal margin of `0.1`, and loss weights of `0.2`, `0.2`, and `0.1` for the keep, drop, and low-attention terms, respectively.

## Evaluation and Evidence Maps

The training script evaluates the configured test split after each epoch. It reports top-1 accuracy (printed as `precision`), mean average precision (mAP), and recall for classes with at least one correct prediction. A checkpoint is saved when top-1 accuracy improves. The file contains the epoch, model weights, and optimizer state and follows the naming pattern `<mode>_<epoch>_<accuracy>_<mAP>.pth.tar`.

For evidence visualization, `network.forward()` returns `(logits, visualization_info)`. The dictionary contains `heatmap_1x1`, `heatmap_2x2`, `heatmap_4x4`, and `heatmap_fused`. Each map has shape `[R, 1, 224, 224]`, where `R` is the total number of target pairs in the batch.

## Contact

For questions about the paper or code, open a GitHub issue or contact [Wang Tang](mailto:wangtang@cuit.edu.cn).
