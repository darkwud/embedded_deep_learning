# Flower Image Classifier — ResNet-18 Transfer Learning

A 5-class flower image classifier built with PyTorch using transfer learning on a ResNet-18 architecture. Developed as part of the EE566 Deep Learning for Embedded Systems course.

## Classes

Daisy, Dandelion, Roses, Sunflowers, Tulips

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | ResNet architecture (BasicBlock + ResNet class) implemented from scratch |
| `train1.py` | Training script with transfer learning, CSV loss logging, and checkpoint saving |
| `predict.py` | Single-image inference script |
| `my_dataset.py` | Custom PyTorch Dataset for loading flower images |
| `utils.py` | Data splitting, training loop, and evaluation helpers |
| `plot_losses.py` | Generates training loss and validation accuracy plots across learning rates |

## Setup

### Dependencies

- Python 3.12
- PyTorch 2.2.2
- Torchvision
- Pillow
- Matplotlib

### Dataset

Download the [flower_photos dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) and extract it to `data/flower_photos/`.

### Pretrained Weights

Download [resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth) and place it in the project root.

## Usage

### Training

```bash
python train1.py --lr 0.001 --epochs 15 --data-path ./data/flower_photos
```

### Inference

```bash
python predict.py
```

### Plot Loss Curves

```bash
python plot_losses.py
```

## Results

Training was conducted with three learning rates over 15 epochs:

| Learning Rate | Final Val Accuracy |
|---|---|
| 0.1 | 48.6% |
| 0.001 | 93.6% |
| 0.00001 | 93.5% |

A learning rate of 0.001 provided the best balance of convergence speed and accuracy for fine-tuning the pretrained model.
