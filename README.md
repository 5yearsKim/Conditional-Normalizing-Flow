## Glow in PyTorch

![CIFAR-10 Samples](/samples/epoch_80.png?raw=true "CIFAR-10 Samples")

Implementation of Glow in PyTorch. Based on the paper:

  > [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)\
  > Diederik P. Kingma, Prafulla Dhariwal\
  > _arXiv:1807.03039_

Training script and hyperparameters designed to match the
CIFAR-10 experiments described in Table 4 of the paper.


## Usage

### Environment Setup
  1. Make sure you have [Anaconda or Miniconda](https://conda.io/docs/download.html)
  installed.
  2. Clone repo with `git clone https://github.com/chrischute/glow.git glow`.
  3. Go into the cloned repo: `cd glow`.
  4. Create the environment: `conda env create -f environment.yml`.
  5. Activate the environment: `source activate glow`.

### Train
  1. Make sure you've created and activated the conda environment as described above.
  2. Run `python train.py -h` to see options.
  3. Run `python train.py [FLAGS]` to train. *E.g.,* run
  `python train.py` for the default configuration, or run
  `python train.py --gpu_ids=0,1` to run on
  2 GPUs instead of the default of 1 GPU. This will also double the batch size.
  4. At the end of each epoch, samples from the model will be saved to
  `samples/epoch_N.png`, where `N` is the epoch number.


A single epoch takes about 30 minutes with the default hyperparameters (K=32, L=3, C=512) on two 1080 Ti's.


## Samples (K=16, L=3, C=512)

### Epoch 10

![Samples at Epoch 10](/samples/epoch_10.png?raw=true "Samples at Epoch 10")


### Epoch 20

![Samples at Epoch 20](/samples/epoch_20.png?raw=true "Samples at Epoch 20")


### Epoch 30

![Samples at Epoch 30](/samples/epoch_30.png?raw=true "Samples at Epoch 30")


### Epoch 40

![Samples at Epoch 40](/samples/epoch_40.png?raw=true "Samples at Epoch 40")


### Epoch 50

![Samples at Epoch 50](/samples/epoch_50.png?raw=true "Samples at Epoch 50")


### Epoch 60

![Samples at Epoch 60](/samples/epoch_60.png?raw=true "Samples at Epoch 60")


### Epoch 70

![Samples at Epoch 70](/samples/epoch_70.png?raw=true "Samples at Epoch 70")


### Epoch 80

![Samples at Epoch 80](/samples/epoch_80.png?raw=true "Samples at Epoch 80")


More samples can be found in the `samples` folder.


## Results (K=32, L=3 C=512)

### Bits per Dimension

| Epoch | Train | Valid |
|-------|-------|-------|
| 10    | 3.64  | 3.63  |
| 20    | 3.51  | 3.56  |
| 30    | 3.46  | 3.53  |
| 40    | 3.43  | 3.51  |
| 50    | 3.42  | 3.50  |
| 60    | 3.40  | 3.51  |
| 70    | 3.39  | 3.49  |
| 80    | 3.38  | 3.49  |
# Conditional-Normalizing-Flow
