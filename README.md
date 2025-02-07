### 1. Prerequisites

**Dependencies**

- Ubuntu >= 20.04
- CUDA >= 11.3
- pytorch==1.12.1
- torchvision=0.13.1
- mmcv==2.0.0rc4
- mmengine==0.7.3
- MMDetection==3.0.0rc6

**Step 0.** Create Conda Environment

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 1.** Install [Pytorch](https://pytorch.org)

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

**Step 2.** Install [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install "mmengine==0.7.3"
mim install "mmcv==2.0.0rc4"
```

**Step 3.** Install 

```shell
pip install -v -e .
```
### 2. Training

```shell
python tools/train.py configs/adkd/${CONFIG_FILE} [optional arguments]
```

### 3. Evaluation

```shell
python tools/test.py configs/adkd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```

