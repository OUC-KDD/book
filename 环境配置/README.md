# 安装

## Windows

## Linux

## macOS
``` shell
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

conda create -n OCEAN python=3.11
conda activate OCEAN

pip install torch torchvision
```