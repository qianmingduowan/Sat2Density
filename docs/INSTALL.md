# Sat2Density
## Installation Guide
### Prerequisite
- [Anaconda3](https://www.anaconda.com/products/individual) or [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)
- CUDA 11.1 (or higher) and CUDNN
   - Check your cuda version and CUDNN availability by runnning the following command line
   ```
   nvcc -V && cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```
- [CMake]() 

### Create a conda environment and install required packages
   ```
   conda create -n sat2density python=3.9
   conda activate sat2density 
   # Install PyTorch in the first, please visit https://pytorch.org/get-started/locally/ for the details
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   pip install -r requirements.txt 
   ```
### Install third-party libs requested by [imaginaire](https://github.com/NVlabs/imaginaire/tree/master)
   ```
   CURRENT=$(pwd)
   for p in correlation channelnorm resample2d bias_act upfirdn2d; do
   cd imaginaire/third_party/${p};
   python setup.py install;
   cd ${CURRENT};
   done
   ```
