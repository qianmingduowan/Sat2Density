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
   
   ```

### Install third-party libraries for 

1. Andconda
2. install Cuda11.1 and cudnn (requested by [imaginaire](https://github.com/NVlabs/imaginaire))
   If you are not root user, you can install cuda in you '~' path
3. make sure the 'nvcc -V' \& 'cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2'
    have a right output
4. apt-get update \& apt-get install cmake (if you are not in docker content, you should add 'sudo before every apt-get')
5. conda activate your-env-name
6. pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
   or https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/torch_stable.html
7. test  'torch.cuda.is_available()'
8. bash scripts/install