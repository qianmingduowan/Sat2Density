1. Andconda
2. install Cuda11.1 and cudnn (requested by [imaginaire](https://github.com/NVlabs/imaginaire))
   If you are not root user, you can install cuda in you '~' path
3. make sure the 'nvcc -V' \& 'cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2'
    have a right output
4. apt-get update \& apt-get install cmake (if you are not in docker content, you should add 'sudo before every apt-get')
5. conda activate your-env-name (make sure python>3.8)
6. pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
   or https://mirror.sjtu.edu.cn/pytorch-wheels/cu111/torch_stable.html
7. test  'torch.cuda.is_available()'
8. bash scripts/install