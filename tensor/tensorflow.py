import torch
import tensorflow as tf
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    print(device_lib.list_local_devices())
    tf.config.list_physical_devices('GPU')

    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)

    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    print('학습을 진행하는 기기:', device)
