import os

from urllib.request import urlretrieve


def download(url, name):
    models_path = os.path.join(os.path.dirname(__file__), 'models')

    if not os.path.exists(models_path):
        os.mkdir(models_path)

    urlretrieve(url, os.path.join(models_path, name))


if __name__ == '__main__':
    download('http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy', 'AlexNet.npy')
    download('https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz', 'VGGNet16.npz')
