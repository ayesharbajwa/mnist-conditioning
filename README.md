# mnist-conditioning
MIT 18.0651 Final Project

## Requirements
- Python 3
- Numpy
- Tensorflow

## Running
The following MNIST models are supported:
- `simple`: single-layer softmax
- `cnn`: multi-layer CNN

Each model's performance can be tested against the following adversarial attacks:
- FGSM (Fast Gradient Sign Method)
- FGMT (Fast Gradient Method with Target)

The commands below will train the specified model, generate the specified adversarial examples, and test the model's performance against the generated example:
```
# simple model
python adv_mnist_simple.py [fgsm|fgmt]

# cnn model
python fgsm_mnist_conv.py     # for FGSM
python fgmt_mnist_conv.py     # for FGMT

```
