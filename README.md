## Implementation of sparsification and quantization techniques in distributed machine learning models

## How To Run

python distributed_nn.py --network ResNet --dataset Cifar10

Options:

    --network: VGG11/Resnet50/LeNet
    --dataset: MNIST/Cifar10

## Algorithmic Details

### Quantized Stochasitc Gradient compression

Only gradients are pushed and pulled from the server and not the weights of the model, thereby reducing communication cost.
When averaging those gradients or updating of the models, the gradients are decompressed for computation.

### Top-k Sparsification 
To further reduce the communication data Top-k sparsification technique is applied before QSGD. 
Absolute highest top K percentile values are selected and all other values are made 0. Since we can assume that only Top-k gradients 
dominate the weight updation hence other non significant gradients are assumed to be 0 and not sent by the workers to the server.

## Results
The communication cost among the distributed workers is reduced from 148 MB to 1.48 MB in VGG-11 model and 
from 6.56 MB to 0.06 MB in LeNet model.
