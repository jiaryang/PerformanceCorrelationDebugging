import torch
import torch.nn as nn
import time
import argparse
import os
from torch import Tensor

os.environ['PYTORCH_MIOPEN_SUGGEST_NHWC'] = '1'
class Conv2D(nn.Module):
    def __init__(self, atch_size, input_size, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x: Tensor) -> Tensor:
        return self.conv_layer(x)

def convolution_test(batch_size, input_size, in_channels, out_channels, kernel_size, stride, padding, dtype, iters):
    # Create a random input tensor with the specified size in NHWC format
    x = torch.randn((batch_size, in_channels, input_size, input_size), dtype=dtype).cuda() #.permute(0, 3, 1, 2)

    # Move the input tensor to the GPU, if available
    x = x.to(memory_format=torch.channels_last)

    # Define a simple convolutional layer with NHWC support
    conv_layer = Conv2D(batch_size, input_size, in_channels, out_channels, kernel_size, stride, padding).cuda()
    conv_layer = conv_layer.to(memory_format=torch.channels_last, dtype=dtype)
    #conv_layer = torch.compile(conv_layer)
    conv_layer.eval()

    # Measure the execution time of the convolution on the GPU
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        for i in range(iters):
            output = conv_layer(x)
    torch.cuda.synchronize()
    end_time = time.time()

    # Move the output tensor back to the CPU for printing and change format to NHWC

    # Print the results
    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")
    print(f"GPU Execution time: {end_time - start_time} seconds")

def readArgv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hi', help='Height of input', type=int, required=True)
    parser.add_argument('--wi', help='Weight of input', type=int, required=True)
    parser.add_argument('--ci', help='Number of channels of input', type=int, required=True)
    parser.add_argument('--co', help='Number of channels of output', type=int, required=True)
    parser.add_argument('-k', help='Kernel size', type=int, required=True)
    parser.add_argument('-n', help='Batch size', type=int, required=True)
    parser.add_argument('--stride', help='stride steps of conv kenel', type=int, required=True)
    parser.add_argument('--dtype', help='Height of input', type=str, default='bf16')
    parser.add_argument('--iters', help='iterations for benchmarking', type=int, default=100)

    args = parser.parse_args()

    return args


args=readArgv()
# Configurable parameters
batch_size = args.n
input_size = args.hi  # configurable input size
in_channels = args.ci  # number of input channels (e.g., for RGB images)
out_channels = args.co  # number of output channels (filters)
kernel_size = args.k  # size of the convolutional kernel
stride = args.stride  # stride of the convolution
padding = 0  # padding of the input
dtype=torch.bfloat16
if args.dtype == 'fp16':
    dtype=torch.float16
elif args.dtype == 'fp32':
    dtype=torch.float32

if args.k == 3:
    padding = 1
elif args.k == 7:
    padding = 3

# Run the convolution test with time profiling
convolution_test(batch_size, input_size, in_channels, out_channels, kernel_size, stride, padding, dtype, args.iters)

