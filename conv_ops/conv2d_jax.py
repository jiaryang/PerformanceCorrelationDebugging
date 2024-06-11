import os
import jax
import jax.numpy as jnp
import time
import argparse
from jax import random
import flax.linen as nn
from jax.lib import xla_bridge

def append_xla_flag(new_flag):

    current_xla_flags = os.environ.get('XLA_FLAGS', '')

    if current_xla_flags:
        new_xla_flags = current_xla_flags + ' ' + new_flag
    else:
        new_xla_flags = new_flag

    os.environ['XLA_FLAGS'] = new_xla_flags
    print(f"XLA_FLAGS: {os.environ.get('XLA_FLAGS', '')}")

class Conv2D(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding=((self.padding, self.padding), (self.padding, self.padding)),
            dtype=self.dtype
        )(x)
        return x

def convolution_test(batch_size, input_size, in_channels, out_channels, kernel_size, stride, padding, dtype, iters):
    key = random.PRNGKey(0)
    
    # Create a random input tensor with the specified size in NHWC format
    x = random.normal(key, (batch_size, input_size, input_size, in_channels)).astype(dtype)

    # Define the convolution layer
    conv_layer = Conv2D(in_channels, out_channels, kernel_size, stride, padding, dtype=dtype)
    params = conv_layer.init(key, x)

    # JIT compile the forward pass
    @jax.jit
    def apply_conv(params, x):
        return conv_layer.apply(params, x)
    
    # Measure the execution time of the convolution on the GPU
    xla_device = xla_bridge.get_backend().platform
    print(f"Using device: {xla_device}")
    
    start_time = time.time()
    for i in range(iters):
        output = apply_conv(params, x).block_until_ready()
    end_time = time.time()

    # Print the results
    print(f"Input size: {x.shape}")
    print(f"Output size: {output.shape}")
    print(f"Execution time: {end_time - start_time} seconds")

def readArgv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hi', help='Height of input', type=int, required=True)
    parser.add_argument('--wi', help='Width of input', type=int, required=True)
    parser.add_argument('--ci', help='Number of channels of input', type=int, required=True)
    parser.add_argument('--co', help='Number of channels of output', type=int, required=True)
    parser.add_argument('-k', help='Kernel size', type=int, required=True)
    parser.add_argument('-n', help='Batch size', type=int, required=True)
    parser.add_argument('--stride', help='stride steps of conv kernel', type=int, required=True)
    parser.add_argument('--dtype', help='Data type of input', type=str, default='bf16')
    parser.add_argument('--iters', help='iterations for benchmarking', type=int, default=100)
    parser.add_argument('--data_format', choices=['nhwc', 'nchw'], default='nhwc', help='Data format, either nhwc or nchw, default is nhwc')

    args = parser.parse_args()

    return args


def main():
    args = readArgv()

    # Configurable parameters
    batch_size = args.n
    input_size = args.hi  # configurable input size
    in_channels = args.ci  # number of input channels (e.g., for RGB images)
    out_channels = args.co  # number of output channels (filters)
    kernel_size = args.k  # size of the convolutional kernel
    stride = args.stride  # stride of the convolution
    padding = 0  # padding of the input

    dtype = jnp.bfloat16
    if args.dtype == 'fp16':
        dtype = jnp.float16
    elif args.dtype == 'fp32':
        dtype = jnp.float32

    if args.k == 3:
        padding = 1
    elif args.k == 7:
        padding = 3

    if args.data_format == 'nhwc':
        append_xla_flag('--xla_gpu_force_conv_nhwc')

    # Print the parameters before running the convolution test
    """
    print(f"Parameters: batch_size={batch_size}, input_size={input_size}, "
          f"in_channels={in_channels}, out_channels={out_channels}, "
          f"kernel_size={kernel_size}, stride={stride}, padding={padding}, "
          f"dtype={dtype}, data_format={args.data_format}, iters={args.iters}")
    """

    # Run the convolution test with time profiling
    convolution_test(batch_size, input_size, in_channels, out_channels, kernel_size, stride, padding, dtype, args.iters)

if __name__ == '__main__':
    main()
