### CUDA优化之 LayerNorm性能优化实践
在2020年末，OneFlow发布了Softmax的CUDA性能优化文章 [如何实现一个高效的Softmax CUDA kernel？——OneFlow 性能优化分享](https://zhuanlan.zhihu.com/p/341059988) ，性能大幅超过了CUDNN的Softmax实现，尤其对很多框架没有考虑的half类型也做了充分优化。如今经过一年迭代，Softmax的接口和实现更加成熟，趋于稳定，主要包括以下几个优化：

1、优化了小的num_cols下的性能。

2、同时支持了Softmax和LogSoftmax，适用场景更广。

3、输入输出通过Load/Store结构传递，解耦数据IO和计算，只需要加几行代码就可以快速支持Softmax和其他Kernel Fuse，减少带宽需求，带来很高的性能收益。

Softmax代码在https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/cuda/softmax.cuh 下，可以脱离OneFlow代码直接编译，在include 头文件后，使用以下几行代码就可以实现一个Softmax GPU Kernel，将下面的DispatchSoftmax换成DispatchLogSoftmax就可以实现一个LogSoftmax Kernel。

```
    oneflow::cuda::softmax::DirectLoad<half, float> load(in, cols);
    oneflow::cuda::softmax::DirectStore<float, half> store(out, cols);
    oneflow::cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(
        cuda_stream, load, store, rows, cols);
```

#### LayerNorm性能优化
LayerNorm是语言模型中常用的操作之一，其CUDA Kernel实现的高效性会影响很多网络最终的训练速度，Softmax的优化方法也适用于LayerNorm，LayerNorm的数据也可以表示为(num_rows, num_cols)，计算过程中对每一行的元素做Reduce操作求均值方差。因此我们使用了和Softmax同样的优化方法来优化LayerNorm操作，本文以LayerNorm前向计算为例进行介绍。

##### OneFlow 与 NVIDIA Apex的对比结果

NVIDIA Apex中实现了高效的fused LayerNorm Kernel来扩展PyTorch 算子，我们对OneFlow优化后的LayerNorm Kernel和NVIDIA Apex进行了对比测试，测试结果如下：

横轴为num_cols大小，纵轴为Kernel执行需要的时间(越低越好)：
![image](https://user-images.githubusercontent.com/25500633/145202567-73fb2e94-27fa-4ed6-b3d5-1efa81757458.png)

我们将时间换算成访存带宽，结果如下，纵轴为Kernel达到的有效带宽(越高越好)：

![image](https://user-images.githubusercontent.com/25500633/144731406-04c8c93e-12d9-47d0-89f0-47f01f06d59c.png)

其中测试环境为NVIDIA A100-PCIE-40GB GPU，数据类型为half， Shape =（49152， num_cols），我们将最后一维动态变化，测试了从32到32768不同大小的LayerNorm Kernel，可以看到在所有情况下，OneFlow的Kernel执行时间和有效访存带宽都优于Apex的实现。

##### LayerNorm计算方法

以PyTorch为例，LayerNorm的接口为:
<img width="630" alt="image-20211113230817847" src="https://user-images.githubusercontent.com/25500633/141715671-de6de2fc-7995-43ba-a7f5-1e3a5a84ab0c.png">

其中input 形状为：`[∗×normalized_shape[0]×normalized_shape[1]×…×normalized_shape[−1]]`

第一个参数normalized_shape只能是x_shape的后几维，例如x_shape为(N, C, H, W),  normalized_shape 可以是（W）， (H, W) ，(C, H, W)或(N, C, H, W)。输入x在normalized_shape这几维上求均值和方差。
第三个参数elementwise_affine代表是否要对normalize的结果做变换，即normalize的结果乘gamma，加beta。若elementwise_affine=True，就多了两个模型参数gamma和beta，形状为normalized_shape。
<p align="center">
<img width="300" alt="image-20211113231114822" src="https://user-images.githubusercontent.com/25500633/141715686-9864ee73-7c6a-4ffa-97e5-ddece51dfd63.png">
</p>
例如对于输入x形状为(N, C, H, W)， normalized_shape为(H, W)的情况，可以理解为输入x为(N*C, H*W) ，在N*C个行上，每行有H*W个元素，对每行的元素求均值和方差，得到N*C 个mean和inv_variance，再对输入按如下LayerNorm的计算公式计算得到y。若elementwise_affine=True，则有H*W 个gamma和beta，对每行H*W个的元素做变换。

<img width="260" alt="image-20211113230455783" src="https://user-images.githubusercontent.com/25500633/141715678-5eea5729-82ca-49a6-b9f6-124e86c70cc7.png">                            <img width="300" alt="image-20211121204300613" src="https://user-images.githubusercontent.com/25500633/143554234-76804f3f-1d4b-42e8-b1b8-2a25db8a6a74.png">


##### LayerNorm中求方差的方法：

常见的求方差的方法有two pass 方法、naive 方法、和Welford 算法，本文摘录一些关键的公式和结论，详细的介绍和推导可参考：[Wiki: Algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) ，和  [GiantPandaCV: 用Welford算法实现LN的方差更新](https://mp.weixin.qq.com/s/t0x782mDkMo-ZBVEbK8gPg) 

- two-pass方法

使用的公式是：
<p align="center">
<img width="212" alt="image-20211126170943291" src="https://user-images.githubusercontent.com/25500633/143562521-6c12772a-96aa-4155-ab23-54d5e9f2195b.png">
</p>
two-pass是指这种方法需要遍历两遍数据，第一遍累加x得到均值，第二遍用上面公式计算得到方差。这种方法在n比较小时仍然是数值稳定的。


- naive方法

使用的公式是：
<p align="center">
<img width="324" alt="image-20211126171441597" src="https://user-images.githubusercontent.com/25500633/143562873-235392a5-56c4-4b96-b49c-c70566a79bff.png">
</p>

这种方法是一种single pass方法，在计算方差时只需要遍历一遍数据累加x的平方及累加x，最后按上述公式计算得到方差。这种方法只需要遍历一遍数据，相比two-pass的算法，更容易达到好的性能，但是上面的Wiki参考链接中介绍由于SumSquare和(Sum×Sum)/n可能非常接近，可能会导致计算结果损失精度较大，因此这种方法不建议在实践中使用。


- welford算法

  使用的公式是：
           
<p align="center">
<img width="313" alt="image-20211126174959846" src="https://user-images.githubusercontent.com/25500633/143563076-411e6ed9-1329-410e-b430-6eaa1c9d41f0.png">
</p>

welford算法也是一种single pass方法，且数值稳定性很好，因此现在很多框架都采用这种方法。本文的代码中采用的也是welford方法。

##### OneFlow深度优化LayerNorm CUDA Kernel的技巧

和Softmax一样，LayerNorm也采用分段函数优化，对于不同的num_cols范围，采用不同的实现，以在各种情况下都能达到较高的有效带宽。在每种实现中都采用了一个公共的优化：向量化访存，NVIDIA性能优化的博客[Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)中提到可以通过向量化内存操作来提高CUDA Kernel性能，很多CUDA Kernel都是带宽受限的，使用向量化内存操作可以减少总的指令数，减少延迟，提高带宽利用率。

理论上来说，在计算LayerNorm的过程中，输入x需要读两次，第一次用于计算均值和方差。第二次用于得到均值和方差后的计算过程。而对Global Memory的访问操作是昂贵的，如果能将输入x先存起来，不重复读，就可以提升性能。在GPU中将输入x存起来可以使用寄存器或Shared memory，但是寄存器资源和Shared memory资源都是有限的，如果num_cols过大，就会超出资源的使用限制，因此我们针对不同num_cols采用不同的实现，下面分别进行介绍：

1、针对num_cols <= 1024的情况，以Warp为单位处理一行或两行，将输入x存储到寄存器中。

<p align="center">
<img width="472" alt="image-20211128123219430" src="https://user-images.githubusercontent.com/25500633/143731660-5261fe1e-d2f4-494f-9100-e45a66fd7ec3.png">
</p>

硬件上并行执行的32个线程称之为一个Warp，同一个Warp的32个thread执行同一条指令， Warp是GPU调度执行的基本单元。线程块和元素的对应关系如上图所示，每个Warp的threads处理一行元素，每个block有block_size / warp_size 个Warp，每个block处理block_size / warp_size行元素。

具体的处理流程是，如下图所示，每行有num_cols个元素，每个warp处理一行，因此每个线程需要处理num_cols / warp_size个元素，每个线程读取自己需要处理的元素存储到寄存器中，并用welford算法计算好均值和方差后，Warp中的所有线程执行一次WelfordWarpAllReduce，这样每个线程上就得到了正确的均值和方差参与后续计算。
<p align="center">
<img width="475" alt="image-20211128134337824" src="https://user-images.githubusercontent.com/25500633/143731675-e657a1c9-f5eb-4c0c-8e43-682c6ddae55e.png">
</p>

WelfordWarpAllReduce 由WelfordWarpReduce和Broadcast操作完成，WelfordWarpReduce借助Warp级别同步原语`__shfl_down_sync`实现，Broadcast操作借助`__shfl_sync`实现，代码如下：

```
template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                             T* m2, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                                T* m2, T* count) {
  WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}
```

在这里有个模板参数thread_group_width，当num_cols > pack_size * WarpSize 时，thread_group_width为WarpSize。当num_cols太小，即num_cols<pack_size * WarpSize时，一个Warp内的线程不是全部处理有效的值，此时我们采用更小的thread_group_width，取值可能是16、8、4、2、1，由num_cols决定，并且每个线程处理两行增加并行度。

此外，在读写输入输出时，我们采用向量化访存的优化，在满足条件时，将pack_size个元素pack成更大的数据类型读入，下图为pack_size=2时的示意图，每个线程以更大的数据类型读入元素，可以更好的利用显存带宽。
<p align="center">
<img width="471" alt="image-20211128125204168" src="https://user-images.githubusercontent.com/25500633/143731697-f017d863-4b9a-434d-9866-4d8ddeb09515.png">
</p>
将pack_size个元素pack成更大的数据类型读入，但是x还要参与计算。因此我们定义一个union结构的Pack类型，storage用于从Global Memory中读写，做计算时用elem[i] 取每个元素参与计算，Pack类型定义如下:

```
template<typename T, int N>
union Pack {
  PackType<T, N> storage;
  T elem[N];
};
```

LayerNormWarpImpl Kernel代码如下：

LayerNormWarpImpl的实现有以下几个模板参数：

LOAD、STORE分别代表输入输出，使用`load.template load<pack_size>(ptr, row_id, col_id);`和`store.template store<pack_size>(ptr, row_id, col_id);` 进行读取和写入。使用LOAD和STORE有两个好处：1、可以在CUDA Kernel中只关心计算类型ComputeType，而不用关心具体的数据类型T。2、只需要加几行代码就可以快速支持LayerNorm和其他Kernel Fuse，减少带宽需求，提升整体性能。

ComputeType代表计算类型。pack_size代表向量化访存操作的pack元素的个数，我们将几个元素pack起来读写，提升带宽利用率。cols_per_thread代表每个线程处理的元素个数。

thread_group_width代表处理元素的线程组的宽度，当cols > pack_size * warp_size时，thread_group_width就是warp_size，即32。当cols < pack_size * warp_size时，就根据cols大小用1/2个warp或1/4个warp来处理每行的元素。采用更小的thread_group_width后，WarpAllReduce需要执行的轮次也相应减少。

rows_per_access代表每个thread_group一次处理的行数，当cols较小，thread_group_width不是warp_size 32时，若rows能被2整除，我们就让每个线程处理2行来增加指令并行度，从而提升性能。

padding代表当前是否做了padding，若cols不是warp_size的整数倍，我们会把它padding到最近的整数倍处理。

```
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean,
                                  ComputeType* inv_variance) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t num_global_thread_group = gridDim.x * blockDim.y;
  const int64_t lane_id = threadIdx.x;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows;
       row += num_global_thread_group * rows_per_access) {
    ComputeType thread_mean[rows_per_access];
    ComputeType thread_m2[rows_per_access];
    ComputeType thread_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_mean[row_id] = 0;
      thread_m2[row_id] = 0;
      thread_count[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            WelfordCombine(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                           thread_count + row_id);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = 0; }
        }
      }
    }
    ComputeType warp_mean[rows_per_access];
    ComputeType warp_m2[rows_per_access];
    ComputeType warp_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      int global_row_id = row + row_id;
      ComputeType* row_buf = buf[row_id];
      WelfordWarpAllReduce<ComputeType, thread_group_width>(
          thread_mean[row_id], thread_m2[row_id], thread_count[row_id], warp_mean + row_id,
          warp_m2 + row_id, warp_count + row_id);
      ComputeType row_mean = warp_mean[row_id];
      ComputeType row_variance =
          max(Div(warp_m2[row_id], warp_count[row_id]), static_cast<ComputeType>(0.0));
      ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      if (lane_id == 0) {
        mean[global_row_id] = row_mean;
        inv_variance[global_row_id] = row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
        }
      }
    }
  }
}
```


2、针对num_cols>2048 ，以Block为单位处理一行，利用Shared Memory存储输入数据

对于num_cols > 2048的情况，每个block处理一行元素，将输入x存储到Shared Memory中。

<p align="center">
<img width="458" alt="image-20211128142857055" src="https://user-images.githubusercontent.com/25500633/143733509-2657d032-2b0d-4d6f-9a60-bc0bcdb386c4.png">
</p>

具体的处理流程是，如下图所示，每行有num_cols个元素，每个block处理一行，因此每个线程需要处理num_cols / block_size个元素，每个线程读取自己需要处理的元素存储到Shared Memory中，并用welford算法计算好均值和方差后，block中的所有线程执行一次WelfordBlockAllReduce，这样每个线程上就得到了正确的均值和方差参与后续计算。

<p align="center">
<img width="481" alt="image-20211128143023612" src="https://user-images.githubusercontent.com/25500633/143733520-ddf20e76-407c-4ff3-a17d-f0138f6ba019.png">
</p>

WelfordBlockAllReduce是借助WelfordWarpReduce操作完成的，具体逻辑是，一个Block中最多有32个Warp，对所有的Warp先执行一次WelfordWarpReduce，执行完后，每个warp中的第一个线程，即lane_id=0的线程上得到当前WelfordWarpReduce的结果，再将每个Warp的第一个线程的结果拷贝到一块Shared Memory buffer中，再用第一个Warp的32个线程执行一次WelfordWarpReduce，此时第一个Warp中的lane_id=0的线程上得到的就是block中所有线程reduce的结果。再借助Shared Memory，将该结果broadcast到block中的所有线程上，即完成了WelfordBlockAllReduce的操作。

值得注意的是GPU上Shared Memory资源同样有限，当num_cols超过一定范围时需要占用的Shared Memory可能就超出了最大限制，Kernel就无法启动起来，因此，我们采用cudaOccupancyMaxActiveBlocksPerMultiprocessor函数判断当前硬件资源条件下Kernel是否能成功启动，仅在返回值大于0时采用这种方案。此外，由于 Block 内线程要做同步，当 SM 中正在调度执行的一个 Block 到达同步点时，SM 内可执行 Warp 逐渐减少，若同时执行的 Block 只有一个，则 SM 中可同时执行的 Warp 会在此时逐渐降成0，会导致计算资源空闲，造成浪费，若此时同时有其他 Block 在执行，则在一个 Block 到达同步点时仍然有其他 Block 可以执行。当 block_size 越小时，SM 可同时调度的 Block 越多，因此在这种情况下 block_size 越小越好。但是当在调大 block_size，SM 能同时调度的 Block 数不变的情况下，block_size 应该是越大越好，越大就有越好的并行度。因此代码中在选择 block_size 时，对不同 block_size 都计算了 cudaOccupancyMaxActiveBlocksPerMultiprocessor，若结果相同，使用较大的 block_size。

LayerNormBlockSMemImpl Kernel的代码如下：
```
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                       const int64_t cols, const double epsilon, ComputeType* mean,
                                       ComputeType* inv_variance) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        WelfordCombine(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        pack[i] = (buf[i * num_packs + pack_id] - row_mean) * row_inv_var;
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}
```

3、一个Block处理一行的元素，不使用Shared Memory，重复读输入x

当num_cols较大，无法用Shared Memory存储x时，使用这种实现。这种方法和前面第二种情况线程和元素对应关系一致，唯一的区别在于，第二种方法将输入x存储到Shared Memory中，本方法不存储x，在每次计算时需要再从Global Memory中读入x。这种方法虽然需要多读一份x，但是在实际执行时，部分输入可以被Cache缓存起来，不会实际增加很多时间。值得注意的是，在这种实现中，block_size越大，SM中能同时并行执行的block数就越少，对Cache的需求就越少，就有更多机会命中Cache，因此我们使用较大的block size。

LayerNormBlockUncachedImpl代码如下：

```
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                           const int64_t cols, const double epsilon,
                                           ComputeType* mean, ComputeType* inv_variance) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        WelfordCombine(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      mean[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      const int pack_offset = pack_id * pack_size;
      load.template load<pack_size>(pack, row, pack_offset);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { pack[i] = (pack[i] - row_mean) * row_inv_var; }
      store.template store<pack_size>(pack, row, pack_offset);
    }
  }
}


```