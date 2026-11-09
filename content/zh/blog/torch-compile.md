---
title: "torch.compile 学习笔记"
date: 2026-11-09
---

## 总结

一句话总结： `torch.compile` 是一个专门面对 pytorch 的JIT 编译器, 它优化计算执行过程，减少显存读写和kernel lauch 开销，在不改变计算结果的前提下，提升计算速度。 

展开解释：

它通过 [TorchDynamo]({{< relref "torch-dynamo.md" >}}) 劫持 Python 字节码执行来捕获计算图，用 [AOTAutograd]({{< relref "aot-autograd.md" >}}) 生成前反向联合的函数式 ATen 图，再由 [TorchInductor]({{< relref "torch-inductor.md" >}}) 做[算子融合]({{< relref "operator-fusion.md" >}})并生成 Triton/C++ kernel。目的是减少显存读写和 kernel launch 开销，在不改 eager 语义的前提下提速。

为什么能加速了？

从一次kernel 执行过程来看，cpu 发送  kernel 执行的指令给GPU，这个过程是异步的，但是如果kernel是一些很小计算，执行特别快，cpu发送跟不上，瓶颈会在 cpu 上。 

GPU 执行 一条kernel 指令，会从HBM 读取数据到 cache ,结束之后把数据写回HBM。 

torch compile 主要收益来源 是算子融合 （多个小算子合并为大算子），即减少了 cpu 发送的指令（从而减少kernel lauch 的开销），也减少计算中间结果写回到HBM里面，结果暂时留在 cache 里面（从而减少显存读写开销） 


## 参考

1. [Dynamo Deep-Dive](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html?utm_source=chatgpt.com)
2. [A Walk Through Example of torch.compile](https://depyf.readthedocs.io/en/latest/walk_through.html?utm_source=chatgpt.com)
3. [torch.compile, explained](https://medium.com/pytorch/torch-compile-explained-ae0def293084?utm_source=chatgpt.com)
4. [TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747?utm_source=chatgpt.com)
