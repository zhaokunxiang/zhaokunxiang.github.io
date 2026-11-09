---
title: "TorchInductor"
date: 2026-11-09
---

TorchInductor 是 `torch.compile()` 的默认 backend
它接受一个计算图（可能是[TorchDynamo]({{< relref "torch-dynamo.md" >}})输出的前向计算图，也可能是[AOTAutograd]({{< relref "aot-autograd.md" >}})输出的反向计算图），负责优化和 code generation。GPU 上通常生成 Triton kernel，CPU 上通常生成 C++ / OpenMP 代码；

---

返回 [torch.compile 学习笔记]({{< relref "torch-compile.md" >}})
