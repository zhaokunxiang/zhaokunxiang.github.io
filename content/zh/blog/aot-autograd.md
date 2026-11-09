---
title: "AOTAutograd"
date: 2026-11-09
---

一句话总结：
AOTAutograd 接受一张前向计算图（recall:TorchDynamo 输出是前向计算图），输出一张反向计算图。 

展开解释：
AOTAutograd 用 FakeTensor 做符号执行，只传播 shape、dtype、device 等 metadata，不做真实计算，并在 aten 层记录 op。它先从 inputs（primals）执行到 outputs，再以 tangents（也就是上游传回来的 grad_outputs）为起点对 outputs 做 VJP。
forward 和 backward 两段 op 被记录在同一张图里，就得到了 joint graph。这一切发生在编译期、真实数据到来之前，这就是 “AOT” 的由来

然后需要把 joint graph 切成 forward graph 和 backward graph。切的位置决定了哪些中间张量跨过边界，也就是 saved_tensors。
对每个中间结果，要么在 forward 保存，要么在 backward 重算（例如 pointwise op 重算很便宜，matmul 则不应重算）。partitioner（Inductor 默认使用 min-cut 算法）在这些约束下，选出让保存张量总量最小的切法。这一步就是 AOTAutograd 中的切分（partition）环节。

---

返回 [torch.compile 学习笔记]({{< relref "torch-compile.md" >}})
