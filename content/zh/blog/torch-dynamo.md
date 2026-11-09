---
title: "TorchDynamo"
date: 2026-11-09
---

TorchDynamo 在 python bytecodes 层面将指令分离成为 computation graph 和 plain python code.  只会生成 forward computation graph。它会在遇到第一个无法用计算图表示的操作处切断（graph break）：之前的操作编译为纯 tensor 计算图，该操作回退给 Python 执行，剩余 bytecode 再递归交给 Dynamo 处理。


important features:
1.  inline the possible function call fully composed of computation graph ,reduce the function call cost 
2. dynamic shape support, dynamo will try to generalize the shape not strict to a specify shape , but a sets of shapes 

比如你 call A 
A : x->x^2  纯computation graph 的operation，
那就把 function call A取消掉, 然后直接把 这个加入现在函数computation graph

---

返回 [torch.compile 学习笔记]({{< relref "torch-compile.md" >}})
