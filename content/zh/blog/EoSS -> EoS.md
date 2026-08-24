---
title: "EoSS → EoS"
date: 2026-08-24
tags:
  - optimization
---


## 1.  问题定义

考虑一个二次问题
$$
L_B(\theta)
=\frac12\theta^\top H_B\theta+g_B^\top\theta+c_B,
$$

$$
\nabla L_B(\theta)=H_B\theta+g_B,
$$

$$
\nabla L_B(\theta)-\nabla L_B(\theta^*)
=H_B(\theta-\theta^*).
$$

SGD 更新为

$$
\theta_{t+1}=\theta_t-\eta\nabla L_{B_t}(\theta_t).
$$

$$
\begin{aligned}
\theta_{t+1}-\theta^*
&=\theta_t-\theta^*
-\eta\left[\nabla L_{B_t}(\theta_t)-\nabla L_{B_t}(\theta^*)\right]
+\eta\nabla L_{B_t}(\theta^*)\\
&=(I-\eta H_{B_t})(\theta_t-\theta^*)
+\eta\nabla L_{B_t}(\theta^*).
\end{aligned}
$$


定义误差与二阶矩：

$$
e_t:=\theta_t-\theta^*,
\qquad
\Sigma_t:=\mathbb E[e_te_t^\top].
$$

$$
\begin{aligned}
\mathbb E[e_t^\top e_t]
&=\mathbb E\!\left[\operatorname{tr}(e_te_t^\top)\right]\\
&=\operatorname{tr}(\Sigma_t),
\end{aligned}
$$


$$
A_t:=I-\eta H_{B_t},
\qquad
q_t:=\nabla L_{B_t}(\theta^*),
\qquad
m_t:=\mathbb E[e_t].
$$

误差递推写成

$$
e_{t+1}=A_te_t-\eta q_t.
$$

二阶矩满足

$$
\begin{aligned}
\Sigma_{t+1}
&=\mathbb E\!\left[
(A_te_t-\eta q_t)(A_te_t-\eta q_t)^\top
\right]\\
&=\Phi[\Sigma_t]+G[m_t]+\eta^2C,
\end{aligned}
$$

其中

$$
\Phi[\Sigma]
:=\mathbb E[A_t\Sigma A_t^\top],
$$

$$
G[m]
:=-\eta\mathbb E[A_tmq_t^\top]
-\eta\mathbb E[q_tm^\top A_t^\top],
$$

$$
C:=\mathbb E[q_tq_t^\top]\succeq0.
$$

**我们的目标是**：

$$
\boxed {lim_{t\to\infty}\mathbb E[e_t^\top e_t]<\infty  \quad\Longleftrightarrow\quad \ \rho(\Phi)<1}
$$
**然后从 $\ \rho(\Phi)<1$ 推导出 GD 下经典的结论**
$$
\boxed{
\lambda_{\max}(H)<\frac{2}{\eta}
}.
$$
**和  temperature $\tau:=\frac{\eta}{B}$ 的来源**


展开 $\Sigma_{t+1}$

$$
\begin{aligned}
m_{t+1}
&=\mathbb E[A_t]m_t-\eta\mathbb E[q_t]\\
&=\mathbb E[A_t]m_t\\
&=(I-\eta H)m_t,
\end{aligned}
$$

因为

$$
\mathbb E[q_t]
=\mathbb E[\nabla L_{B_t}(\theta^*)]
=\nabla L(\theta^*)
=0.
$$

递推展开为

$$
\boxed{
\Sigma_t
=\Phi^t[\Sigma_0]
+\sum_{k=0}^{t-1}\Phi^{t-1-k}G[m_k]
+\eta^2\sum_{k=0}^{t-1}\Phi^k[C]
}.
$$



## 2. 充分性：若 $\rho(\Phi)<1$

令

$$
\lambda:=\rho(\Phi)<1.
$$

### 2.1 初值项衰减

把任意 $\Sigma$ 沿 $\Phi$ 的特征矩阵展开：

$$
\Sigma=C_1M_1+C_2M_2+\cdots+C_dM_d,
$$

其中

$$
\Phi[M_i]=\lambda_iM_i.
$$

由线性性，

$$
\Phi^t[\Sigma]
=C_1\lambda_1^tM_1+\cdots+C_d\lambda_d^tM_d.
$$

由于

$$
|\lambda_i|<1,
\qquad
\lim_{t\to\infty}\lambda_i^t=0,
$$

所以

$$
\Phi^t[\Sigma]\to0.
$$

特别地，

$$
\Phi^t[\Sigma_0]\to0.
$$

### 2.2 交叉项衰减

$$
\|\Phi^k[X]\|_F
\leq\lambda^k\|X\|_F.
$$

定义

$$
P_k:=A_{k-1}\cdots A_0,
$$

则

$$
m_k=\mathbb E[P_km_0].
$$
(这里成立是因为batch 是IID )

由 Jensen 不等式，

$$
\begin{aligned}
\|m_k\|^2
&=\|\mathbb E[P_km_0]\|^2\\
&\leq\mathbb E[\|P_km_0\|^2]\\
&=\mathbb E[(P_km_0)^\top P_km_0]\\
&=\operatorname{tr}\!\left(
\mathbb E[P_km_0m_0^\top P_k^\top]
\right)\\
&=\operatorname{tr}\!\left(
\Phi^k[m_0m_0^\top]
\right).
\end{aligned}
$$

又因为

$$
|\operatorname{tr}(X)|
\leq\sqrt d\,\|X\|_F,
$$

所以

$$
\begin{aligned}
\|m_k\|^2
&\leq\sqrt d\,
\|\Phi^k[m_0m_0^\top]\|_F\\
&\leq\sqrt d\,\lambda^k
\|m_0m_0^\top\|_F\\
&=\sqrt d\,\lambda^k\|m_0\|^2.
\end{aligned}
$$

因此

$$
\boxed{
\|m_k\|
\leq d^{1/4}\|m_0\|\lambda^{k/2}
}.
$$

接着估计 $G[m]$：

$$
\begin{aligned}
\|G[m]\|_F
&=\eta\left\|
\mathbb E[A_tmq_t^\top]
+\mathbb E[q_tm^\top A_t^\top]
\right\|_F\\
&\leq\eta\left(
\|\mathbb E[A_tmq_t^\top]\|_F
+\|\mathbb E[q_tm^\top A_t^\top]\|_F
\right)\\
&\leq2\eta\,
\mathbb E[\|A_tmq_t^\top\|_F]\\
&=2\eta\,
\mathbb E[\|A_tm\|\,\|q_t\|]\\
&\leq2\eta\,
\mathbb E[\|A_t\|_{\mathrm{op}}\|q_t\|],\|m\|\\
&\leq C_G\|m\|.
\end{aligned}
$$
（注 等号 是rank1 matrix 特有的） 

 $C_G$ 是固定常数,(batch 是 i i d)

于是

$$
\|G[m_k]\|_F
\leq C_Gd^{1/4}\|m_0\|\lambda^{k/2}
=:C_0\lambda^{k/2}.
$$

令

$$
S_t:=\sum_{k=0}^{t-1}\Phi^{t-1-k}G[m_k].
$$

则

$$
\begin{aligned}
\|S_t\|_F
&\leq\sum_{k=0}^{t-1}
\|\Phi^{t-1-k}G[m_k]\|_F\\
&\leq C_0\sum_{k=0}^{t-1}
\lambda^{t-1-k}\lambda^{k/2}\\
&\to0.
\end{aligned}
$$



### 2.3 噪声协方差项收敛

当 $\rho(\Phi)<1$ 时，

$$
\sum_{k=0}^{\infty}\Phi^k
=(I-\Phi)^{-1}.
$$

因此

$$
\eta^2\sum_{k=0}^{t-1}\Phi^k[C]
\to
\eta^2(I-\Phi)^{-1}[C].
$$

综合三项，

$$
\boxed{
\lim_{t\to\infty}\Sigma_t
=\Sigma_\infty
=\eta^2(I-\Phi)^{-1}[C]
}.
$$

充分性得证。

## 3. 必要性：$\rho(\Phi)>1$ 时构造发散初值

假设

$$
\rho(\Phi)>1.
$$

选取半正定特征矩阵 $V_*\succeq0$，使得

$$
\Phi[V_*]=\rho(\Phi)V_*.
$$

选择初始分布满足

$$
m_0=0,
\qquad
\Sigma_0=V_*.
$$

例如可令

$$
e_0=V_*^{1/2}z,
\qquad
\mathbb E[z]=0,
\qquad
\mathbb E[zz^\top]=I.
$$

由于

$$
m_{t+1}=(I-\eta H)m_t,
$$

所以

$$
m_t=0,
\qquad\forall t,
$$

进而

$$
G[m_t]=0.
$$

此时

$$
\Sigma_{t+1}=\Phi[\Sigma_t]+\eta^2C,
$$

从而

$$
\Sigma_t
=\Phi^t[V_*]
+\eta^2\sum_{k=0}^{t-1}\Phi^k[C].
$$

由于第二项为半正定矩阵，

$$
\Sigma_t
\succeq\Phi^t[V_*]
=\rho(\Phi)^tV_*.
$$

因此

$$
\operatorname{tr}(\Sigma_t)
\geq\rho(\Phi)^t\operatorname{tr}(V_*),
$$

并且

$$
\mathbb E[\|e_t\|^2]
=\operatorname{tr}(\Sigma_t)
\to\infty.
$$

所以，若要对任意初始条件都有

$$
\lim_{t\to\infty}\mathbb E[\|e_t\|^2]<\infty,
$$

则必须有

$$
\boxed{\rho(\Phi)<1}.
$$

边界情况忽略

## 4. Temperature 的直接来源：分解 $\Phi$

将 batch Hessian 分解为

$$
H_B=H+\Delta_B,
\qquad
\mathbb E[\Delta_B]=0.
$$

代入二阶矩算子：

$$
\begin{aligned}
\Phi[\Sigma]
&=
\mathbb E\!\left[
(I-\eta H-\eta\Delta_B)
\Sigma
(I-\eta H-\eta\Delta_B)
\right]\\
&=(I-\eta H)\Sigma(I-\eta H)
+\eta^2\mathbb E[\Delta_B\Sigma\Delta_B].
\end{aligned}
$$

两个交叉项因为 $\mathbb E[\Delta_B]=0$ 而消失。

若 batch 包含 $B$ 个 i.i.d. 样本，则

$$
\Delta_B
=\frac1B\sum_{i\in B}(H_i-H).
$$

独立性给出

$$
\mathbb E[\Delta_B\Sigma\Delta_B]
=\frac1B\mathcal N[\Sigma],
$$

其中曲率异质性算子为

$$
\mathcal N[\Sigma]
:=
\mathbb E_i[(H_i-H)\Sigma(H_i-H)].
$$

因此

$$
\boxed{
\Phi[\Sigma]
=(I-\eta H)\Sigma(I-\eta H)
+\frac{\eta^2}{B}\mathcal N[\Sigma]
}.
$$

定义

$$
\boxed{
\tau:=\frac{\eta}{B}
},
$$


为了比较不同 learning rate 下的动力学，定义优化时间

$$
s:=t\eta.
$$

一个 step 推进的优化时间是 $\Delta s=\eta$。因此，固定 step 数并不是在比较相同的优化进程；应当固定 $s$，即令两组实验满足

$$
t_1\eta_1=t_2\eta_2=s.
$$

例如，$\eta_1=0.1$ 运行 $10$ 步，与 $\eta_2=0.01$ 运行 $100$ 步，都对应 $s=1$。这里的公平只指比较相同的优化时间，不是比较相同的 step 数、样本数或计算量。

因此，将单步变化除以 $\Delta s=\eta$，就是把“每个 step 的变化”换算成“每单位优化时间的变化”：

$$
\frac{\Phi[\Sigma]-\Sigma}{\eta}
=-(H\Sigma+\Sigma H)
+\eta H\Sigma H
+\tau\mathcal N[\Sigma].
$$

因此 $\eta$ 控制有限步长的确定性离散化效应，而

$$
\tau=\frac{\eta}{B}
$$

控制单位优化时间内的曲率噪声强度：

## 5. Full batch：从 $\rho(\Phi)<1$ 到 $\lambda_{\max}(H)<2/\eta$


在 i.i.d. 抽样极限 $B\to\infty$ 时（有限数据集对应 $B=n$），batch Hessian 的随机性消失，$H_B=H$，因此

$$
\Phi[\Sigma]=(I-\eta H)\Sigma(I-\eta H).
$$

设 population Hessian 的特征对为

$$
Hu_i=\lambda_i(H)u_i.
$$

取协方差方向 $\Sigma=u_iu_i^\top$，则

$$
\begin{aligned}
\Phi[u_iu_i^\top]
&=(I-\eta H)u_iu_i^\top(I-\eta H)\\
&=(1-\eta\lambda_i(H))^2u_iu_i^\top.
\end{aligned}
$$

所以 $\Phi$ 在该方向上的特征值是

$$
(1-\eta\lambda_i(H))^2.
$$

$\Phi$ 的其他特征值形如

$$
(1-\eta\lambda_i(H))(1-\eta\lambda_j(H)).
$$

由 $|ab|\leq\max\{a^2,b^2\}$，最大模一定在某个对角方向取得，因此

$$
\rho(\Phi)
=\lambda_{\max}(\Phi)
=\max_i(1-\eta\lambda_i(H))^2.
$$

长期均方收敛要求

$$
\rho(\Phi)<1,
$$

等价于对所有 $i$，

$$
|1-\eta\lambda_i(H)|<1.
$$

因此

$$
0<\eta\lambda_i(H)<2.
$$

在 $H\succ0$、$\eta>0$ 下，最终得到

$$
\boxed{
\lambda_{\max}(H)<\frac{2}{\eta}
}.
$$

即

$$
\boxed{
\rho(\Phi)<1
\iff
\max_i(1-\eta\lambda_i(H))^2<1
\iff
\lambda_{\max}(H)<\frac{2}{\eta}
}.
$$
