# boosting

[TOC]

###基本思想:

​	将弱分类器组装成一个强分类器.在PAC（概率近似正确）学习框架下，则一定可以将弱分类器组装成一个强分类器.(后面会详细介绍PCA)

- Boosting通过关注弱规则的错误而逐渐组合成强规则，所以它是一种错误驱动的方法（每一个弱分类器的生成就是为了纠正前面的错误）

###两个核心问题:

> 1）在每一轮如何改变训练数据的权值或概率分布？
>
> 通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果．

> 2）通过什么方式来组合弱分类器？
>
> 通过加法模型将弱分类器进行线性组合．比如AdaBoost通过加权多数表决的方式，即增大错误率小的分类器的权值，同时减小错误率较大的分类器的权值．而提升树通过拟合残差的方式逐步减小残差，将每一步生成的模型叠加得到最终模型．

## adaboost

这里会首先介绍adaboost整个算法流程,将在下一讲再来窥探算法背后的细节

###算法陈述:

不失一般性、我们以二类分类问题来进行讨论，易知此时我们的弱模型、强模型和最终模型为弱分类器、强分类器和最终分类器。再不妨假设我们现在有的是一个二类分类的训练数据集：

​					$D={(x_1,y_1),(x_2,y_2),…,(x_n, y_n)}$

其中，每个样本点都是由实例 $x_i$ 和类别 $y_i$ 组成,且：

​					$x_i∈X⊆R^n ;y_i∈Y=\{−1, +1\}$

这里的 $X$ 是样本空间、$Y$ 是类别空间。AdaBoost 会利用如下的步骤、从训练数据中训练出一系列的弱分类器、然后把这些弱分类器集成为一个强分类器：

​	1.**输入**：训练数据集（包含 $N$ 个数据）、弱学习算法及对应的弱分类器、迭代次数 $M$

到	2.**过程**：

​		1.初始化训练数据的权值分布  

​						$W_0=(w_{01},…,w_{0N})$

​			其中,$w_{0*}=\frac1N$,即每个样本权重相等.	

​		2.对$k=1,2,…, M$：

​			1.使用权值分布为 $W_k$ 的训练数据集训练弱分类器:

​						$g_{k}(x):X→\{−1, +1\}$

​			      $g_k(x)$ 是使得加权训练数据集误差最小的分类器.

​			2.计算 $g_{k}(x)$ 在训练数据集上的加权错误率:  

​						$e_{k}=\sum_{i=1}^Nw_{ki}I(g_{k}(x_i)≠y_i)$

​			3.根据加权错误率计算 $g_{k}(x)$ 的重要性:

​						$α_{k}=\frac12\ln\frac{1−e_{k}}{e_{k}}$

​			4.根据 $g_{k}(x)$的表现更新训练数据集的权值分布：被 $g_{k}(x)$误分的样本 $(y_ig_{k}x_i)<0$ 的样本)要相对

​			    地(以 $e^{α_{k}}$为比例)增大其权重，反之则要(以 $e^{−α_{k}}$为比例地）减少其权重:  						  

​						$w_{k+1,i}=\frac{w_{ki}}{Z_k}⋅\exp(−α_{k}y_ig_{k}(x_i))$

​						$W_{k+1}=(w_{k+1,1},…,w_{k+1,N})$

​			     这里的 $Z_k$ 是规范化因子:

​						$Z_k=\sum_{i=1}^Nw_{ki}⋅exp(−α_{k}y_ig_{k+1}(x_i))$

​			     它的作用是将 $W_{k+1}$ 归一化成为一个概率分布 

​		3.加权集成弱分类器 

​						$f(x)=\sum_{k=1}^Mα_kg_k(x)$

​	3.**输出**：最终分类器 $G(x)$:

​						$G(x)=sign(f(x))=sign(\sum_{k=1}^Mα_kg_k(x))$



接下来,将对2.2.3步骤对 $g$ 进行重要性分配,以及步骤2.2.4样本的权重跟新做推导(只关注算法流程可跳过)

###算法推导:

####**1.adaboost指数损失函数推导:**

上述步骤2.3中,我们构造的各个基本分类器的线性组合:

​						$f(x)=\sum_{k=1}^Mα_kg_k(x)$

#####**1.1加法模型和前向分步算法:**

如下式所示便是一个加法模型:

​						$f(x)=\sum_{k=1}^Mα_kb(x;\gamma_k )$

​	其中, $b(x;\gamma_k )$ 为基函数, $\gamma_k$ 为基函数参数, $\alpha_k$ 为基函数系数.

在给定训练数据及损失函数 $L(y,f(x))$ 的条件下，学习加法模型 $f(x)$ 成为经验风险极小化问题，即损失函数极小化问题：

​						$\min_{\beta_k,\gamma_k}\sum_{i=1}^NL(y_i,\sum_{k=1}^M\alpha_kb(x_i;\gamma_k))$

将该问题简化:从前向后,每一步只学习<u>**一个基函数**</u>及其系数，逐步逼近上式，即：每步只优化如下损失函数：

​						$\min_{\beta,\gamma}\sum_{i=1}^NL(y_i,\alpha*b(x_i;\gamma))$

这个优化方法便就是所谓的前向分步算法.

下面，咱们来具体看下**前向分步算法**的算法流程：

- 输入：训练数据集 $T=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_Y)\}$

- 损失函数: $L(y,f(x))$

- 基函数集: $\{b(x;\gamma_k)\}$,其中 $k=\{1,2,3,...,M\}$

- 输出:加法模型 $f(x)$

- 算法步骤:

  - 初始化 $f_0(x) = 0$

  - 对于 $k=\{1,2,3,..,M\}$

    - 极小化损失函数:

      ​			$(\alpha_k,\gamma_k) = \arg\min_{\alpha,\gamma}\sum_{i=1}^NL(y_i,f_{k-1}(x_i)+\alpha b(x_i;\gamma))$

      ​得到参数 $(\beta_k,\gamma_k)$

    - 更新:

      ​			$f_k(x)=f_{k-1}(x)+α_kb(x;\gamma_k )$

    - 最终得到加法模型:

      ​			$f(x)=f_M(x)=\sum_{k=1}^Mα_kb(x;\gamma_k )$

就这样，前向分步算法将同时求解从 $k=1$ 到 $M$ 的所有参数 $(\alpha_k,\gamma_k)$ 的优化问题简化为逐次求解各个 $\alpha_k,\gamma_k(1 \leq k \leq M)$ 的优化问题

#####**1.2,前向分步算法与adaboost的关系**

现在咱们就来看adaboost 的另外一种理解，即可以认为其模型是加法模型、**损失函数为指数函数**、学习算法为前向分步算法的二类分类学习方法.

起始adaboost就是前向分步算法的一个特例,各个基分类器 $g$ 就是前向步算法的基函数 $b$ ,加法模型等价与adaboost的最终分类器:

​						$f(x)=\sum_{k=1}^Mα_kg_k(x)$

下面,咱们来**证明:当前向分布算法的损失函数为指数函数:**

​						$L(y,f(x))=\exp(-y*f(x))$

时,其学习过程的具体操作**等价于adaboost的学习过程**

假设经过 $k-1$ 轮迭代，前向分步算法已经得到 $f_{k-1}(x)$ :

​						$f_{k-1}(x) = f_{k-2}(x)+\alpha_{k-1}*g_{k-1}(x)$

​							$= \alpha_1*g_1(x)+...+\alpha_{k-1}*g_{k-1}(x)$

而后在第 $k$ 轮迭代得到 $\alpha_k,g_k(x)$ 和 $f_k(x)$ .其中, $f_k(x)$ 为:

​						$f_k(x)=f_{k-1}(x)+\alpha_k*g_k(x)$ 

现在 $\alpha_k,g_k(x)$ 未知,现在的目标便是根据前向分步算法训练 $\alpha_k,g_k(x)$ ,使得最终的 $f_k(x)$ 在训练数据集 $T$ 上指数损失最小.即:

​						$(\alpha_k,g_k(x))=\arg\min_{\alpha,g}\sum_{i=1}^N\exp(-y_i(f_{k-1}(x_i)+\alpha*g(x_i)))$

针对这种需要求解多个参数的情况，可以先固定其它参数，求解其中一两个参数，然后逐一求解剩下的参数。例如我们可以固定 $g_1(x),...,g_{k-1}(x)$ 和 $\alpha_1,...,\alpha_{k-1}$ ,只针对 $(\alpha_k,g_k(x))$ 做优化,换言之,在面对 $g_1(x),...,g_{k-1}(x),g_k(x)$ 和   $\alpha_1,...,\alpha_{k-1},\alpha_k$ 这 $2m$ 个参数都未知的情况下可以考虑:

​	1.先假定  $g_1(x),...,g_{k-1}(x)$ 和 $\alpha_1,...,\alpha_{k-1}$ 已知,求解出$\alpha_k$ 和 $g_k(x)$

​	2.实际也是如此,第 $k$ 轮之前已经解出了 $g_1(x),...,g_{k-1}(x)$ 和 $\alpha_1,...,\alpha_{k-1}$ 

且考虑到上式中的 $\exp(-y_if_{k-1}(x_i))$ 既不依赖 $\alpha$ 也不依赖 $g$ ,所以是个与最小化无关的固定值,记为 $w_{ki}$,

即 $w_{ki}=\exp(-y_if_{k-1}(x_i))$,则上式可以表示为:

​						$(\alpha_k,g_k(x))=\arg\min_{\alpha,g}\sum_{i=1}^Nw_{ki}\exp(-y_i\alpha g(x_i)))$

下面会多次用到这个式子,简记为: $(\alpha_k,g_k(x))$

需要注意的是, $w_{ki}$ 虽然与当前第k轮最小化损失函数无关,但 $w_{ki}$ 与$f_{k-1}(x)$ ,随着每一轮的迭代而发生变化.

接下来,便要证:

​			 **使得上式达到最小的 $\alpha_k^*$** **和** **$g_k^*(x)$ 就是adaboost算法所求解得到的 $\alpha_k$ 和 $g_k(x)$**

为求解上式 $(\alpha_k,g_k(x))$,先求解 $g_k^*(x)$ ,再求解 $\alpha_k$

​	1.首先求解 $g_k^*(x)$ .对于任意 $\alpha\gt0$ ,<u>使上式 $(\alpha_k,g_k(x))$ 最小的 $g(x)$ 由下式得到</u>(为什么???):

​						$g_k^*(x)=\arg\min_g\sum_{i=1}^Nw_{ki}I(y_i\neq g(x_i))$

​	    别忘了, $w_{ki}=\exp(-y_if_{k-1}(x_i))$.

​	    跟步骤2.2.2所述的加权误差计算公式对比如下:

​						$e_{k}=\sum_{i=1}^Nw_{ki}I(g_{k}(x_i)≠y_i)$

​	    可知，上面得到的 $g_k^*(x)$ 便是Adaboost算法的基本分类器 $g_k(x)$,因为它是当前第 $k$ 轮加权训练数据分类误差率最小的基本分类器.换言之, 这个 $g_k^*(x)$ 便是adaboost算法所要求的 $g_k(x)$ ,别忘了,步骤2.2.1中,说明 $g_k(x)$ 是使得误差率最低的分类器.

​	2.然后求解 $\alpha_k^*$ ,回到上式 $(\alpha_k,g_k(x))$:

​						$(\alpha_k,g_k(x))=\arg\min_{\alpha,g}\sum_{i=1}^Nw_{ki}\exp(-y_i\alpha g(x_i)))$

​	   式子后半部分进一步化简,得:

​						$\sum_{i=1}^Nw_{ki}\exp(-y_i\alpha g_k(x_i))$

​	   此时有两种情形:

​			$y_i = g_k(x):w_{ki}\exp(-\alpha)$

​			$y_i\neq g_k(x):w_{ki}\exp(\alpha)$

​						$=\sum_{i=1}^NI(y_i= g_k(x_i))w_{ki}\exp(-\alpha)+\sum_{i=1}^NI(y_i\neq g_k(x_i))w_{ki}\exp(\alpha)$

​						$=(\exp(\alpha)-\exp(-\alpha))\sum_{i=1}^Nw_{ki}I(y_i\neq g_k(x_i))+\exp(-\alpha)\sum_{i=1}^Nw_{ki}$

​						$=(\exp(\alpha)-\exp(-\alpha))e_k+\exp(-\alpha)\sum_{i=1}^Nw_{ki}$

​	    	其中, $\sum_{i=1}^Nw_{ki}=1​$ , 为什么? 因为步骤2.2.4在第 $k-1​$ 轮计算 $w_{ki}​$ 时对其进行了归一化,还不明白?继续往下看!

​		上式对 $\alpha$ 求导,其令其等于 $0$ ,即可得到使得 $(\alpha_k,g_k(x))$ 一式最小的 $\alpha$ ,即:

​						$\alpha_k^*=\frac12\ln\frac{1-e_k}{e_k}$

​	   这里的 $\alpha_k^*$ 跟步骤2.2.3中 $\alpha_k$ 的计算公式完全一致.

就这样，结合模型 $f_k(x)=f_{k-1}+\alpha_kg_k(x)$ 和 $w_{ki}=\exp(-y_if_{k-1}(x_i))$ .

可以推出:

​						$w_{k+1,i}=\exp(-y_if_k(x_i))$

​						$=\exp[-y_i(f_{k-1}(x_i)+\alpha_kg_k(x_i))]$

​						$=\exp[-y_if_{k-1}(x_i)] \exp[-y_i\alpha_kg_k(x_i)]$

从而有:

​						$w_{k+1,i}=w_{k,i}\exp(-y_i\alpha_kg_k(x_i))$

相比步骤2.2.4,只相差一个规范化因子,即后者多了:

​						$Z_k=\sum_{i=1}^Nw_{ki}⋅exp(−α_{k}y_ig_{k+1}(x_i))$

是为了将权重之和归一化,保证 $\sum_{i=1}^Nw_{k+1,i}=1$ .此时也解释了上面为什么 $\sum_{i=1}^Nw_{ki}=1$ .

整个过程解释了 $g_k(x)$(步骤2.2.1) 及其系数 $\alpha_k$ (步骤2.2.3)的由来,以及对样本权重更新公式(步骤2.2.4)的解释.

在这里抛一个问题,上面提到adaboost用指数函数作为损失函数,这个指数损失函数是如何设计而来???



## gradient boosting

上一节推导了adaboost是用**指数函数作为损失函数**的加法模型,学习算法为前向分步算法的二分类模型

现在将其推广到**任意形式的损失函数** :$L(y,f(x))$ 

###监督学习概念回顾

- 注解:$x_i\in R^d$ 第 $i$ 个训练样本, 维数是 $d$ 

- 模型:给定训练样本 $x_i$ 如何预测出 $\hat{y_i}$
  - 线性模型: $\hat{y_i}=\sum_{i=1}^d$ (包括线性回归,逻辑回归)
  - 根据任务的不同,预测分数 $\hat{y_i}$ 有不同的解释
    - 线性回归: $\hat{y_i}$ 就是预测分数
    - 逻辑回归: $\frac1{1+\exp(-\hat{y_i})}$ 预测正样本为正的概率
    - 其他等等...,例如,排序算法的 $\hat{y_i}$ 就是一个排序分数

- 参数:这需要从训练数据中学习得到

  - 线性模型: $\theta=\{w_j|j=1,..,d\}$

    - 目标函数:通用形式如下				

      ​			$Obj(\theta) = L(\theta) + \Omega(\theta)$

  - $L(\theta)$ :训练损失,评估模型对数据的拟合好坏
  - $\Omega(\theta)$ :正则项,限制模型的复杂度

- 训练数据的损失函数: $L=\sum_{i=1}^nl(y_i,\hat{y_i})$

  - 平方损失函数: $L(y_i,\hat{y_i}=(y_i-\hat{y_i})^2$
  - Logistic Loss(Cross Entropy): $l(y_i,\hat{y_i}=y_i\ln(1+e^{-\hat{y_i}})+(1-y_i)\ln(1+e^{\hat{y_i}})$ 

- 正则化:模型的复杂程度

  - L2正则(ridge): $\Omega(w)=\lambda||w||^2$
  - L1正则(lasso): $\Omega(w)=\lambda||w||_1$

- 岭回归(Ridge regression): $L=\sum_{i=1}^n(y_i-w^Tx_i)^2+\lambda||w||^2$

  - 线性模型,平方误差,L2正则

- Lasso: $L=\sum_{i=1}^n(y_i-w^Tx_i)^2+\lambda||w||_1$

  - 线性模型,平方误差,L1正则

- 逻辑回归(Logistic regression): $L=\sum_{i=1}^n[y_i\ln(1+e^{-w^Tx_i})+(1-y_i)\ln(1+e^{w^Tx_i})]+\lambda||w||^2$

  - 线性模型,Logistic loss,L2正则

- 将模型,参数,目标函数(模型损失函数+正则项)的概念区分开,在工程上会带来好处

  - 想想用SGD来同时实现ridge regression 和logistic regression


    ​				$Obj(\theta) = L(\theta) + \Omega(\theta)$

- 为什么这里的目标函数包含两部分

- 优化训练损失为了促进预测模型

  - 在训练数据拟合得好至少有助于找到训练数据的分布

- 优化正则项为了促进简化模型

  - 简单的模型可以在预测时有更小的方差(variance),使得模型更稳定

### 回归树和集成学习(学到的模型)

- 回归树(也被称为cart,详细讲解见决策树部分)
  - 决策规则和决策树一样
  - **每个叶子节点包含一个分数(score)**

树模型的集成方法

- 使用范围广,像GBM,random forest
  - 半数的数据挖掘竞赛赢家都是各种树模型的集成方法
- 输入的伸缩不变行,所以不用关心特征的正规化
- 可扩展,用于工业界

模型与参数:

- 模型:假设有 $K$ 棵树

  ​				$\hat{y_i}=\sum_{k=1}^Kf_k(x_i),f_k\in F$

  其中, $F$ :包含所有树模行的函数空间

  注意:**回归树起始就是一个从属性(attributes)映射(map)到分数(score)的函数**

- 参数:

  - 包含每一棵数的结构和每一个叶子节点的分数

  - 或者使用函数作为参数

    ​			$\Theta=\{f_1,f_2,...,f_K\}$

  - 需要学习一颗树(一个函数),而不是从 $R^d$ 中学习权重

  - 每一棵数的具体方法见决策树部分

树集成模型的目标函数

​					$Obj=\sum_{i=1}^nl(y_i,\hat{y_i})+\sum_{k=1}^K\Omega(f_k)$

​	其中, $\sum_{i=1}^nl(y_i,\hat{y_i})$ :训练损失, $\sum_{k=1}^K\Omega(f_k)$ 

- 如何定义 $\Omega$ ?
  - 树叶节点的数量
  - 叶节点权重的L2正则

客观解与启发式

- 谈到决策树,通常用启发式的方法
  - 基于信息增益(比)切分数据
  - 树的剪枝
  - 树的最大深度
  - 叶子节点值的平滑处理
- 大多数启发式映射和客观节一样,从形式上看我们知道我们学的是什么(好好理解)
  - 信息增益 --> 训练损失
  - 剪枝 --> 基于叶节点定义的正则项
  - 最大深度 --> 函数空间 $F$ 的范围
  - 叶节点值的平滑处理 --> 叶节点权重的L2正则化

回归数不仅仅用于回归

- 回归树集成是用来预测一个分数,它可以用来
  - Regression,Classification,Ranking,....
- 其最终取决于你如何定义目标函数
- 所以可以学到的有:
  - 平方损失: $l(y_i,\hat{y_i})=(y_i-\hat{y_i})^2$
    - 其结果是熟知的 gradient boosted machine
  - 使用cross entropy: $l(y_i,\hat{y_i})=y_i\ln(1+e^{-\hat{y_i}})+(1-y_i)\ln(1+e^{\hat{y_i}})$
    - 其结果是LogitBoost

本质

- 处处存在偏差(Bias)与方差(Variance)的权衡
- loss + regularization 的目标函数模式被应用于回归树的学习
- 学习尽量简单的模型
- 定义需要学习的是什么(目标函数,模型)
- 具体如何学习呢,下一节



### Gradient Boosting(应该如何学)

####如何学习

- 模型: $\hat{y_i}=\sum_{k=1}^Kf_k(x_i),f_k\in F$


- 目标函数: $Obj=\sum_{i=1}^nl(y_i,\hat{y_i})+\sum_{k=1}^K\Omega(f_k)$

- 将不使用SGD去找 $f$ (因为 $f$ 是树模型,而不仅仅是数值向量)

- 求解方式:加法模型(Boosting)

- 从0开始不断预测,每次加入一个新的函数

  ​		$\hat{y_i}^{(0)} = 0$

  ​		$\hat{y_i}^{(1)} = \hat{y_i}^{(0)} +f_1(x_i)$

  ​		$\hat{y_i}^{(2)} =f_1(x_i)+f_2(x_i)= \hat{y_i}^{(1)} +f_2(x_i)$

  ​		$...$

  ​		$\hat{y_i}^{(t)} =\sum_{k=1}^tf_k(x_i)= \hat{y_i}^{(t-1)} +f_k(x_i)$

  其中, $\hat{y_i}^{(t)}$ :第 $t$ 轮训练得到的模型, $f_k(x_i)$ :第 $t$ 轮新加函数,

#### 加法训练(additive Training)

- 如何决定 $F$ 中哪个 $f$ 被加进来

  - 优化目标函数!!!!

- 第 $t$ 轮的预测 $\hat{y_i}^{(t)} = \hat{y_i}^{(t-1)} +f_k(x_i)$

  - <u>$f_k$ 就是在本轮需要找的函数</u>

  ​			$Obj^{(t)}=\sum_{i=1}^nl(y_i,\hat{y_i}^{(t)})+\sum_{k=1}^t\Omega(f_k)$

  ​				   $=\sum_{i=1}^nl(y_i,\hat{y_i}^{(t-1)}+f_t(x_i))+\Omega(f_t)+constant$

- 考虑平方误差

  ​			$Obj^{(t)}=\sum_{i=1}^n[y_i-(\hat{y_i}^{(t-1)}+f_t(x_i))]^2+\Omega(f_t)+constant$

  ​				   $=\sum_{i=1}^n[2(\hat{y_i}^{(t-1)}-y_i)f_t(x_i)+f_t(x_i)^2]+\Omega(f_t)+contant$

  其中,$2(\hat{y_i}^{(t-1)}-y_i)$ 通常被称为前一轮结果的残差

#### 目标函数泰勒展开式

- 目标:

  ​			$Obj^{(t)}=\sum_{i=1}^nl(y_i,\hat{y_i}^{(t-1)}+f_t(x_i))+\Omega(f_t)+constant$

  - 似乎还是很困难,除了平方损失函数

- 目标函数的泰勒展开式

  - 泰勒公式: $f(x+\Delta x)\approx f(x)+f^{'}(x)\Delta x+\frac12f^{''}(x)\Delta x^2$ !!!!

  - 定义: $g_i=\partial_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$ , $h_i=\partial^2_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$

    ​		$Obj^{(t)}\approx \sum_{i=1}^n[l(y_i,\hat{y_i}^{(t-1)})+g_if_t(x_i)+\frac12h_if^2_t(x_i)]+\Omega(f_t)+constant$

- 如果以上形式很复杂,考虑平方误差:

  ​	$g_i=\partial_{\hat{y}^{t-1}}(\hat{y}^{t-1}-y_i)^2=2(\hat{y}^{(t-1)}-y_i)$	$h_i=\partial^2_{\hat{y}^{t-1}}(\hat{y}^{t-1}-y_i)^2=2$

  对比与前面的结果,完全一样!!

  注意:**传统的GBM与xgboost的最大区别在于,传统GBM使用的是泰勒公式的一阶展开,而xgboost用的泰勒公式的二阶展开**

#### 新目标函数

- 目标函数,删除常数项:

  ​				$Obj^{(t)}\approx \sum_{i=1}^n[g_if_t(x_i)+\frac12h_if^2_t(x_i)]+\Omega(f_t)$

  其中: $g_i=\partial_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$ , $h_i=\partial^2_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$

- 为什么花费大量时间推到目标函数,而不是直接从树模型出发:

  - 理论上的好处:知道学习的是什么
  - 工程上的好处:回想一些监督学习要素
    - $g_i$ 和 $h_i$ 来自于目标函数的定义
    - 目标函数的学习**仅仅依赖与目标变量 $g_i$ 和 $h_i$**
    - 想想同时实现square loss 和 logistic loss 的 boosted tree,如何分离代码块

#### 树的定义

- 使用叶子节点的得分向量来定义树,映射函数是叶子下标到叶子实例的映射

  ​				$f_t(x)=w_{q(x)},w\in R^T,q:R^{d} \rightarrow \{1,2,..,T\}$

  其中,$w$ :树的叶子节点权重向量,$T$ 维   $q$ :$d$维实数向量(特征向量)到叶子节点下标的映射

#### 树的复杂度定义

​					$\Omega(f_t)=\gamma T+\frac12\lambda\sum_{j=1}^Tw^2_j$

​	其中,$T$ 叶子节点数量, $\sum_{j=1}^Tw^2_j$ :叶子节点得分的L2正则项

#### 回顾目标函数

- 定义一个叶节点的实例(一个训练样本为一个实例)集合

  ​				$I_j=\{i|q(x_i)=j\}$

  其中,$i$ :实例x的下标, $j$为叶子节点下标,$I_j$代表被分在叶节点$j$的所有实例(样本)集合

- 根据叶节点重组目标函数

  ​				$Obj^{(t)}\approx \sum_{i=1}^n[g_if_t(x_i)+\frac12h_if^2_t(x_i)]+\Omega(f_t)$

  ​					   $=\sum_{i=1}^n[g_iw_{q(x_i)}+\frac12h_iw^2_{q(x_i)}]+\gamma T+\lambda\frac12\sum_{j=1}^Tw^2_j$

  ​					   $=\sum_{j=1}^T[(\sum_{i\in I_j}g_i)w_j+\frac12(\sum_{i\in I_j}h_i+\lambda)w^2_j]+\gamma T$

  这是 $T$ 个独立二次函数的和

#### 结构分数

- 单边量二次函数的两个定理

  ​	$\arg\min_xGx+\frac12Hx^2=-\frac GH,H>0$		$\min_x Gx+\frac12Hx^2=-\frac12\frac{G^2}H$

- 定义 $G_j=\sum_{i\in I_j}g_i$    $H_j=\sum_{i\in I_j}h_j$

  ​				$Obj^{(t)}=\sum_{j=1}^T[(\sum_{i\in I_j}g_i)w_j+\frac12(\sum_{i\in I_j}h_i+\lambda)w^2_j]+\gamma T$

  ​					   $=\sum_{j=1}^T[G_jw_j+\frac12(H_j+\lambda)w_j^2]+\gamma T$

- 假设树结构( $g(x)$ )固定,叶子节点权重的最优解,目标函数的结果为:

  ​		$w_j^*=-\frac{G_j}{H_j+\lambda}$ 		$Obj=-\frac12\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T$

  其中,$-\frac12\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}$ 测量树结构有多好!!!

如何求解 $g$

#### 单棵树的搜索算法

- 枚举所有可能的树结构q

- 计算q的结构得分:

  ​				$Obj=-\frac12\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T$

- 找到最佳树结构,使用最优的叶子权重:

  ​				$w_j^*=-\frac{G_j}{H_j+\lambda}$

- 但是,树结构可以有无数种....(懵逼脸)

#### 树的贪心学习

- 实践中,采用贪心算法生成树

  - 从深度为0开始

  - 对于树的每一个叶子节点,尝试添加一个切分,计算切分之后的目标函数值

    ​			$Gain=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma$

    其中, $\frac{G_L^2}{H_L+\lambda}$:左叶节点得分,$\frac{G_R^2}{H_R+\lambda}$ :右叶节点得分, $\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}$:未切分时的得分

- 剩余问题:如何有效找到一个最好的切分?

  - 对于每个节点,枚举所有的特征
    - 针对每个特征,对每个实例(样本)的此特征值进行排序
    - 使用线性扫描的方式寻找该特征的最佳切分
    - 找到所有特征中的那个最佳切分,也就是找到 $Gain$ 最大的切分点进行切分

  细心的同学可能已经发现:找到 $Gain$ 最大的点进行切分,更新的 

  ​				$Obj=Obj-Gain$

  ​					$=-\frac12\sum_{j=1}^{T+1}\frac{G_j^2}{H_j+\lambda}+\gamma(T+1)-\frac12(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda})$

  结果不对!!

  再来看 $Gain=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma$ ,其中 $\gamma$ 为定值, $Gain$ 最大, 所以 $\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}$ 最大, 乘以 $\frac12$ 也最大,所以:

  $Gain$最大,等价于 $\frac12(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda})-\gamma=\frac12Gain-\gamma$ 最大

  所以 $Obj$ 的正确更新方式:

  ​				 $Obj^{(T+1)}=Obj^{(T)}-(\frac12Gain-\frac12\gamma)$

  ​					 	 $=-\frac12\sum_{j=1}^{T+1}\frac{G_j^2}{H_j+\lambda}+\gamma(T+1)$

  其中, $T$ :切分前的叶节点数

如何处理类别特征

- 实际上不用专门为类别特征设计算法,直接使用one-hot编码,向量的长度为特征类别数,第 $j$ 个类别满足 $[0,0,..0_{j-1},1_j,0_{j+1},..,0]$
- 当一个特征类别多时向量为稀疏向量,学习算法可以很完美的解决稀疏数据(怎么解决)

#### 剪枝与正则化

- 回忆节点切分的信息增益(gain),可以为负:

  ​				$Gain=\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}-\gamma$

  - 当训练损失$\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}$ 小于 $\gamma$ 时,$Gain$ 为负
  - 需要在模型复杂度与预测精度之间权衡

- 提前结束

  - 如果最佳切分的 $Gain$ 为负,则停止切分
  - 但是可能切分之后会有利于后面的切分....

- 最后剪枝

  - 让树完全长成,递归的减去所有信息增益($Gain$)为负的切分点

### 算法概述:Boosted Tree Algorithm

- 每一次迭代加入一颗新树

- 每次迭代之前,计算:

  ​		 $g_i=\partial_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$ , $h_i=\partial^2_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$

- 使用统计,贪婪的生长一棵树$f_t(x)$:

  ​				$Obj=-\frac12\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T$

- 添加$f_t(x)$到模型 $\hat{y_i}^{(t)}=\hat{y_i}^{(t-1)}+f_t(x)$

  - 通常,实作上会使用 $y^{(t)}=y^{(t-1)}+\eta f_t(x)$
  - $\eta$ 被称作步长,通常设为0.1
  - 这意味着每一次迭代不会进行全面的优化,防止过度拟合