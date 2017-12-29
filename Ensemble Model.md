# ensemble model

[TOC]

## blending

##bagging

​	首先介绍一个概念,Bootstraping，即自助法：它是一种有放回的抽样方法.



## boosting

- 具体思想笔者会放到下一讲boosting的内容讲解

  数学形式:

## stacking

## Bagging，Boosting二者之间的区别

- 1）样本选择上：
  - Bagging：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
  - Boosting：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。
- 2）样例权重：
  - Bagging：使用均匀取样，每个样例的权重相等
  - Boosting：根据错误率不断调整样例的权值，错误率越大则权重越大。
- 3）预测函数：
  - Bagging：所有预测函数的权重相等,也可以根据每个分类器的正确率来分配权重。
  - Boosting：每个弱分类器都有相应的权重，对于分类误差小的分类器会有更大的权重。
- 4）并行计算：
  - Bagging：各个预测函数可以并行生成
  - Boosting：各个预测函数只能顺序生成，因为后一个模型参数需要前一轮模型的结果。当然,xgboost就是将这种顺序模型可并行化,具体做法我们后面再看.

## 常用集成模型概览

- 1）Bagging + 决策树 = 随机森林
- 2）AdaBoost + 决策树 = 提升树
- 3）Gradient Boosting + 决策树 = GBDT

## 总结

- 集成模型具有相当不错的正则化能力、但该正则化能力并不是必然存在的

## 参考文献

[1] 林轩田，机器学习技法。

[2\] [Bagging和Boosting 概念及区别](http://www.cnblogs.com/liuwu265/p/4690486.html)