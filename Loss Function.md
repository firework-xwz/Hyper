# Loss Function

## Keras.io

* mean_squared_error或mse

* mean_absolute_error或mae

* mean_absolute_percentage_error或mape

* mean_squared_logarithmic_error或msle

* squared_hinge

* hinge

* binary_crossentropy（亦称作对数损失，logloss）

* categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如`(nb_samples, nb_classes)`的二值序列

* sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：`np.expand_dims(y,-1)`

* kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.

* cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数

  

### mean_squared_error或mse

　　顾名思义，意为均方误差，也称标准差，缩写为MSE，可以反映一个数据集的离散程度。

　　标准误差定义为各测量值误差的平方和的平均值的平方根，故又称为均方误差。

　　公式：![img](http://images2015.cnblogs.com/blog/984577/201609/984577-20160916100944180-165787821.png)

 

　　公式意义：可以理解为一个从n维空间的一个点到一条直线的距离的函数。



### mean_absolute_error或mae

译为平均绝对误差，缩写MAE。

　　平均绝对误差是所有单个观测值与算术平均值的偏差的绝对值的平均。

　　**公式**：（fi是预测值，yi是实际值,绝对误差)



### mean_absolute_percentage_error或mape

译为平均绝对百分比误差 ，缩写MAPE。

　　**公式：**（At表示实际值，*F**t*表示预测值）



### mean_squared_logarithmic_error或msle

译为均方对数误差,缩写MSLE。

　　**公式：**![img](http://images2015.cnblogs.com/blog/984577/201609/984577-20160905172213285-1114618916.png)（n是整个数据集的观测值，pi为预测值，ai为真实值）



### squared_hinge

公式为max(0,1-y_true*y_pred)^2.mean(axis=-1)，取1减去预测值与实际值的乘积的结果与0比相对大的值的平方的累加均值。



### hinge

公式为为max(0,1-y_true*y_pred).mean(axis=-1)，取1减去预测值与实际值的乘积的结果与0比相对大的值的累加均值。

Hinge Loss 最常用在 SVM 中的最大化间隔分类中，对可能的输出 *t* = ±1 和分类器分数 *y*，预测值 *y* 的 hinge loss 定义如下：

　　L(y) = max(0,1-t*y)

　　看到 y 应当是分类器决策函数的“原始”输出，而不是最终的类标。例如，在线性的 SVM 中

 　　y = w*x+b

　　可以看出当 *t* 和 *y* 有相同的符号时（意味着 *y* 预测出正确的分类）

 　　|y|>=1

　　此时的 hinge loss

 　　L(y) = 0

　　但是如果它们的符号相反

　　L(y)则会根据 y 线性增加 one-sided error。

### binary_crossentropy

即对数损失函数，log loss，与sigmoid相对应的损失函数。

　　**公式：**L(Y,P(Y|X)) = -logP(Y|X)

　　该函数主要用来做极大似然估计的，这样做会方便计算。因为极大似然估计用来求导会非常的麻烦，一般是求对数然后求导再求极值点。

　　损失函数一般是每条数据的损失之和，恰好取了对数，就可以把每个损失相加起来。负号的意思是极大似然估计对应最小损失

### categorical_crossentropy

多分类的对数损失函数，与softmax分类器相对应的损失函数，理同上。

　　tip：此损失函数与上一类同属对数损失函数，sigmoid和softmax的区别主要是，sigmoid用于二分类，softmax用于多分类



### sparse_categorical_crossentrop

在上面的多分类的对数损失函数的基础上，增加了稀疏性（即数据中多包含一定0数据的数据集），如目录所说，需要对数据标签添加一个维度np.expand_dims(y,-1)。



### kullback_leibler_divergence

对于离散随机变量，其概率分布*P* 和 *Q*的KL散度可按下式定义为

　　即按概率**P**求得的**P**和**Q**的对数差的平均值。KL散度仅当概率**P**和**Q**各自总和均为**1**，且对于任何**i**皆满足

　　**Q(i)>0**及**P(i)>0**时，才有定义。式中出现**0Ln0**的情况，其值按**0**处理。

　　对于连续随机变量，其概率分布*P*和*Q*可按积分方式定义为 

　　

　　其中*p*和*q*分别表示分布*P*和*Q*的密度。

　　更一般的，若*P*和*Q*为集合*X*的概率测度，且*Q*关于*P*绝对连续，则从*P*到*Q*的KL散度定义为

　　

　　其中，假定右侧的表达形式存在，则为*Q*关于*P*的R–N导数。

　　相应的，若*P*关于*Q*绝对连续，则

　　即为*P*关于*Q*的相对熵，用以度量两个分布的差异。



### cosine_proximity

此方法用余弦来判断两个向量的相似性。

　　设向量 A = (A1,A2,...,An)，B = (B1,B2,...,Bn)，则有

　　![img](http://images2015.cnblogs.com/blog/984577/201609/984577-20160916101244883-1062324365.png)

　　![img](http://images2015.cnblogs.com/blog/984577/201609/984577-20160907091633660-304209462.png)

　　余弦值的范围在[-1,1]之间，值越趋近于1，代表两个向量的方向越趋近于0，他们的方向更加一致。相应的相似度也越高。

