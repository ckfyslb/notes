# NLP知识点

![img](https://upload-images.jianshu.io/upload_images/1667471-37315f7baaee75f4.jpg)



## 1. 什么是词嵌入

⾃然语⾔是⼀套⽤来表达含义的复杂系统。在这套系统中，词是表义的基本单元。顾名思义，词向量是⽤来表⽰词的向量，也可被认为是词的特征向量或表征。**把词映射为实数域向量的技术也叫词嵌入（word embedding）。**近年来，词嵌⼊已逐渐成为⾃然语⾔处理的基础知识。

在NLP(自然语言处理)领域，文本表示是第一步，也是很重要的一步，通俗来说就是把人类的语言符号转化为机器能够进行计算的数字，因为普通的文本语言机器是看不懂的，必须通过转化来表征对应文本。早期是**基于规则**的方法进行转化，而现代的方法是**基于统计机器学习**的方法。

**数据决定了机器学习的上限,而算法只是尽可能逼近这个上限，**在本文中数据指的就是文本表示，所以，弄懂文本表示的发展历程，对于NLP学习者来说是必不可少的。接下来开始我们的发展历程。文本表示分为**离散表示**和**分布式表示**：



## 2.离散表示

### 2.1 One-hot表示

One-hot简称读热向量编码，也是特征工程中最常用的方法。其步骤如下：

1. 构造文本分词后的字典，每个分词是一个比特值，比特值为0或者1。
2. 每个分词的文本表示为该分词的比特位为1，其余位为0的矩阵表示。

例如：**John likes to watch movies. Mary likes too**

**John also likes to watch football games.**

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10} **

每个词典索引对应着比特位。那么利用One-hot表示为：

**John: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] **

**likes: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]** .......等等，以此类推。

One-hot表示文本信息的**缺点**：

- 随着语料库的增加，数据特征的维度会越来越大，产生一个维度很高，又很稀疏的矩阵。
- 这种表示方法的分词顺序和在句子中的顺序是无关的，不能保留词与词之间的关系信息。



### 2.2 词袋模型

词袋模型(Bag-of-words model)，像是句子或是文件这样的文字可以用一个袋子装着这些词的方式表现，这种表现方式不考虑文法以及词的顺序。

**文档的向量表示可以直接将各词的词向量表示加和**。例如：

**John likes to watch movies. Mary likes too**

**John also likes to watch football games.**

以上两句可以构造一个词典，**{"John": 1, "likes": 2, "to": 3, "watch": 4, "movies": 5, "also": 6, "football": 7, "games": 8, "Mary": 9, "too": 10} **

那么第一句的向量表示为：**[1,2,1,1,1,0,0,0,1,1]**，其中的2表示**likes**在该句中出现了2次，依次类推。

词袋模型同样有一下**缺点**：

- 词向量化后，词与词之间是有大小关系的，不一定词出现的越多，权重越大。
- 词与词之间是没有顺序关系的。



### 2.3 TF-IDF

TF-IDF（term frequency–inverse document frequency）是一种用于信息检索与数据挖掘的常用加权技术。TF意思是词频(Term Frequency)，IDF意思是逆文本频率指数(Inverse Document Frequency)。

**字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章。**

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-21_10-7-23.png)

分母之所以加1，是为了避免分母为0。

那么，![](https://latex.codecogs.com/gif.latex?TF-IDF=TF*IDF)，从这个公式可以看出，当w在文档中出现的次数增大时，而TF-IDF的值是减小的，所以也就体现了以上所说的了。

**缺点：**还是没有把词与词之间的关系顺序表达出来。



### 2.4 n-gram模型

n-gram模型为了保持词的顺序，做了一个滑窗的操作，这里的n表示的就是滑窗的大小，例如2-gram模型，也就是把2个词当做一组来处理，然后向后移动一个词的长度，再次组成另一组词，把这些生成一个字典，按照词袋模型的方式进行编码得到结果。改模型考虑了词的顺序。

例如：

**John likes to watch movies. Mary likes too**

**John also likes to watch football games.**

以上两句可以构造一个词典，**{"John likes”: 1, "likes to”: 2, "to watch”: 3, "watch movies”: 4, "Mary likes”: 5, "likes too”: 6, "John also”: 7, "also likes”: 8, “watch football”: 9, "football games": 10}**

那么第一句的向量表示为：**[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]**，其中第一个1表示**John likes**在该句中出现了1次，依次类推。

**缺点：**随着n的大小增加，词表会成指数型膨胀，会越来越大。



### 2.5 离散表示存在的问题

由于存在以下的问题，对于一般的NLP问题，是可以使用离散表示文本信息来解决问题的，但对于要求精度较高的场景就不适合了。

- 无法衡量词向量之间的关系。
- 词表的维度随着语料库的增长而膨胀。
- n-gram词序列随语料库增长呈指数型膨胀，更加快。
- 离散数据来表示文本会带来数据稀疏问题，导致丢失了信息，与我们生活中理解的信息是不一样的。



## 3. 分布式表示

科学家们为了提高模型的精度，又发明出了分布式的表示文本信息的方法，这就是这一节需要介绍的。

**用一个词附近的其它词来表示该词，这是现代统计自然语言处理中最有创见的想法之一。**当初科学家发明这种方法是基于人的语言表达，认为一个词是由这个词的周边词汇一起来构成精确的语义信息。就好比，物以类聚人以群分，如果你想了解一个人，可以通过他周围的人进行了解，因为周围人都有一些共同点才能聚集起来。



### 3.1 共现矩阵

共现矩阵顾名思义就是共同出现的意思，词文档的共现矩阵主要用于发现主题(topic)，用于主题模型，如LSA。

局域窗中的word-word共现矩阵可以挖掘语法和语义信息，**例如：**

- I like deep learning.	
- I like NLP.	
- I enjoy flying

有以上三句话，设置滑窗为2，可以得到一个词典：**{"I like","like deep","deep learning","like NLP","I enjoy","enjoy flying","I like"}**。

我们可以得到一个共现矩阵(对称矩阵)：

![image](https://wx2.sinaimg.cn/large/00630Defly1g2rwv1op5zj30q70c7wh2.jpg)

中间的每个格子表示的是行和列组成的词组在词典中共同出现的次数，也就体现了**共现**的特性。

**存在的问题：**

- 向量维数随着词典大小线性增长。
- 存储整个词典的空间消耗非常大。
- 一些模型如文本分类模型会面临稀疏性问题。
- **模型会欠稳定，每新增一份语料进来，稳定性就会变化。**



## 4.神经网络表示

### 4.1 NNLM

NNLM (Neural Network Language model)，神经网络语言模型是03年提出来的，通过训练得到中间产物--词向量矩阵，这就是我们要得到的文本表示向量矩阵。

NNLM说的是定义一个前向窗口大小，其实和上面提到的窗口是一个意思。把这个窗口中最后一个词当做y，把之前的词当做输入x，通俗来说就是预测这个窗口中最后一个词出现概率的模型。

![image](https://wx3.sinaimg.cn/large/00630Defly1g2vb5thw9rj30eq065dg4.jpg)

以下是NNLM的网络结构图：

![image](https://wx3.sinaimg.cn/large/00630Defly1g2t1f4bqilj30lv0e2adl.jpg)

- input层是一个前向词的输入，是经过one-hot编码的词向量表示形式，具有V*1的矩阵。

- C矩阵是投影矩阵，也就是稠密词向量表示，在神经网络中是**w参数矩阵**，该矩阵的大小为D*V，正好与input层进行全连接(相乘)得到D\*1的矩阵，采用线性映射将one-hot表示投影到稠密D维表示。

  ![image](https://wx3.sinaimg.cn/large/00630Defly1g2t1s20jpnj30f107575i.jpg)

- output层(softmax)自然是前向窗中需要预测的词。

- 通过BP＋SGD得到最优的C投影矩阵，这就是NNLM的中间产物，也是我们所求的文本表示矩阵，**通过NNLM将稀疏矩阵投影到稠密向量矩阵中。**

### 4.2 Word2Vec

谷歌2013年提出的Word2Vec是目前最常用的词嵌入模型之一。Word2Vec实际是一种浅层的神经网络模型，它有两种网络结构，**分别是CBOW（Continues Bag of Words）连续词袋和Skip-gram。**Word2Vec和上面的NNLM很类似，但比NNLM简单。

**CBOW**

CBOW获得中间词两边的的上下文，然后用周围的词去预测中间的词，把中间词当做y，把窗口中的其它词当做x输入，x输入是经过one-hot编码过的，然后通过一个隐层进行求和操作，最后通过激活函数softmax，可以计算出每个单词的生成概率，接下来的任务就是训练神经网络的权重，使得语料库中所有单词的整体生成概率最大化，而求得的权重矩阵就是文本表示词向量的结果。

![image](https://ws2.sinaimg.cn/large/00630Defly1g2u6va5fvyj30gf0h0aby.jpg)

**Skip-gram**：

Skip-gram是通过当前词来预测窗口中上下文词出现的概率模型，把当前词当做x，把窗口中其它词当做y，依然是通过一个隐层接一个Softmax激活函数来预测其它词的概率。如下图所示：

![](https://gitee.com/kkweishe/images/raw/master/ML/2019-8-20_20-34-0.jpg)

**优化方法**：

- **层次Softmax：**至此还没有结束，因为如果单单只是接一个softmax激活函数，计算量还是很大的，有多少词就会有多少维的权重矩阵，所以这里就提出**层次Softmax(Hierarchical Softmax)**，使用Huffman Tree来编码输出层的词典，相当于平铺到各个叶子节点上，**瞬间把维度降低到了树的深度**，可以看如下图所示。这课Tree把出现频率高的词放到靠近根节点的叶子节点处，每一次只要做二分类计算，计算路径上所有非叶子节点词向量的贡献即可。

> **哈夫曼树(Huffman Tree)**：给定N个权值作为N个[叶子结点](https://baike.baidu.com/item/叶子结点/3620239)，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。

![image](https://ws4.sinaimg.cn/large/00630Defly1g2u762c7nwj30jb0fs0wh.jpg)

- **负例采样(Negative Sampling)：**这种优化方式做的事情是，在正确单词以外的负样本中进行采样，最终目的是为了减少负样本的数量，达到减少计算量效果。将词典中的每一个词对应一条线段，所有词组成了[0，1］间的剖分，如下图所示，然后每次随机生成一个[1, M-1]间的整数，看落在哪个词对应的剖分上就选择哪个词，最后会得到一个负样本集合。

  ![image](https://wx3.sinaimg.cn/large/00630Defly1g2u7vvrgjnj30lu07d75v.jpg)

**Word2Vec存在的问题**

- 对每个local context window单独训练，没有利用包 含在global co-currence矩阵中的统计信息。
- 对多义词无法很好的表示和处理，因为使用了唯一的词向量



### 4.3 sense2vec

word2vec模型的问题在于词语的多义性。比如duck这个单词常见的含义有水禽或者下蹲，但对于 word2vec 模型来说，它倾向于将所有概念做归一化平滑处理，得到一个最终的表现形式。



## 5. 词嵌入为何不采用one-hot向量

虽然one-hot词向量构造起来很容易，但通常并不是⼀个好选择。⼀个主要的原因是，one-hot词向量⽆法准确表达不同词之间的相似度，如我们常常使⽤的余弦相似度。由于任何两个不同词的one-hot向量的余弦相似度都为0，多个不同词之间的相似度难以通过onehot向量准确地体现出来。

word2vec⼯具的提出正是为了解决上⾯这个问题。它将每个词表⽰成⼀个定⻓的向量，并使得这些向量能较好地表达不同词之间的相似和类⽐关系。





## 过拟合与欠拟合

### 过拟合

**现象**

- **训练集**效果很好，但是**验证集**很差，这种现象称为过拟合，表现为**高方差**。

#### 常见解决方法

- 训练数据不足
  - 数据增强
    - NLP-EDA
    
      **(1) 同义词替换（SR: Synonyms Replace）：**不考虑stopwords，在句子中随机抽取n个词，然后从同义词词典中随机抽取同义词，并进行替换。
    
      Eg: “我非常喜欢这部电影” —> “我非常喜欢这个影片”，句子仍具有相同的含义，很有可能具有相同的标签。
    
      **(2) 随机插入(RI: Randomly Insert)：**不考虑stopwords，随机抽取一个词，然后在该词的同义词集合中随机选择一个，插入原句子中的随机位置。该过程可以重复n次。
    
      Eg : “我非常喜欢这部电影” —> “爱我非常喜欢这部影片”。
    
      **(3) 随机交换(RS: Randomly Swap)：**句子中，随机选择两个词，位置交换。该过程可以重复n次。
    
      Eg: “如何评价 2017 知乎看山杯机器学习比赛?” —> “2017 机器学习?如何比赛知乎评价看山杯”。
    
      **(4) 随机删除(RD: Randomly Delete)：**句子中的每个词，以概率p随机删除。
    
      Eg: “如何评价 2017 知乎看山杯机器学习比赛?" —> “如何 2017 看山杯机器学习 ”。
    
      > **停用词stopwords是指在文本中频繁出现但通常没有太多有意义的词语**。这些词语往往是一些常见的功能词、虚词甚至是一些标点符号，如介词、代词、连词、助动词等，比如中文里的"的"、“是”、“和”、“了”、“。“等等，英文里的"the”、“is”、“and”、”…"等等。
      >
      > 停用词的作用是在文本分析过程中过滤掉这些常见词语，从而减少处理的复杂度，提高算法效率，并且在某些任务中可以改善结果的质量，避免分析结果受到这些词的干扰。
      >
    
    - CV-切割、裁剪、旋转
    
    - Mixup
    
      ![image-20240611141740508](C:\Users\CZY\AppData\Roaming\Typora\typora-user-images\image-20240611141740508.png)
    
  - 对抗训练
  
- 训练数据中存在噪声：
  - 交叉验证
  - 集成学习（Bagging）
  
- 模型复杂度较高：
  - 正则化项
  - Dropout
  - Early Stoping
  - 降低模型复杂度
    - 减少模型参数
    - 使用简单模型

### 欠拟合

**现象**

- **训练集**和**验证集**的效果都很差，这种现象称为欠拟合，表现为**高偏差**；

#### 常见解决方法

- 特征工程
  - FM、FFM
- 集成学习（Boosting）
- 提高模型复杂度
  - 增加模型参数
  - 使用复杂模型

### 参考

- [Bagging和Boosting的区别（面试准备） - Earendil - 博客园](https://www.cnblogs.com/earendil/p/8872001.html)
- [NLP-Interview-Notes/过拟合和欠拟合.md at main · km1994/NLP-Interview-Notes](https://github.com/km1994/NLP-Interview-Notes/blob/main/BasicAlgorithm/%E8%BF%87%E6%8B%9F%E5%90%88%E5%92%8C%E6%AC%A0%E6%8B%9F%E5%90%88.md)

## 常见正则化方法

### 数据增强

- 简单数据增强
  - NLP（EDA：增删改、替换）- EDA：
  
    **(1) 同义词替换（SR: Synonyms Replace）：**不考虑stopwords，在句子中随机抽取n个词，然后从同义词词典中随机抽取同义词，并进行替换。
  
    Eg: “我非常喜欢这部电影” —> “我非常喜欢这个影片”，句子仍具有相同的含义，很有可能具有相同的标签。
  
    **(2) 随机插入(RI: Randomly Insert)：**不考虑stopwords，随机抽取一个词，然后在该词的同义词集合中随机选择一个，插入原句子中的随机位置。该过程可以重复n次。
  
    Eg : “我非常喜欢这部电影” —> “爱我非常喜欢这部影片”。
  
    **(3) 随机交换(RS: Randomly Swap)：**句子中，随机选择两个词，位置交换。该过程可以重复n次。
  
    Eg: “如何评价 2017 知乎看山杯机器学习比赛?” —> “2017 机器学习?如何比赛知乎评价看山杯”。
  
    **(4) 随机删除(RD: Randomly Delete)：**句子中的每个词，以概率p随机删除。
  
    Eg: “如何评价 2017 知乎看山杯机器学习比赛?" —> “如何 2017 看山杯机器学习 ”。
  
    > **停用词stopwords是指在文本中频繁出现但通常没有太多有意义的词语**。这些词语往往是一些常见的功能词、虚词甚至是一些标点符号，如介词、代词、连词、助动词等，比如中文里的"的"、“是”、“和”、“了”、“。“等等，英文里的"the”、“is”、“and”、”…"等等。
    >
    > 停用词的作用是在文本分析过程中过滤掉这些常见词语，从而减少处理的复杂度，提高算法效率，并且在某些任务中可以改善结果的质量，避免分析结果受到这些词的干扰。
  
  - 图像（切割、裁剪、旋转等）
  
- 数据融合
  - Mixup：
  
    ![image-20240611141740508](C:\Users\CZY\AppData\Roaming\Typora\typora-user-images\image-20240611141740508.png)

### 为损失函数添加正则化项

- 常见的正则化项有 L1、L2 正则项；

**参考**

- L1、L2 正则化的特点与区别：[NLP-Interview-Notes/正则化.md at main · km1994/NLP-Interview-Notes](https://github.com/km1994/NLP-Interview-Notes/blob/main/BasicAlgorithm/%E6%AD%A3%E5%88%99%E5%8C%96.md)
- [机器学习中正则化项L1和L2的直观理解_阿拉丁吃米粉的博客-CSDN博客](https://blog.csdn.net/jinping_shi/article/details/52433975)

#### L1 正则化

#### L2 正则化

### 提前结束训练（Early Stopping）

- 当模型在验证集上的性能开始下降时，提前结束训练；
- 一般会配合交叉验证来使用；

**参考**

- [【关于 早停法 EarlyStopping 】那些你不知道的事](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=2247488608&idx=1&sn=5c484d2d0177dc968265ca1f3c9221e0&scene=21#wechat_redirect)


### Dropout

**作用**

- 提升泛化能力，减少过拟合；

**原理**

- Dropout 提供了一种廉价的 Bagging 集成近似（模型平均）；

**思想**

- 遗传算法，通过随机变异（随机删除神经元），来促使整个种群的进化；

#### 常见问题

##### Dropout 在训练和测试时有什么区别？为什么？

- 训练时，经过 Dropout 的输出值会乘以 $\frac{1}{1-p}$；测试时不会。  
- 经过 Dropout 后，输入 `x` 的期望输出将变为 `p*0 + (1-p)*x = (1-p)x`（`p` 的可能变为 0，`1-p` 的可能保持不变）；
- 为了还原未经过 Dropout 的期望值，故需要乘以 $\frac{1}{1-p}$

##### 为什么 Dropout 能防止过拟合？

- 直观上，Dropout 会使部分神经元失活，减小了模型容量，从而降低了模型的拟合能力；
- 宏观上，Dropout 提供了一种廉价的 Bagging 集成方法（共享权重）；  
- 隐藏单元经过 Dropout 后，必须学习与不同采样的神经元合作，使得神经元具有更强的健壮性（减少神经元之间复杂的共适应关系）；

#### PyTorch 实现

- 【训练阶段】前向传播时，对每个神经元以概率 `p` 失活（即乘以 `0.`），而其他未失活的单元则乘以 `1/(1-p)`（放大）
- 【测试阶段】使 dropout 失效，即正常使用所有神经元；

```python
class Dropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p  # 以 p 的概率丢弃

    def forward(self, x):
        if not self.training:
            return x
        
        mask = (torch.rand(x.shape) > self.p).float()
        return x * mask / (1.0 - self.p)
```

### R-Drop

> [[2106.14448] R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)

**动机 & 作法**

- 尝试解决 Dropout 在训练与预测时使用不一致的问题；
- **Dropout 本身不是已经尝试解决了这个不一致问题吗？它的解决方案有什么问题？**
  - Dropout 通过缩放神经元的输出值来缓解训练与预测时不一致的影响。Dropout 的本意是为了得到一个“模型平均”的结果，而这种通过缩放来还原实际上是一种“权重平均”（见 Dropout 的推导），这两者未必等价；
  - 具体来说，Dropout 的正确使用方式应该是预测时打开 Dropout，然后计算多次预测的平均值作为结果；但实际并不是这样使用的。
- **R-Drop 是怎么解决这个问题的？**
  - 通过对同一样本 Dropout 两次，然后加入 KL 散度来保持不同 Dropout 下预测结果的一致性；
- **KL 散度损失是怎么保证预测一致性的？**
  - 交叉熵损失只关注目标类的得分，非目标类的得分不影响最终 loss；相当于训练目标是 “**不同 Dropout 下目标类的得分都大于非目标类的得分**”。
  - 举例来说 `[0.5, 0.2, 0.3]`、`[0.5, 0.3, 0.2]` 与 `[1, 0, 0]` 的交叉熵损失是一样的，都是 `-log(0.5)`，非目标类的 `0.2` 和 `0.3` 都没有起作用；
  - KL 散度则会关注每一个类别的得分，相当于训练目标是“**不同 Dropout 下每个类别的得分一致**”
  - 就上例来说，计算 `[0.5, 0.2, 0.3]`、`[0.5, 0.3, 0.2]` 与 `[1, 0, 0]` 的 KL 散度都会产生非零损失；

#### PyTorch 实现

```python
class RDrop(nn.Module):

    def __init__(self, encoder, kl_alpha=1.0):
        super().__init__()

        self.encoder = encoder
        self.kl_alpha = kl_alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss()

    def forward(self, x, labels):
        logits1 = self.encoder(x)
        logits2 = self.encoder(x)
        ce_loss = (self.ce(logits1, labels) + self.ce(logits2, labels)) / 2
        kl_loss1 = self.kl(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1))
        kl_loss2 = self.kl(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1))
        return ce_loss + self.kl_alpha * (kl_loss1 + kl_loss2) / 2
```

**参考**

- [又是Dropout两次！这次它做到了有监督任务的SOTA - 苏剑林](https://kexue.fm/archives/8496)


### 各种 Normalization

#### 参考

- [详解深度学习中的 Normalization，BN/LN/WN - 知乎](https://zhuanlan.zhihu.com/p/33173246)

#### Batch Normalization

`per channel per batch`

**使用场景**：CV

**前向过程**

$$
\begin{aligned}
    \mu &= \frac{1}{m} \sum_{i=1}^m x_i                         &//&\text{batch mean} \\
    \sigma^2 &= \frac{1}{m} \sum_{i=1}^m (x_i-\mu)^2            &//&\text{batch variance} \\
    \hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}   &//&\text{normalization} \\
    y_i &= \gamma \hat{x}_i + \beta                             &//&\text{scale and shift}
\end{aligned}
$$

##### PyTorch实现

```python
def batch_norm(x, eps=1e-5):
    """x: [batch_size, time_step, channel]"""
    C = x.shape[-1]
    mean = torch.mean(x, dim=(0, 1), keepdim=True)  # [1, 1, C]
    std = torch.std(x, dim=(0, 1), unbiased=False, keepdim=True)  # [1, 1, C]
    gamma = torch.nn.Parameter(torch.empty(C))
    beta = torch.nn.Parameter(torch.empty(C))
    output = gamma * (x - mean) / (std + eps) + beta
    return output
```

**参考**

- [【深度学习】深入理解Batch Normalization批标准化 - 郭耀华 - 博客园](https://www.cnblogs.com/guoyaohua/p/8724433.html)
- [Batch Normalization的通俗解释 - 知乎](https://zhuanlan.zhihu.com/p/54073204)


#### Layer Normalization

`per sample per layer`

**使用场景**：NLP

**前向过程**

```python

```


#### Instance Normalization

`per sample per channel`

**作用**

**使用场景**：CV 风格迁移

#### Group Normalization

`per sample per group`

#### Weight Normalization


#### 常见问题

##### BN 与 LN 的区别

- BN 在 batch 维度为归一；
- LN 在 feature 维度做归一；

##### 为什么 BN 一般不用于 NLP ？


### 对抗训练





## 梯度消失和梯度爆炸

- 原因：

（1）**深层网络**：梯度反向传播时，通过链式法则求前面层的导数，随着层数增加，大于1的导数不断相乘导致梯度爆炸，小于1的导数不断不断相乘导致梯度消失

（2）**不合适的激活函数**：如果激活函数选择不合适，比如使用sigmoid，梯度消失就会很明显了，如果使用sigmoid作为损失函数，其梯度是不可能超过0.25的，这样经过链式求导之后，很容易发生梯度消失。同理，tanh作为激活函数，比sigmoid要好一些，但是它的导数仍然是小于1的。

![aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTEzNDIyNjc1](D:\typora图片\aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTEzNDIyNjc1.png)                                  ![image-20231026162148961](D:\typora图片\image-20231026162148961.png)

- 解决办法：

（1）**重新设计网络模型**

1. 在深度神经网络中，梯度爆炸可以通过重新设计层数更少的网络来解决。
2. 使用更小的批尺寸对网络训练也有好处。
3. 在循环神经网络中，训练过程中在更少的先前时间步上进行更新（沿时间的截断反向传播，truncated Backpropagation through time）可以缓解梯度爆炸问题。

（2）**梯度截断（Gradient Clipping）**

主要是针对梯度爆炸提出的，其思想是设置一个梯度剪切阈值，然后更新梯度的时候，如果梯度超过这个阈值，那么就将其强制限制在这个范围之内。这可以防止梯度爆炸。

（3）**权重正则化（Weight Regularization）**

比较常见的是 L1 正则和 L2 正则。正则化是通过对网络权重做正则限制过拟合，仔细看正则项在损失函数的形式：

$$Loss = (y − W^Tx)^2 + α||W||^2$$

其中，α是指正则项系数，因此，如果发生梯度爆炸，权值的范数就会变的非常大，通过正则化项，可以部分限制梯度爆炸的发生。

```注：事实上，在深度神经网络中，往往是梯度消失出现的更多一些。```

（4）**使用ReLU、Leaky ReLU、ELU等激活函数**

如果激活函数的导数为1，那么就不存在梯度消失爆炸的问题了，每层的网络都可以得到相同的更新速度，ReLU就这样应运而生。

**ReLU**的主要贡献在于：

- 解决了梯度消失、爆炸的问题
- 计算方便，计算速度快
- 加速了网络的训练

同时也存在一些**缺点**：

- 由于负数部分恒为0，会导致一些神经元无法激活（可通过设置小学习率部分解决）

- 输出不是以0为中心的

**Leaky ReLU**解决了0区间带来的影响，而且包含了ReLU的所有优点

Leaky ReLU = max(k ∗ x , x) 其中k是leak系数，一般选择0.01或者0.02，或者通过学习而来。

![aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNzAyMjExMDAxNTE3](D:\typora图片\aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwNzAyMjExMDAxNTE3.png)

**ELU**激活函数也是为了解决ReLU的0区间带来的影响，其数学表达为：![aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTM0NjAzMDc5](D:\typora图片\aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTM0NjAzMDc5.png)
其函数及其导数数学形式为：

![aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTM0NjE0MTIx](D:\typora图片\aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTM0NjE0MTIx.png)

但是ELU相对于Leaky ReLU来说，计算要更耗时间一些

（5）==**Batch Normalization**==

内部协变量偏移（Internal Covariate Shift）：在深层网络训练的过程中，由于网络中参数变化而引起内部结点数据分布发生变化的这一过程。就是后面隐藏层的输入都来自上一层的输出，上一层的输出又是由它的输入和参数计算得到的，参数由于网络更新不断发生变化，导致后面网络的输入的分布不断变化。
带来两个问题：
（1）上层网络需要不停调整来适应输入数据分布的变化，导致网络学习速度的降低
（2）网络的训练过程容易陷入梯度饱和区（即自变量进入某个区间后，梯度变化会非常小），减缓网络收敛速度

通过BN解决：

（1）在mini-batch的基础上对每个特征进行独立的normalization，使得输入每个特征的分布均值为0，方差为1。（缓解ICS问题，让每一层网络的输入数据分布都变得稳定）

![image-20231026215007986](D:\typora图片\image-20231026215007986.png)

![image-20231026215639999](D:\typora图片\image-20231026215639999.png)

（2）只做normalization会导致数据表达能力的缺失。也就是我们通过变换操作改变了原有数据的信息表达（representation ability of the network），使得底层网络学习到的参数信息丢失。另一方面，通过让每一层的输入分布均值为0，方差为1，会使得输入在经过sigmoid或tanh等饱和性激活函数时，容易陷入非线性激活函数的线性区域（梯度饱和区）。因此，BN又引入了两个可学习（learnable）的参数$$\gamma$$与$$\beta$$。这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换，即$$\tilde Z_j=\gamma_j\hat Z_j+\beta_j$$。特别地，当$$\gamma^2=\sigma^2, \beta=\mu$$时，可以实现等价变换（identity transform）并且保留了原始输入特征的分布信息。

![image-20231026215400997](D:\typora图片\image-20231026215400997.png)

总的来说，BN通过将每一层网络的输入进行normalization，保证输入分布的均值与方差固定在一定范围内，减少了网络中的Internal Covariate Shift问题，并在一定程度上缓解了梯度消失，加速了模型收敛；并且BN使得网络对参数、激活函数更加具有鲁棒性，降低了神经网络模型训练和调参的复杂度；最后BN训练过程中由于使用mini-batch的mean/variance作为总体样本统计量估计，引入了随机噪声，在一定程度上对模型起到了正则化的效果。

（6）**残差结构**

![aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTQ0MTA1NzYw](D:\typora图片\aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcxMjIwMTQ0MTA1NzYw.png)

相比较于以前网络的直来直去结构，残差中有很多这样的跨层连接结构，这样的结构在反向传播中具有很大的好处，见下式：

![v2-c36d0a801d329b333e3dd35b20003710_720w](D:\typora图片\v2-c36d0a801d329b333e3dd35b20003710_720w.webp)

式子的第一个因子$$\frac{\partial loss}{\partial x_L}$$表示的损失函数到达 L 的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。

（7）**LSTM**

LSTM不那么容易发生梯度消失，主要原因在于LSTM内部复杂的“门” (gates)，LSTM通过它内部的“门”可以接下来更新的时候“记住”前几次训练的”残留记忆“，因此，经常用于生成文本中。



## 文本表示方法

文本表示成计算机能够运算的数字或向量的方法一般称为**词嵌入**（Word Embedding）方法。词嵌入将不定长的文本转换到定长的空间内，是文本分类的第一步。

### 1. One-hot

将每一个单词使用一个离散的向量表示。具体将每个字/词编码一个索引，然后根据索引进行赋值。

> （1）容易受维数灾难的困扰。词数量过大会占用很大内存；
>
> （2）词汇鸿沟。不能很好地刻画词与词之间的相似性，任意两个词之间都是孤立的，从这两个向量中看不出两个词是否有关系；
>
> （3）强稀疏性。当维度过度增长的时候，整个向量中有用的信息特别少，几乎就没法做计算。

### 2. Bag of Words

词袋表示，也称为Count Vetors，每个文档的字/词可以使用其出现次数（词频）来进行表示。

### 3. N-gram

与Count Vectors类似，不过加入了相邻单词组合成为新的单词，并进行计数。如果N取值为2，则相邻的2个词组成新词再进行计数。

### 4. TF-IDF

由两部分组成：第一部分是**词语频率**（Term Frequency），第二部分是**逆文档频率**（Inverse Document Frequency）。其中计算语料库中文档总数除以含有该词语的文档数量，然后再取对数就是逆文档频率。

``` 
TF(t) = 该词语在当前文档出现的次数 / 当前文档中词语的总数
IDF(t) = log（语料库的文档总数 / （出现该词语的文档数 + 1））
TF-IDF(t) = TF(t) x IDF(t)
```

TF-IDF与一个词在文档中的出现次数成正比，与该词在整个语料库中的出现次数成反比。eg：自动提取关键词的算法就是计算出文档的每个词的TF-IDF值，然后按降序排列，取排在最前面的几个词。

**缺点**：有时候用词频来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。

> 前面4种文本表示方法（One-hot、Bag of Words、N-gram、TF-IDF）都或多或少存在一定的问题：转换得到的向量维度很高，需要较长的训练实践；没有考虑单词与单词之间的关系，只是进行了统计。
> 与这些表示方法不同，**深度学习**也可以用于文本表示，还可以将其映射到一个低纬空间。其中比较典型的例子有：FastText、Word2Vec和Bert。

###  5. FastText

是一种典型的深度学习词向量的表示方法，它非常简单通过Embedding层将单词映射到稠密空间，然后将句子中所有的单词在Embedding空间中进行平均，进而完成分类操作。所以FastText是一个三层的神经网络，输入层、隐藏层和输出层。

FastText在文本分类任务上，是优于TF-IDF的：

- FastText用单词的Embedding叠加获得的文档向量，将相似的句子分为一类
- FastText学习到的Embedding空间维度比较低，可以快速进行训练

### 6. Word2Vec

Word2Vec 是语言模型中的一种，它是从大量文本语料中以无监督方式学习语义知识的模型，被广泛地应用于自然语言处理中。

Word2Vec 是用来生成词向量的工具。

Word2Vec 是轻量级的神经网络，其模型仅仅包括输入层、隐藏层和输出层，模型框架根据输入输出的不同，主要包括 CBOW 和 Skip-gram 模型。 CBOW 的方式是在知道词 $w_t$ 的上下文 $w_{t-2}$ ， $w_{t-1}$ ， $w_{t+1}$ ， $w_{t+2}$ 的情况下预测当前词 $w_t$。而 Skip-gram 是在知道了词 $w_t$ 的情况下，对词 $w_t$ 的上下文进行预测， $w_{t-2}$ ， $w_{t-1}$ ， $w_{t+1}$ ， $w_{t+2}$ 如下图所示：

![u=3165528252,3781957337&fm=253&app=138&f=JPEG](D:\typora图片\u=3165528252,3781957337&fm=253&app=138&f=JPEG.png)

[深入浅出Word2Vec原理解析 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/114538417)





## LLM预训练、微调

[深度学习笔记（一）：模型微调fine-tune_深度学习 fine tune-CSDN博客](https://blog.csdn.net/sinat_36831051/article/details/84988174)

**预训练**：

就是指预先训练的一个模型或者指预先训练模型的过程。

预训练模型就是已经用数据集训练好了的模型。现在我们常用的预训练模型就是他人用常用模型和大型数据集来做训练集训练好的模型参数。

**微调**：

就是指将预训练过的模型作用于自己的数据集，并使参数适应自己数据集的过程。

微调步骤：

1. 在源数据集（如 ImageNet 数据集）上预训练一个神经网络模型，即源模型。
2. 创建一个新的神经网络模型，即目标模型。它复制了源模型上除了输出层外的所有模型设计及其参数。假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集；还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
3. 为目标模型添加一个输出大小为目标数据集类别个数的输出层，并随机初始化该层的模型参数。
4. 在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。

![20181213145907678](D:\typora图片\20181213145907678.png)

普通预训练模型用了大型数据集做训练，已经具备了提取浅层基础特征和深层抽象特征的能力。

不做微调需要（1）从头开始训练，需要大量的数据、计算时间和计算资源；（2）存在模型不收敛，参数不够优化，准确率低，模型泛化能力低，容易过拟合等风险。

使用微调可以有效避免了上述可能存在的问题。

### 预训练模型的轻量化微调

#### 背景

> [从遗忘问题到预训练轻量化微调 - 李rumor](https://mp.weixin.qq.com/s/C_6qlTq63IBnRSEMnDO7SQ)

- “**预训练+微调**”范式有效的前提是，我们假设模型在海量数据上学到了大量**先验知识**，而这些知识被存储在模型的**参数**中；
- 对整个预训练模型进行微调，意味着会**改动**这些参数；如果变动太大，那么就可能会带来“**灾难性遗忘**”的问题；
  - 一个简单的验证方法：“有的时候，大家可以试试学习率大一些跑跑，会发现学几代以后loss就会骤变，这个其实就是重现遗忘最简单的方式。”
  - 不过参数变动不代表知识一定会丢失，大多数情况下，“预训练+微调”依然是有效的；

#### 三种解决思路

1. **Replay**: 重播，就是在新任务中，把老的内容复习一遍，能让模型保留住。
   - 虽然实现简单，但是现在的预训练模型和数据都很“重”，成本很大；
2. **Regularization-Based**: 在损失函数上应用正则化方法，使新模型和原模型之间的差距不会很大（跟蒸馏的思想很接近）
   - 实现简单，对模型的改动很小；
   - 这个方法有两个问题：1）需要微调整个模型，效率低；2）可能会导致**训练目标的偏移**，即得到的不是最优解；
3. **Lightweight Finetuning**: 轻量化微调，目前比较常见的方法是**参数孤立化**（Parameter Isolation），即冻住预训练模型，加入新的模块，只微调该模块来避免遗忘的问题。
   - 目前比较主流的方法，兼顾了“遗忘问题”和训练效率；

#### 轻量化微调（Lightweight Finetuning）

- 目前**轻量化微调**的主要方法是**参数孤立化**，即不动预训练模型，而是在预训练模型的基础上增加新的可训练参数



## 循环神经网络

RNN/LSTM/GRU 和 Transformer 都用于处理序列数据，在计算量上有所不同。

比如 GRU 是一种RNN的变体，它通过使用门控机制来克服传统 RNN 中的梯度消失问题。GRU 的计算量相对较小，因为它的参数量较少，并且它是一种逐步处理输入序列的模型。在每个时间步，GRU 只需计算一些简单的线性变换和非线性激活函数。

相比之下，Transformer 是一种基于注意力机制的神经网络架构，用于处理序列数据。它引入了**自注意力机制，允许模型在不同位置对输入序列的各个元素进行加权关注**。由于 Transformer 需要计算全连接的注意力矩阵，它的计算量较大。

GRU 相对较简单，计算量相对较小，适用于较小规模的序列数据。而 Transformer 计算量较大，适用于处理更大规模的序列数据，如机器翻译或语言建模等任务。

### 1. RNN

（Recurrent Neural Network）

![1027162-20161113162104764-927733222](D:\typora图片\1027162-20161113162104764-927733222.png)

![1027162-20161113162105295-307972897](D:\typora图片\1027162-20161113162105295-307972897.png)

**RNN前向过程：**
$$
\begin{aligned}
    y_t &= W[h_{t-1},x_t] + b \\ 
    h_t &= \tanh(a_t) 
\end{aligned}
$$

>  $[x1,x2]$ 表示**向量拼接**；RNN 为递推结构，其中 $h_0$ 一般初始化为 0

**RNN节点内部连接：**

![1027162-20161113162111280-1753976877](D:\typora图片\1027162-20161113162111280-1753976877.png)

在 RNN 中，每个时间步都有一个隐藏状态（hidden state)，它可以接收当前时间步的输入和上一个时间步的隐藏状态作为输入。隐藏状态的输出不仅取决于当前时间步的输入，还取决于之前所有时间步的输入。这种循环连接使得RNN**可以处理变长序列**，并且能够捕捉到序列中的时序信息。与传统的前馈神经网络不同，RNN具有循环连接，使得它可以在处理序列时**保持记忆状态**。

每一 time step 中使用的参数 W, b 是一样的，也就是说每个步骤的参数都是共享的，这是RNN的重要特点。

RNN 在自然语言处理（NLP）等领域有广泛的应用，例如语言建模、机器翻译、情感分析等任务。由于 RNN 能够处理变长序列，并且可以保持记忆状态，它在处理自然语言时可以考虑上下文的信息，捕捉到词语之间的依赖关系和语义信息。

此外，RNN 也可以应用于时间序列预测，例如股票价格预测、天气预测等。RNN 可以根据过去的时间序列数据预测未来的趋势，对于具有时序依赖的数据具有一定的优势。

- **优点**：

① RNN 很**适合处理序列数据**，因为考虑了之前的信息

② 可以和 CNN 一起使用得到更好的任务效果

- **缺点**：

① **梯度消失、梯度爆炸**

② RNN 较其他 CNN 和全连接**要用更多的显存空间，更难训练**

③ RNN 有短期记忆问题（越晚的输入影响越大，越早的输入影响越小），**无法处理太长的输入序列**

RNN 对于长时记忆的困难主要来源于梯度爆炸 / 消失问题：[RNN - LSTM - GRU - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/60915302)

**梯度爆炸的解决办法：**

(1) **Truncated Backpropagation through time**：每次只 BP 固定的 time step 数，类似于 mini-batch SGD。缺点是丧失了长距离记忆的能力。

(2) **Clipping Gradients**： 当梯度超过一定的 threshold 后，就进行 element-wise 的裁剪，该方法的缺点是又引入了一个新的参数 threshold。同时该方法也可视为一种基于瞬时梯度大小来自适应 learning rate 的方法

**梯度消失的解决办法：**

(1) **使用 LSTM、GRU等升级版 RNN**，使用各种 gates 控制信息的流通。

(2) 将权重矩阵 W 初始化为正交矩阵。

(3) 反转输入序列。像在机器翻译中使用 seq2seq 模型，若使用正常序列输入，则输入序列的第一个词和输出序列的第一个词相距较远，难以学到长期依赖。将输入序列反向后，输入序列的第一个词就会和输出序列的第一个词非常接近，二者的相互关系也就比较容易学习了。这样模型可以先学前几个词的短期依赖，再学后面词的长期依赖关系。

### 2. LSTM

（Long Short-Term Memory）

LSTM 分别采用两大策略来解决上述的缺点。首先，**针对梯度消失问题，采用门机制来解决** ，效果非常好。 而对于短期记忆覆盖长期记忆的问题， LSTM **采用一个 cell state 来保存长期记忆**， 再配合门机制对信息进行过滤，从而达到对长期记忆的控制。

![aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhkY2VQV2RtTmFoSVRpY3VHVUhPNlh1MDBadWpKUThpYjlpYWFvMVNtRDc2eHMyaWJ1TGhEcTRCcU5BLzY0MA](D:\typora图片\aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhkY2VQV2RtTmFoSVRpY3VHVUhPNlh1MDBadWpKUThpYjlpYWFvMVNtRDc2eHMyaWJ1TGhEcTRCcU5BLzY0MA.png)

LSTM模型的核心思想是“细胞状态”（cell state）。“细胞状态”类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。

<img src="D:\typora图片\微信图片_20240423142023.jpg" alt="微信图片_20240423142023" style="zoom: 67%;" />

LSTM 有通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种**让信息选择式通过**的方法。他们**包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作**。

![v2-835e18c5697bbcda6c564864450b373e_720w](D:\typora图片\v2-835e18c5697bbcda6c564864450b373e_720w.webp)

1. **遗忘门（forget gate）**

   **决定从细胞状态中丢弃什么信息**。输入 $$ℎ_{t−1}$$ 和 $$x_t$$ ，Sigmoid 层输出一个 0 到 1 之间的数。1 代表“完全保留”，而 0代表“完全舍弃”。

   <img src="D:\typora图片\image-20231102134116992.png" alt="image-20231102134116992" style="zoom: 150%;" />

   ![image-20231102134232962](D:\typora图片\image-20231102134232962.png)

   **输入门（input gate）**

   **确定细胞状态所存放的新信息**。这一步有两个部分， sigmoid 层作为“输入门层”决定哪些数据是需要更新的。 tanh 层创建一个新的候选值向量 $$\widetilde C_t$$加入到状态中。下一步，我们要将这两个部分合并以创建对细胞状态的更新。

   <img src="D:\typora图片\image-20231102134350567.png" alt="image-20231102134350567" style="zoom:150%;" />

   ![image-20231102134301093](D:\typora图片\image-20231102134301093.png)

   在决定需要遗忘和需要加入的记忆之后，更新细胞状态 $$C_{t-1}$$ 为 $$C_t$$了。把旧的状态 $$C_{t-1}$$ 与 $$f_t$$ 相乘，遗忘我们先前决定遗忘的东西，然后我们加上 $$i_t * \widetilde C_t$$，这可以理解为新的记忆信息，当然，这里体现了对状态值的更新度是有限制的，我们可以把 $$i_t$$ 当成一个权重。

   <img src="D:\typora图片\image-20231102134337537.png" alt="image-20231102134337537" style="zoom:150%;" />

   ![image-20231102134322474](D:\typora图片\image-20231102134322474.png)

   **输出门（output gate）**

   **确定输出**。这个输出基于我们的细胞状态，但会是一个过滤后的值。首先，我们运行一个 sigmoid 层，这个也就是输出门，以决定细胞状态中的哪个部分是我们将要输出的。然后把细胞状态通过 tanh 层处理（将数值压到 -1 和 1 之间），并将它与 sigmoid 门的输出相乘，这样就只输出了我们想要的部分了。

   <img src="D:\typora图片\image-20231102134212043.png" alt="image-20231102134212043" style="zoom:150%;" />

   ![image-20231102134415957]( D:\typora图片\image-20231102134415957.png)


**综上，整个LSTM前向过程为：**
$$
\begin{aligned}
    f_t &= \sigma(W_f[h_{t-1},x_t] + b_t) \\
    i_t &= \sigma(W_i[h_{t-1},x_t] + b_i) \\
    \tilde{C}_t &= \tanh(W_C[h_{t-1},x_t] + b_C) \\
    C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \\
    o_t &= \sigma(W_o[h_{t-1},x_t] + b_o) \\
    h_t &= o_t * \tanh(C_t)
\end{aligned}
$$

> $[x1,x2]$ 表示**向量拼接**；$*$ 表示**按位相乘**；  
> $f_i$：长期记忆的遗忘比重；  
> $i_i$：短期记忆的保留比重；  
> $\tilde{C}_t$：当前时间步的 Cell 隐状态，即短期记忆；也就是普通 RNN 中的 $h_t$  
> $C_{t-1}$：历史时间步的 Cell 隐状态，即长期记忆；  
> $C_t$：当前时间步的 Cell 隐状态；  
> $o_t$：当前 Cell 隐状态的输出比重；  
> $h_t$：当前时间步的隐状态（输出）；

**门机制的好处：**

- 首先，门机制极大的减轻了梯度消失问题，极大的简化了我们的调参复杂度。
- 其次，门机制提供了**特征过滤**，将有用的特征保存，没用的特征丢弃，这极大的丰富了我们向量的表示信息。

**$$h_t$$ 与 $$C_t$$ 的传递关系:**

- 首先，先理解 $$h_t$$ 与 $$C_t$$ 的本质， $$C_t$$ 本质上是 $$0−t$$ 时刻的全局信息，而 $$h_t$$ 表示的是在 $$0−t−1$$ 时刻的全局信息的影响下， 当前 $$t$$ 时刻的上下文表示。
- 具体到公式中，我们看到 $$h_t$$ 本质是先将 $$C_t$$ 经过 tanh 激活函数压缩为 (-1, 1)之间的数值，然后再通过输出门对 $$C_t$$ 进行过滤，来获得当前单元的上下文信息。这意味着当前时刻的上下文信息 $$h_t$$ 不过是全局信息 $$C_t$$ 的一部分信息。
  而对于全局信息 $$C_t$$ ， 其是由上一时刻全局信息 $$C_{t-1}$$ 与当前时刻信息 $$x_t$$ 通过输入门与遗忘门结合而成的。

实际应用中一般不采用单层的lstm，而是多层，在很多时序数据中双向的表现也很不错。

**双向LSTM**：

![aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhIMmVsdlRmYXFQbmhoSnFDR3JsSFJBaWEyek1TOU5leDllYkVWR2NDWXh6TVYyc2NyY3lwOWV3LzY0MA](D:\typora图片\aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhIMmVsdlRmYXFQbmhoSnFDR3JsSFJBaWEyek1TOU5leDllYkVWR2NDWXh6TVYyc2NyY3lwOWV3LzY0MA.png)

**深层双向LSTM**：

![aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhMczA0Q29Ta1FLWlBhVWtCaEpRWVM1TzRCN3ppYWRkaWE3TG5jNjJIRTZkaWNBRUJKaGhQZ01lVWcvNjQw](D:\typora图片\aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhMczA0Q29Ta1FLWlBhVWtCaEpRWVM1TzRCN3ppYWRkaWE3TG5jNjJIRTZkaWNBRUJKaGhQZ01lVWcvNjQw.png)

LSTM 的一个初始化技巧就是将 forget gate 的 bias 置为正数（例如 1 或者 5，如 tensorflow 中的默认值就是 1.0 ），这样一来模型刚开始训练时 forget gate 的值都接近 1，不会发生梯度消失 (反之若 forget gate 的初始值过小则意味着前一时刻的大部分信息都丢失了，这样很难捕捉到长距离依赖关系)。 随着训练过程的进行，forget gate 就不再恒为 1 了。不过，一个训好的模型里各个 gate 值往往不是在 [0, 1] 这个区间里，而是要么 0 要么 1，很少有类似 0.5 这样的中间值，其实相当于一个二元的开关。假如在某个序列里，forget gate 全是 1，那么梯度不会消失；某一个 forget gate 是 0，模型选择遗忘上一时刻的信息。

### 3. GRU

（Gated Recurrent Unit）

同时将细胞状态和隐藏状态合并

![1027162-20161117222054654-754777391](D:\typora图片\1027162-20161117222054654-754777391.png)

![image-20231102104757481](D:\typora图片\image-20231102104757481.png)

GRU 由两个门构成，分别是重置门 reset gate $$r_t$$ 和更新门 update gate $$z_t$$。GRU 相比 LSTM ，减少了参数量和计算量，收敛速度更快。

对比一下 GRU 与 LSTM 的公式：

- 首先，门的计算公式大同小异。
- 其次，$$\widetilde C_t$$ 与 $$\widetilde h_t$$ 其实是一样的，都是表示当前信息。
- 最后，LSTM 中的 $$C_t$$ 与 GRU 中的 $$h_t$$ 计算公式相差无几。

从公式上看，可以发现 GRU 抛弃了 LSTM 中的 $$h_t$$ ，它认为既然 $$C_t$$ 中已经包含了 $$h_t$$ 中的信息了，那还要 $$h_t$$ 做什么，于是，它就把 $$h_t$$ 干掉了。 然后，GRU 又发现，在生成当前时刻的全局信息时，我当前的单元信息与之前的全局信息是此消彼长的关系，直接用 1−$$z_t$$ 替换 $$f_t$$ ，简单粗暴又高效 。

GRU很聪明的一点就在于，**我们使用了同一个门控 $$z_t$$ 就同时可以进行遗忘和选择记忆（LSTM则要使用多个门控）**。

- $$(1−z_t)*ℎ_{t-1}$$ ：表示对原本隐藏状态的选择性“遗忘”。这里的 $$1-z_t$$ 可以想象成遗忘门（forget gate），忘记 $$h_{t-1}$$ 维度中一些不重要的信息。
- $$z_t*\widetilde h_t$$ ：表示对包含当前节点信息的  $$\widetilde h_t$$  进行选择性”记忆“。与上面类似，这里的 $$1-z_t$$ 同理会忘记 $$\widetilde h_t$$ 维度中的一些不重要的信息。或者，这里我们更应当看做是对 $$\widetilde h_t$$ 维度中的某些信息进行选择。
- $$h_t = (1−z_t)*ℎ_{t-1}+z_t*\widetilde h_t$$ ：结合上述，这一步的操作就是忘记传递下来的 $$ℎ_{t-1}$$ 中的某些维度信息，并加入当前节点输入的某些维度信息。

> 可以看到，这里的遗忘 $$z_t$$ 和选择 $$1-z_t$$ 是联动的。也就是说，对于传递进来的维度信息，我们会进行选择性遗忘，则遗忘了多少权重 （$$z_t$$），我们就会使用包含当前输入的 $$\widetilde h$$ 中所对应的权重进行弥补 $$1-z_t$$ 。以保持一种”恒定“状态。

### 4. 总结

#### RNN和LSTM的区别

> 相关问题：**Cell 状态的作用**、**LSTM 是如何实现长短期记忆的？**

- LSTM 相比 RNN 多了一组 **Cell 隐状态**，记 $C$（Hidden 隐状态两者都有）；
  - $C$ 保存的是当前时间步的隐状态，具体包括来自之前（所有）时间步的隐状态 $C_{t-1}$ 和当前时间步的**临时隐状态** $\tilde{C}_t$。
- 由于 Cell 的加入，使 LSTM 具备了控制**长期/短期记忆比重**的能力，具体来说：
  - 如果**长期记忆**（之前时间步）的信息不太重要，就**减小** $C_{t-1}$ 的比重，反映在遗忘门的输出 $f_t$ 较小；
  - 如果**短期记忆**（当前时间步）的信息比较重要，就**增大** $\tilde{C}_t$ 的比重，反映在记忆门的输出 $i_t$ 较大；
- **参考**：
  - [对LSTM的理解 - 知乎](https://zhuanlan.zhihu.com/p/332736318)

#### Cell state 和 Hidden state 的关系

> [如何理解 LSTM 中的 cell state 和 hidden state? - 知乎](https://www.zhihu.com/question/68456751?sort=created)

- 从前向过程可以看到 Cell 存储了全部时间步的信息，而 Hidden 由 Cell 经过输出门后得到，可以把 Hidden 看做是网络在 Cell 基础上进行特征选择的结果；
- 一种说法是 Cell 偏向长期记忆，Hidden 偏向短期记忆；

#### LSTM 中各个门的作用是什么？

- “**遗忘门**”控制前一步记忆状态（$C_{t-1}$）中的信息有多大程度被遗忘；
- “**输入门**（记忆门）”控制当前的新状态（$\tilde{C}_t$）以多大的程度更新到记忆状态中；
- “**输出门**”控制当前输出（$h_t$）多大程度取决于当前的记忆状态（$C_t$）；

#### LSTM 前向过程（门的顺序）

- 遗忘门 -> 输入门 -> 输出门

#### GRU 中各门的作用

- “更新门”用于控制前一时刻的状态信息被融合到当前状态中的程度；
- “重置门”用于控制忽略前一时刻的状态信息的程度

#### GRU 与 LSTM 的区别

- 合并 “遗忘门” 和 “记忆门” 为 “更新门”；
  - 其实更像是移除了 “输出门”；
- 移除 Cell 隐状态，直接使用 Hidden 代替；

![aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhNTHFNZnVmTjNwM3Byd3AzN0V4czJuSWRQVmVvQ05VQllQVHhDbjJadXlBNHR5bG96Y3I1QlEvNjQw](D:\typora图片\aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWhNTHFNZnVmTjNwM3Byd3AzN0V4czJuSWRQVmVvQ05VQllQVHhDbjJadXlBNHR5bG96Y3I1QlEvNjQw.png)

### 5. ConvLSTM和ConvGRU

为了构建时空序列预测模型，**同时掌握时间和空间信息**，所以将 LSTM 中的全连接权重改为卷积。

![aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWg2UHlJa000ZlN1R1FOaWJQMWlhNDl1ZUd4aDNmeTdZejJaWnBCUHlxRjRIVUZhSXJvSTFmSll6US82NDA](D:\typora图片\aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9GSXpPRWliOFZRVXEzSVVYMXdLTngzUERJRUN0MHFQNWg2UHlJa000ZlN1R1FOaWJQMWlhNDl1ZUd4aDNmeTdZejJaWnBCUHlxRjRIVUZhSXJvSTFmSll6US82NDA.png)

## Seq2Seq

输入是一个序列，输出也是一个序列。输入序列和输出序列的长度是可变的。

**最基础的Seq2Seq模型**包含了三个部分，即Encoder、Decoder以及连接两者的中间状态向量，Encoder通过学习输入，将其编码成一个固定大小的状态向量$C$，继而将$C$传给Decoder，Decoder再通过对状态向量$C$的学习来进行输出。
下图中的矩形$(h_1,h_2,...,h_m;H_1,H_2,...H_n)$代表了RNN单元，通常是LSTM或者GRU。

![image-20210816173523464](D:\typora图片\image-20210816173523464.png)

==Seq2Seq模型是基于Encoder-Decoder框架设计的==，用于解决序列到序列问题的模型。

Seq2Seq模型缺点包括了RNN模块存在的缺点，和基础Encoder-Decoder框架存在的问题：

1. 中间语义向量$C$无法完全表达整个输入序列的信息；
2. 中间语义向量$C$对$y_1,y_2,...,y_{n-1}$所产生的贡献都是一样的；
3. 随着输入信息长度的增加，先前编码好的信息会被后来的信息覆盖，丢失很多信息。

为了解决Seq2Seq模型的缺陷，**引入了Attention机制**，不再将整个输入序列编码为固定长度的 “ 中间向量$C$ “ ，而是编码成一个向量的序列$C_1,C_2,...$，如下图所示。

![2019-10-28-attention](D:\typora图片\2019-10-28-attention.png)





## Encoder-Decoder模型



## Attention

### 1. 计算过程

第一步： query 和 key 进行相似度计算，得到权值

第二步：将权值进行归一化，得到直接可用的权重

第三步：将权重和 value 进行加权求和

![attention原理3步分解](https://easyai.tech/wp-content/uploads/2022/08/efa5b-2019-11-13-3step.png)



### 2. Attention的类型

<img src="https://easyai.tech/wp-content/uploads/2022/08/7fc4f-2019-11-13-types.png" alt="Attention的种类" style="zoom: 50%;" />

**1. 计算区域**

根据Attention的计算区域，可以分成以下几种：

1）**Soft** Attention，这是比较常见的Attention方式，对所有key求权重概率，每个key都有一个对应的权重，是一种全局的计算方式（也可以叫Global Attention）。这种方式比较理性，参考了所有key的内容，再进行加权。但是计算量可能会比较大一些。

2）**Hard** Attention，这种方式是直接精准定位到某个key，其余key就都不管了，相当于这个key的概率是1，其余key的概率全部是0。因此这种对齐方式要求很高，要求一步到位，如果没有正确对齐，会带来很大的影响。另一方面，因为不可导，一般需要用强化学习的方法进行训练。（或者使用gumbel softmax之类的）

3）**Local** Attention，这种方式其实是以上两种方式的一个折中，对一个窗口区域进行计算。先用Hard方式定位到某个地方，以这个点为中心可以得到一个窗口区域，在这个小区域内用Soft方式来算Attention。

**2. 所用信息**

假设我们要对一段原文计算Attention，这里原文指的是我们要做attention的文本，那么所用信息包括内部信息和外部信息，内部信息指的是原文本身的信息，而外部信息指的是除原文以外的额外信息。

1）**General** Attention，这种方式利用到了外部信息，常用于需要构建两段文本关系的任务，query一般包含了额外信息，根据外部query对原文进行对齐。

比如在阅读理解任务中，需要构建问题和文章的关联，假设现在baseline是，对问题计算出一个问题[向量](https://easyai.tech/ai-definition/vector/)q，把这个q和所有的文章词向量拼接起来，输入到[LSTM](https://easyai.tech/ai-definition/lstm/)中进行建模。那么在这个模型中，文章所有词向量共享同一个问题向量，现在我们想让文章每一步的词向量都有一个不同的问题向量，也就是，在每一步使用文章在该步下的词向量对问题来算attention，这里问题属于原文，文章词向量就属于外部信息。

2）**Local** Attention，这种方式只使用内部信息，key和value以及query只和输入原文有关，==在self attention中，key=value=query==。既然没有外部信息，那么在原文中的每个词可以跟该句子中的所有词进行Attention计算，相当于寻找原文内部的关系。

还是举阅读理解任务的例子，上面的baseline中提到，对问题计算出一个向量q，那么这里也可以用上attention，只用问题自身的信息去做attention，而不引入文章信息。

**3. 结构层次**

结构方面根据是否划分层次关系，分为单层attention，多层attention和多头attention：

1）单层Attention，这是比较普遍的做法，用一个query对一段原文进行一次attention。

2）多层Attention，一般用于文本具有层次关系的模型，假设我们把一个document划分成多个句子，在第一层，我们分别对每个句子使用attention计算出一个句向量（也就是单层attention）；在第二层，我们对所有句向量再做attention计算出一个文档向量（也是一个单层attention），最后再用这个文档向量去做任务。

3）多头Attention，这是Attention is All You Need中提到的multi-head attention，==用到了多个query对一段原文进行了多次attention，每个query都关注到原文的不同部分==，相当于重复做多次单层attention：
$$
head_i=Attention(q_i,K,V)
$$
最后再把这些结果拼接起来：
$$
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O
$$
**4. 模型方面**

从模型上看，Attention一般用在CNN和LSTM上，也可以直接进行纯Attention计算。

**1）CNN+Attention**

CNN的卷积操作可以提取重要特征，我觉得这也算是Attention的思想，但是CNN的卷积感受视野是局部的，需要通过叠加多层卷积区去扩大视野。另外，Max Pooling直接提取数值最大的特征，也像是hard attention的思想，直接选中某个特征。

CNN上加Attention可以加在这几方面：

a. 在卷积操作前做attention，比如Attention-Based BCNN-1，这个任务是文本蕴含任务需要处理两段文本，同时对两段输入的序列向量进行attention，计算出特征向量，再拼接到原始向量中，作为卷积层的输入。

b. 在卷积操作后做attention，比如Attention-Based BCNN-2，对两段文本的卷积层的输出做attention，作为pooling层的输入。

c. 在pooling层做attention，代替max pooling。比如Attention pooling，首先我们用LSTM学到一个比较好的句向量，作为query，然后用CNN先学习到一个特征矩阵作为key，再用query对key产生权重，进行attention，得到最后的句向量。

**2）LSTM+Attention**

LSTM内部有Gate机制，其中input gate选择哪些当前信息进行输入，forget gate选择遗忘哪些过去信息，我觉得这算是一定程度的Attention了，而且号称可以解决长期依赖问题，实际上LSTM需要一步一步去捕捉序列信息，在长文本上的表现是会随着step增加而慢慢衰减，难以保留全部的有用信息。

LSTM通常需要得到一个向量，再去做任务，常用方式有：

a. 直接使用最后的hidden state（可能会损失一定的前文信息，难以表达全文）

b. 对所有step下的hidden state进行等权平均（对所有step一视同仁）。

c. Attention机制，对所有step的hidden state进行加权，把注意力集中到整段文本中比较重要的hidden state信息。性能比前面两种要好一点，而方便可视化观察哪些step是重要的，但是要小心过拟合，而且也增加了计算量。

**3）纯Attention**

Attention is all you need，没有用到CNN/RNN，乍一听也是一股清流了，但是仔细一看，本质上还是一堆向量去计算attention。

**5. 相似度计算方式**

在做attention的时候，我们需要计算query和某个key的分数（相似度），常用方法有：

1）点乘：最简单的方法， $s(q,k)=q^Tk$

2）矩阵相乘：$s(q,k)=q^Tk$

3）cos相似度：$s(q,k)=\frac{q^Tk}{||q||\cdot||k||}$

4）串联方式：把q和k拼接起来，$s(q,k)=W[q;k]$

5）用多层感知机也可以：$s(q,k)=v_a^Ttanh(Wq+Uk)$







Multi-head Self Attention

前向过程（PyTorch 实现）

```python
def forward(x, mask, H, D):
    q = k = v = x  # [B, L, N]
    B, L, N = x.shape

    # linear
    q = W_q(q).reshape([B, L, H, D]).transpose(1, 2)  # [B, H, T, D]
    k = W_k(k).reshape([B, L, H, D]).transpose(1, 2)  # [B, H, T, D]
    v = W_v(v).reshape([B, L, H, D]).transpose(1, 2)  # [B, H, T, D]

    # attention
    logits = matmul(q, k.transpose(-2, -1)) / sqrt(D) + mask
    a = softmax(logits)

    # output
    o = matmul(a, v)
    o = W_o(o).reshape([B, L, N])
    return o
```







## Transformer

RNN是序列模型，无法并行，Transformer可以并行

- 原始 Transformer 指的是一个基于 Encoder-Decoder 框架的 Seq2Seq 模型，用于解决机器翻译任务；
- 后其 Encoder 部分被用于 BERT 而广为人知，因此有时 Transformer 也特指其 Encoder 部分；



## Bert

预训练两大任务：MLM（Masked Language Model）；NSP（Next Sentence Prediction）



## RoBERTa





## DeBERTa

### DeBERTa v1

![10dc9509df43c25e1c39df95420f9b52](D:\typora图片\10dc9509df43c25e1c39df95420f9b52.jpeg)

DeBERTa 模型使用了两种新技术改进了 BERT 和 RoBERTa 模型，同时还引入了一种新的微调方法以提高模型的泛化能力。

两种新技术的改进：

- **注意力解耦机制**：图1右侧黄色部分
- **增强的掩码解码器** ：图1左侧 Enhanced Mask Decoder 部分（V3版本中，用其他方法替换）

新的微调方法：**虚拟对抗训练方法**。

结果表明，这些技术显著提高了模型预训练的效率以及自然语言理解（NLU）和自然语言生成（NLG）下游任务的性能。

与 Large-RoBERTa 相比，基于一半训练数据训练的 DeBERTa 模型在很多 NLP 任务中始终表现得更好，MNLI 提高了+0.9%（90.2%–>91.1%），SQuAD v2.0提高了+2.3%（88.4%–>90.7%），RACE提高了+3.6%（83.2%–>86.8%）。

同时，通过训练由48个Transformer层和15亿参数组成的Large-DeBERTa模型。其性能得到显著提升，单个 DeBERTa 模型在平均得分方面首次超过了 SuperGLUE 基准测试上的表现，同时集成的 DeBERTa 模型目前位居榜首。 截至 2021 年 1 月 6 日，SuperGLUE 排行榜已经超过了人类基线（90.3 VS 89.8）。

==**优化点解释**==

a) 注意力解耦（Disentangled attention）

- 在BERT 中，输入层的每个单词都使用一个向量来表示，该向量是其单词（内容）嵌入和位置嵌入的总和。

![2a0edcb5931c094a9770d0b857c59c75](D:\typora图片\2a0edcb5931c094a9770d0b857c59c75.jpeg)


图2：BERT输入表示. 输入嵌入是token embeddings, segmentation embeddings 和position embeddings 之和。

- 而 DeBERTa 中的每个单词使用两个对其内容和位置分别进行编码的向量来表示，并且注意力单词之间的权重分别使用基于它们的内容和相对位置的解码矩阵来计算。

- 这么做的原因是，经观察发现，单词对的注意力权重不仅取决于它们的内容，还取决于它们的相对位置。 例如，“deep”和“learning”这两个词相邻出现时，它们之间的依赖性比它们出现在不同句子中时要强得多。

- 对于序列中位置 i 处的 token，微软使用了两个向量， {Hi} 和{ Pi|j }表示它，它们分别表示其内容和与位置 j 处的token的相对位置。 token i 和 j 之间的交叉注意力得分的计算可以分解为四个部分:

  ![759473b7bce7dd79e4609dfdda53328a](D:\typora图片\759473b7bce7dd79e4609dfdda53328a.jpeg)

- 也就是说，一个单词对的注意力权重可以使用其内容和位置的解耦矩阵计算为四个注意力（内容到内容，内容到位置，位置到内容和位置到位置）的得分总和。

- 现有的相对位置编码方法在计算注意力权重时使用单独的嵌入矩阵来计算相对位置偏差。 这等效于仅使用上等式中的“内容到内容”和“内容到位置”来计算注意力权重。微软认为位置到内容也很重要，因为单词对的注意力权重不仅取决于它们的内容，还会和相对位置有关。根据它们的相对位置，只能使用内容到位置和位置到内容进行完全建模。 由于微软使用相对位置嵌入，因此位置到位置项不会提供太多附加信息，因此在实现中将其从上等式中删除。

  ![5d28e7ed37911ba4f833133119ec3347](D:\typora图片\5d28e7ed37911ba4f833133119ec3347.jpeg)

b) 增强的掩码解码器
DeBERTa和BERT模型一样，也是使用MLM进行预训练的，在该模型中，模型被训练为使用 mask token 周围的单词来预测mask词应该是什么。 DeBERTa将上下文的内容和位置信息用于MLM。 解耦注意力机制已经考虑了上下文词的内容和相对位置，但没有考虑这些词的绝对位置，这在很多情况下对于预测至关重要。

如：给定一个句子“a new store opened beside the new mall”，并用“store”和“mall”两个词 mask 以进行预测。 仅使用局部上下文（即相对位置和周围的单词）不足以使模型在此句子中区分store和mall，因为两者都以相同的相对位置在new单词之后。 为了解决这个限制，模型需要考虑绝对位置，作为相对位置的补充信息。 例如，句子的主题是“store”而不是“mall”。 这些语法上的细微差别在很大程度上取决于单词在句子中的绝对位置。

有两种合并绝对位置的方法。 BERT模型在输入层中合并了绝对位置。 在DeBERTa中，微软在所有Transformer层之后将它们合并，然后在softmax层之前进行 mask token 预测，如图3 所示。这样，DeBERTa捕获了所有Transformer层中的相对位置，同时解码被mask的单词时将绝对位置用作补充信息 。 此即为 DeBERTa 增强型掩码解码器(EMD)。

![6a06cce1354b40b9dc8b4b6ebad9c66b](D:\typora图片\6a06cce1354b40b9dc8b4b6ebad9c66b.jpeg)

图3： 解码器比较

EMD的结构如图2b 所示。 EMD有两个输入(即 I , H I,HI,H)。 H HH 表示来自先前的transformer层的隐藏状态，并且 I II 可以是用于解码的任何必要信息，例如，绝对位置嵌入或从先前的EMD层输出。 n nn 表示 n nn 个EMD堆叠层，其中每个EMD层的输出将是下一个EMD层的输入I II，最后一个EMD层的输出将直接输出到语言模型头。 n nn 层可以共享相同的权重。 在实验中，微软设定n = 2 n=2n=2 ，即2层共享相同的权重，以减少参数的数量，并使用绝对位置嵌入作为第一个EMD层的I II。 当 I = H I=HI=H 和 n = 1 n=1n=1 时，EMD与BERT解码器层相同。 不过，EMD更通用、更灵活，因为它可以使用各种类型的输入信息进行解码。

c) 虚拟对抗训练方法
对抗训练是NLPer经常使用的技术，在做比赛或者公司业务的时候一般都会使用对抗训练来提升模型的性能。DeBERTa预训练里面引入的对抗训练叫SiFT，它攻击的对象不是word embedding，而是embedding之后的layer norm。

规模不变微调(Scale-invariant-Fine-Tuning SiFT)算法一种新的虚拟对抗训练算法， 主要用于模型的微调。

虚拟对抗训练是一种改进模型泛化的正则化方法。 它通过对抗性样本提高模型的鲁棒性，对抗性样本是通过对输入进行细微扰动而创建的。 对模型进行正则化，以便在给出特定于任务的样本时，该模型产生的输出分布与该样本的对抗性扰动所产生的输出分布相同。

对于之前的NLP任务，一般会把扰动应用于单词嵌入，而不是原始单词序列。 但是嵌入向量值的范围在不同的单词和模型之间有所不同。 对于具有数十亿个参数的较大模型，方差会变大，从而导致对抗训练有些不稳定。

受层归一化的启发，微软提出了SiFT算法，该算法通过应用扰动的归一化词嵌入来提高训练稳定性。 即在实验中将DeBERTa微调到下游NLP任务时，SiFT首先将单词嵌入向量归一化为随机向量，然后将扰动应用于归一化的嵌入向量。 实验表明，归一化大大改善了微调模型的性能。

### DeBERTa v2

2021年2月微软放出的 V2 版本在 V1 版本的基础上又做了一些改进：

**1.词表：**在 v2 中，tokenizer扩的更大，从V1中的50K，变为 128K 的新词汇表。

2.nGiE(nGram Induced Input Encoding) v2 模型在第一个转换器层之外使用了一个额外的卷积层，以更好地学习输入标记的依赖性。

**3.共享位置和内容的变换矩阵：**通过实验表明，这种方法可以在不影响性能的情况下保存参数。

**4.应用桶方法对相对位置进行编码：**v2 模型使用对数桶对相对位置进行编码。

![35e80e9d047c1c16d87c1a6aeed86e6d](D:\typora图片\35e80e9d047c1c16d87c1a6aeed86e6d.jpeg)

优化结果：2.0版几个变更对模型的影响，增大词典效果最显著。

### DeBERTa v3

2021年11月微软又放出了 V3 版本。这次的版本在模型层面并没有修改，而是将预训练任务由掩码语言模型（MLM）换成了ELECTRA一样类似GAN的RTD (Replaced token detect) 任务。

我们知道BERT模型只使用了编码器层和MLM进行训练。而ELECTRA 使用 GAN 的思想，利用生成对抗网络构造两个编码器层进行对抗训练。其中一个是基于MLM训练的生成模型，另一个是基于二分类训练的判别模型。生成模型用于生成不确定的结果同时替换输入序列中的掩码标记，然后将修改后的输入序列送到判别模型。判别模型需要判断对应的 token 是原始 token 还是被生成器替换的 token。

不同的训练方法实验尝试：

1、[ES]生成模型和判别模型共享embedding层训练；
2、[NES]生成模型和判别模型不共享embedding层，也不共享参数层，交替训练；
3、[GDES]生成模型和判别模型部分共享。
通过实验表明：

![image-20231021121430371](D:\typora图片\image-20231021121430371.png)

DeBERTa V3 在某些任务中相比之前模型有不小的涨幅，其中GDES模式优化效果最好。

### 总结

1.DeBERTa V1 相比 BERT 和 RoBERTa 模型的改进：
两种技术改进：
注意力解耦机制
增强的掩码解码器
新的微调方法：虚拟对抗训练方法(SiFT)。
2.DeBERTa V2 改进
tokenizer扩的更大（优化效果最明显）
transformer外使用了额外的卷积层
共享位置和内容的变换矩阵
应用桶方法对相对位置进行编码
3.DeBERTa V3 改进
在模型层面并没有修改，将预训练任务MLM换成了ELECTRA一样类似GAN的RTD任务



## GPT

使用 wikipedia 和 GPT3.5 生成数据

- 比赛的目标是回答由大型语言模型（LLM）编写的基于科学的难题。

- 该比赛旨在帮助研究人员了解 LLM 测试自身的能力，并探索 LLM 可以在资源受限的环境中运行的潜力。

- 大型语言模型能力的范围正在扩大，研究人员正在使用 LLM 来描述他们自己。

- 对于最先进的模型来说，许多现有的自然语言处理基准（benchmark）已经变得微不足道，因此需要创建更具挑战性的任务来测试日益强大的模型。

- 竞赛的数据集是通过向 gpt3.5 模型提供关于各种科学主题的文本片段，并要求它编写多项选择题（带有已知答案）来生成的。简单的问题被过滤掉了。

- 手语输入的人工智能识别远远落后于语音输入，甚至手势输入，因为健壮的数据集以前不存在。

- 目前在 Kaggle 运行的最大型模型有大约100亿个参数，而 gpt3.5 有1750亿个参数。

- 这个比赛旨在探索一个比 gpt3.5 小10倍以上的问答模式是否能有效地回答 gpt3.5写的问题。研究结果将有助于了解 LLM 的基准测试和自我测试能力。

**引入依赖项**：openai、wikipediaapi

```python
import openai
import wikipediaapi
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
openai.api_key = user_secrets.get_secret("openai_api")
```

**数据收集**

1. 列出科学、技术、工程、数学的主题
2. 随机选择一个类别或页面
3. 从选定的页面提取文本
4. 向 LLM 模型编写消息
5. 合并管道的所有元素

**数据收集增强**











## 文本分类实例

### 1. 基于机器学习

实操主要包括以下几个任务：

1. 基于文本统计特征的特征提取（包括词频特征、TF-IDF特征等）
2. 如何划分训练集（用于参数选择、交叉验证）
3. 结合提取的不同特征和不同模型（线性模型、集成学习模型）完成训练和预测

实验步骤：

1. 数据加载及预处理

2. 数据集划分

   一般策略是：

   (1) **在选择模型、确定模型参数时**：

   先把当前竞赛给的**训练集划分为三个部分：训练集、验证集、测试集**。其中，训练集用于训练，验证集用于调参，测试集用于评估线下和线上的模型效果。

   特别的，为了进行交叉验证，划分后的训练集留10%用于测试；剩下90%输入交叉验证。

   注意：测试集最好固定下来，可以先打乱然后记录下索引值，之后直接取用。

   有一些模型的参数需要选择，这些参数会在一定程度上影响模型的精度，那么如何选择这些参数呢？

   - 通过阅读文档，要弄清楚这些参数的大致含义，哪些参数会增加模型的复杂度
   - 通过在验证集上进行验证模型精度，找到模型是否过拟合还是欠拟合

   (2) **在确定最佳模型参数（best parameters）后**：

   可以把原始的完整训练集全部扔给设置为best parameters的模型进行训练，得到最终的模型（final model），然后利用final model对真正的测试集进行预测。

   **注意：由于训练集中，不同类别的样本个数不同，在进行数据集划分时可以考虑使用分层抽样，根据不同类别的样本占比进行抽样，以保证标签的分布与整个数据集的分布一致。**

3. 特征提取

   词频特征、TF-IDF特征、......

4. 模型训练和预测

   词频特征+线性模型、TFIDF 特征+线性模型、TFIDF 特征+Adaboost、TFIDF 特征+XGBoost、TFIDF 特征+随机森林、TFIDF+LightGBM、交叉验证模型效果、交叉验证+网格搜索选择参数、TFIDF+随机森林+交叉验证+网格搜索、TFIDF+LightGBM+交叉验证+网格搜索、......

   **评价指标**：precision_score、recall_score、f1_score、accuracy_score、roc_auc_score、......

   

### 2. 基于深度学习

步骤和前面机器学习文本分类一样，词向量表示方法改为基于深度学习的FastText、Word2Vec等。

#### 2.1 Word2Vec+TextCNN+BiLSTM+Attention分类

模型结构如下图所示，主要包括WordCNNEncoder、SentEncoder、SentAttention和FC模块。

![Word2Vec+TextCNN+BiLSTM+Attention模型框架](D:\typora图片\Word2Vec+TextCNN+BiLSTM+Attention模型框架.png)

最终需要做的是文档分类任务，从文档的角度出发，文档由多个句子序列组成，而句子序列由多个词组成，因此我们可以考虑**从词的embedding --> 获取句子的embedding --> 再获得文档的embedding --> 最后根据文档的embedding对文档分类**。

CNN的卷积核在文本数据上，卷积核的宽度和word embedding的维度相同。

**WordCNNEncoder**包括三个不同卷积核大小的CNN层和相应的三个max pooling层，用于对一个句子卷积，然后max pooling得到一个句子的embedding。

**SentEncoder**包括多个BiLSTM层，将一篇文档中的句子序列作为输入，得到一篇文档中各个句子的embedding。

**Attention**中输入的一篇文档中各个句子的embedding首先经过线性变化得到`key`，`query`是可学习的参数矩阵，`value`和`key`相同，得到每个句子embedding重要性加权的一篇文档的embedding。

每个batch由多个文档组成；文档由多个句子序列组成；句子序列由多个词组成。所以输入整体模型的batch形状为：batch_size, max_doc_len, max_sent_len；

- 输入WordCNNEncoder的batch形状为：batch_size * max_doc_len, max_sent_len（只输入词的id）；

1. 利用word2vec embedding（固定）和随机初始化的权重（需要被训练）构建word embedding（二者相加）：batch_size * max_doc_len, 1(添加的一个channel维度，方便做卷积), max_sent_len, word_embed_size；
2. 分别经过卷积核为2、3、4的三个CNN层：batch_size * max_doc_len，sentence_len,  hidden_size；
3. 再分别经过三个相应的max pooling层：batch_size * max_doc_len，1,  hidden_size；
4. 拼接三个max pooling层的输出：batch_size * max_doc_len，1,  3 * hidden_size(sent_rep_size)；
5. 输出：batch_size * max_doc_len,  sent_rep_size；

- 输入SentEncoder的batch形状为：batch_size,  max_doc_len，sent_rep_size；
- 输入Attention的batch形状为：batch_size,  max_doc_len，2 * hidden_size of lstm；
- 输入FC的batch形状为：batch_size， 2 * hidden。

#### 2.2 基于Bert预训练和微调进行文本分类

##### （1）利用Huggingface的transformer包进行预训练

主要包括以下几个步骤：

1. 用数据集训练Tokenizer；
2. 加载数据及数据预处理；
3. 设定预训练模型参数，初始化预训练模型；
4. 设定训练参数，加载训练器；
5. 训练并保存模型。

##### （2）微调模型进行分类

主要包括以下几个步骤：

1. 训练集划分；
2. 数据预处理；
3. 加载预训练模型、设置微调参数；
4. 微调训练下游任务模型并保存。






