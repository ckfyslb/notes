# 一、Transformer

### 1. 工作流程

1. 获取句子中每个词的向量表示（**Embedding** + **位置编码**）。
2. 词向量矩阵（单词数n行 * 词向量维度d列）输入Encoder（Encoder_block * 6）第一个block，后面每个block的输入为前一个block的输出，最后的输出就是句子所有单词的编码信息矩阵（其中每个block输出矩阵维度与输入一致）。
3. Decoder（Decoder_block * 6）的第一个block输入为Encoder输出的矩阵 和 Decoder前面的预测输出（后面还没预测的mask掉，最开始没有前一个输出的时候输入开始符），后面的每个block输入为Encoder的输出和前一个block的输出，最后的输出经过全连接和softmax得到最终输出。

### 2. 结构

**Encoder_block结构：**就是一个多头注意力，它的输出再做一个残差（注意力的输出和输入相加，防止梯度消失）和标准化（Layer Normalization）之后输入一个前馈网络（全连接），前馈网络的输出再做一个残差和标准化，就得到了block的输出。

**Decoder_block结构：**比Encoder_block多了一个多头注意力，第一个多头注意力（masked）的输入的是之前的输出，然后它的输出做残差和标准化之后再输入下一个多头注意力用于计算q和后面的残差，Encoder的输出输入这个多头注意力计算k和v，然后输出之后做残差和标准化，再输入前馈网络，输出做残差和标准化，就得到了block的输出。

### 3. Warm-up

学习率往往设置为**常数**或者**逐渐衰减** (decay)，都不能让Transformer很好地收敛。

在优化Transformer结构时，除了设置初始学习率与它的衰减策略，往往还需要在训练的初始阶段设置一个非常小（接近0）的学习率，让它经过一定的迭代轮数后逐渐增长到初始的学习率，这个过程称作**warm-up阶段（学习率预热）**。

Warm-up是原始Transformer结构优化时的一个必备学习率调整策略。Transformer结构对于warm-up的超参数（**持续轮数、增长方式、初始学习率**等）非常敏感，若调整不慎，往往会使得模型无法正常收敛。

Transformer结构的优化非常困难，其具体表现在：

- - **warm-up阶段超参数敏感；**
  - **优化过程收敛速度慢。**

### 4. 为什么要position embedding？

self-attention无法表达位置信息（对位置信息不敏感），如果不添加位置编码，那么无论单词在什么位置，它的注意力分数都是确定的。

为了理解单词顺序，Transformer为每个输入的词嵌入添加了一个向量，这样能够更好的表达词与词之间的关系。词嵌入与位置编码相加，而不是拼接，他们的效率差不多，但是拼接的话维度会变大，所以不考虑。
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d})
\\
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d})
$$

> 其中，pos表示单词在句子中的位置，d表示PE的维度（与词嵌入维度一样），2i和2i+1分别表示向量的偶数维度和奇数维度。

### 5. 与 RNN/CNN 的比较

> Transformer 为什么比 RNN/CNN 更好用？优势在哪里？  

#### （1）RNN

- 特点/优势（Transformer之前）：

  - 适合解决线性序列问题；天然能够捕获位置信息（相对+绝对）；

    > 绝对位置：每个 token 都是在固定时间步加入编码；相对位置：token 与 token 之间间隔的时间步也是固定的；

  - 支持不定长输入；

  - LSTM/Attention 的引入，加强了长距离语义建模的能力；

- 劣势：

  - 串行结构难以支持并行计算；

  - 依然存在长距离依赖问题；

  - 单向语义建模（Bi-RNN 是两个单向拼接）

#### （2）CNN

- 特点/优势：
  - 捕获 n-gram 片段信息（局部建模）；
  - 滑动窗口捕获相对位置特征（但 Pooling 层会丢失位置特征）；
  - 并行度高（滑动窗口并行、卷积核并行），计算速度快；
- 劣势：
  - 长程建模能力弱：受感受野限制，无法捕获长距离依赖，需要空洞卷积或加深层数等策略来弥补；
  - Pooling 层会丢失位置信息（目前常见的作法会放弃 Pooling）；
  - 相对位置敏感，绝对位置不敏感（平移不变性）

#### （3）Transformer

- 特点/优势：
  - 通过位置编码建模相对位置和绝对位置特征；
  - Self-Attention 同时编码双向语义和解决长距离依赖问题；
  - 支持并行计算；
- 缺点/劣势：
  - 不支持不定长输入（通过 padding 填充到定长）；
  - 计算复杂度高；

#### （4）Transformer 能完全取代 RNN 吗？（不行）

从三个方面进行比较：

1. **上下文语义特征**

在抽取上下文语义特征（方向+距离）方面：**Transformer > RNN > CNN**

- RNN 只能进行单向编码（Bi-RNN 是两个单向）；在**长距离**特征抽取上也弱于 Transformer；

- CNN 只能对短句编码（N-gram）；

- Transformer 可以同时**编码双向语义**和**抽取长距离特征**；

2. **序列特征**

在抽取序列特征方面：**RNN > Transformer > CNN**

- Transformer 的序列特征完全依赖于位置编码，当序列长度没有超过 RNN 的处理极限时，位置编码对时序性的建模能力是不及 RNN 的

3. **计算速度**

在计算速度方面：**CNN > Transformer > RNN**

- RNN 因为存在时序依赖难以并行计算；
- Transformer 和 CNN 都可以并行计算，但 Transformer 的计算复杂度更高

### 6. transformer和Bert中位置编码的区别

**为什么要对位置进行编码？**
Attention提取特征的时候，可以获取全局每个词对之间的关系，但是并没有显式保留时序信息，或者说位置信息。就算打乱序列中token的顺序，最后所得到的Attention结果也不会变，这会丢失语言中的时序信息，因此需要额外对位置进行编码以引入时序信息。

- Transformer的位置编码是一个固定值，只能标记位置，不能标记这个位置有什么用。

  在Transformer中，位置编码是由sin/cos函数生成的固定值。

  具体做法：**用不同频率的正余弦函数对位置信息进行编码**，位置编码向量的维度与文本编码向量的维度相同 ，即$d_{model}$。因此二者可以直接相加作为token最终的编码向量。

  $$PE_(pos,2i)=sin(pos/10000^{2i/d_{model}})$$

  $$PE_(pos,2i+1)=cos(pos/10000^{2i/d_{model}})$$
  其中，$pos$表示位置，$i$表示所在维度。

- BERT的位置编码是可学习的Embedding，不仅可以标记位置，还可以学习到这个位置有什么用。

  在BERT中，与一般的词嵌入编码类似，位置编码也是**随机生成且可训练的**，维度为`[seq_length, width]`，其中seq_length代表序列长度，width代表每一个token对应的向量长度。

# 二、Transformer各模块的作用

### 1. QKV Projection

#### 为什么在 Attention 前要对 Q/K/V 做一次投影？

- 首先在 Transformer-Encoder 中，Q/K/V 是相同的输入；
- 加入这个全连接的目的就是为了将 Q/K/V 投影到不同的空间中，增加多样性；
- 如果没有这个投影，在之后的 Attention 中相当于让相同的 Q 和 K 做点积，那么 attention 矩阵中的分数将集中在对角线上，即每个词的注意力都在自己身上；这与 Attention 的初衷相悖——**让每个词去融合上下文语义**；

### 2. Self-Attention

#### （1）原理

1. 对任一个词，它的 query 和 所有的每个词的 key 进行相似度计算，得到权值，除以$\sqrt{d_k}$做 scale
2. 将权值进行归一化，即矩阵$QK^T$的每一行归一化
3. 将权重和 value 进行加权求和得到包含全局信息的词向量

$$
𝑆𝑜𝑓𝑡𝑚𝑎𝑥(\frac{𝑄𝐾^𝑇}{\sqrt{𝑑_𝑘}})𝑉
$$

```python
import torch
import torch.nn as nn
class Self_Attention(nn.Module):
    def __init__(self, embed_dim, dk, dv):
        super(Self_Attention, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(dim, dk)
        self.k = nn.Linear(dim, dk)
        self.v = nn.Linear(dim, dv)
	def forward(self, x):
        q = self.q(x)
        k = self.q(k)
        v = self.q(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        return x
att = Self_Attention(embed_dim=2, dk=2,dv=3)
x = torch.rand((1, 4, 2))  # (batchsize, token_num, embed_dim)
output = att(x)
```

#### （2）为什么使用多头

多头和 CNN 中多通道的思想类似，目的是期望不同的注意力头能学到不同的特征。不同的头会关注不同的信息。

通过增加参数量来增强网络的容量从而提升网络表达能力。

经过多头之后，还需要线性层来做线性变换，以自动决定（通过训练）对每个头的输出赋予多大的权重，从而在最终的输出中强调一些头的信息，而忽视其他头的信息。这是一种自适应的、数据驱动的方式来组合不同头的信息。

#### （3）为什么 Transformer 中使用的是乘性 Attention（点积），而不是加性 Attention？

- 在 GPU 场景下，矩阵乘法的效率更高（原作说法）；

- **在不进行 Scaled 的前提下**，随着 d（每个头的特征维度）的增大，乘性 Attention 的效果减弱，加性 Attention 的效果更好；

#### （4）为什么要Scaled

- **目的**：防止梯度消失；

- **解释**：在 Attention 模块中，注意力权重通过 Softmax 转换为概率分布；但是 Softmax 对输入比较敏感，当输入的方差越大，它计算出的概率分布就会大部分概率集中到少数几个分量位置。极端情况下，其概率分布将退化成一个 One-Hot 向量；其结果就是雅可比矩阵（偏导矩阵）中绝大部分位置的值趋于 0，即梯度消失；通过缩放操作可以使注意力权重的方差重新调整为 1，从而缓解梯度消失的问题；

  （假设$Q$和$K$各分量独立同分布，且均服从均值为0，方差为1的正态分布，未经缩放的注意力权重矩阵$QK^T$当中的每个分量将服从均值为0，方差为d的正态分布，d是每个向量的维度，当d越大的时候，就会出现前面说的梯度消失的问题，那么这个时候把注意力权重矩阵除以 $\sqrt{d}$ 就会使得里面的分量重新服从标准正态分布，从而缓解梯度消失的问题）

#### （5）在 Softmax 之前加上 Mask 的作用是什么？

> 相关问题：为什么将被 mask 的位置是加上一个极小值（-1e9），而不是置为 0？

使无意义的 token 在 softmax 后得到的概率值（注意力）尽量接近于 0；从而使正常 token 位置的概率和接近 1

### 3. Add & Norm

#### （1）加入残差的作用

- 在求导时加入一个恒等项，以**减少梯度消失问题**；

#### （2）加入 LayerNorm 的作用

- 提升网络的泛化性；
- 加在激活函数之前，避免激活值落入饱和区，减少梯度消失问题；

任何norm的意义都是为了让使用norm的网络的输入的数据分布变得更好，也就是转换为标准正态分布，数值进入敏感度区间，以减缓梯度消失，从而更容易训练。

#### （3）为什么不用BN？

如果在不同样本的相同维度间进行normalization，那么就舍弃了同一个样本它不同维度的信息，这对于自然语言来说是没有意义的

![img](https://pic1.zhimg.com/80/v2-3342429b035696b3e732aeeab49fe394_720w.webp)

对比上面的图片，LN相当于要做3次normalization

#### （4）Pre-LN 和 Post-LN 的区别

![img](https://pic1.zhimg.com/v2-d5c994ac883ec5bf58580a6664714c7c_r.jpg)

- Post-LN（BERT 实现）：
  $$x_{n+1} = \text{LN}(x_n + f(x_n))$$

  - 先做完残差连接，再归一化；
  - 优点：保持主干网络的方程比较稳定，使模型泛化能力更强，性能更好；
  - 缺点：把恒等路径放在 norm 里，使模型收敛更难（反向传播时梯度变小，残差的作用被减弱）

- Pre-LN：
  $$x_{n+1} = x_n + f(\text{LN}(x_n))$$

  - 先归一化，再做残差连接；
  - 优点：**加速收敛**
  - 缺点：效果减弱

  当使用Pre-LN结构时，warm-up阶段已不再是必需，并且Pre-LN结构可以大幅提升Transformer的收敛速度

### 4. Feed-Forward Network

- 前向公式
  $$W_2 \cdot \text{ReLU}(W_1x + b_1) + b_2$$

#### （1）FFN 层的作用是什么？

- 功能与 1*1 卷积类似：
  1）跨通道的特征融合/信息交互；
  2）**通过激活函数增加非线性**；

- **之前操作都是线性的**：
  1）Projection 层并没有加入激活函数；
  2）Attention 层只是线性加权；

#### （2）FFN 中激活函数的选择

> 相关问题：BERT 为什么要把 FFN 中的 ReLU 替换为 GeLU？

- 背景：原始 Transformer 中使用的是 **ReLU**；BERT 中使用的是 **GeLU**；
- GeLU 在激活函数中引入了正则的思想，越小的值越容易被丢弃；相当于综合了 ReLU 和 Dropout 的功能；而 ReLU 缺乏这个随机性；
- 为什么不使用 sigmoid 或 tanh？——这两个函数存在饱和区，会使导数趋向于 0，带来梯度消失的问题；不利于深层网络的训练；

# 三、Bert知识点

### 1. Bert基本原理

Bert由Transformer的Encoder叠加而成，base有12层，large有24层。

输入长度为512的序列，其中每个token维度为768。

通过两大任务来预训练：

- MLM（Masked Language Model）
  就是在输入一句话的时候，随机地选一些要预测的词，然后用一个特殊的符号[MASK]来代替它们，之后让模型根据上下文去学习这些地方该填的词。

  BERT中有15%的token会被随机掩盖，这15%的token中80%用[MASK]这个token来代替，10%用随机的一个词来替换，10%保持这个词不变。

  最后只计算[MASK]部分的损失。

- NSP（Next Sentence Prediction）
  预测输入 BERT 的两段文本是否为连续的文本。

  语料中50%的句子，选择其相应的下一句一起形成上下句，作为正样本；其余50%的句子随机选择一句非下一句一起形成上下句，作为负样本。

BERT 相较于原来的 RNN、LSTM 可以做到并发执行，同时提取词在句子中的关系特征，并且能在多个不同层次提取关系特征，进而更全面反映句子语义。相较于 word2vec，能根据句子上下文获取词义，从而避免歧义出现。同时缺点也是显而易见的，模型参数太多，而且模型太大，少量数据训练时，容易过拟合。

### 2. Bert为什么三个embedding可以相加？

token embedding：词的向量表示

segment embedding：向量表示，区分成对的输入序列第一个句子的部分在向量中为0，第二个句子的部分在句子中为1

position embedding：位置编码

加法不会导致”信息损失“，本质上神经网络中每个神经元收到的信号也是“权重”相加得来。

### 3. 为什么Bert中要用WordPiece/BPE这样的subword Token？

避免OOV（Out Of Vocabulary），也就是词汇表外的词。在NLP中，通常会预先构建一个词汇表，包含所有模型能够识别的词。然而，总会有一些词没有出现在预先构建的词汇表中，这些词就是 OOV。

传统的处理方式往往是将这些 OOV 映射到一个特殊的符号，如 `<UNK>`，但这种方式无法充分利用 OOV 中的信息。例如，对于词汇表中没有的词 "unhappiness"，如果直接映射为 `<UNK>`，则模型就无法理解它的含义。

WordPiece/Byte Pair Encoding (BPE) 等基于子词的分词方法提供了一种解决 OOV 问题的方式。现在更多的语言大模型选择基于BPE的方式，只不过BERT时代更多还是WordPiece。**BPE 通过将词分解为更小的单元（子词或字符），可以有效地处理词汇表外的词。**对于上面的 "unhappiness" 例子，即使 "unhappiness" 本身不在词汇表中，但是它可以被分解为 "un"、"happiness" 等子词，而这些子词可能在词汇表中。这样，模型就可以通过这些子词来理解 "unhappiness" 的含义。

另一方面就是，BPE本身的语义粒度也很合适，一个token不会太大，也不会小到损失连接信息（如一个字母）。

### Bert中为什么要在开头加个[CLS]?

具体来说，我们想让[CLS]做的事情就是利用好BERT强大的表示能力，这个表示能力不仅限于token层面，而且我们尝试要得到整个seqence的表示。因此，[CLS]就是做这个事情的。具体来说，整个encoder的最后一层的[CLS]学到的向量可以很好地作为整句话的语义表示，从而适配一些setence层面的任务，如整句话的情感分类。

那关键点就在于，为什么[CLS]可以建模整句话的语义表征呢？简单来说也很好理解，因为“这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义。”

——为什么无明显语义？因为训练的时候BERT发现每个句子头都有，这样他能学到什么语义呢？

——为什么要公平？因为控制变量，我们不希望做其他下游任务的时候用于区分不同样本间特征的信息是有偏的。

当然，不放在句子开头的其他位置是否可行？一个未经考证的臆想是，任何其他位置的position embedding都无法满足放在开头的一致性。所以不同句子间可能会有一定的不同，这并不是我们做一些句间的分类问题想要的。

### 不用[CLS]的语义输出，有其他方式可以代替吗？

这个问题还是考察到了[CLS]的核心内涵，也就是如何获得整个sentence的语义表示。既然不让使用特意训好的[CLS]，那我们就从每个token得到的embedding入手，把所有的token弄到一起。

很直观的思路，就是对BERT的所有输出词向量（忽略[CLS]和[SEP]）应用MaxPooling和AvgPooling，然后将得到的两个向量拼接起来，作为整个序列的表示。这样做的话可以同时保留序列中最显著的特征（通过MaxPooling）和整体的，均衡的特征（通过AvgPooling）。

当然这种做法我本人并没有尝试过，或许也不是一个很好做的研究/工作方向。

### Bert中有哪些地方用到了mask?

预训练任务Masked Language Model (MLM)

self-attention的计算

下游任务的decoder

### 预训练阶段的mask有什么用？

虽然MLM现在被主流LLM抛弃了，但是也是一项很重要的任务。

主要的思想是，把输入的其中一部分词汇随机掩盖，模型的目标是预测这些掩盖词汇。这种训练方式使得每个位置的BERT都能学习到其上下文的信息。

### attention中的mask有什么用？（BERT中）

这是nlp任务很重要的问题，就是不同样本的seq_len不一样。但是由于输出的seq_len需要一致，所以需要通过补padding来对齐。而在attention中我们不希望一个token去注意到这些padding的部分，因为实际场景下它们是不存在的，所以attention中的mask就是来处理掉这些无效的信息的。

具体来说就是在softmax前每个都设为-inf（或者实际的场景一个很小的数就可以），然后过完softmax后"padding"部分的权重就会接近于零，query token就不会分配注意力权重了。

### decoder中的mask有什么用？

![img](https://pic4.zhimg.com/80/v2-c6a5ef50451428a50de3ac3e92a2ccbf_720w.webp)

这就是decoder-only的模型（自回归模型）架构问题，一般称为future mask，通常为一个上三角矩阵。

简单来说，就是模拟inference的过程。比如，在做next token prediction的时候，模型是根据前面已有的tokens来做的，也就是看不到未来的tokens的信息。而在我们训练的过程中，通常采用teacher forcing的策略，也就是我们当然会把完整的标签喂给模型，但是由于在一个一个生成next token的过程中，模型应该是一个一个往外“蹦“字的过程（想想chatgpt回复你的样子）所以我们会遮盖掉seqence中当前位置之后信息，以防止模型利用未来信息，也就是信息泄露。mask掉后模型的注意力只会集中在此前的序列上。

### Bert中self attention 计算复杂度如何？

𝑂(𝑑𝐿2) ，你可以参考上面那张图，因为输入的序列的每一个token都要对这个序列上的所有token去求一个attention score。

### 有什么技术降低复杂度提升输入长度的？

比如Sparse Attention，放弃对全文的关注，只关心局部的语义组合，相当于self-attention上又加了一些mask，这样的话就可以降低复杂度，而且下游任务的语义关联性的体现往往是局部/稀疏的。

### Bert是如何处理传统方法难以搞定的溢出词表词(oov)的语义学习的？

前面提到了，关键词是subword。

### 中文是如何处理溢出词表词(oov)的语义学习的？

subword处理中文都是字符级别的，所以就不会有词级别oov的问题了。

### 为什么以前char level/subword level的NLP模型表现一般都比较差，但是到了bert这里就比较好？

还是归功于Transformers，因为对于字符级别（char-level）或者子词级别（subword-level）的NLP模型，挑战在于需要模型能够理解字符或者子词组合起来形成词语和句子的语义，这对模型的能力有很高的要求。

然而，以前NLP模型没办法做到很深，两层lstm基本就到极限了，非线性成长路线过分陡峭，所以增加网络容量的时候，降低了泛化能力。

Bert降低了输入的复杂度，提升了模型的复杂度。模型多层产生的非线性增长平滑，可以加大网络容量，同时增强容量和泛化能力。

解释一下：

——什么叫非线性成长路线过分陡峭？

如果我们将模型的深度看作是 x 轴，模型的复杂度或训练难度看作是 y 轴，那么随着深度的增加，y 值的增长可能会变得非常快。

——BERT为什么降低了输入复杂度？

WordPiece这种subword的做法不至于像char level那样基本完全抛弃了自身的语义信息（因为切的太细就会太复杂），也不至于像word leve那样，因此可以减小词汇表大小。当然也避免了OOV的问题。

——BERT为什么提升了模型复杂度？

Transformers可以把网络做深，本身内部网络容量也很够。

### Bert为什么要使用warmup的学习率trick

主要是考虑到训练的初始阶段params更新比较大，可能会使模型陷入local minima或者overfitting。

warmup就是把lr从一个较小的值线性增大到预设，以减缓参数震荡，让训练变得比较smooth，当模型参数量上来后这种策略变得更重要了。

### 为什么说GPT是单向的Bert是双向的？

这也是decoder-only和encoder-only的区别。

decoder-only架构的生成模型在输出的时候只能看到当前位置前的tokens，也就是屏蔽了序列后面的位置，以适配NTP任务。

encoder-only架构的编码模型在输出的时候可以利用前后位置的tokens，以适配MLM任务。

具体的做法是self-attention加不加casual mask，也就是遮不遮住序列后面的内容。

### Bert如何处理一词多义？

一词多义指的是在不同句子中token有不同的含义。

这正是self-attention解决的，搭配上MLM的任务，就可以让每个token会注意到上下文的其他token来得到自己的embedding。

### Bert中的transformer和原生的transformer有什么区别？

其实很多，如果我们只讨论模型架构，也就是对比[Attention is All You Need](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)的encoder和BERT的话，最重点的区别在于位置编码。

原生的transformer是最经典的Sinusoidal绝对位置编码。

而BERT中变成了可以学习的参数，也就是可学习位置编码。

变得可学了的话，只要模型学习能力强，数据量够，确实不会差。可以类比卷积核从手工变成了模型自己学。

关于位置编码，如果你有时间的话，建议从下面的链接一直往后看，苏神的内容质量都很高。位置编码确实大有可为，最近RoPE+NTK的方法来外推context length也挺让人热血沸腾的。

### Albert是通过什么方法压缩网络参数的？有什么问题？

两个技巧，其一是参跨层数共享，其二是对嵌入参数化进行因式分解，也就是“不再将 one-hot 向量直接映射到大小为 H 的隐藏空间，先映射到一个低维词嵌入空间 E，然后再映射到隐藏空间”。

问题也是“模型压缩”通用的问题，网络表达能力和容量下降。然后推理速度也不会有很直观的提升。

### attention计算方式以及参数量，attention layer手写，必考。

如果你找的工作是比较基础的，比如说本科生找llm相关实习，那基本会让你手写多头。

如果你想比较方便地一站对比各个Transformer模型的源码，可以来这个库：[GitHub - OpenBMB/ModelCenter](https://link.zhihu.com/?target=https%3A//github.com/OpenBMB/ModelCenter)

这里我就来做ModelCenter的Attention源码解析吧，都以注释的形式放在里面，其实源码本身写的很全面了。

```python
class Attention(bmt.DistributedModule): #这里bmt.DistributedModule你不用在意，简单理解为torch.nn.Module即可

    """ Attention module consisting procedure of Q, K, V combination and its output projection. 
    For more detail, see `Attention is All you Need <https://arxiv.org/abs/1706.03762>`_.

    Args:
        dim_in (int): input dimension.
        dim_head (int): dimension of each heads used in attention.
        num_heads (int): number of heads used in attention.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attetion module. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in attention module. Defaults to 0.02.
        bias (bool, optional): whether to use bias term in fully-connected layers used in attention module. Defaults to False.
        mask_value (float, optional): mask value of the masked position. Defaults to `-inf`.
        pos_bias_type (str, optional): `relative` for relative position bias, `rotary` for ratery position embedding. Defaults to `none`.
        attn_scale (bool, optional): whether to scale before softmax, i.e., :math:`\text{softmax}({Q K^T \over \sqrt{\text{dim_model}}})`. Default to False.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, dim_in : int, 
                       dim_head : int,
                       num_heads : int, 
                       dim_out : int = None,
                       dtype = torch.half,
                       int8 = False, 
                       init_mean = 0.0, 
                       init_std = 0.02,
                       bias = False,
                       mask_value : float = float("-inf"),
                       pos_bias_type : str = "none",
                       length_scale : bool = False,
                       attn_scale : bool = False,
                       dropout_p : float= 0,
                       shared_key_and_value = False,
        ): 

        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        num_heads_kv = 1 if shared_key_and_value else num_heads #这里可以选择Multi-Query Attention(MQA)，MHA/MQA/GQA的对比可以看https://zhuanlan.zhihu.com/p/644698212

        #下面是四个最重要的线性层project_q,project_k,project_v,attention_out
        #注意这里矩阵的输出维度，有高并行的优点。除了输入输出外其他的一些参数继承自线性层的实现即可。
        self.project_q = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        #project_k和project_v的头数是num_heads_kv
        self.project_k = Linear(
            dim_in = dim_in,
            dim_out = num_heads_kv * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_v = Linear(
            dim_in = dim_in,
            dim_out = num_heads_kv * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        #将多头转换回输出维度
        self.attention_out = Linear(
            dim_in = num_heads * dim_head,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
        
        self.init_mean = init_mean
        self.init_std = init_std
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.int8 = int8
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.mask_value = mask_value
        self.dtype = dtype
        self.dropout_p = dropout_p
        self.shared_key_and_value = shared_key_and_value

        if dropout_p:
            self.attention_dropout = torch.nn.Dropout(dropout_p)
        else:
            self.attention_dropout = None
        
        self.bias = bias
        self.pos_bias_type = pos_bias_type
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query : torch.Tensor,
                      key_value : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : Optional[torch.Tensor] = None,
                      use_cache: bool = False,
                      past_key_value = None,
        ):

        """ This model inherits from bmt.DistributedModule. 

        Args:
            query (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            key_value (:obj:`torch.Tensor` of shape ``(batch, len_k, dim_model)``): Length of input sequence before padding.  
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, len_q, len_k)``): Used to avoid performing attention on padding token indices.
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, len_q, len_k)`` or ``(1, num_heads, len_k, len_q)``): Provide positional information about tensor `key_value` and `query`. 

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, len_q, dim_model)``): The attention output.
        """

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = self.project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = self.project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = self.project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        #拆头
        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)

        # if self.shared_key_and_value:
        #     h_k = h_k.repeat(1, self.num_heads, 1, 1)
        #     h_v = h_v.repeat(1, self.num_heads, 1, 1)

        h_q = h_q.contiguous()      # (batch, num_heads, len_q, dim_head)
        h_k = h_k.contiguous()      # (batch, num_heads, len_k, dim_head)
        h_v = h_v.contiguous()      # (batch, num_heads, len_k, dim_head)

        #自回归常用的优化trick，decoding到第t步的时候，前t-1步已经计算了key和value，所以保存下来，避免重复计算。
        #encoding不需要，因为输入是固定的，一个step就可以，不需要recursive地生成。这是auto-regressive特有的trick
        if past_key_value is not None:
            h_k = torch.cat([past_key_value[0], h_k], dim=-2)
            h_v = torch.cat([past_key_value[1], h_v], dim=-2)
            len_k = h_k.size(-2)

        current_key_value = (h_k, h_v) if use_cache else None
        
        #如果模型采用RoPE位置编码的话，在这里要为h_q, h_k赋予位置信息
        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # (batch, num_heads, len_q, dim_head) @ (batch, num_heads_kv, len_k, dim_head)T 
        # => (batch, num_heads, len_q, len_k)

        #算Attn score
        score = torch.matmul(h_q, h_k.transpose(2, 3))
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        # (batch, num_heads, len_q, len_k) 
        # score = score.view(batch_size, self.num_heads, len_q, len_k)
        
        #其他相对位置编码直接加在Attn score上
        if self.pos_bias_type == "relative":
            if position_bias is not None:
                # (batch, num_heads, len_q, len_k) + (1, num_heads, len_q, len_k) 
                score = score + position_bias
        
        #对score填充mask，第二个参数矩阵中True表示要填充，attention_mask本身非0表示有效
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(self.mask_value, device=score.device, dtype=score.dtype)
        )   # (batch, num_heads, len_q, len_k)

        #过softmax
        score = self.softmax(score)

        # avoid nan in softmax，一些数值稳定相关的问题
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )
        #.view(batch_size * self.num_heads, len_q, len_k) # (batch * num_heads, len_q, len_k)

        #如果需要，加dropout
        if self.attention_dropout is not None:
            score = self.attention_dropout(score)

         # (batch * num_heads, len_q, len_k) @ (batch * num_heads, len_k, dim_head) = (batch * num_heads, len_q, dim_head)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, self.num_heads * self.dim_head) # (batch, len_q, num_heads * dim_head)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        score = self.attention_out(score)

        #还是decoding的时候是否使用past_key_value的策略。
        if use_cache:
            return score, current_key_value
        else:
            return score
```

当然，上面是实际使用版，这里提供一个简易面试专用版，不考虑那么多tricks，本科生找实习专用。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        # Define the dimension of each head or subspace
        self.d_k = d_model // self.num_heads

        # These are still of dimension d_model. They will be split into number of heads 
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        K_length = K.size(-2)
    
        # Scaling by d_k so that the soft(arg)max doesn't explode
        QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    
        # Apply the mask
        if mask is not None:
            QK = QK.masked_fill(mask.to(QK.dtype) == 0, float('-inf'))
    
        # Calculate the attention weights (softmax over the last dimension)
        weights = F.softmax(QK, dim=-1)
    
        # Apply the self attention to the values
        attention = torch.matmul(weights, V)
    
        return attention, weights


    def split_heads(self, x, batch_size):
        """
        The original tensor with dimension batch_size * seq_length * d_model is split into num_heads 
        so we now have batch_size * num_heads * seq_length * d_k
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # linear layers
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        # split into multiple heads
        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  

        # self attention
        scores, weights = self.scaled_dot_product_attention(q, k, v, mask)

        # concatenate heads
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)

        # final linear layer
        output = self.W_o(concat)

        return output, weights
```

在面试中还被问到过一点，出于运算速度的考虑，我们认为“一次大的矩阵乘法的执行速度实际上比多次较小的矩阵乘法更快”，因此你也可以：

```python
self.qkv = nn.Linear(d_model, 3 * d_model)

# 在forward方法中
qkv = self.qkv(x)  # (batch_size, seq_len, 3 * d_model)
q, k, v = torch.split(qkv, d_model, dim=-1)  # split into three tensors
```

### NLU以及NLG各种任务的差异

NLU（自然语言理解）& NLG（自然语言生成），“望文生义”，其实只是在任务目标上有区别，本人认为不用太重点区分。

![img](https://pic1.zhimg.com/80/v2-461248c25150f59c00428b71358996dc_720w.webp)

### tokenizer的细节，tokenizer的计算方式，各种tokenizer的优缺点。

tokenizer真又是一大块工作，而且也很值得优化，因为相当于模型的眼睛，它怎么去看待文字。

### 各种norm方式的优缺点。

我们常说的norm有以下四种：LN/BN/IN/GN，分别对应layer/batch/instance/group

NLP下很直观的一个图：

![img](https://pic4.zhimg.com/80/v2-420930342112f06a21923fe3e0f9a7ab_720w.webp)

具体来说：

- Batch Norm：把每个Batch中，每句话的相同位置的字向量看成一组做归一化。
- Layer Norm：在每一个句子中进行归一化。
- Instance Norm：每一个字的字向量的看成一组做归一化。
- Group Norm：把每句话的每几个字的字向量看成一组做归一化。

为了更方便理解上代码：

```python3
# 创建一个输入张量，其中有两个句子（即 batch_size=2），每个句子有4个词（即 seq_len=4），并且每个词的词向量维度是3（即 dim=3）。这样，我们的输入张量的形状就是 (2,4,3)
X = np.array([
    [[0.1, 0.2, 0.3],
     [1.1, 1.2, 1.3],
     [2.1, 2.2, 2.3],
     [3.1, 3.2, 3.3]],
    
    [[4.1, 4.2, 4.3],
     [5.1, 5.2, 5.3],
     [6.1, 6.2, 6.3],
     [7.1, 7.2, 7.3]]
])

#batch_norm ，会发现一句话里的每个token都一样了，虽然是例子有点特殊，但是也说明了batch_norm不适合NLP
array([[[-1.04912609, -0.99916771, -0.94920932],
        [-1.04912609, -0.99916771, -0.94920932],
        [-1.04912609, -0.99916771, -0.94920932],
        [-1.04912609, -0.99916771, -0.94920932]],

       [[ 0.94920932,  0.99916771,  1.04912609],
        [ 0.94920932,  0.99916771,  1.04912609],
        [ 0.94920932,  0.99916771,  1.04912609],
        [ 0.94920932,  0.99916771,  1.04912609]]])

#layer_norm 
 array([[[-1.42728248, -1.33807733, -1.24887217],
         [-0.53523093, -0.44602578, -0.35682062],
         [ 0.35682062,  0.44602578,  0.53523093],
         [ 1.24887217,  1.33807733,  1.42728248]],
 
        [[-1.42728248, -1.33807733, -1.24887217],
         [-0.53523093, -0.44602578, -0.35682062],
         [ 0.35682062,  0.44602578,  0.53523093],
         [ 1.24887217,  1.33807733,  1.42728248]]]),

#instance_norm（同样每个token都一样，完全忽略了上下文的关系）
 array([[[-1.22474487e+00,  0.00000000e+00,  1.22474487e+00],
         [-1.22474487e+00,  0.00000000e+00,  1.22474487e+00],
         [-1.22474487e+00,  0.00000000e+00,  1.22474487e+00],
         [-1.22474487e+00,  0.00000000e+00,  1.22474487e+00]],
 
        [[-1.22474487e+00,  0.00000000e+00,  1.22474487e+00],
         [-1.22474487e+00,  0.00000000e+00,  1.22474487e+00],
         [-1.22474487e+00,  0.00000000e+00,  1.22474487e+00],
         [-1.22474487e+00,  0.00000000e+00,  1.22474487e+00]]]),

#group_norm(如果n=2，也就是每两个token一起算)
 array([[[-1.18431305, -0.98692754, -0.78954203],
         [ 0.78954203,  0.98692754,  1.18431305],
         [-1.18431305, -0.98692754, -0.78954203],
         [ 0.78954203,  0.98692754,  1.18431305]],
 
        [[-1.18431305, -0.98692754, -0.78954203],
         [ 0.78954203,  0.98692754,  1.18431305],
         [-1.18431305, -0.98692754, -0.78954203],
         [ 0.78954203,  0.98692754,  1.18431305]]]))
```

其实只要仔细看上面的例子，就很容易能想到NLP中每一种norm的优缺点：

**Batch Normalization（Batch Norm）**：
**缺点**：在处理序列数据（如文本）时，Batch Norm可能不会表现得很好，因为序列数据通常长度不一，并且一次训练的Batch中的句子的长度可能会有很大的差异；此外，Batch Norm对于Batch大小也非常敏感。对于较小的Batch大小，Batch Norm可能会表现得不好，因为每个Batch的统计特性可能会有较大的波动。

**Layer Normalization（Layer Norm）**：
**优点**：Layer Norm是对每个样本进行归一化，因此它对Batch大小不敏感，这使得它在处理序列数据时表现得更好；另外，Layer Norm在处理不同长度的序列时也更为灵活。

**Instance Normalization（Instance Norm）**：
**优点**：Instance Norm是对每个样本的每个特征进行归一化，因此它可以捕捉到更多的细节信息。Instance Norm在某些任务，如风格迁移，中表现得很好，因为在这些任务中，细节信息很重要。
**缺点**：Instance Norm可能会过度强调细节信息，忽视了更宏观的信息。此外，Instance Norm的计算成本相比Batch Norm和Layer Norm更高。

**Group Normalization（Group Norm）**：
**优点**：Group Norm是Batch Norm和Instance Norm的折中方案，它在Batch的一个子集（即组）上进行归一化。这使得Group Norm既可以捕捉到Batch的统计特性，又可以捕捉到样本的细节信息。此外，Group Norm对Batch大小也不敏感。
**缺点**：Group Norm的性能取决于组的大小，需要通过实验来确定最优的组大小。此外，Group Norm的计算成本也比Batch Norm和Layer Norm更高。

### 比较

RNN是序列模型，无法并行，Transformer可以并行

- 原始 Transformer 指的是一个基于 Encoder-Decoder 框架的 Seq2Seq 模型，用于解决机器翻译任务；
- 后其 Encoder 部分被用于 BERT 而广为人知，因此有时 Transformer 也特指其 Encoder 部分；



## Transformer Encoder 代码


```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class TransformerEncoder(nn.Module):

    def __init__(self, n_head, d_model, d_ff, act=F.gelu):
        super().__init__()

        self.h = n_head
        self.d = d_model // n_head
        # Attention
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)
        # LN
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        # FFN
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.act = act
        #
        self.dropout = nn.Dropout(0.2)

    def attn(self, x, mask):
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = einops.rearrange(q, 'B L (H d) -> B H L d', H=self.h)
        k = einops.rearrange(k, 'B L (H d) -> B H d L', H=self.h)
        v = einops.rearrange(v, 'B L (H d) -> B H L d', H=self.h)
        a = torch.softmax(q @ k / math.sqrt(self.d) + mask, dim=-1)  # [B H L L]
        o = einops.rearrange(a @ v, 'B H L d -> B L (H d)')
        o = self.O(o)
        return o

    def ffn(self, x):
        x = self.dropout(self.act(self.W1(x)))
        x = self.dropout(self.W2(x))
        return x

    def forward(self, x, mask):
        x = self.LN1(x + self.dropout(self.attn(x, mask)))
        x = self.LN2(x + self.dropout(self.ffn(x)))
        return x


model = TransformerEncoder(2, 4, 8)
x = torch.randn(2, 3, 4)
mask = torch.randn(1, 1, 3, 3)
o = model(x, mask)

model.eval()
traced_model = torch.jit.trace(model, (x, mask))

x = torch.randn(2, 3, 4)
mask = torch.randn(1, 1, 3, 3)

assert torch.allclose(model(x, mask), traced_model(x, mask))
```



# 四、Bert改进模型

### 1. RoBERTa

#### （1）动态掩码：comparable or slightly better

Bert使用的是静态掩码。但是这样会存在一个现象，比如我训练40个epoches，那么每次epoches都是使用同一批数据。

这其实不是什么大问题，我们在深度学习训练模型的时候，每个epoches基本都没咋变过。

不过对于Bert，其实本质是一个自监督模型。每次的训练输入如果是不同的，对于模型肯定是更好的。

比如我们句子为：今天去哪里吃饭啊？

mask之后为：今天去哪里[mask]饭啊？

每次训练使用同一个mask样本，那么模型见得就少。

如果换一个mask：[mask]天去哪里吃饭啊？

模型对于同一个句子，在预测不同的单词，那么模型对句子的表达能力直觉上肯定是会上升的。

所以为了缓解这种静态掩码的问题，Bert的操作是这样的：

复制原始样本10份，每份都做不同的静态mask，然后进行训练。

我们想一下这个过程：比如我仍然是训练40个epoches，复制了十份样本，相当于每4个epoches使用的是同一个mask的样本。

这个操作确实缓解了静态掩码的问题，但是毕竟还有重复mask的情况出现。

这个时候其实有个朴素的思想，为啥不直接复制40份，然后分在40个epoches中进行训练，这个到时候写Bert的时候再说。

RoBERTa 是咋做的呢？

动态掩码，也就是不是在最开始的时候的数据处理的过程中就生成mask样本，而是在送入到模型之前才进行mask，这样同一个句子，在40epoches中，每次mask都不同。

#### （2）去掉NSP任务并且更改数据输入格式为全部填充可以跨越多个文档

我们先说RoBERTa 的四种输入形式和实验效果，然后再详细分析：

1. SEGMENT-PAIR+NSP：就是Bert的输入形式
2. SENTENCE-PAIR+NSP：输入的是一对句子，即前后是单个句子
3. FULL-SENTENCES：输入为全量的句子，填满512的长度，采集样本的时候可以跨越文章的界限，去除了NSP loss
4. DOC-SENTENCES：输入和FULL-SENTENCE类似，但是一个样本不能跨越两个document

实验效果：

<img src="https://pic3.zhimg.com/80/v2-492bcf7392f2da5c523151bb6d065e02_720w.webp" alt="img" style="zoom:50%;" />

对上面这个图一个最简单的总结就是NSP没啥用。然后我们来详细说一下这个事情。

首先Bert的消融实验证明，NSP是应该有的，如果没有NSP，在部分任务上效果有损失。

但是上图RoBERTa实验证明，NSP没啥效果，可以没有。

一个直观的解释，或者说猜测是因为，可能是Bert在消融实验去除NSP的时候，仍然保持的是原始的输入，即有NSP任务的时候的输入形式。

这就相当于，构造了好了符合NSP任务的数据，但是你删去了针对这个任务的损失函数，所以模型并没有学的很好，在部分任务精读下降。

但是RoBERTa这里不是的，它删除NSP任务的时候，同时改变了输入格式，并不是使用上下两句的输入格式，而是类似文档中的句子全部填满这512个字符的格式进行输入。

简单说就是，去掉了NSP任务的同时，去掉了构造数据中NSP数据格式。

比较SEGMENT-PAIR和DOC-SENTENCES两个模式的时候，证明没有NSP效果更好。其实看起来好像并没有控制变量，因为变了两个地方，一个是是否有NSP，一个是输入格式。

这种情况下，就只能去看在下游任务中的效果了。

#### （3）更多数据，更大bsz，更多的步数，更长训练时间

1. 数据：Bert：16G；RoBERTa：160G；十倍
2. bsz：Bert：256；RoBERTa：8K
3. steps：Bert：1M；RoBERTa：300K/500K

#### （4）动态掩码那里，说到一个复制10份的细节，那里是针对的Bert，RoBERTa是每次输入之前才mask，注意区分，不要搞混



RoBERTa 模型是BERT 的改进版(从其名字来看，A Robustly Optimized BERT，即简单粗暴称为强力优化的BERT方法)。 在模型规模、算力和数据上，与BERT相比主要有以下几点改进：

- 更大的模型参数量（论文提供的训练时间来看，模型使用 1024 块 V100 GPU 训练了 1 天的时间）
- 更大batch size。RoBERTa 在训练过程中使用了更大的bacth size。尝试过从 256 到 8000 不等的bacth size。
- 更多的训练数据（包括：CC-NEWS 等在内的 160GB 纯文本。而最初的BERT使用16GB BookCorpus数据集和英语维基百科进行训练）

另外，RoBERTa在训练方法上有以下改进：

- 去掉下一句预测(NSP)任务。
- 动态掩码。BERT 依赖随机掩码和预测 token。原版的 BERT 实现在数据预处理期间执行一次掩码，得到一个静态掩码。 而 RoBERTa 使用了动态掩码：每次向模型输入一个序列时都会生成新的掩码模式。这样，在大量数据不断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语言表征。
- 文本编码。Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。

### 2. ALBERT

ALBERT是紧跟着RoBERTa出来的，也是针对Bert的一些调整，重点在减少模型参数，对速度倒是没有特意优化。

提出了两种能够大幅减少预训练模型参数量的方法

- Factorized embedding parameterization。将embedding matrix分解为两个大小分别为V x E和E x H矩阵。这使得embedding matrix的维度从O(V x H)减小到O(V x E + E x H)。
- Cross-layer parameter sharing，即多个层使用相同的参数。参数共享有三种方式：只共享feed-forward network的参数、只共享attention的参数、共享全部参数。ALBERT默认是共享全部参数的。

使用Sentence-order prediction (SOP)来取代NSP。具体来说，其正例与NSP相同，但负例是通过选择一篇文档中的两个连续的句子并将它们的顺序交换构造的。这样两个句子就会有相同的话题，模型学习到的就更多是句子间的连贯性。

### 3. DeBERTa

#### （1）V1 相比 BERT 和 RoBERTa 模型的改进：

encoder由11层 Transformer组成，decoder由2层参数共享的Transformer和一个Softmax输出层组成

- **解耦注意力机制**：为了更充分利用相对位置信息，输入的input embedding不再加入pos embeding, 而是input在经过transformer编码后，在encoder段与“decoder”端 通过**相对位置**计算**分散注意力**

将位置信息和内容信息分别和交叉做attention，得到**一个单词对的注意力权重是 内容到内容，内容到位置，位置到内容和位置到位置 四个注意力的得分的总和**。

（对于序列中位置i处的token，使用两个向量， {𝐻𝑖} 和 {𝑃𝑖|𝑗} 表示它，一个表示它的内容，一个表示与另一个位置j的token的相对位置。 token i和j之间的交叉注意力得分的计算可以分解为四个部分。但是，**由于使用相对位置嵌入，因此位置到位置项不会提供太多附加信息，因此在实现中将其从计算注意力公式中删除。**）

- **考虑绝对位置的MLM任务**

原始的BERT存在预训练和微调不一致问题。预训练阶段，隐层最终的输出输入到softmax预测被mask掉的token，而微调阶段则是将隐层最终输出输入到特定任务的decoder。这个decoder根据具体任务不同可能是一个或多个特定的decoder，如果输出是概率，那么还需要加上一个softmax层。为消除这种不一致性，DeBERTa将MLM与其他下游任务同等对待，并将原始BERT中输出层的softmax替换为**「增强后的mask decoder(EMD)」**，EMD包含一个或多个Transformer层和一个softmax输出层。至此，结合了BERT和EMD的DeBERTa成为了一个encoder-decoder模型。

DeBERTa是使用MLM进行预训练的，模型被训练为使用mask token周围的单词来预测mask词应该是什么。 DeBERTa将上下文的内容和位置信息用于MLM。 **解耦注意力机制已经考虑了上下文词的内容和相对位置，但没有考虑这些词的绝对位置**。

给定一个句子“a new store opened beside the new mall”，并用“store”和“mall”两个词mask以进行预测。 仅使用局部上下文(例如，相对位置和周围的单词)不足以使模型在此句子中区分store和mall，**因为两者都以相同的相对位置在new单词之后。 为了解决这个限制，模型需要考虑绝对位置，作为相对位置的补充信息。** ==如果句子中不同位置出现了相同的词，那么这两个词的附近的词对于这个词来说都是以相同的相对位置出现在它附近，不加入绝对位置的话不足以区分这两个词==

**有两种合并绝对位置的方法。 BERT模型在输入层中合并了绝对位置。 在DeBERTa中，我们在所有Transformer层之后将它们合并**，然后在softmax层之前进行mask token预测。这样，DeBERTa捕获了所有Transformer层中的相对位置，仅将绝对位置用作补充信息。

- **预训练时引入对抗训练**
  - 训练时加入了一些数据扰动
    （将DeBERTa微调到下游NLP任务时，**首先将单词嵌入向量归一化为随机向量，然后将扰动应用于归一化的嵌入向量。** ）
  - mask策略中不替换词，变为替换成词的pos embedding
    （将bert的训练策略中，mask有10%的情况是不做任何替换，这种情况attention偏向自己会非常明显，DeBeta将不做替换改成了换位该位置词绝对位置的pos embedding，实验中明显能看到这种情况下的attention对自身依赖减弱）

#### （2）V2 改进

- 更换tokenizer，将**词典扩大**了。（优化效果最明显）
- 在第一个transformer block后加入**卷积**
- 共享位置和内容的变换矩阵
- 把相对位置编码换成了log bucket，各个尺寸模型的bucket数都是256

#### （3）V3 改进

- 在模型层面并没有修改，将预训练任务由掩码语言模型（MLM）换成了ELECTRA一样类似GAN的Replaced token detect任务



# 五、其他大模型

### GPT



# 六、强化学习

## 基础知识

## MAPPO

## RLHF

### instructGPT的原理，讲讲RLHF、SFT、和reward。

instructGPT是一种基于强化学习的文本生成模型，其核心原理涉及两个概念：RLHF（Reinforcement Learning from Human Feedback）和reward shaping（奖励塑造）。

> RLHF：在训练instructGPT时，首先**使用有人类生成的示例对模型进行预训练**。然后，**通过与人类评估者进行交互，收集评估结果，以创建一个用于强化学习的数据集。该数据集包含了人类评估者对生成结果的评分或反馈，用于指导模型的强化学习训练。**
> Reward shaping：为了更好地引导模型的训练，reward shaping用于调整模型的奖励信号。通过将人类评估者的反馈与模型生成的文本进行比较，可以计算出一个差异度量，用作奖励信号的一部分。这样，模型可以根据这个奖励信号进行训练，并进行强化学习的训练。模型根据当前的状态（对话历史）生成文本，并通过奖励信号来评估生成文本的质量。模型的目标是最大化预期累积奖励，从而生成更高质量的文本。

通过RLHF和reward shaping的结合，instructGPT能够通过人类评估者的反馈指导模型的生成过程，并逐步提升生成文本的质量和一致性。监督微调（SFT）和人类反馈强化学习（RLHF）是两种用于微调大型语言模型的方法，它们的目的是使模型的输出更符合人类的偏好和价值观。它们的基本思想和步骤如下：

监督微调（SFT）：SFT是一种利用**人工标注的数据**来训练模型的方法，它可以使模型学习到一些基本的规则和约束，例如遵循人类的指令、避免有害或无用的输出等。SFT的步骤包括：

> 准备数据集：收集一些包含人类指令和期望输出的数据，例如Helpful and Harmless数据集，它包含了一些常见的对话场景和相应的标签。
> 训练模型：使用一个预训练好的语言模型，例如GPT-4，并在数据集上进行微调，使模型能够根据输入的指令生成合适的输出。
> 评估模型：使用一些评价指标，例如准确率、BLEU分数、ROUGE分数等，来衡量模型的性能和质量。

人类反馈强化学习（RLHF）：RLHF是一种**利用人类对模型输出的评价**来训练模型的方法，它可以使模型更好地适应人类的偏好和价值观，例如生成更有趣、更友好、更安全的输出等。RLHF的步骤包括：

> 训练奖励模型：收集一些包含人类对模型输出的评价或排名的数据，例如HumanEval数据集，它包含了一些由人类评价员对不同模型输出进行打分或排序的数据。
> 使用一个预训练好的语言模型，例如GPT-4，并在奖励模型上进行微调，使奖励模型能够根据输入和输出给出一个奖励值。
> 训练策略模型：使用一个预训练好的语言模型，例如GPT-4，并使用一种强化学习算法，例如近端策略优化（Proximal Policy Optimization，PPO），来更新模型参数。PPO算法会根据奖励模型给出的奖励值来调整模型生成不同输出的概率。
> 评估模型：使用一些评价指标，例如奖励值、人类标注、对话质量等，来衡量模型的性能和质量。





### 10.RLHF完整训练过程是什么？RL过程中涉及到几个模型？



### 11.RLHF过程中RM随着训练过程得分越来越高，效果就一定好吗？有没有极端情况？



### 12.encoder only，decoder only，encoder-decoder 划分的具体标注是什么？典型代表模型有哪些？







Transformer 常见面试问题
===

## Transformer Encoder 代码


```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class TransformerEncoder(nn.Module):

    def __init__(self, n_head, d_model, d_ff, act=F.gelu):
        super().__init__()

        self.h = n_head
        self.d = d_model // n_head
        # Attention
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)
        # LN
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        # FFN
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.act = act
        #
        self.dropout = nn.Dropout(0.2)

    def attn(self, x, mask):
        q, k, v = self.Q(x), self.K(x), self.V(x)
        q = einops.rearrange(q, 'B L (H d) -> B H L d', H=self.h)
        k = einops.rearrange(k, 'B L (H d) -> B H d L', H=self.h)
        v = einops.rearrange(v, 'B L (H d) -> B H L d', H=self.h)
        a = torch.softmax(q @ k / math.sqrt(self.d) + mask, dim=-1)  # [B H L L]
        o = einops.rearrange(a @ v, 'B H L d -> B L (H d)')
        o = self.O(o)
        return o

    def ffn(self, x):
        x = self.dropout(self.act(self.W1(x)))
        x = self.dropout(self.W2(x))
        return x

    def forward(self, x, mask):
        x = self.LN1(x + self.dropout(self.attn(x, mask)))
        x = self.LN2(x + self.dropout(self.ffn(x)))
        return x


model = TransformerEncoder(2, 4, 8)
x = torch.randn(2, 3, 4)
mask = torch.randn(1, 1, 3, 3)
o = model(x, mask)

model.eval()
traced_model = torch.jit.trace(model, (x, mask))

x = torch.randn(2, 3, 4)
mask = torch.randn(1, 1, 3, 3)

assert torch.allclose(model(x, mask), traced_model(x, mask))
```

</details>


## Transformer 与 RNN/CNN 的比较

> 其他提法：Transformer 为什么比 RNN/CNN 更好用？优势在哪里？  
> 参考资料：
>
> - [自然语言处理三大特征抽取器（CNN/RNN/Transformer）比较 - 知乎](https://zhuanlan.zhihu.com/p/54743941)
>   - [CNN/RNN/Transformer比较 - 简书](https://www.jianshu.com/p/67666ada573b)
>   - [NLP常用特征提取方法对比 - CSDN博客](https://blog.csdn.net/u013124704/article/details/105201349)

### RNN

- 特点/优势（Transformer之前）：

  - 适合解决线性序列问题；天然能够捕获位置信息（相对+绝对）；

    > 绝对位置：每个 token 都是在固定时间步加入编码；相对位置：token 与 token 之间间隔的时间步也是固定的；

  - 支持不定长输入；

  - LSTM/Attention 的引入，加强了长距离语义建模的能力；

- 劣势：

  - 串行结构难以支持并行计算；

  - 依然存在长距离依赖问题；

    > 有论文表明：RNN 最多只能记忆 50 个词左右的距离（How Neural Language Models Use Context）；

  - 单向语义建模（Bi-RNN 是两个单向拼接）

### CNN

- 特点/优势：
  - 捕获 n-gram 片段信息（局部建模）；
  - 滑动窗口捕获相对位置特征（但 Pooling 层会丢失位置特征）；
  - 并行度高（滑动窗口并行、卷积核并行），计算速度快；
- 劣势：
  - 长程建模能力弱：受感受野限制，无法捕获长距离依赖，需要空洞卷积或加深层数等策略来弥补；
  - Pooling 层会丢失位置信息（目前常见的作法会放弃 Pooling）；
  - 相对位置敏感，绝对位置不敏感（平移不变性）

### Transformer

- 特点/优势：
  - 通过位置编码（position embedding）建模相对位置和绝对位置特征；
  - Self-Attention 同时编码双向语义和解决长距离依赖问题；
  - 支持并行计算；
- 缺点/劣势：
  - 不支持不定长输入（通过 padding 填充到定长）；
  - 计算复杂度高；

### Transformer 能完全取代 RNN 吗？

> [有了Transformer框架后是不是RNN完全可以废弃了？ - 知乎](https://www.zhihu.com/question/302392659?sort=created)

- 不行；

<!-- 
下面主要从三个方面进行比较：对**上下文语义特征**和**序列特征**的表达能力（主要），以及**计算速度**；

### 1. 上下文语义特征

在抽取上下文语义特征（方向+距离）方面：**Transformer > RNN > CNN**

- RNN 只能进行单向编码（Bi-RNN 是两个单向）；  
  在**长距离**特征抽取上也弱于 Transformer；有论文表明：RNN 最多只能记忆 50 个词左右的距离；

    > How Neural Language Models Use Context

- CNN 只能对短句编码（N-gram）；

- Transformer 可以同时**编码双向语义**和**抽取长距离特征**；

### 2. 序列特征

在抽取序列特征方面：**RNN > Transformer > CNN**

- Transformer 的序列特征完全依赖于 Position Embedding，当序列长度没有超过 RNN 的处理极限时，位置编码对时序性的建模能力是不及 RNN 的；
- CNN 的时序特征 TODO；

### 3. 计算速度

在计算速度方面：**CNN > Transformer > RNN**

- RNN 因为存在时序依赖难以并行计算；
- Transformer 和 CNN 都可以并行计算，但 Transformer 的计算复杂度更高；
  -->

## Transformer 中各模块的作用

### QKV Projection

#### 为什么在 Attention 之前要对 Q/K/V 做一次投影？

- 首先在 Transformer-Encoder 中，Q/K/V 是相同的输入；
- 加入这个全连接的目的就是为了将 Q/K/V 投影到不同的空间中，增加多样性；
- 如果没有这个投影，在之后的 Attention 中相当于让相同的 Q 和 K 做点击，那么 attention 矩阵中的分数将集中在对角线上，即每个词的注意力都在自己身上；这与 Attention 的初衷相悖——**让每个词去融合上下文语义**；

### Self-Attention

#### 为什么要使用多头？

> 其他提法：多头的加入既没有增加宽度也没有增加深度，那加入它的意义在哪里？

- 这里的多头和 CNN 中多通道的思想类似，目的是期望不同的注意力头能学到不同的特征；

#### 为什么 Transformer 中使用的是乘性 Attention（点积），而不是加性 Attention？

- 在 GPU 场景下，矩阵乘法的效率更高（原作说法）；

- **在不进行 Scaled 的前提下**，随着 d（每个头的特征维度）的增大，乘性 Attention 的效果减弱，加性 Attention 的效果更好（原因见下一个问题）；

  > [小莲子的回答 - 知乎](https://www.zhihu.com/question/339723385/answer/811341890)

#### Attention 计算中 Scaled 操作的目的是什么？

> 相似提法：为什么在计算 Q 和 K 的点积时要除以根号 d？  
> 参考内容：[Transformer 中的 attention 为什么要 scaled? - 知乎](https://www.zhihu.com/question/339723385)

- **目的**：防止梯度消失；

- **解释**：在 Attention 模块中，注意力权重通过 Softmax 转换为概率分布；但是 Softmax 对输入比较敏感，当输入的方差越大，其计算出的概率分布就越“尖锐”，即大部分概率集中到少数几个分量位置。极端情况下，其概率分布将退化成一个 One-Hot 向量；其结果就是雅可比矩阵（偏导矩阵）中绝大部分位置的值趋于 0，即梯度消失；通过缩放操作可以使注意力权重的方差重新调整为 1，从而缓解梯度消失的问题；

  - 假设 $Q$ 和 $K$ 的各分量 $\vec{q_i}$ 和 $\vec{k_i}$ 相互独立，且均值为 $0$，方差为 $1$；

    > 在 Embedding 和每一个 Encoder 后都会过一个 LN 层，所以可以认为这个假设是合理的；

  - 则未经过缩放的注意力权重 $A$ 的各分量 $\vec{a_i}$ 将服从均值为 $0$，方差为 $d$ 的正态分布；

  - $d$ 越大，意味着 $\vec{a_i}$ 中各分量的差越大，其结果就是经过 softmax 后，会出现数值非常小的分量；这样在反向传播时，就会导致**梯度消失**的问题；

  - 此时除以 $\sqrt{d}$ 会使 $\vec{a_i}$ 重新服从标准的正态分布，使 softmax 后的 Attention 矩阵尽量平滑，从而缓解梯度消失的问题；

  <details><summary><b>数学推导与代码验证</b></summary> 


  - 数学推导：

    > [Transformer 中的 attention 为什么要 scaled? - TniL的回答（已删除）](https://www.zhihu.com/question/339723385/answer/782509914)

    - 定义 $Q=[\vec{q_1}, \vec{q_2}, .., \vec{q_n}]^T$, $K=[\vec{k_1}, \vec{k_2}, .., \vec{k_n}]^T$，其中 $\vec{q_i}$ 和 $\vec{k_i}$ 都是 $d$ 维向量；

    - 假设 $\vec{q_i}$ 和 $\vec{k_i}$ 的各分量都是服从标准正态分布（均值为 0，方差为 1）的随机变量，且相互独立，记 $q_i$ 和 $k_i$，即 $E(q_i)=E(k_i)=0$, $D(q_i)=D(k_i)=1$；

    - 根据期望与方差的性质，有 $E(q_ik_i)=0$ 和 $D(q_ik_i)=1$，推导如下：

      $$\begin{align*}
          E(q_ik_i) &= E(q_i)E(k_i) = 0 \times 0 = 0 \\
          D(q_ik_i) &= E(q_i^2k_i^2) - E^2(q_ik_i) \\
          &= E(q_i^2)E(k_i^2) - E^2(q_i)E^2(k_i) \\
          &= [E(q_i^2) - E^2(q_i)] [E(k_i^2) - E^2(k_i)] - 0^2 \times 0^2 \\
          &= D(q_i)D(k_i) - 0 \\
          &= 1
      \end{align*}$$

    - 进一步，有 $E(\vec{q_i}\vec{k_i}^T)=0$ 和 $D(\vec{q_i}\vec{k_i}^T)=d$，推导如下：

      $$\begin{align*}
          E(\vec{q_i}\vec{k_i}^T) &= E(\sum_{i=1}^d q_ik_i) = \sum_{i=1}^d E(q_ik_i) = 0 \\
          D(\vec{q_i}\vec{k_i}^T) &= D(\sum_{i=1}^d q_ik_i) = \sum_{i=1}^d D(q_ik_i) = d
      \end{align*}$$

    - 根据 attention 的计算公式（softmax 前）, $A'=\frac{QK^T}{\sqrt{d}}=[\frac{\vec{q_1}\vec{k_1}^T}{\sqrt{d}}, \frac{\vec{q_2}\vec{k_2}^T}{\sqrt{d}}, .., \frac{\vec{q_n}\vec{k_n}^T}{\sqrt{d}}]=[\vec{a_1}, \vec{a_2}, .., \vec{a_n}]$，可知 $E(\vec{a_i})=0$, $D(\vec{a_i})=1$，推导如下：

      $$\begin{align*}
          E(\vec{a_i}) &= E(\frac{\vec{q_i}\vec{k_i}^T}{\sqrt{d}}) = \frac{E(\vec{q_i}\vec{k_i}^T)}{\sqrt{d}} = \frac{0}{\sqrt{d}} = 0 \\
          D(\vec{a_i}) &= D(\frac{\vec{q_i}\vec{k_i}^T}{\sqrt{d}}) = \frac{D(\vec{q_i}\vec{k_i}^T)}{(\sqrt{d})^2} = \frac{d}{d} = 1
      \end{align*}$$

  - 代码验证

    ```python
    import torch
    
    def get_x(shape, eps=1e-9):
        """创建一个 2d 张量，且最后一维服从正态分布"""
        x = torch.randn(shape)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x - mean) / (std + eps)
    
    d = 400  # 数字设大一些，否则不明显
    q = get_x((2000, d))
    k = get_x((2000, d))
    
    # 不除以 根号 d
    a = torch.matmul(q, k.transpose(-1, -2))  # / (d ** 0.5)
    print(a.mean(-1, keepdim=True))  # 各分量接近 0
    print(a.var(-1, keepdim=True))  # 各分量接近 d
    
    # 除以根号 d
    a = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
    print(a.mean(-1, keepdim=True))  # 各分量接近 0
    print(a.var(-1, keepdim=True))  # 各分量接近 1
    ```

  </details>


#### 在 Softmax 之前加上 Mask 的作用是什么？

> 相关问题：为什么将被 mask 的位置是加上一个极小值（-1e9），而不是置为 0？

- 回顾 softmax 的公式；
- 其目的就是使无意义的 token 在 softmax 后得到的概率值（注意力）尽量接近于 0；从而使正常 token 位置的概率和接近 1；

### Add & Norm

#### 加入残差的作用是什么？

- 在求导时加入一个恒等项，以减少梯度消失问题；

#### 加入 LayerNorm 的作用是什么？

- 提升网络的泛化性；（TODO：详细解释）
- 加在激活函数之前，避免激活值落入饱和区，减少梯度消失问题；

#### Pre-LN 和 Post-LN 的区别

- Post-LN（BERT 实现）：
  $$x_{n+1} = \text{LN}(x_n + f(x_n))$$
  - 先做完残差连接，再归一化；
  - 优点：保持主干网络的方程比较稳定，是模型泛化能力更强，性能更好；
  - 缺点：把恒等路径放在 norm 里，使模型收敛更难（反向传播时梯度变小，残差的作用被减弱）
- Pre-LN：
  $$x_{n+1} = x_n + f(\text{LN}(x_n))$$
  - 先归一化，再做残差连接；
  - 优点：加速收敛
  - 缺点：效果减弱

### Feed-Forward Network

- 前向公式
  $$W_2 \cdot \text{ReLU}(W_1x + b_1) + b_2$$

#### FFN 层的作用是什么？

- 功能与 1*1 卷积类似：1）跨通道的特征融合/信息交互；2）通过激活函数增加非线性；

  > [1*1卷积核的作用_nefetaria的博客-CSDN博客](https://blog.csdn.net/nefetaria/article/details/107977597)

- 之前操作都是线性的：1）Projection 层并没有加入激活函数；2）Attention 层只是线性加权；

#### FFN 中激活函数的选择

> 相关问题：BERT 为什么要把 FFN 中的 ReLU 替换为 GeLU？

- 背景：原始 Transformer 中使用的是 **ReLU**；BERT 中使用的是 **GeLU**；
- GeLU 在激活函数中引入了正则的思想，越小的值越容易被丢弃；相当于综合了 ReLU 和 Dropout 的功能；而 ReLU 缺乏这个随机性；
- 为什么不使用 sigmoid 或 tanh？——这两个函数存在饱和区，会使导数趋向于 0，带来梯度消失的问题；不利于深层网络的训练；



## BERT 相关面试题

- [daily-interview/BERT面试题.md at master · datawhalechina/daily-interview](https://github.com/datawhalechina/daily-interview/blob/master/AI%E7%AE%97%E6%B3%95/NLP/%E7%89%B9%E5%BE%81%E6%8C%96%E6%8E%98/BERT/BERT%E9%9D%A2%E8%AF%95%E9%A2%98.md)



## 参考资料

- [深入剖析PyTorch中的Transformer API源码_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1o44y1Y7cp/?spm_id_from=333.788)
- [超硬核Transformer细节全梳理！_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1AU4y1d7nT)
- [Transformer、RNN 与 CNN 三大特征提取器的比较_Takoony的博客-CSDN博客](https://blog.csdn.net/ningyanggege/article/details/89707196)





# LLM核心参数

[大模型核心参数解析（Top-k、Top-p、Temperature、frequency penalty、presence penalty）-CSDN博客](https://blog.csdn.net/u012856866/article/details/140308083)