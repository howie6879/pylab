# 读 - Character-level Convolutional Networks for Text Classiﬁcation

> 这篇论文提供了一个关于字符级卷积网络(`ConvNets`)在文本分类中应用实证研究，我们构建了几个大型数据集，以表明字符级卷积网络可以实现最先进的或竞争性的结果，针对传统模型（如词袋，n-gram及其TFIDF变体）和深度学习模型（如基于单词的ConvNets和循环神经网络）进行比较

## Introduction
文本分类是自然语言处理中的一个经典问题，需要给自由文本文档分配预定义的类别。文本分类的研究范围包括从设计最佳特征到选择最佳机器学习分类器。到目前为止，几乎所有的文本分类技术都是基于单词的，一些单词有序组合的简单统计，如 N-gram，通常表现最好

另一方面，许多研究人员发现卷积网络（`ConvNets`）从计算机视觉应用到语音识别等，对于从原始信号中提取信息都有不错的效果，特别是，在深度学习研究的早期使用的延时网络本质上是对序列数据进行建模的卷积网络

在这篇文章中，我们探讨在字符层面上把文本作为一种原始信号，并将其应用于一维随机网络中。对于本文我们只使用文本分类任务作为例证来说明`ConvNets`理解文本的能力，从历史上看，我们知道`ConvNets`通常需要大规模数据集才能工作，因此我们也构建了其中的几个，提供了一系列与传统模型和其他深度学习模型的比较

将卷积网络应用于文本分类或自然语言处理是文献研究的热点，研究表明，`ConvNets`可以直接应用于分布式或离散的嵌入词，而不需要任何语言的句法或语义结构的知识，事实证明，这些方法与传统模型相比具有竞争力

还有一些相关的工作使用字符级特性进行语言处理，这些包括使用[带有线性分类器的字符级n-gram](http://www.icsd.aegean.gr/lecturers/stamatatos/papers/ijait-spam.pdf)，并将字符级特性合并到`ConvNets`，特别是，这些`ConvNet`方法以单词为基础，在单词或单词n-gram层提取字符级特征，形成分布式表示。对词性标注和信息检索的改进进行了观察

本文是第一篇只将字符应用于`ConvNets`的文章，我们证明，在大规模数据集上训练时，深度`ConvNets`不需要词汇知识，此外以前的研究结论认为，`ConvNets`不需要语言的句法或语义结构知识，这种工程上的简化对于能够适用于不同语言的单一系统来说是至关重要的，因为字符总是构成一个必要的结构，无论能否做到分割成词，仅处理字符还有一个好处，那就是可以自然而然地学会拼写错误和表情符号等不正常的字符组合

## Character-level Convolutional Networks 
在这部分，我们介绍了用于文本分类的字符级`ConvNets`的设计，设计是模块化的，其中通过反向传播获得梯度以执行优化

### Key Modules 
主要组件是简单卷积模块，它简单地计算一维卷积。假设我们有一个离散输入函数：

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyhuie0ffwj303m00ja9t.jpg)

和一个离散核函数：

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyhum22ucbj303s00ja9t.jpg)

具有步幅d的`f(x)`和`g(y)`之间的卷积:

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyhuy2yjqwj307600kdfo.jpg)

被定义为：

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld02654sj30j901e745.jpg)

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld0mj7h8j30jn06labv.jpg)

其中`c = k - d + 1`是偏移常数，这个非常合用的模块使我们能够对 convnet 进行比6层更深的训练，而其他所有层都失败了。 [A theoretical analysis of feature pooling in visual recognition.](https://www.di.ens.fr/sierra/pdfs/icml2010b.pdf)的分析可能对这个问题有所帮助

该模型使用的非线性函数是`h(x) = max{0, x}`，这使得我们的卷积层类似于校正的线性单位(ReLUs)，使用的算法是128个小批量的随机梯度下降（SGD），使用动量0.9和初始步长0.01，每3个时间段减半，持续10次。 每个时期采用固定数量的随机训练样本，这些样本在类之间均匀采样。稍后将详细说明每个数据集的此数字。 实现使用 Torch 7完成

### Character quantization
我们的模型接受一系列编码字符作为输入，通过为输入语言规定大小为m的字母表来完成编码，然后使用1-m编码（或“one-hot”编码）量化每个字符。然后，将字符序列变换为具有固定长度l0的这种m大小的向量的序列。忽略超过长度l0的任何字符，并且包括空白字符的不在字母表中的任何字符被量化为全零向量。字符量化顺序是向后的，因此字符的最新读数总是放在输出开始的附近，使得完全连接的图层可以很容易地将权重与最新读数联系起来

我们所有模型中使用的字母由70个字符组成，包括26个英文字母，10个数字，33个其他字符加一个全零向量:

```python
abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}
```

稍后我们将会生成一个字母表加上大小写的模型并进行比较

### 模型设计
论文中设计了`large`和`small`两种卷积网络，都由6个卷积层和3个全连接层共9层神经网络组成

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyldz7mgmoj30jh05xt9n.jpg)

由于字母表长度为70，因此我们的输入特征为70，输入特征长度论文设定为1014，看起来1014个字符已经可以捕获大部分感兴趣的文本，我们还在三个全连接层之间插入两个dropout层以实现模型正则化，dropout概率为0.5，表1列出了卷积层的配置，表2列出了全连接层的配置：

表1：实验中的卷积层，卷积层的步幅为1，并且池化层都是非重叠的，因此我们省略了他们的步幅描述

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld29gcn0j30jd04v74p.jpg)

使用`Gaussian distribution`初始化权重，初始化模型的均值和标准差分别为：

- large model：(0, 0.02)
- small model：(0, 0.05)

表2：实验中的全连接层，最后一层的输出单元数由问题决定，例如，对于10分类问题将是10

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld2le6wij30j803674f.jpg)

对于不同的问题，我们的输入长度可能也是不同的（比如此例子: `l0`=1014），特征长度也是如此，从我们的模型设计可以很容易地知道，给定输入长度 l0，在最后一个卷积层之后(任何一个完全连接层之前)的特征长度为 `l6` = (`l0`-96) / 27，这个数字与第6层的特征大小相乘将给出第一个完全连接的层接受的输入维度

### Data Augmentation using Thesaurus
许多研究人员发现，适当的数据扩充技术有助于降低深度学习模型的泛化误差，当我们能够找到模型可能具有的合适的不变性属性时，这些技术通常能够很好地工作。就文本而言，不能像图像或语音识别那样进行数据转换来增加数据，因为字符的顺序具有严格的句法和语义意义，因此最好的数据扩充方式应该是使用人类的重述文本类，但是这样昂贵且不切实际的，因为数据集太庞大了，综上，比较常见的做法是使用词汇和词组的近义词进行替代扩充。

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld4tuovoj30jd03v40e.jpg)

## Comparison Models
为了与竞争模型进行公平的比较，我们用传统和深度学习方法进行了一系列的实验。我们尽最大努力来选择可以提供可比性和有竞争力的模型，并且在没有偏向某个模型的情况下忠实的汇报结果

### Traditional Methods
我们提到的传统方法都是使用手动提取特征并构建一个线性分类器， 在所有这些模型中，所使用的分类器都是一个多项式 Logit模型

- Bag-of-words & TF-IDF
- Bag-of-ngrams & TF-IDF
- Bag-of-means & TF-IDF

### Deep Learning Methods
近来深度学习模型开始应用于文本分类，我们选用两个简单且经典的模型来进行对比：

- word-based ConvNet
- long-short term memory (LSTM)  recurrent neural network model

### Choice of Alphabet
对于英语字母表，一个明显的选择就是是否要区分大写字母和小写字母，结论发现不区分时候结果比较好，一种可能的解释是语义不会随着不同的字母情况而改变，因此带来了正则化的好处

## Large-scale Datasets and Results
以前在不同的领域已经证明ConvNets通常能够很好地处理大规模的数据集，特别是在我们的例子中，当模型采用了低层次的原始特征(比如字符)时。然而，大多数用于文本分类的开放数据集都很小，大规模数据集被分割成一个比测试小得多的训练集，因此，我们不是通过使用它们来混淆我们的社区，而是为我们的实验构建了几个大型数据集，范围从数十万到数百万个样本，总结在表3

表3：我们的大型数据集的统计数据，`Epoch`大小是每次迭代的`minibatches`数量大小：

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld56uee5j30jf061gmu.jpg)

表4：测试所有模型的错误，以百分比表示，`Lg`代表`大`，`Sm`代表`小`，`w2v`是`word2vec`的缩写形式，`Lk`表示`lookup table`，`Th`代表`thesaurus`，有`Full`标签的表示`ConvNets`区分大小写：

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld5f1ko0j30k50d2gq3.jpg)

基于所有合适的模型对数据进行训练，表格4列出来了误差百分比，请注意，由于我们没有中文词库，Sogou News数据集使用词库扩充部分没有任何结果，我们标记蓝色为好的结果，红色为不好的：

## Discussion
![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyld5la9jhj30jy0a575n.jpg)

为了进一步了解表4中的结果，我们在本节中提供了一些实证分析，为了便于我们的分析，我们在比较模型中给出了图3中的相对误差。每个图都计算自我们的比较的模型和字符级ConvNet模型之间的差异，然后除以比较的模型误差，图中的所有ConvNets分别是具有词库扩充的大型模型。

- **Character-level ConvNet is an effective method**，我们实验结论最重要的一点就是`character-level ConvNets`可以在不需要单词的情况下对文本进行分类，这强烈表明语言也可以被认为是一种与任何其他类型无异的信号
- **Dataset size forms a dichotomy between traditional and ConvNets models**，数据量越大，性能越好
- **ConvNets may work well for user-generated data**，擅长识别拼写错误、表情符号等奇异的符号组合，在人为生成的数据上效果较好
- **Choice of alphabet makes a difference**，比较大量的数据集可以不用考虑大小写
- **Semantics of tasks may not matter**，字符级ConvNets分类时，与语义无关
- **Bag-of-means is a misuse of word2vec**，Bag-of-means模型表现很差，这表明如此简单的使用分布式单词表示可能不会给我们带来文本分类的优势
- **There is no free lunch**，每一种模型都是适合特定数据集的，做不到一种模型适用于所有数据集

## Conclusion and Outlook
本文提供了一个关于文本分类的字符级卷积网络的实证研究，我们使用几个大型数据集与大量传统和深度学习模型进行了比较，一方面，分析显示`character-level ConvNet`是一个有很高效率的方法，另一方面，我们的模型在比较中的表现取决于许多因素，例如数据集大小，文本是否被策划以及字母表的选择。

## Resources
参考文章：

- [字符级卷积神经网络（Char-CNN）实现文本分类—模型介绍与TensorFlow实现 - 呜呜哈的博客 - CSDN博客](https://blog.csdn.net/liuchonge/article/details/70947995)
- [Character-level Convolutional Networks for Text Classification之每日一篇 - gentelyang的博客 - CSDN博客](https://blog.csdn.net/gentelyang/article/details/80833942)
- [《Character-level Convolutional NNetworks for Text Classification》论文学习 - 程序员大本营](http://www.pianshen.com/article/5803133093/)

开源项目：

- [GitHub - mhjabreel/CharCNN](https://github.com/mhjabreel/CharCNN)
- [GitHub - mhjabreel/CharCnn_Keras: The implementation of text classification using character level convoultion neural networks using Keras](https://github.com/mhjabreel/CharCnn_Keras)
- [GitHub - lc222/char-cnn-text-classification-tensorflow: Character-level Convolutional Networks for Text Classification论文仿真实现](https://github.com/lc222/char-cnn-text-classification-tensorflow) - 这篇感觉代码有参考上述，看上面两个即可
- [GitHub - gaussic/text-classification-cnn-rnn: CNN-RNN中文文本分类，基于TensorFlow](https://github.com/gaussic/text-classification-cnn-rnn)

我参考上面的代码实现在[character_level_convolutional_networks_for_text_classiﬁcation](https://github.com/howie6879/pylab/tree/master/pylab/papers/character_level_convolutional_networks_for_text_classi%EF%AC%81cation)

![](https://ws1.sinaimg.cn/large/007i3XCUgy1fyjvmwzoq2j30p00dwmzl.jpg)