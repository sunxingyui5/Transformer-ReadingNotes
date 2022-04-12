## Attention Is All You Need  
**阅读地址：** [Attention Is All You Need](https://readpaper.com/paper/2963403868)  
**推荐学习视频：** [Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.999.0.0)  
**被引用次数：**  

### Transformer的价值  
开创了继MLP、CNN、RNN之后的第四大类深度学习模型，泛用能力很强，在NLP和CV领域均有很大作用  
是第一个仅使用attention做序列转录的模型，解决了seq2seq 模型处理长期依赖的局限性  
Attention用了一个更广泛的归纳偏置，使得它能处理更一般化的信息，并没有做空间上的假设  

### 摘要  
模型主体上的encoder&decoder架构  
是一个简单模型，总体上做的是序列到序列的生成  
在encoder和decoder之间使用attention机制而且仅使用attention机制，没有使用循环或卷积  
模型性能上更好，并行度更好，在机器翻译任务上达到了非常好的效果  

### 简介  
RNN：输入一个序列，把这个序列从左往右地往前做，对第![](http://latex.codecogs.com/svg.latex?t)个词会计算一个输出![](http://latex.codecogs.com/svg.latex?h_t)（也叫隐藏状态），由第![](http://latex.codecogs.com/svg.latex?t)个词本身和![](http://latex.codecogs.com/svg.latex?h_{t-1})共同决定的  
> 问题：无法并行计算，下一个输出极度依赖上一步结果，计算上性能差，内存开销大  
最近的工作通过 factorization 分解 tricks 和 conditional computation 并行化来提升计算效率，但sequential computation的问题本质依然存在

attention在RNN上的应用：怎么把encoder更有效地传递给decoder  
Transformer不再使用传统的循环神经层，而是纯注意力机制  

### 相关工作  
CNN卷积块的感受野很小，要和离得远的数据建模的话需要很多次卷积，故CNN对比较长的序列难以建模。而使用注意力机制的话，每次能看到所有的像素数据，一层就能看到整个序列  
CNN比较好的地方是可以做多个输出通道，即它可以识别多种不一样的模式。故Transformer提出了multi-head attention模拟CNN的多个输出通道  
**self-attention**是一种将单个序列的不同位置相关联的注意力机制，以计算序列的表示，在阅读理解、抽象摘要、文本蕴涵等都有应用  
**创新点：** Transformer 是第一个完全依赖自注意力来计算其输入和输出表示而不使用序列对齐 RNN 或卷积的转导模型  

### 模型架构  
现有的序列模型里，比较好的是encoder-decoder架构  
> encoder将一个长为n的输入（如句子）：![](http://latex.codecogs.com/svg.latex?x_1,x_2,...,x_n)映射成![](http://latex.codecogs.com/svg.latex?Z=z_1,z_2,...,z_n)，输入![](http://latex.codecogs.com/svg.latex?x_t)对应机器学习可以理解的向量![](http://latex.codecogs.com/svg.latex?z_t) 
decoder拿到encoder的输出，会生成一个长为m的序列![](http://latex.codecogs.com/svg.latex?y_1,y_2,...,y_m)，n和m不一样长，编码时可以一次性给你，解码时只能一个个生成（auto-regressive模型）  

Transformer使用了encoder-decoder架构，具体来说是将一些self-attention，point-wise，fully connection堆在一起的  
![Transformer](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/Transformer.jpg)  

### Encoder and Decoder Stacks  
**Input Embedding：** 输入经过一个 Embedding层, 一个词进来之后表示成一个向量  
得到的向量值会和 Positional Encoding相加  
**Encoder：** 重复六个layers，每个layers会有两个sub-layers
>multi-head、self-attention 
>    
> position-wise、fully connection、feed-forward network（就是MLP）  

![layers](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/layers.png)

对每个子层使用residual connection（残差连接）  
最后使用layer normalization  
子层公式：![](http://latex.codecogs.com/svg.latex?Layer Norm(x+Sub-layer(x)))  
把每层输出维度变成512（固定了），调参也就调一个参数就行了，另一个参数是复制多少块N  
**Decoder：** 由N=6个层构成，与Encoder不一样的地方是它有第三个sub-layer，即Masked Multi-Head Attention  
>![decoder](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/decoder.jpg)  

对每个子层使用residual connection（残差连接）  
最后使用layer normalization   
**输出：** decoder的输出进入一个 Linear 层，做一个 softmax，得到输出  
Linear + softmax: 一个标准的神经网络的做法  
**总结：** Transformer 是一个比较标准的 encoder - decoder 架构。  
>区别：encoder、decoder 内部结构不同，encoder 的输出 如何作为 decoder 的输入有一些不一样  
    
### Attention的每个子层具体怎么定义的  
注意力函数是将一个query和一系列key-value对映射成一个输出的函数，所有的query，keys，values和output都是向量  
output是values的一个加权和，故output和values的维度是一样的  
权重value是对应的key和query的相似度算来的  
相似度（compatibility function）不同的注意力机制有不同的算法  
    
### Scaled Dot-Product Attention（Transformer自己用到的注意力机制）  
queries和keys等长，都等于![](http://latex.codecogs.com/svg.latex?d_k)，values长为![](http://latex.codecogs.com/svg.latex?d_v)  
具体计算：对每个query和key做内积，作为相似度  
如果两个向量的nove是一样的，内积越大，相似度越高（等于0，向量正交，没有相似度）  
再除以![](http://latex.codecogs.com/svg.latex?\sqrt{d_k})，再用softmax来得到权重（得到n个非负的，加起来和为1的权重，再作用到value上就得到输出了）  
原因：防止softmax函数的梯度消失  
queries可以写成矩阵Q  
>![queries](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/queries.jpg)

Scaled Dot-Product Attention和别的注意力机制的区别
>additive：加性注意力机制，可以处理queries和keys不等长的情况  
dot-product（multi-plicative）点积的注意力机制

（点乘机制会更高效）  
>当![](http://latex.codecogs.com/svg.latex?d_k)不大时，除不除都无所谓  
当![](http://latex.codecogs.com/svg.latex?d_k)比较大，两个向量长度都很长，做点积的时候值会比较大或比较小，会造成偏激的结果，向两端靠拢，这时梯度会很小，跑不动  

对于![](http://latex.codecogs.com/svg.latex?t)时刻的![](http://latex.codecogs.com/svg.latex?q_t)，应该只去看![](http://latex.codecogs.com/svg.latex?k_1,k_2,...,k_{k-1})，而不去看![](http://latex.codecogs.com/svg.latex?k_t)和![](http://latex.codecogs.com/svg.latex?k_t)以后的东西，因为当前时刻还没有在注意力机制中，![](http://latex.codecogs.com/svg.latex?k_t)会对所有keys全部做运算，不用到后面的东西就行了  
**Mask：** 对于![](http://latex.codecogs.com/svg.latex?q_t)和![](http://latex.codecogs.com/svg.latex?k_t)之后计算的值换成一个非常大的负数，在softmax后会变成0
![ScaledDot-ProductAttention](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/ScaledDot-ProductAttention.jpg)  

### Multi-Head Attention  
**思路：** 与其做一个单个的注意力函数，不如把整个queries，keys，values投影到低维，投影![](http://latex.codecogs.com/svg.latex?h)次，然后再做![](http://latex.codecogs.com/svg.latex?h)次的注意力函数  
把每个函数输出并在一起，再投影回来会得到最终的输出
![Multi-HeadAttention](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/Multi-HeadAttention.png)
为了实现不一样的模式，会使用不一样的计算相似度的办法  
给h次机会，希望投影的时候能学到不一样的投影方法，使得在投影进去的度量空间里面能够去匹配不同模式需要的相似函数，最后并一起再做投影  
![](http://latex.codecogs.com/svg.latex?MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O\space, where\space head_i=Attention(QW_i^Q,KW_i^K,VW_i^V))  
**输出：** 不同的头输出concat起来，再投影到$W^O$里面  
对每个head：把Q，K，V通过不同的、可以学习的W投影到低维上面，再做注意力机制  
**实际上：** h=8（即8个头）投影的是输出的维度除以h（$d_k=d_v=\frac{d_{model}}{h}$即$\frac{512}{8}=64$）  

### 在Transformer中如何使用注意力机制  
![applyAttention](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/applyAttention.jpg)  

### Position-wise Feed-Forward Networks（就是一个MLP）  
把MLP对每个词都走一次，对每个词作同样的MLP  
![FFN.jpg](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/FFN.jpg)  
简单的实现思路案例  
![attention](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/attention.jpg)  

### Embedding层和Softmax层  
**Embedding：** 将输入的一个词映射成为一个长为d的向量来表示整个词（此处d=512）  
encoder和decoder的输入都有一个Embedding，在Softmax前也有一个Embedding  
上述三个Embedding共享权重，训练起来更简单  
将权重乘以$\sqrt{d_{model}}$即$\sqrt{512}$，学习Embedding 的时候，会把每一个向量的$L_2Norm$学的比较小  
维度大的话，学到的一些权重值就会变小，但之后还需要加上Positional Encoding（不会随着维度的增加而变化）   

### Position Encoding  
Attention输出是不会有时序信息的  
所以要在Attention的输入里面加入时序信息（如把位置i加入输入）  
>$PE(pos,2i)=\sin (pos/10000^{2i/d_{model}})$  
>$PE(pos,2i+1)=\cos (pos/10000^{2i/d_{model}})$  

用长为512的向量来表示一个数，用周期不一样的sin和cos值算出来  
和Embedding相加，就可以完成把时序信息加入输入的做法
![PositionEncoding](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/PositionEncoding.jpg)  

### Self-Attention为什么好  
相对于循环层和卷积层，用自注意力机制更好
![whyAttention](https://github.com/sunxingyui5/Transformer-ReadingNotes/blob/main/img/whyAttention.jpg)
Attention对模型的假设更少，导致需要很多的数据，模型才能训练出来  

### Optimizer训练器  
使用Adam Optimizer，学习率计算方法为：  
$lrate=d_{model}^{-0.5}\cdot {\min (step\_num^{-0.5},step\_num\cdot warmup\_steps^{-1.5})}$    

### 正则化  
**Residual Dropout：** 对每个子层的输出上，在进入残差连接之前使用了一个dropout  
$P_{drop}=0.1$（即给10%的元素值权重$\times 0$）  
**Label Smoothing：** 用Softmax学习一个东西时，标号正确的是1，错误的是0  
$\epsilon_{ls}=0.1$：表示对于正确的值，只要求Softmax输出值为0.1，这样会使不确信度增加
