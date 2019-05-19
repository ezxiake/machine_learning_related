# Machine learning related programming code
> 不同代码放置在不同的 Branch 中

## 1. Andrew Ng - Machine Learning 课程代码（Python3）

跳转 ：[Branch ： Machine-Learning-AndrewNg-Python3](https://github.com/ezxiake/machine_learning_related/tree/Machine-Learning-AndrewNg-Python3)

* 代码用Python3实现，是从Octave版本的作业翻译过来的，最大化的保持了和Octave版的一致（包括代码，以及画出来的图片风格）。
* 资源中包括了课件，练习说明的pdf，源代码，数据集信息等。
* 代码都是在 jupyter notebook 中实现的。 运行代码可以预装一个 jupyter notebook local version。
* 运行时，如缺少什么包，可以 pip3 install xxx 去安装。
* 所有代码，都极少的调用封装后的高级包（基本上就是Numpy 和 Matplotlib），如果调用也是作为额外项，了解用。为了学习目的，能自己敲的代码都自己敲了。
* 有些画图的部分做了调整，全部都画在了一起，方便对比了解，例如方差与偏差那个练习。
* 这8个练习中，只有svm的第二部分，即`垃圾邮件识别`有些不一样，其中未实现对词干提取（porterStemmer）的那个方法，Andrew 的Octave的代码中，我对词干提取的功能也进行了debugging，发现那个方法好像在octave中也未起作用（未深入去探索），它 Applies the Porter Stemming algorithm as presented in the following paper: Porter, 1980, An algorithm for suffix stripping, Program, Vol. 14, no. 3, pp 130-137 [Original code modeled after the C version provided](http://www.tartarus.org/~martin/PorterStemmer/c.txt)。所以此处忽略了对词干的提取步骤。SVM作业的其他功能均已实现，但是结尾还是有一些不对，就是在`垃圾邮件分类`练习的最后一步，输入自己的email，然后转化成features，并进行预测，发现预测的结果都是1，对垃圾邮件分类感兴趣的你可以深入了解并debugging一下。


