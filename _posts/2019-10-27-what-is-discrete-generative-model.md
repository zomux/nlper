---
layout: post
title: What is A Discrete Generative Model and How to Learn It
---

### Discriminative model

First, may be we are more familiar with a discriminative model, or we usually call it a classification model. Here are some examples of classification models:

| Input  | Output (label) |
|--|--|
| Image  | Whether it's a animal (1 or 0)  |
| Email | Whether it's a spam (1 or 0) |
| A sentence | Next word in the sentence (a word) |
| Voice record | Text (word sequence) |

In machine learning, a discriminative model does not directly give the label, but the probability of the label, and that is $p(y|x)$ . $x$ is the input data and $y$ is the label. In the case of spam detection, we can say the email is a spam when $p(y=1|x) > 0.5$. We can also be more conservative, and only say a email is spam when $p(y=1|x) > 0.9$. So you see, we have a recall and precision trade-off.

The training objective given a datapoint $(x_d,y_d)$ is that we want to maximize the probability of $y_d$ when we observe $x_d$, or here we call it likelihood. It can be written in

$$ \mathop{\mathrm{argmax}}\limits_\theta \log p(y=y_d|x=x_d;\theta) $$

Here, we use the logarithm becu

### Generative model

![variables](https://i.imgur.com/A8nKrSp.png)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMDIzODY0ODcsLTUyNTYzNjMxNywtMT
AyODA5OTA4Nl19
-->