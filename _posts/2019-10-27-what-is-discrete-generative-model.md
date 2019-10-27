---
layout: post
title: What is a discrete generative model and how to learn it
---

### Discriminative model

First, may be we are more familiar with a discriminative model, or we usually call it a classification model. Here are some examples of classification models:

| Input  | Output (label) |
|--|--|
| Image  | Whether it's a animal (1 or 0)  |
| Email | Whether it's a spam (1 or 0) |
| A sentence | Next word in the sentence (a word) |
| Voice record | Text (word sequence) |

In machine learning, a discriminative model does not directly give the label, but the probability of the label, and that is $p(y|x)$ . $x$ is the input data and $y$ is the label. In the case of spam detection, we can say the email is a spam when $p(y=1|x) > 0.5$. 



### Generative model

![variables](https://imgur.com/7s9PNS7.png)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc2MDUzOTQ3MSwtMTAyODA5OTA4Nl19
-->