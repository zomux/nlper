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

Here, we use the logarithm because given multiple datapoints, we can do a summarization on the log-likelihoods instead of production. $\theta$ is the parameter of our model, it may be the parameters of a neural network.

### Generative model

Now as we know the discriminative model and how to train it. We move on to the generative model. In generative model, instead of the output label, we want to know the probability of the input data $p(x)$. Is this awkward? But consider the email spam detection example, if we know a distribution that generates spamming emails $p_{\mathrm{spam}}(x;\theta)$. Please note here $\theta$ is still the parameters of this probability model. Then,  given a new email $x_d$, we can just plug in the data into the probability   $p_{\mathrm{spam}}(x;\theta)$ and say the mail is a spam when  $p_{\mathrm{spam}}(x=x_d;\theta) > 0.5$.

So, because the probability distribution $p(x)$ try to capture the generation process of $x$, we call it a generative model. Then, similar to the discriminative case, we train the generative model by maximizing the log-likelihood:

$$\mathop{\mathrm{argmax}}\limits_\theta \log p(x=x_d;\theta) $$

Wait, there is a problem, the model now only has the output $x$, but no input is provided.  So are we going to create something to map nothing to $x$ ? 

![variables](https://i.imgur.com/A8nKrSp.png)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNzg0NDYzMCwtNTI1NjM2MzE3LC0xMD
I4MDk5MDg2XX0=
-->