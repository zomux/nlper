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

Wait, there is a problem, the model now only has the output $x$, but no input is provided. For example, it $x$ is an image, then we are basically predicting all the pixels in the image. So are we going to create a model that maps nothing to $x$ ? Hmm, we have no idea on how to compute such a model and it's log-likelihood.

But, the good thing is that we know how to compute a lower-bound of the log-likelihood by introducing another variable $z$. We call it a latent variable, and the probability of $x$ depends on $z$ as illustrated in the diagram. 


![variables](https://i.imgur.com/A8nKrSp.png)

When we translate this assumption into equations, it will be

$$p(x,z) = p(x|z) p(z).$$

Simple, right? Then, let's do some surgeries to the log-likelihood. We first marginalize w.r.t. $z$:

$$\log p(x) = \log \int p(x,z) dz.$$

This is easy to understand. Let's say our mail is $x=\text{``Amazing discount"}$, the latent variable only has two cases: $z=\text{``spam"}$ and $z=\text{``not spam"}$. So the probability of this mail is just

$$ p(x=\text{``Amazing discount"}) \\
= p(x=\text{``Amazing discount"}, z=\text{``spam"})  \\
   +p(x=\text{``Amazing discount"}, z=\text{``not spam"})$$

Next, we introduce a Q distribution, which makes the equation:

$$\log \int p(x,z) dz = \log \int q(z|x) \frac{p(x,z)}{q(z|x)} dz.$$

The template $\int q(z|x) ... dz$ is actually an expectation $\mathbb{E}_{z \sim q(z|x)}[...]$. So the equation can be written in

$$\log p(x) = \log \mathbb{E}_{z \sim q(z|x)}[\frac{p(x,z)}{q(z|x)}].$$

Hmm, well, we still don't know how to compute this equation. Wait, can we use Jensen's inequality here? Remember that Jensen's inequality tells us

$$ \log \mathbb{E}[...]  \ge \mathbb{E}[\log  ...].$$

The reason is because the logarithm is a convex function. And now we find a lower-bound of the log-likelihood:

$$\log p(x) \ge \mathbb{E}_{z \sim q(z|x)}[ \log \frac{p(x,z)}{q(z|x)}].$$

Remember our assumption $p(x,z) = p(x|z)p(z)$, we just plug it into the equation to make it
$$\log p(x) \ge \mathbb{E}_{z \sim q(z|x)}[ \log \frac{p(x|z)p(z)}{q(z|x)}].$$

Oh, multiplication and division in logarithm, let's decompose them:

$$\log p(x) \ge \mathbb{E}_{z \sim q(z|x)}[ \log p(x|z) + \log p(z) - \log q(z|x)].$$

OMG, we just find that two probabilistic distributions $q(z|x)$ and $p(x|z)$ are mapping something to something just like our discriminative model, and certainly, we can create such models. $p(z)$ is just the prior distribution of the latent variable, let just set it to be a standard Gaussian $p(z) = N(0,1)$.

Cool. Let's further clean up the equation:

$$\log p(x) \ge \mathbb{E}_{z \sim q(z|x)}[ \log p(x|z)] + \mathbb{E}_{z \sim q(z|x)}[  \log p(z) - \log q(z|x)].$$

The second half is what we call [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), or in short, KL divergence, or in short, KL. It is also called relative entropy. Therefore, we usually write the lower bound equation as

$$\log p(x) \ge \mathbb{E}_{z \sim q(z|x)}[ \log p(x|z)] - \mathrm{KL}(q(z|x) | p(z)).$$

Anyway, the good news is that when both $q(z|x)$ and $p(z)$ are Gaussians, the KL divergence can be analytically solved. If you are interested, take a look at this stackexchange post: https://stats.stackexchange.com/a/7449 .

So, to compute the lower-bound, we just need to sample a $z$ from $q(z|x)$,  compute the left part $\mathbb{E}_{z \sim q(z|x)}[ \log p(x|z)]$ and the right part, which is KL divergence. Because this conclusion is so important, we give the equation a name, we call it evidence lower-bound, or in short, ELBO. Suppose the two conditional distributions are parameterized by $\theta$ and $\phi$, then ELBO is defined as

$$\mathrm{ELBO}(x;\theta, \phi) = \mathbb{E}_{z \sim q(z|x;\phi)}[ \log p(x|z;\theta)] - \mathrm{KL}(q(z|x;\phi) | p(z)).$$

Instead of maximizing the log-likelihood, we maximize the lower-bound:

$$\mathop{\mathrm{argmax}}\limits_{\theta, \phi} \mathrm{ELBO}(x;\theta, \phi) $$

### Understanding ELBO

The left part of the ELBO $\mathbb{E}_{z \sim q(z|x;\phi)}[ \log p(x|z;\theta)]$  basically says that if we sample a $z$ from $q(z|x)$, can we really reconstruct the original $x$ with $p(x|z)$ ? So, this is a reconstruction objective. Of course, when q(z|x) is a complicated distribution, then $z$ can carry more information,

| Input  | Output (label) |
|--|--|
| Image  | Whether it's a animal (1 or 0)  |
| Email | Whether it's a spam (1 or 0) |
| A sentence | Next word in the sentence (a word) |
| Voice record | Text (word sequence) |
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc4NTA5OTkyLC01MjU2MzYzMTcsLTEwMj
gwOTkwODZdfQ==
-->