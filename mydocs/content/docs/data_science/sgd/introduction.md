---
title: Introduction
weight: 1
---

# Introduction

The fundamental concept behind gradient descent is to minimize a function {{< katex >}}E = E(w){{< /katex >}}. The gradient {{< katex >}}\nabla_\mathbf{W} E{{< /katex >}} indicates the direction in which {{< katex >}}E{{< /katex >}} increases the fastest at a point, and the objective is to modify the values of {{< katex >}}\mathbf{W}{{< /katex >}} iteratively to minimize {{< katex >}}E{{< /katex >}}. This is achieved by updating the values of {{< katex >}}w{{< /katex >}} as follows:


{{< katex display >}}
\mathbf{W}^{\prime}=\mathbf{W}-\eta \nabla_\mathbf{W} E 
{{< /katex >}}

Here, {{< katex >}}\eta{{< /katex >}} is the learning rate. In the subsequent sections, we will derive the equations necessary for this process, while keeping the text succinct and to the point.

## Forward Propagation

Forward propagation refers to the computation of the output of a neural network given an input {{< katex >}}\mathbf{x}{{< /katex >}}. The equations for forward propagation are:

{{< katex display >}}
\begin{aligned}
E & = \frac{1}{2}(t-y)^{2} \\
\mathbf{y} & = \mathbf{W}_n \mathbf{z}_{n-1} + \mathbf{b}_n \\
\mathbf{z}_n & = h(\mathbf{a}_n) \\
\mathbf{a}_n & = \mathbf{W}_n \mathbf{z}_{n-1} + \mathbf{b}_n \\
\mathbf{a}_1 & = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1 \\
\end{aligned}
{{< /katex >}}

We can visualize the forward propagation process as follows:

{{< centerfigure src="/data_science_blog/images/sgd/introduction/forward_propagation.jpg" alt="Forward Propagation" caption="" >}}

## Backward Propagation

Backward propagation is used to compute the gradients of the error with respect to the weights and biases in a neural network. The equations for backward propagation are:

{{< katex display >}}
\begin{aligned}
\mathbf{\Delta y} 
&= \nabla_{\mathbf{y}} E = \mathbf{t} - \mathbf{y} \\
\mathbf{\Delta a} 
&= \nabla_{\mathbf{a}} E = (\nabla_{\mathbf{y}} \mathbf{a})^\top \mathbf{\Delta y} \\
\mathbf{\Delta z}_{n-1} 
&= \mathbf{W}_{n}^{\top} \mathbf{\Delta a}_{n} \\
\mathbf{\Delta a}_{n-1} 
&= h'(\mathbf{a}_{n-1}) \odot \mathbf{\Delta z}_{n} \\
&= h'(\mathbf{a}_{n-1})\mathbf{W}_{n}^{\top} \mathbf{\Delta a}_{n}
\end{aligned}
{{< /katex >}}

where {{< katex >}}\Delta{{< /katex >}} denotes the gradient of the error with respect to a variable. The visualization of backward propagation is shown below:

{{< centerfigure src="/data_science_blog/images/sgd/introduction/backward_propagation.jpg" alt="Backward propagation" caption="" >}}

## Symbol Definitions

Table 1 lists the symbols used in this document, along with their names, alternative names, and alternative symbols.

{{< centerfigure src="/data_science_blog/images/sgd/introduction/symbolic_definition_table.jpg" alt="An elephant at sunset" caption="Table1 : Symbols, main names, alternative names, and alternative symbols" >}}