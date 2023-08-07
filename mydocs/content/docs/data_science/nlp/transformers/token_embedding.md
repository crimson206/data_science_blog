# Token Embedding

Token embeddings provide a numerical representation for words, capturing their semantic meaning and allowing machines to process and understand text. 

This article delves into the mechanics of token embeddings, from tokenization to weight adjustments in neural networks.

## Contents

- [Tokenizing](#1tokenizing)
- [Embedding](#2embedding)
- [Forward](#3forwardpropagation)
- [Backward](#4backpropagation)
- [WrappingUp](#5-wrapping-up)


---
## 1.Tokenizing
Assume that We have vocabulary items (or tokens)


```python
token_size = 5
vocas = [f"voca{i}" for i in range(token_size)]
tokens = [i for i in range(token_size)]

print("\nvocas:\n", vocas)
print("\ntokenized_vocas:\n", tokens)
```

    
    vocas:
     ['voca0', 'voca1', 'voca2', 'voca3', 'voca4']
    
    tokenized_vocas:
     [0, 1, 2, 3, 4]
    


---
## 2.Embedding
A Embedding class is defined.
All the things will be explained. 

Skip understanding for now.


```python
import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size, d_embed, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.learning_rate = learning_rate
        # Initialize the weight matrix randomly
        np.random.seed(42)
        self.embedding_weights = np.random.randint(-2, 3, size=(vocab_size, d_embed)).astype(np.float64)
        # Save the last input for backward pass
        self.last_input = None
        self.grad_weights = None
    
    def forward(self, x):
        # Save the input for the backward pass
        self.last_input = x
        # Use the input indices to look up the corresponding vectors in the weight matrix
        return self.embedding_weights[x]

    def backward(self, gradient_output):
        # Initialize a gradient matrix for the weights, filled with zeros
        self.grad_weights = np.zeros_like(self.embedding_weights)
        # Accumulate the gradients at the positions corresponding to the input indices
        for i, index in enumerate(self.last_input):
            self.grad_weights[index] += gradient_output[i]
        # Update the weights using the calculated gradient
        self.embedding_weights -= self.learning_rate * self.grad_weights
```

---
Each of 5 token is converted into 3 numbers.

This is called "embedding".

Weights are used for embedding


```python
d_embed = 3
print("Number of tokens:", len(tokens))
print("Embedding dimention:", d_embed)

custom_embedding = TokenEmbedding(token_size, d_embed)

print("\nweights\n:", custom_embedding.embedding_weights)

print("\nvocas[0] ~ (tokenized)\ntokens[0] => (embedding)\nweights[0]=\n", custom_embedding.embedding_weights[0])
```

    Number of tokens: 5
    Embedding dimention: 3
    
    weights
    : [[ 1.  2.  0.]
     [ 2.  2. -1.]
     [ 0.  0.  0.]
     [ 2.  1.  0.]
     [ 2. -1.  1.]]
    
    vocas[0] ~ (tokenized)
    tokens[0] => (embedding)
    weights[0]=
     [1. 2. 0.]
    


---
## 3.ForwardPropagation
Embedding Sentences
Using tokens, we can write sentences.


```python
input_ids = np.array([[0, 1], [2, 2]])

print("sentence0, voca0 voca1 \nsentence1, voca2 voca2\n")
print("input_ids size:s (n_batch,seq_len)")
print("tokenized sentences:\n", input_ids)
```

    sentence0, voca0 voca1 
    sentence1, voca2 voca2
    
    input_ids size:s (n_batch,seq_len)
    tokenized sentences:
     [[0 1]
     [2 2]]
    


---
We embed the sentences.


```python
# Pass the input through the custom embedding layer
embeddings = custom_embedding.forward(input_ids)

print("\nembedded sentences=\n\n", custom_embedding.embedding_weights[input_ids])
```

    
    embedded sentences=
    
     [[[ 1.  2.  0.]
      [ 2.  2. -1.]]
    
     [[ 0.  0.  0.]
      [ 0.  0.  0.]]]
    


---
In equations, they are expressed followings :

sentence = S,\
input = X,\
weights = W,\
output = Y


{{< katex display >}}
\begin{aligned}
\text{X} &= \sum_{i} \text{s}_i \bar{e}_i = \sum_{i,j} x_{ij} \bar{e}_i \bar{e}_j^T \\
\text{y}_i &= \sum_{j,k} w_{x_{ij},k} \bar{e}_k^T \\
\text{Y} &= \sum_{i} \text{y}_i \bar{e}_i = \sum_{i,j,k} w_{x_{ij},k} \bar{e}_i \bar{e}_k^T
\end{aligned}
{{< /katex >}}


,where {{< katex >}}\bar{e}{{< /katex >}} refers to sentence axis, and {{< katex >}}\bar{e^{T}}{{< /katex >}}
refers to embedding vector axis



Check the shapes of input and output


```python
print("Input shape:", input_ids.shape, "= (n_sentences, len_sentences)")
print("Output shape:", embeddings.shape, "= (n_sentences, len_sentences, dim_embedding)")
```

    Input shape: (2, 2) = (n_sentences, len_sentences)
    Output shape: (2, 2, 3) = (n_sentences, len_sentences, dim_embedding)
    


---
## 4.BackPropagation
By Back propagation, we adjust the weights.



{{< katex >}}\Delta_{Y_{ij}}{{< /katex >}} is the error caused by {{< katex >}}X_{ij}{{< /katex >}}.\
Remember that {{< katex >}}X_{ij}{{< /katex >}} is the index of the voca index.

Simply, all the errors caused by {{< katex >}}X_{ij}{{< /katex >}} whose value is 3,\
are actually caused by {{< katex >}}W_3{{< /katex >}}.

Therefore,
$$\Delta_{W_{i}} = \sum_{j, k} \Delta_{Y_{jk}} \text{   for all j, k  such that } X_{jk} = i$$

Lets check our assumption


```python
print("Input_ids:\n", input_ids)

np.random.seed(42)
error_tensor = np.random.randint(-2, 3, size=embeddings.shape).astype(np.float64)

print("\nError:\n", error_tensor)

print(f"\nExpected dW[2] = \nerror_tensor[1, 0] + error_tensor[1, 1] =\n", error_tensor[1, 0] + error_tensor[1, 1])

custom_embedding.backward(error_tensor)

print(f"\nResult dW[2] = \n{custom_embedding.grad_weights[2]}")
```

    Input_ids:
     [[0 1]
     [2 2]]
    
    Error:
     [[[ 1.  2.  0.]
      [ 2.  2. -1.]]
    
     [[ 0.  0.  0.]
      [ 2.  1.  0.]]]
    
    Expected dW[2] = 
    error_tensor[1, 0] + error_tensor[1, 1] =
     [2. 1. 0.]
    
    Result dW[2] = 
    [2. 1. 0.]
    


---
This error means, the embedding weights must be adjusted.


---

## 5. Wrapping Up

- Token embedding effectively transforms tokens into meaningful numerical features, termed as embeddings.
- Through training and iterative adjustments, this layer refines its representations, optimizing how tokens are embedded.
- This transition from simple tokens to intricate vectors is pivotal in bridging the gap between machine computations and human language understanding.
