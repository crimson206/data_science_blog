# Attention Calculation

In the following section, we'll explore the core concept of attention calculation in Transformers, an integral mechanism that empowers them to capture relationships within sequences.

Discover how Queries, Keys, and Values interact to create this impactful feature.

## Contents

- [Query, Key and Value](#1-initializing-query-key-and-value)
- [Attention Score](#2-attention-score)
- [Attention Probability](#3-attention-probability)
- [Output](#4-output)
- [Mask](#5-masked-attention-probabilities)
- [Fuction : calculate_attention](#6-function-for-attention-computation)
- [Setup (Run this first to practice with codes)](#7setup-run-this-first-to-practice-with-codes)

## 1. Initializing Query, Key, and Value

In Transformers, we deal with three main components: Query, Key, and Value.

All three share the shape 
{{< katex >}}
(n_{\text{batch}}, n_{\text{sequence}}, n_{\text{embed}})
{{< /katex >}}.


Consider embeddings as defining features for each sequence.

While random weights won't yield insightful outputs, remember:

- Q, K, and V arise from a "single input" but via "distinct weights".
- Through weight optimization, Transformers refine and produce significant outputs.


```python
np.random.seed(42)
data = np.random.randn(1, 10, 4)  # 1 batch, sequence length of 10, and embedding dimension of 4

W_q = np.random.randn(data.shape[-1], 4)  # Weight for projecting data to Query
W_k = np.random.randn(data.shape[-1], 4)  # Weight for projecting data to Key
W_v = np.random.randn(data.shape[-1], 4)  # Weight for projecting data to Value

query = np.matmul(data, W_q)
key = np.matmul(data, W_k)
value = np.matmul(data, W_v)

# Similarity injection.
query[0][2] = [2, -2, 2, -2]
key[0][5] = [2, -2, 2, -2]

d_k = key.shape[-1]
attention_score = np.matmul(query, np.transpose(key, (0, 2, 1))) # Q x K^T, (n_batch, seq_len, seq_len)

# a precaution to handle the potential problem of having large dot products
attention_score = attention_score / np.sqrt(d_k)
```


---
## 2. Attention Score

Each sequence is represented by an embedding vector. 

The attention score between any sequence pair from the Query and Key is derived from their dot product. This means both similarity and magnitude play roles.

Note the high score at our point of observation.


```python
draw_mat_multi(
    query[0],
    key[0].T,
    attention_score[0],
    ["Fearues", "Sequences", "Query"],
    ["Sequences", "Fearues", "Key.T"],
    ["Key Sequences", "Query Sequences", "Attention Score"],
    pairs=np.array([[2, 5], [2, 3]]),
    figsize=[5,5],
    width_ratios=[2,5],
    height_ratios=[2,5],
    s=50,
)
```


    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/calculate_attention_files/calculate_attention_8_0.png" alt="Attention Score" caption="" >}}



---
## 3. Attention Probability

Using the softmax function, attention scores are normalized across key sequences for each query.

Recall: Q and K originate from the same input.

Attention probabilities depict the unique influence each sequence receives from other sequences.



```python
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp_x

attention_prob = np.apply_along_axis(softmax, -1, attention_score)

draw_mat_horizon_transfer(
    attention_score[0],
    attention_prob[0],
    info_a=["Key Sequences", "Query Sequences", "Attention Score"],
    info_b=["Key Sequences", "", "Attention Probabilities"],
)
```


    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/calculate_attention_files/calculate_attention_10_0.png" alt="Attention Probability" caption="" >}}




---
## 4. Output

Q and K dictate the information flow proportion, while V carries the information itself. 

Every feature_i in a sequence is a linear blend of feature_i's from all sequences.



```python
draw_mat_multi(
    attention_prob[0],
    value[0],
    output[0],
    ["Key Sequences", "Query Sequences", "Attention Probabilities"],
    ["Fearues", "Sequences", "Value"],
    ["Sequences","Fearues",  "Output"],
    pairs=np.array([[2, 3], [2, 3]]),
    figsize=[5,5],
    width_ratios=[5,2],
    height_ratios=[1,1],
)

```


    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/calculate_attention_files/calculate_attention_12_0.png" alt="Output" caption="" >}}



---
## 5. Masked Attention Probabilities

To make layers prioritize neighboring word contexts, use the following mask. 

Of course, masks can be tailored as needed.



```python
seq_len = 10
mask = np.zeros((seq_len, seq_len))

for i in range(seq_len):
    mask[i, max(0, i-1):min(seq_len, i+2)] = 1

# Expand dimensions for batch size
mask = np.expand_dims(mask, 0)

sns.heatmap(mask[0], cmap='RdBu', center=0)
plt.show()
```


{{< centerfigure src="/data_science_blog/images/nlp/transformers/calculate_attention_files/calculate_attention_14_0.png" alt="Mask" caption="" >}}



---
We computed the probabilities again using the masked scores.
Note that, the high probability in the point (2,5) disappeared.


```python
masked_attention_score = np.where(mask == 0, -1e9, attention_score)
masked_attention_prob = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1), -1, masked_attention_score)

draw_mat_horizon_transfer(
    masked_attention_score[0],
    masked_attention_prob[0],
    info_a=["Key Sequences", "Query Sequences", "Attention Score"],
    info_b=["Key Sequences", "", "Attention Probabilities"],
)
```


{{< centerfigure src="/data_science_blog/images/nlp/transformers/calculate_attention_files/calculate_attention_16_0.png" alt="Masked Attention Probabilities" caption="" >}}



---
## 6. Function for Attention computation


```python
def calculate_attention(query, key, value, mask=None):
    # query, key, value: (n_batch, seq_len, d_k)
    # mask: (n_batch, seq_len, seq_len)
    
    d_k = key.shape[-1]
    attention_score = np.matmul(query, np.transpose(key, (0, 2, 1))) # Q x K^T, (n_batch, seq_len, seq_len)
    attention_score = attention_score / np.sqrt(d_k)
    
    if mask is not None:
        attention_score = np.where(mask == 0, -1e9, attention_score)

    attention_prob = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1), -1, attention_score)
    out = np.matmul(attention_prob, value) # (n_batch, seq_len, d_k)

    return out
```

## 7.Setup (Run this first to practice with codes)

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
```


```python
def draw_mat_multi(mat_a, mat_b, mat_ab, info_a=None, info_b=None, info_ab=None, pairs=np.array([[0, 0],[1, 1]]), figsize=(5, 5), width_ratios=[1, 1], height_ratios=[1, 1], linewidth=2, s=100):
    # Adjusting pairs for visualization
        
    if pairs is not None:
        pairs = pairs + 0.5

    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios)

    # Attention Probabilities visualization
    ax0 = plt.subplot(gs[1, 0])
    sns.heatmap(mat_a, cmap='RdBu', center=0, cbar=True)
    if pairs is not None:
        ax0.axhline(y=pairs[0][0], color='blue', linewidth=linewidth)
        ax0.axhline(y=pairs[1][0], color='red', linewidth=linewidth)
    if info_a is not None:
        ax0.set_xlabel(info_a[0])
        ax0.set_ylabel(info_a[1])
        ax0.set_title(info_a[2])

    # Value matrix visualization
    ax1 = plt.subplot(gs[0, 1])
    sns.heatmap(mat_b, cmap='RdBu', center=0, cbar=True)
    if pairs is not None:
        ax1.axvline(x=pairs[0][1], color='blue', linewidth=linewidth)
        ax1.axvline(x=pairs[1][1], color='red', linewidth=linewidth)
    if info_b is not None:
        ax1.set_xlabel(info_b[0])
        ax1.set_ylabel(info_b[1])
        ax1.set_title(info_b[2])

    # Output of attention mechanism visualization
    ax2 = plt.subplot(gs[1, 1])
    sns.heatmap(mat_ab, cmap='RdBu', center=0, cbar=True)
    if pairs is not None:
        ax2.scatter(pairs[0][1], pairs[0][0], color='blue', s=s)
        ax2.scatter(pairs[1][1], pairs[1][0], color='red', s=s)
    if info_ab is not None:
        ax2.set_xlabel(info_ab[0])
        ax2.set_ylabel(info_ab[1])
        ax2.set_title(info_ab[2])

    plt.tight_layout()
    plt.show()
```


```python
def draw_mat_horizon_transfer(mat_a, mat_b, info_a, info_b):
    # Create a gridspec for matrix multiplication visualization
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Attention Probabilities visualization
    ax0 = plt.subplot(gs[0, 0])
    sns.heatmap(mat_a, cmap='RdBu', center=0, cbar=True)
    ax0.axhline(y=2.5, color='blue', linewidth=2)
    if info_a is not None:
        ax0.set_xlabel(info_a[0])
        ax0.set_ylabel(info_a[1])
        ax0.set_title(info_a[2])

    # Value matrix visualization
    ax1 = plt.subplot(gs[0, 1])
    sns.heatmap(mat_b, cmap='RdBu', center=0, cbar=True)
    ax1.axhline(y=2.5, color='blue', linewidth=2)
    if info_b is not None:
        ax1.set_xlabel(info_b[0])
        ax1.set_ylabel(info_b[1])
        ax1.set_title(info_b[2])

```