# Positional Encoding

Positional encoding is a technique used in Transformers to provide sequence-awareness. 

By adding unique patterns to token embeddings based on their position in a sequence, the Transformer model can discern token order.



## Contents

- [PositionalEncoding](#0-layer-implementation)
- [ArbitraryInput](#1-arbitrary-input)
- [PosionalEncoding](#2-visualization-of-encoded-information-and-output)
- [PatternAnalysis](#3-pattern-analysis)
- [PositionPrediction](#4-position-information-recovery)
- [Post-Processing?](#5-there-is-the-post-processing)
- [Summary](#6-summary)


---
## 0. Layer Implementation
In this article, we will the class,

```python
import numpy as np
import math
import matplotlib.pyplot as plt

class PositionalEncoding:
    def __init__(self, d_embed, max_len=400):
        encoding = np.zeros((max_len, d_embed))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_embed, 2) * -(np.log(10000.0) / d_embed))
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term)
        self.encoding = encoding.reshape(1, max_len, d_embed)
        self._pos_embed = None

    def forward(self, x):
        seq_len = x.shape[1]
        self._pos_embed = self.encoding[:, :seq_len, :]
        out = x + self._pos_embed
        return out
```

---
## 1. Arbitrary input

Lets prepare the input data and visualize.


```python
n_batch = 2
n_words = 300
d_embed = 20

pos_encoding = PositionalEncoding(d_embed)
np.random.seed(42)
x = np.random.randn(n_batch, n_words, d_embed) / 2

# Plotting the positionally encoded input
plt.figure(figsize=(5, 3))
plt.pcolormesh(x[0].squeeze().T, cmap='viridis')
plt.title('Input')
plt.colorbar()
plt.show()
```

{{< centerfigure src="/data_science_blog/images/nlp/transformers/position_embedding_files/position_embedding_3_0.png" alt="" caption="" >}}


## 2. Visualization of encoded information and output

back up the input data, and encode the positional information


```python
x_backup = x
output = pos_encoding.forward(x)
pos_encoding_array = pos_encoding._pos_embed[0]

# Plotting the positional encoding
plt.figure(figsize=(5, 3))
plt.pcolormesh(pos_encoding_array.T, cmap='viridis')
plt.title('Positional Encoding')
plt.colorbar()
plt.axvline(x=50, color='red', linestyle='--')  # First vertical dotted line at position 5
plt.axvline(x=100, color='red', linestyle='--')  # Second vertical dotted line at position 10
plt.xlabel('Axis with patterns we focus on')  # Highlighting the axis of interest
plt.show()

# Plotting the positionally encoded input
plt.figure(figsize=(5, 3))
plt.pcolormesh(output[0].squeeze().T, cmap='viridis')
plt.title('Positionally Encoded output')
plt.colorbar()
plt.axvline(x=50, color='red', linestyle='--')  # First vertical dotted line at position 5
plt.axvline(x=100, color='red', linestyle='--')  # Second vertical dotted line at position 10
plt.xlabel('Axis with patterns we focus on')  # Highlighting the axis of interest
plt.show()
```


    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/position_embedding_files/position_embedding_5_0.png" alt="" caption="" >}}

{{< centerfigure src="/data_science_blog/images/nlp/transformers/position_embedding_files/position_embedding_5_1.png" alt="" caption="" >}}



---
The visualized output is from the first batch.

Please note that, 300 is the number of words.

The positional encoding gives unique patterns to each word.

Compare the patterns highlighted by the dot red lines.



---
## 3. Pattern analysis

If two pattenrs are the same, their dot product give large similarity score.

See the simplest example.

[1, 0, 1] * [1, 0, 1] = [1, 0, 1] => 2
[0, 1, 0] * [1, 0, 1] = [0, 0, 0] => 0

Using this characteristic, we predict the positions of words.


```python
pos30_predictor = pos_encoding_array[30]  # the 30th positional encoding

similarity_scores = np.dot(pos_encoding_array, pos30_predictor)

# Plotting the similarity scores
plt.figure(figsize=(5, 3))
plt.plot(similarity_scores)
plt.title('Self Similarity with 30th Position Encoding')
plt.axvline(x=30, color='red', linestyle='--')  # First vertical dotted line at position 5
plt.xlabel('Position')
plt.ylabel('Similarity Score')
plt.grid(True)
plt.show()

```


    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/position_embedding_files/position_embedding_8_0.png" alt="" caption="" >}}



---
As positional encodings are designed to be unique for each position, 

we see the highest similarity score at the 30th position (as it is compared with itself), 

and lower similarity scores for other positions.

Do the same with the encoded output


```python
similarity_scores = np.dot(output[0], pos30_predictor)

# Find the index of the highest similarity score
max_index = np.argmax(similarity_scores)

# Plotting the similarity scores
plt.figure(figsize=(5, 3))
plt.plot(similarity_scores)
plt.title('Output Similarity with 30th Position Encoding')
plt.axvline(x=30, color='red', linestyle='--', label='30th position')  # Vertical line at position 30
plt.axvline(x=max_index, color='blue', linestyle='--', label='Max similarity position')  # Vertical line at the position of highest similarity
plt.xlabel('Position')
plt.ylabel('Similarity Score')
plt.legend()  # Adding a legend to distinguish between the lines
plt.grid(True)
plt.show()
```


    
    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/position_embedding_files/position_embedding_10_0.png" alt="" caption="" >}}





---
## 4. Position information recovery

The word at position 60 was predicted to be at position 30.

How good would this prediction be in general?

For each word, we will apply all positional encoding patterns, and predict its position as the one that gives the highest similarity score.

Here's how we can do it:

Calculate the similarity scores between each word and every positional encoding pattern.
For each word, find the position that gives the highest similarity score.


```python
# Compute the similarity scores matrix
similarity_scores_matrix = np.dot(output[0], pos_encoding_array.T)

# For each word, predict its position as the one that gives the highest similarity score
predicted_positions = np.argmax(similarity_scores_matrix, axis=1)

plt.figure(figsize=(5, 3))
plt.plot(predicted_positions, 'bo-', linewidth=0.5, markersize=2)
plt.title('Predicted Positions for Each Word')
plt.xlabel('True Position')
plt.ylabel('Predicted Position')
plt.grid(True)
plt.show()

```


    
    
{{< centerfigure src="/data_science_blog/images/nlp/transformers/position_embedding_files/position_embedding_12_0.png" alt="" caption="" >}}




---
## 5. Is There a Need for Post-Processing?

Indeed. 

Deep learning models, like Transformers, inherently recognize and interpret patterns.

Through training, they will independently discern and effectively utilize these positional encodings.


---

## 6. Summary

- Positional encoding is essential for embedding sequence order into Transformers.
- Unique patterns are added to token embeddings, enabling the model to determine token positions within a sequence.
