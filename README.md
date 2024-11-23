# Rotary Positional Embedding: !(Paper)[https://arxiv.org/pdf/2410.06205]

## What is RoPE?
- RoPE encodes positional information by multiplying the context embeddings (query `q` and key `k`) with rotation matrices based on their absolute positions. This approach allows the inner product of context embeddings to depend solely on their **relative positions** rather than their absolute positions.
- The key feature of RoPE is its ability to capture relative positional information in a way that preserves the context between tokens while improving scalability and efficiency.


### What is Rotation Matrix?
<img width="637" alt="Screenshot 2024-11-23 at 7 51 11 PM" src="https://github.com/user-attachments/assets/79944b6a-bb07-44ec-ac9c-307a98b940b7">

![image](https://github.com/user-attachments/assets/faa7b1b7-5d7a-483d-b60e-4a1bf7ec215c)

