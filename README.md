# Rotary Positional Embedding: !(Paper)[https://arxiv.org/pdf/2410.06205]

## What is RoPE?
- RoPE encodes positional information by multiplying the context embeddings (query `q` and key `k`) with rotation matrix based on their absolute positions. This approach allows the inner product of context embeddings to depend solely on their **relative positions** rather than their absolute positions.
- The key feature of RoPE is its ability to capture relative positional information in a way that preserves the context between tokens while improving scalability and efficiency.


### What is Rotation Matrix?
<img width="637" alt="Screenshot 2024-11-23 at 7 51 11 PM" src="https://github.com/user-attachments/assets/79944b6a-bb07-44ec-ac9c-307a98b940b7">

![image](https://github.com/user-attachments/assets/faa7b1b7-5d7a-483d-b60e-4a1bf7ec215c)

<img width="813" alt="Screenshot 2024-11-23 at 7 56 05 PM" src="https://github.com/user-attachments/assets/a8609438-28b3-42ae-9219-7c40fedfd949">
<img width="996" alt="Screenshot 2024-11-23 at 7 56 27 PM" src="https://github.com/user-attachments/assets/29302398-74bf-4319-89b5-383253923ccb">
<img width="1010" alt="Screenshot 2024-11-23 at 7 56 53 PM" src="https://github.com/user-attachments/assets/81b6f994-924c-4c6b-bd2e-cf453e26a4e1">
##### Reference from: https://www.youtube.com/watch?v=SMBkImDWOyQ
