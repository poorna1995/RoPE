import torch


def compute_freq(dim: int, seq_len: int, theta):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'seq_len'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        seq_len (int): seq_len index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.



    """
    # Each group contains two components of an embedding,
    # calculate the corresponding rotation angle theta_i for each group.
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    print("frequwncy", freqs)
    # Generate token sequence index m = [0, 1, ..., sequence_length - 1]
    t = torch.arange(seq_len, device=freqs.device)  # type: ignore
    # Calculate m * theta_i
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    print("frequwncy of comple pane", freq_complex)
    return freq_complex


def reshape_for_broadcast(freq_complex: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to match the shape of the target tensor 'x'
    for broadcasting during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    # Reshape `freqs_cis` for broadcasting to match x's shape
    return freq_complex.view(*([1] * (ndim - 2)), *freq_complex.shape)




def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freq_complex: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freq_complex'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freq_complex (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    # Reshape and convert xq and xk to complex number
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freq_complex = reshape_for_broadcast(freq_complex, xq_)
    # Apply rotation operation, and then convert the result back to real numbers.
    xq_out = torch.view_as_real(xq_ * freq_complex).flatten(3)
    xk_out = torch.view_as_real(xk_ * freq_complex).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)









dim = 64  # Embedding dimension
seq_len = 10  # Sequence length
theta = 10000.0  # Frequency scaling factor


# Define dummy query and key tensors
batch_size = 2
num_heads = 4
seq_len = seq_len

# Queries and keys with dimensions [batch_size, num_heads, seq_len, dim]
xq = torch.randn(batch_size, num_heads, seq_len, dim)
xk = torch.randn(batch_size, num_heads, seq_len, dim)
freqs_cis = compute_freq(dim=dim, seq_len=seq_len, theta=theta)

# Apply rotary embeddings
xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

# Print the results
print("Input Query Tensor Shape (xq):", xq.shape)
print("Input Key Tensor Shape (xk):", xk.shape)
print("Precomputed Frequency Tensor (freqs_cis):", freqs_cis.shape)

