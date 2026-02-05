# Rust Tensor Einsum

A minimal, standalone Rust implementation of Einstein Summation (einsum) for tensor operations, designed for educational and experimental purposes.

This library provides a `Tensor` struct and a generalized `einsum` function that allows for concise expression of tensor contractions, mimicking the behavior of NumPy's `einsum`.

## Features
- purely safe Rust (std only, no external dependencies).
- Support for arbitrary rank tensors.
- String-based einsum notation (e.g., `"ij,jk->ik"`).
- Helper methods for tensor creation and CSV export.

## Usage

You can run the included demo to see the tensor operations in action, including examples of:
1.  **Transitivity**: Matrix multiplication representing logical value propagation.
2.  **Syllogism**: Modeling logical cuts as tensor contractions.
3.  **Contextual Inference**: Handling multi-dimensional contexts in reasoning.

```bash
cargo run --example demo
```

## Example Code
```rust
use rust_tensor_einsum::Tensor;

let a = Tensor::from_vec2(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
let b = Tensor::from_vec2(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);

// Matrix multiplication: C_ik = sum_j A_ij * B_jk
let c = Tensor::einsum("ij,jk->ik", &[&a, &b]);
c.print_tensor();
```

## License
MIT
