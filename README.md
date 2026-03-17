# Programming-Massively-Parallel-Processors

Programming exercises based on the book *Programming Massively Parallel Processors*.

## Chapter 2: Vector Addition

This repository now includes a runnable CUDA vector addition example in:

- `chapter_2/vecAdd.cu`
- `chapter_2/vec_add.ipynb`

The program:

- Implements CPU vector addition (`vecAddH`)
- Implements GPU vector addition (`vecAddD`) using a CUDA kernel
- Runs both on vector size `10000`
- Reports timing and speedup
- Verifies correctness with max absolute error

## Run in Google Colab

1. Open `chapter_2/vec_add.ipynb` in Colab.
2. Set runtime type to `GPU`.
3. Run the notebook cells:
   - First cell writes `vecAdd.cu`
   - Second cell compiles and runs:
     - `nvcc -std=c++14 vecAdd.cu -o vecAdd`
     - `./vecAdd`
