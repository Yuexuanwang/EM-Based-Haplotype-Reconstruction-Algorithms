# EM-Based-Haplotype-Reconstruction-Algorithms
This repository provides code for post-processing haplotype reconstruction algorithms based on the E-M framework, as detailed in "Reconstructing Haplotype Structure and Frequencies from Reads Data" (Wang, 2024). It includes a coloring-based and residual-based method to enhance accuracy and computational efficiency for short-read sequencing data.
Here is the updated README to include the details about `Color_based_code.py`, `Largest_Residual_code.py`, `run.py`, and `backends`:

## Features

### Algorithms:
1. **Color-Based Algorithm (`Color_based_code.py`)**  
   Implements a graph-coloring technique during the M-step to optimize the candidate haplotype pool for updating the S matrix. Suitable for exploring a wide candidate space but computationally intensive for large datasets.
   
2. **Largest Residual Algorithm (`Largest_Residual_code.py`)**  
   Targets rows with the highest residual errors during the M-step to iteratively refine the S matrix. Designed for better efficiency and performance in practical scenarios.

### Supporting Scripts:
- **`run.py`**  
  A script to define input data and user-defined parameters for running the algorithms. It uses parallelization for running multiple cases efficiently.
  
- **`backends`**  
  A package that facilitates parallel computation, ensuring scalability and faster processing of large datasets.


## Usage

### Running the Algorithms
1. Use `run.py` to define input data paths and parameters for the algorithms.
2. Run the desired algorithm:
   - **Color-Based Algorithm**:
     ```bash
     python Color_based_code.py --AF allele_frequency_matrix.txt --Data data.pkl --S_es initial_haplotype_structure.txt --W_es initial_haplotype_weights.txt --M_iter 3
     ```
   - **Largest Residual Algorithm**:
     ```bash
     python Largest_Residual_code.py --AF allele_frequency_matrix.txt --Data data.pkl --S_es initial_haplotype_structure.txt --W_es initial_haplotype_weights.txt --M_iter 3 --S_updates 5
     ```

### Parameters
- `--AF`: Path to the allele frequency matrix.
- `--Data`: Path to the serialized reads data file (`pickle` format).
- `--S_es`: Path to the initial haplotype structure matrix.
- `--W_es`: Path to the initial haplotype weight matrix.
- `--M_iter`: Number of iterations for the E-M algorithm (default: 3).
- `--S_updates`: Number of rows updated in each iteration (only for the residual-based algorithm default:5).

### Parallel Execution
To run multiple cases with parallelization, use the `run.py` script:
```bash
python run.py
```

## Output
- Results are saved in `.npz` format, including:
  - Updated haplotype structure matrix (`S`)
  - Updated haplotype weight matrix (`W`)
