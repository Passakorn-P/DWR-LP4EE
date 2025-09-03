# Dynamically Weighted Regularized Linear Programming for Effort Estimation (DWR-LP4EE)

This repository provides the Python implementation of **Dynamically Weighted Regularized Linear Programming for Effort Estimation (DWR-LP4EE)**, a novel enhancement to the **Linear Programming for Effort Estimation (LP4EE) framework**. The implementation is designed for software effort estimation (SEE) tasks, demonstrating significant improvements in estimation accuracy while maintaining comparable computational efficiency to the baseline LP4EE model. Additionally, a Python implementation of the original LP4EE is included to facilitate adoption by practitioners, eliminating the need for the cumbersome R framework that may hinder its practical application.

## Files

- `DWR_LP4EE.py`: The core implementation of the DWR-LP4EE model.
- `ORIG_LP4EE.py`: The implementation of the original LP4EE model for comparison.
- `LP4EE_Regularized.py`: A regularized version of LP4EE using the OSQP solver, which serves as a component for DWR-LP4EE.
- `example.py`: A script demonstrating how to use the models, evaluate their accuracy, and compare their prediction speed using the Maxwell dataset.

## Dependencies

The following Python libraries are required to run the code:

- pandas
- numpy
- scipy
- scikit-learn
- osqp

They all can be installed using pip:

```bash
pip install pandas numpy scipy scikit-learn osqp
```

## Dataset

The `example.py` script includes a sample of the Maxwell dataset, which is used to demonstrate the functionality of the models.

## Usage

The `example.py` script provides several functions to illustrate the use of the DWR-LP4EE and original LP4EE models. To run the examples, execute the file from your terminal:

```bash
python example.py
```

### Making a Single Prediction

The `DWR_LP4EE_example()` function shows how to predict the effort for a single test instance from the training data.

**Expected Output:**
```
Actual Effort: 7871.00, Predicted Effort: 6563.80
```

### Evaluating Accuracy

The `accuracy_example()` function performs a leave-one-out cross-validation (LOOCV) to compare the Mean absolute error (MAE) and Median absolute error (MdAE) of DWR-LP4EE and the original LP4EE models on the Maxwell dataset.

**Expected Output:**
```
DWR-LP4EE Maxwell's MAE, MdAE:  3937.92, 1741.06
ORIG-LP4EE Maxwell's MAE, MdAE: 4898.40, 3801.94
```

### Comparing Prediction Speed

The `speed_example()` function runs each model multiple times on single experiment of the Maxwell dataset to provide an average time per prediction, offering a simple performance benchmark.

**Example Output** (times may vary based on hardware; The results below obtained on a single core of a MacBook Air M1 with a Geekbench v6 score of approximately 2350):
```
DWR-LP4EE - Avg. time per prediction: 3.3383 seconds
ORIG-LP4EE - Avg. time per prediction: 0.0032 seconds
```