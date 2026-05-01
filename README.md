
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)

# 🙏 Thank you for "jingtt" who made the incredible library
This library is inspired by the ideas and approaches of **jingtt** The purpose of this project is **not** to compete with or replace jingtt, but to **learn from it, experiment, and explore improvements** in areas such as usability, code structure, extensibility and performance (where mainly focus).

## Acknowledgement

Credit and respect go to **jingtt** for the original inspiration. This project would not exist without the ideas and prior work from that library.
> This is an independent project and is **not affiliated with, endorsed by, or maintained by** the jingtt authors.

## Contributions
Contributions, discussions, and suggestions are welcome. If you like jingtt and want to help explore new directions, feel free to join!

# VarClusHi Optimization
## ✏️ Short Recap
This module implements the Variable Clustering (VarClus), a hierarchical dimension‑reduction algorithm that groups variables based on shared variance rather than observations.

VarClus works by iteratively splitting clusters. A selected cluster is decomposed using the first two principal components, followed by an orthoblique rotation. Variables are assigned to the component with which they have the highest squared correlation. An iterative reassignment step then refines cluster membership to maximize variance explained by each cluster component.

The result is an interpretable hierarchical clustering of variables, useful for multicollinearity reduction, feature selection, and exploratory data analysis.

Key features
- Hierarchical variable clustering
- PCA-based splits with orthoblique rotation
- Iterative reassignment to maximize explained variance
- Interpretable alternative to standard dimensionality reduction techniques

### For who?
- Analysts familiar with `PROC VARCLUS` in SAS Software: If you have used the `PROC VARCLUS` procedure before but struggled to find a correct and reliable Python implementation, this module is designed for you.
- Python users new to the `PROC VARCLUS` algorithm: The well-documented source code is intended to be educational, helping you build a deep understanding of the mathematics and logic behind variable clustering.





## 🎯 Project Goals
- Reimplement core ideas in a clean, pythonic way
- Experiment with alternative design choices **(Speed Up! 🚀)**
- Make the library easier to extend and customize
- Learn and share knowledge with the community

## 👉 What I did...
There are six targeted optimisations applied to the original `VarClusHi` python class. Each change reduces redundant computation while preserving numerical outputs to within floating-point tolerance (np.allclose defaults).

### 1. `correig` - Eigendecomposition cleanup
#### Issue
The original code calls `np.argsort` on the full eigenvalue array, then uses advanced indexing to reorder both arrays, producing three temporary arrays, even though only the top-2 components are needed.
#### Solution
A dedicated `_eigh_sorted` helper uses `np.arange` with a reverse step _(no argsort)_ and slices to `n_pcs` immediately. The resulting eigvals/eigvecs are returned directly.

```python
# Original
idx = np.argsort(raw_eigvals)[::-1]           #Creates temp array
eigvals = raw_eigvals[idx][:n_pcs]            #Another copy
eigvecs = raw_eigvecs[:, idx][:, :n_pcs]      #And another copy

# Optimized
idx = np.arange(len(raw_eigvals)-1, -1, -1)
eigvals = raw_eigvals[idx][:n_pcs]
eigvecs = raw_eigvecs[:, idx][:, :n_pcs]
```

### 2. `_reassign` - Set membership and shared correlation matrix
#### Issue
Every candidate feature moves inside `_reassign` called `_calc_tot_var`, which internally recomputed `np.corrcoef` from raw data. For `k` features, each pass through the loop triggered 2k full correlation-matrix computations on the same unchanged data.
#### Solution
Compute `np.corrcoef` once for the full feature pool at the start of `_reassign`. Sub-matrices for candidate clusters are extracted via `np.ix_` index slicing. Feature membership is tracked with Python sets for O(1) add/remove instead of O(k) list scans.

```python
# Original - Corrcoef inside every candidate moves
new_var = VarClusHi._calc_tot_var(df, new_clus1, new_clus2)[0]

# Optimized - Compute once, slice per candidate
corr_vals = np.corrcoef(df[feat_list].values.T)       #Once
feat_idx  = {f: i for i, f in enumerate(feat_list)}

def _var(s1, s2):
    i1 = np.array([feat_idx[f] for f in s1])
    i2 = np.array([feat_idx[f] for f in s2])
    return _tot_var_from_corr(corr_vals, i1, i2)      #np.ix_ slice only
```

### 3. `_varclusspu` - Global correlation matrix cache
#### Issue
Inside the main clustering loop, correig was called for every cluster being evaluated and for every sub-cluster produced after a split. Each call recomputed np.corrcoef from raw data, resulting in O(p² × splits) redundant floating-point operations.
#### Solution
Compute the full p×p correlation matrix once at the top of `_varclusspu`. A local helper `_cluster_info_from_corr` extracts per-cluster sub-matrices with integer index slicing from this single cached array.

```python
# Optimized - global_corr computed once
global_corr = np.corrcoef(self.df[self.feat_list].values.T)
feat_pos    = {f: i for i, f in enumerate(self.feat_list)}

def _cluster_info_from_corr(clus):
    idx = [feat_pos[f] for f in clus]

    # O(k²) slice, no recompute
    sub = global_corr[np.ix_(idx, idx)]
    eigvals, eigvecs, varprops = VarClusHi._eigh_sorted(sub, n_pcs=2)
    return float(eigvals[0]), float(eigvals[1]), eigvecs, float(varprops[0])
```

### 4. Feature assignment - Vectorised projection
#### Issue
Assigning each feature to a cluster required computing `np.dot` separately per feature inside a python iteration `n` individual dot products for `n` features in a split cluster.
#### Solution
Replace the per-feature loop with a single matrix–vector multiplication. With multiplying the sub-correlation matrix by the rotated eigenvector once to obtain projection scores for all features simultaneously. Assignment becomes a vectorised boolean comparison.

```python
# Original - Python loop, one np.dot per feature
for feat in split_clus:
    cov1 = np.dot(r_eigvecs[:, 0], split_corrs[feat].values.T)
    cov2 = np.dot(r_eigvecs[:, 1], split_corrs[feat].values.T)
    corr_pc1 = cov1 / sigma1
    ...

# Optimized - Single matmul, all features at once
cov1 = split_corr @ re0 #shape (n_split,)
cov2 = split_corr @ re1
corr_pc1 = cov1 / sigma1
corr_pc2 = cov2 / sigma2
clus1 = [f for f,c1,c2 in zip(split_clus,corr_pc1,corr_pc2) if abs(c1)>=abs(c2)]
```

### 5. `info` Table - Avoid quadratic DataFrame growth  
#### Issue
The original info property appended rows one by one using `df.loc[n_row]`. Each pandas assignment copies the entire internal block array, making this O(n²) in memory allocations for n clusters.
#### Solution
Collect all rows in a plain python list, then call `pd.DataFrame()` once at the end. This is the canonical pandas best practice for building DataFrames iteratively.

```python
# Original - O(n²) repeated internal copy
info_table = pd.DataFrame(columns=cols)
for i, clusinfo in self.clusters.items():
    info_table.loc[n_row] = row #Copies entire df each time


# Optimized - O(n) single construction
rows = [
    [repr(i), repr(len(ci.clus)), ci.eigval1, ci.eigval2, ci.varprop]
    for i, ci in self.clusters.items()
]
return pd.DataFrame(
    rows,
columns = ['Cluster','N_Vars','Eigval1','Eigval2','VarProp']
)
```

### 6. `rsquare` — Vectorised R-Square matrix  
#### Issue
Both rsquare paths computed R² of each variable against each cluster PC using individual np.corrcoef calls inside nested loops O(n_vars × n_clusters) corrcoef invocations total.
#### Solution
Build a proj matrix of shape (n_clusters, n_features) in one pass for each cluster k, multiply the cached global correlation row by the cluster's PC1 eigenvector. Squaring yields all R² values at once. `RS_NC` is a masked row-max on this matrix.

```python
# Optimized - Full R-Square matrix in one vectorised pass
proj = np.empty((n_clus, n_feat))
for k, (_, ci) in enumerate(clus_items):
    c_idx = np.array([feat_pos[f] for f in ci.clus])
    v = ci.eigvecs[:, 0]
    sub = global_corr[np.ix_(c_idx, c_idx)]
    sigmas[k] = math.sqrt(v @ sub @ v)
    proj[k] = (v @ global_corr[c_idx, :]) / sigmas[k]

rs_matrix = proj ** 2                    #(n_clus, n_feat) —-> all R² at once
rs_nc = rs_matrix[other_mask, fi].max()  #RS_NC via masked row-max
```

### Summary
| Function | Before | After | Impact |
|----------|----------|----------|----------|
| correig | O(p²) + argsort overhead | O(p²), no argsort  | Constant factor reduction |
| _reassign (inner) | O(k²) corrcoef per move | O(k²) matrix slice only | No recomputation |
| _varclusspu (corr) | O(p²) per cluster eval | O(k²) slice from cache | p² computed once total |
| Feature assignment | O(k) dot loop | O(k) single matmul | BLAS-level throughput |
| info | O(n²) df.loc append | O(n) list + constructor | pandas best practice |
| rsquare | O(v·c) corrcoef calls | O(c·p) matmul + index | Single matrix pass |

# Testing and Performance
## ✔️ Testing
After optimization coded, re-running to test the calcualtion logic by the same demo dataset. See dataset - [winequality-red.csv](https://github.com/naenumtou/varclushi_opt/blob/master/data/winequality-red.csv). The result is shown below:

```python
  Cluster N_Vars   Eigval1   Eigval2   VarProp
0       0      3  2.141357  0.658413  0.713786
1       1      3  1.766885  0.900991  0.588962
2       2      2  1.371260  0.628740  0.685630
3       3      2  1.552496  0.447504  0.776248
4       4      1  1.000000  0.000000  1.000000
```
Running the test script between 2 outputs resulting from original and optimized verseions. The result is shown below:

 ```python
────────────────────────────────────────────────────────────
  CORRECTNESS CHECK
────────────────────────────────────────────────────────────
  ✓ PASS  N clusters match
  ✓ PASS  N_Vars match
  ✓ PASS  Eigval1 close
  ✓ PASS  Eigval2 close
  ✓ PASS  RS_Own close
  ✓ PASS  RS_NC close
  ✓ PASS  RS_Ratio close

  All checks passed — outputs are numerically identical.

DONE
```

## 🕙 Performance
### cProfile Performance Comparison
To compare the profiling results between the Original and Optimized implementations using cProfile, focusing on call frequency and cumulative execution cost.

 ```python
────────────────────────────────────────────────────────────
  cPROFILE — Original
────────────────────────────────────────────────────────────
  ncalls  tottime  percall  cumtime  function
    3266    0.004    0.000    0.006  _handle_fromlist
     218    0.000    0.000    0.001  simplefilter
      13    0.000    0.000    0.000  __call__
       4    0.000    0.000    0.000  dumps
       1    0.000    0.000    0.000  __subclasscheck__
     654    0.000    0.000    0.000  __enter__
     654    0.000    0.000    0.000  __exit__
       2    0.000    0.000    0.000  _type_convert
     218    0.000    0.000    0.000  _get_context
     218    0.000    0.000    0.001  __enter__
     218    0.000    0.000    0.000  __exit__
       1    0.000    0.000    0.000  namedtuple
       1    0.000    0.000    0.000  __exit__
      13    0.000    0.000    0.000  __new__
       6    0.000    0.000    0.000  <genexpr>
```

 ```python
────────────────────────────────────────────────────────────
  cPROFILE — Optimized
────────────────────────────────────────────────────────────
  ncalls  tottime  percall  cumtime  function
      18    0.000    0.000    0.000  _handle_fromlist
     222    0.000    0.000    0.001  simplefilter
      13    0.000    0.000    0.000  __call__
       4    0.000    0.000    0.000  dumps
       1    0.000    0.000    0.000  __subclasscheck__
     666    0.000    0.000    0.000  __enter__
     666    0.000    0.000    0.000  __exit__
       2    0.000    0.000    0.000  _type_convert
     222    0.000    0.000    0.000  _get_context
     222    0.000    0.000    0.001  __enter__
     222    0.000    0.000    0.001  __exit__
       1    0.000    0.000    0.000  namedtuple
       1    0.000    0.000    0.000  __exit__
      13    0.000    0.000    0.000  __new__
       6    0.000    0.000    0.000  <genexpr>
```
Significant reduction in import‑related overhead. The function _handle_fromlist shows a drastic drop in call count:
- Original: 3,266 calls
- Optimized: 18 calls

This strongly indicates that the optimized version eliminates repeated module import resolution, which is a common hidden performance cost in Python.

Context manager and filter calls remain stable. Functions such as:
- `simplefilter`
- `__enter__` / `__exit__`
- `_get_context`

Showing similar call counts between versions:
| Function | Original | Optimized |
| --- | --- | --- |
| `simplefilter` | 218 | 222 |
| `__enter__` | 654 | 666 |
| `__exit__` | 654 | 666 |

This suggests that the core logical behavior is unchanged and the pptimization did not introduce unnecessary control‑flow overhead.

### Running times
Re-running the code on the same demo dataset **200 times** resulted in an approximately **~4.x** speedup.

 ```python
────────────────────────────────────────────────────────────
  TIMING BENCHMARK BY RUN 200 TIMES
────────────────────────────────────────────────────────────

▶ Original
  mean       0.0533s ± 0.0059s          
  min        0.0469s
  max        0.1093s

▶ Optimized
  mean       0.0141s ± 0.0012s          
  min        0.0122s
  max        0.0209s

────────────────────────────────────────────────────────────
  SUMMARY
────────────────────────────────────────────────────────────
          Mean (s) Std (s) Min (s) Max (s)
Version                                   
Original    0.0533  0.0059  0.0469  0.1093
Optimized   0.0141  0.0012  0.0122  0.0209

  Speedup: 3.79x
```

<p align="center">
<img width="855" height="547" alt="ใช้ Python ทำ proc varclus เหมือนบน SAS ได้แล้วนะ" src="https://github.com/user-attachments/assets/858db566-cccc-447f-9643-f72f2291103d" />
</p>

For a larger dataset with **20k rows and 200 features for 5 times** (capped at this size due to excessive runtime in the original version), resulted in an approximately **~10.x** speedup.

 ```python
────────────────────────────────────────────────────────────
  TIMING BENCHMARK BY RUN 5 TIMES
────────────────────────────────────────────────────────────

▶ Original
  mean       14.8464s ± 0.9600s         
  min        13.5937s
  max        16.4938s

▶ Optimized
  mean       1.4380s ± 0.6753s          
  min        0.8294s
  max        2.3690s

────────────────────────────────────────────────────────────
  SUMMARY
────────────────────────────────────────────────────────────
          Mean (s) Std (s)  Min (s)  Max (s)
Version                                     
Original   14.8464  0.9600  13.5937  16.4938
Optimized   1.4380  0.6753   0.8294   2.3690

  Speedup: 10.32x
```

<p align="center">
<img width="842" height="547" alt="ใช้ Python ทำ proc varclus เหมือนบน SAS ได้แล้วนะ" src="https://github.com/user-attachments/assets/da14eb9b-49c9-4158-bd38-04955803b5ff" />
</p>

For a larger dataset with scaling by **sizing 50,000, 500,000 and 1,000,000 rows and 100, 150 and 200 features for 5 times**. The result is shown below:

```python
────────────────────────────────────────────────────────────
  SUMMARY
────────────────────────────────────────────────────────────
  n_obs  n_var    mean     std     min     max
  50000    100  0.1914  0.0101  0.1732  0.2020
  50000    150  0.9047  0.3507  0.3682  1.4711
  50000    200  1.5565  0.4828  0.8359  2.3473
 500000    100  1.4523  0.1825  1.2335  1.6914
 500000    150  3.1839  0.1799  2.8887  3.4096
 500000    200 14.6079 14.5148  5.8391 43.4754
1000000    100  3.1791  0.1345  2.9873  3.3308
1000000    150 46.2291  4.1337 39.5784 52.1834
1000000    200 21.8285 18.6833  6.5975 52.7565
```

<p align="center">
<img width="989" height="590" alt="ใช้ Python ทำ proc varclus เหมือนบน SAS ได้แล้วนะ" src="https://github.com/user-attachments/assets/4ae13739-cfdf-406f-91a3-76c2495bbe76" />
</p>





# Example (Usage is fully compatible with original version.)
## See [demo.ipynb](https://github.com/jingtt/varclushi/blob/master/demo.ipynb) for more details.

```python
import pandas as pd
from varclushi import VarClusHi
```

Create a VarClusHi object and pass the dataframe (df) to be analyzed as a parameter, you can also specify 
- a feature list (feat_list, default all columns of df)
- max second eigenvalue (maxeigval2, default 1)
- max clusters (maxclus, default None)

Then call method varclus(), which performs hierachical variable clustering algorithm

```python
demo1_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
demo1_df.drop('quality',axis=1,inplace=True)
demo1_vc = VarClusHi(demo1_df,maxeigval2=1,maxclus=None)
demo1_vc.varclus()
```

Call info, you can get the number of clusters, number of variables in each cluster (N_vars), variance explained by each cluster (Eigval1), etc.

```python
demo1_vc.info
```

# Installation

- Requirements: Python 3.4+

- Install by pip:

```
pip install varclushi
```


