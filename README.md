# popman

A Matlab solver for **p**reconditioned Riemannian **o**ptimization methods on **p**roduct **man**ifolds.



## Problems

This solver is applicable for the following optimization problem

min f(U,V), s. t. (U,V) \in M = St\_{M_1}(p,m) \times St_{M_2}(p,n),

where St\_{M\_1}(p,m) := \{U in \mathbb{R}^{m\times p}:\ U^T M_1 U=I\_p\} is the _generalized Stiefel Manifold_. 



The search space M is endowed with a preconditioned metric

g\_(U,V) (xi,eta)=< xi_1 , B11(U,V) eta_1 B12(U,V) > + < xi_2, B21(U,V) eta_2 B22(U,V) >            for xi,eta in T\_(U,V) M,

where B11(U,V) is an m-by-m matrix, B12(U,V) is an p-by-p matrix, B21(U,V) is an n-by-n matrix, B22(U,V) is an p-by-p matrix. 



## How to run?

1. Make sure you have installed the package [Manopt](http://manopt.org). 

2. Run ``test_CCA.m`` for Canonical correlation analysis 

3. Run ``test_SVD.m`` for truncated singular value decomposition 



## References

[Bin Gao](https://www.gaobin.cc/), Renfeng Peng, [Ya-xiang Yuan](http://lsec.cc.ac.cn/~yyx/index.html)

- [Optimization on product manifolds under a preconditioned metric](https://arxiv.org/abs/2306.08873)



## Authors

- Renfeng Peng (AMSS, China)



## Copyright

Copyright (C) 2023, Bin Gao, Renfeng Peng, Ya-xiang Yuan

This solver is based on a third-party package [Manopt](http://manopt.org). 

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)
