This code is the implementation of the algorithms and experiments in our paper "A Unified Framework for Rank-based Loss Minimization".

* `existing_methods\lerm_main`: the source code for the algorithms in paper "Stochastic Algorithms for Spectral Risk Measures", which can be found in https://github.com/ronakdm/lerm.git.
* `existing_methods\SoRR`: the source code for the algorithms in paper "Sum of Ranked Range Loss for Supervised Learning", which can be found in https://github.com/discovershu/SoRR.git.

Note that some modifications have been made to the source code for the purpose of the comparison experiments, which have no impact on the efficiency of the original algorithm.

### Introduction
This code can be run in python 3.8.10.

To avoid compatibility issues, refer to the following package versions,
| package     |   version |
| :----:      | :----:    |
| libsvmdata  |    0.4.1  |             
| matplotlib  |    3.7.1  |  
| numpy       |   1.24.2  | 
| pandas      |    2.0.0  |
| openpyxl    |    3.1.2  |
| scikit-learn|    1.2.2  |             
| scipy       |   1.10.1  |
| torch       |   1.13.1+rocm5.2 (cpu version)  |

* `SGD_solver.py`, `LSVRG_solver.py`: the code to call the SGD algorithm and LSVRG algorithm in `existing_methods\lerm_main`.
* `DCA_solver.py`: the code to call the DCA alogirthm in `existing_methods\SoRR`.
* `run_demo.py`: a demo of how our algorithmic code which can be found in `src` would run.

### How to get the results
* To run the SRM framework experiments, please `python run_SRM.py`
* To run the EHRM framework experiments, please `python run_EHRM.py`
* To run the AoRR framework experiments, please `python run_AoRR_fixed.py` for the real dataset and `python run_AoRR_ratio.py` for the synthetic dataset.

If you need to draw graphs and store data, change `need_log = False` in each `run_*.py` to `need_log = True`.
The results will be saved in folders `figure` and `table`.

### Citation
If you want to reference our paper in your research, please consider citing us by using the following BibTeX:

```bib
@inproceedings{NEURIPS2023_a10946e1,
 author = {Xiao, Rufeng and Ge, Yuze and Jiang, Rujun and Yan, Yifan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {51302--51326},
 publisher = {Curran Associates, Inc.},
 title = {A Unified Framework for Rank-based Loss Minimization},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/a10946e1f46e1ffc0daf37cb2abfdcad-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

If you have any questions, please contact us.
