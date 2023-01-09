# SIHR_Python
This repository is changing [SIHR](https://github.com/vitorsr/SIHR) codes MATLAB/GNU Octave to python.

At this moment, I implement Akashi2016 algorithm using numpy and jax.

This implementation is much faster than original.

The other algorithms are coming soon.

---
# Comparison

The table below is the time of each model to process 1820x1024x3 size single image.</br>
And I marked time per iteration for fair comparison in table.</br>
Because Akashi algorithm has random variables in code, so each implementation has different iterations.

|Year| Method             |       Original        |         Numpy         |           Jax         |
|:--:|--------------------|:---------------------:|:---------------------:|:---------------------:|
|2016| Akashi and Okatani |    0.7714 / it   |    0.2727 / it   |    0.015 / it   | 
---



# Acknowledgement
This repository is largely based on the SIHR repository by vitorsr.



# Reference
Y. Akashi and T. Okatani, “Separation of reflection components by sparse non-negative matrix factorization,” Computer Vision and Image Understanding, vol. 146, pp. 77–85, May 2016, http://dx.doi.org/10.1016/j.cviu.2015.09.001
