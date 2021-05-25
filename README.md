# Code for Adaptive Margin Based Metric Learning for Unbalanced Point Cloud Classification and Retrieval



## dataset and pretrained model
[Modelnet40](https://drive.google.com/file/d/1Skw5GTsL00MmQW3CzTPOwFKCQe5JlMnh/view?usp=sharing) dataset

[pretrained model](https://drive.google.com/file/d/1zI8yryM8ZI5J5cMfN6epi-b9j6467-sQ/view?usp=sharing)

## requirments
 * Python 3.6
 * Pytorch 1.2
 * ff3d-core (when CUDA is available, not necessary)

```python
pip install -r requirements.txt
```


### Point cloud Classfication and Retrieval
```python
python test.py
```

#### Performance on ModelNet40


| cuda_available   | Classification   |Retrieval| 
---- | --- | ---
| True    | 93.4%            | 92.2%   |
| False   | 92.6%            | 91.5%   |


Note that when CUDA is available, 1024 point cloud is sampled from the origin 2048 point cloud in the furthest sampling way, which depends on CUDA. When CUDA is not available, 1024 point cloud is sampled in the random sampling way and thus the performance is sacrificed with a small margin.




    
