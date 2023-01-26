# DT-SNE
Implementation of the DT-SNE algorithm, as used in the paper: "DT-SNE: t-SNE Discrete Visualizations as Decision Tree Structures." Neurocomputing (to appear).

# Build
Use
```
$ python setup.py build_ext --inplace
```
to build the cython code.

After that, DT-SNE can be imported in python with
```python
from dt_sne import dt_sne
```

# Authors
Adrien Bibal (University of Colorado Anschutz Medical Campus) &
Valentin Delchevalerie (University of Namur)

Parts of the code are inspired by the t-SNE implementation of Laurens van der Maaten (https://lvdmaaten.github.io/tsne/), and a DT implementation of Géraldin Nanfack.
