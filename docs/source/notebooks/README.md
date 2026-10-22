# Notebooks

`Experiments_mean.csv` is **not distributed with this repository**. It holds the aggregated
experimental results that `graphics.ipynb`, `graphics_SOCO.ipynb` and `WilcoxonTest.ipynb`
read, and it lives as a release asset:

```bash
curl -L -O https://github.com/ari-dasci/S-ReVel/releases/download/data-v1.0.3/Experiments_mean.csv
```

Run that command in this directory before executing those three notebooks.

The file used to be committed here. It was removed from the working tree and from the git
history because at 44 MB it dominated the size of every clone, and a data file of that size
does not belong in version control. The asset is byte-identical to the committed file.
