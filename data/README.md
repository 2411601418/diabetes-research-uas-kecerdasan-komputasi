# Data

Folder ini berisi dataset utama:

```text
diabetes_012_health_indicators_BRFSS2015.csv
```

Dataset berasal dari Kaggle:

https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Target asli `Diabetes_012` memiliki tiga kelas:

- `0`: non-diabetes
- `1`: prediabetes
- `2`: diabetes

Pada eksperimen, target dikonversi menjadi biner:

- `0` tetap `0`
- `1` dan `2` digabung menjadi `1`

File `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` tidak disimpan lagi di repository. Jika file tersebut tersedia secara lokal, script dapat menggunakannya sebagai dataset eksternal untuk cross-dataset evaluation.
