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

Dataset eksternal pembanding tidak disimpan di repository. Jika tersedia file CSV lain dengan fitur yang sama, script dapat menggunakannya untuk cross-dataset evaluation melalui argumen `--cross-data`.
