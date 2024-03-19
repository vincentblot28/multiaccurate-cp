# Prepare data

```
$ python multiaccurate_cp/main.py prepare --data-path=data --patch-size=572 --pad-size=92 --overlap=92
```

# Train model

```
$ python multiaccurate_cp/main.py train --ml-data-dir=data/02_prepared_data --output-dir=data/03_model_weights --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```

# Inference
## Calibration data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/03_model_weights --model-name=20240318_1244 --data-dir=data/02_prepared_data --ml-set=cal --output-dir=data/04_predictions/ --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```

## Test data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/03_model_weights --model-name=20240318_1244 --data-dir=data/02_prepared_data --ml-set=test --output-dir=data/04_predictions/ --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```