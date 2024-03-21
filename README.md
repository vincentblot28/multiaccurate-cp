# Prepare data

```
$ python multiaccurate_cp/main.py prepare --data-path=data --patch-size=572 --pad-size=92 --overlap=92
```

# Train model

```
$ python multiaccurate_cp/main.py train --ml-data-dir=data/02_prepared_data --output-dir=data/03_model_weights/unet --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```

# Inference
## Calibration data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/03_model_weights/unet --model-name=20240318_1244 --data-dir=data/02_prepared_data --ml-set=cal --output-dir=data/04_predictions/ --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```

## Residual data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/03_model_weights/unet/unet --model-name=20240318_1244 --data-dir=data/02_prepared_data --ml-set=res --output-dir=data/04_predictions/ --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```

## Test data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/03_model_weights/unet --model-name=20240318_1244 --data-dir=data/02_prepared_data --ml-set=test --output-dir=data/04_predictions/ --mean-RGB-values-path=data/01_raw_images/rgb_means.npy
```

# Train residual

```
$ python multiaccurate_cp/main.py train-residual --ml-data-dir=data/02_prepared_data --probas-dir=data/04_predictions --output-dir=data/03_model_weights/resnet --mean-RGB-values-path=data/01_raw_images/rgb_means.npy --model.resnet=resnet50 --model.model-input=image_and_probas
```

# Inference residual
## Calibration data
```
$ python multiaccurate_cp/main.py infer-residual --model-dir=data/03_model_weights/resnet --model-name=20240321_1625 --data-dir=data/02_prepared_data --pred-proba-dir=data/04_predictions --ml-set=cal
```

## Test data
```
$ python multiaccurate_cp/main.py infer-residual --model-dir=data/03_model_weights/resnet --model-name=20240321_1625 --data-dir=data/02_prepared_data --pred-proba-dir=data/04_predictions --ml-set=test
```