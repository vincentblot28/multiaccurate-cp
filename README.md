# Prepare data

```
$ python multiaccurate_cp/main.py prepare --data-path=data --patch-size=572 --pad-size=92 --overlap=92
```

# Train model

```
$ python multiaccurate_cp/main.py train --ml-data-dir=data/aerial/02_prepared_data --output-dir=data/aerial/03_model_weights/unet --mean-RGB-values-path=data/aerial/01_raw_images/rgb_means.npy
```

# Inference
## Calibration data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/aerial/03_model_weights/unet --model-name=20240318_1244 --data-dir=data/aerial/02_prepared_data --ml-set=cal --output-dir=data/aerial/04_predictions/ --mean-RGB-values-path=data/aerial/01_raw_images/rgb_means.npy
```

## Residual data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/aerial/03_model_weights/unet/unet --model-name=20240318_1244 --data-dir=data/aerial/02_prepared_data --ml-set=res --output-dir=data/aerial/04_predictions/ --mean-RGB-values-path=data/aerial/01_raw_images/rgb_means.npy
```

## Test data
```
$ python multiaccurate_cp/main.py infer --model-dir=data/aerial/03_model_weights/unet --model-name=20240318_1244 --data-dir=data/aerial/02_prepared_data --ml-set=test --output-dir=data/aerial/04_predictions/ --mean-RGB-values-path=data/aerial/01_raw_images/rgb_means.npy
```

# Train residual

```
$ python multiaccurate_cp/main.py train-residual --ml-data-dir=data/aerial/02_prepared_data --probas-dir=data/aerial/04_predictions --output-dir=data/aerial/03_model_weights/resnet --mean-RGB-values-path=data/aerial/01_raw_images/rgb_means.npy --model.resnet=resnet50 --model.model-input=image_and_probas --model.embedding-size=1024
```

# Inference residual
## Calibration data
```
$ python multiaccurate_cp/main.py infer-residual --model-dir=data/aerial/03_model_weights/resnet --model-name=20240322_1039 --data-dir=data/aerial/02_prepared_data --pred-proba-dir=data/aerial/04_predictions --ml-set=cal
```

## Test data
```
$ python multiaccurate_cp/main.py infer-residual --model-dir=data/aerial/03_model_weights/resnet --model-name=20240322_1039 --data-dir=data/aerial/02_prepared_data --pred-proba-dir=data/aerial/04_predictions --ml-set=test
```

# Inference PraNet
```
$ python multiaccurate_cp/main.py infer-polyp --data-dir=data/polyp/02_prepared_data --output-dir=data/polyp/04_predictions/ --model-dir=data/polyp/03_model_weights/ ml-set=res
```