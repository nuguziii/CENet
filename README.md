# CENet

Implementation of paper: Deeply Self-Supervised Contour Embedded Neural Network Applied to Liver Segmentation (CMPB 2020)

You can find paper in [link](https://www.sciencedirect.com/science/article/pii/S0169260719305012)

## To Run

1. Clone: 
```
git clone https://github.com/nuguziii/CENet.git
cd CENet
```

2. Dataset

- [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- [BTCV](https://www.synapse.org/#!Synapse:syn3193805)

3. Train

```
python main.py --train --data_dir {path_to_dataset} --output_dir {path_to_output} --description {experiment_name} --log_dir {path_to_log}
```

4. Test

```
python main.py --test --data_dir {path_to_dataset} --output_dir {path_to_output} --description {experiment_name} --log_dir {path_to_log} --model_name {model_name.pth}
```
