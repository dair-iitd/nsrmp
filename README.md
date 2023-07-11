# Neuro-symbolic Robot Manipulation (NSRM)

[**Learning Neuro-symbolic Programs for Language Guided Robot Manipulation**](https://arxiv.org/abs/2211.06652)  
Namasivayam Kalithasan*, Himanshu Singh*, Vishal Bindal*, Arnav Tuli, Vishwajeet Agrawal, Rahul Jain, Parag Singla, Rohan Paul     
[ICRA 2023](https://www.icra2023.org/) 

For the latest updates, see: [nsrmp.github.io](https://nsrmp.github.io)

Given a natural language instruction, and an input and an output scene, our goal is to train a neuro-symbolic model which can output a manipulation program that can be executed by the robot on the input scene resulting in the desired output scene. Our approach is neuro-symbolic and can handle linguistic as well as perceptual variations, is end-to-end differentiable requiring no intermediate supervision, and makes use of symbolic reasoning constructs which operate on a latent neural object-centric representation, allowing for deeper reasoning over the input scene. Our experiments on a simulated environment with a 7-DOF manipulator, consisting of instructions with varying number of steps, as well as scenes with different number of objects, and objects with unseen attribute combinations, demonstrate that our model is robust to such variations, and significantly outperforms existing baselines, particularly in generalization settings.


## Index

- Getting Started: [Setup](#setup), [Quickstart](#quickstart), [Downloads](#downloads), [Hardware Requirements](#hardware-requirements)
- Guides: [Training](#training), [Data generation](#data-generation)
- Miscellaneous: [Acknowledgements](#acknowledgements), [Citation](#citation)

## Setup

- Install [Jacinle](https://github.com/vacancy/Jacinle): Clone the package, and add the bin path to your PATH environment.

        git clone https://github.com/vacancy/Jacinle --recursive
        export PATH=<path_to_jacinle>/bin:$PATH

- Clone the NSRM repository

        git clone https://github.com/dair-iitd/nsrmp.git
	
 - Add the root directory to PATH and PYTHONPATH
 
        cd nsrmp
        export PYTHONPATH=$(pwd):$PYTHONPATH
        export PATH=$(pwd):$PATH

- Create a conda environment from [nsrmp_conda_environment.yaml](nsrmp_conda_environment.yaml). (*Prerequisite*: [conda](https://docs.conda.io/en/latest/miniconda.html))

        conda env create -f nsrmp_conda_environment.yaml
        conda activate nsrm

## Quickstart

This section contains details of evaluating trained model checkpoints on pre-generated data, and simulating examples. See [Training](#training) for training models, and [Data generation](#data-generation) for generating new data.

### Setting up downloads

See [Downloads](#downloads) for links to model checkpoints and dataset. The paths for these would be referred to thereon as:
- `<path_to_nsrm_checkpoint>` for trained NSRM checkpoint
- `<path_to_baseline_checkpoint>` for trained baseline checkpoint
- `<path_to_dataset>` for dataset directory. Note that after unzipping the download, this directory should have the following files
        
        <path_to_dataset>
		└── instructions-*.json
		└── scenes-*.json
		└── train
		└── test
		└── val
		└── vocab.json
	
- `<path_to_dataset_general>` for dataset (generalizzation experiments). Note that after unzipping the download, this directory should have the following files
        
        <path_to_dataset_general>
		└── color_comb
		└── multi-objects
		└── multi-step
		└── type-combinatorial

### Evaluate NSRM

TODO
```bash
jac-crun 0 scripts/eval.py --dataset roboclevr --datadir <path_to_dataset> --vocab_json --instruction_transform basic --use_cuda True --batch_size 32  --load_model_from_file <path_to_nsrm_checkpoint> 
```

### Evaluate baseline

Run the command below to evaluate the performance of baseline model on given dataset:  
`<baseline_type>`: Type of baseline used ('single': only supports single-step instructions, 'multi': supports instructions of arbitrary length)  
`<path_to_model>`: Absolute path to the baseline checkpoint

```bash
python3 scripts/test_baseline.py --datadir <path_to_dataset> --type <baseline_type> --load_model <path_to_model>
```

[OPTIONAL] Arguments:  
**--use_cuda**: Use CUDA or not (default: True)  
**--num_steps**: Range of lengths of instructions. For e.g., `--num_steps a b` filters examples where the number of steps in execution are > a and <= b (default: 0 2)  
**--num_objects**: Range of number of objects in scene. For e.g., `--num_objects a b` filters examples where the number of objects in scene are > a and <= b (default: 0 5)  
**--language_complexity**: Filter complexity of natural language instructions. For e.g., `simple` and `complex` (default: None)  
**--remove_relational**: Filter out examples involving spatial reasoning (default: False)  
**--only_relational**: Filter examples involving spatial reasoning (default: False)  

### Simulate an example

To render a pybullet simulation for any example, run the below command (Note: <path_to_dataset>/test/00297 contains an example for a 2-step instruction)
```
cd nsrmp
jac-run scripts/simulate.py --model_path <path_to_nsrm_checkpoint> \ 
                            --example_path <path_to_dataset>/test/00297 \
                            --predicted True
```
For other options, see [simulate.py](nsrmp/scripts/simulate.py) 

### Visualise plan for an example

TODO

### Reconstruct final scene for an example

TODO

## Downloads

- [Trained NSRM checkpoint](./model_saves/model_release.pth)
- [Trained baseline checkpoint]()
- [Dataset (train/val/test)]()
- [Dataset (Generalization experiments)]()

## Hardware requirements

We have trained and tested the code on
- **GPU** - NVIDIA Quadro RTX 5000
- **CPU** - Intel(R) Xeon(R) Gold 6226R
- **RAM** - 16GB
- **OS** - Ubuntu 20.04

## Training

### Training NSRM End-to-End
```bash
jac-crun 0 scripts/train.py --dataset roboclevr --datadir <path_to_dataset>  --vocab_json <path_to_dataset>/vocab.json --instruction_transform program_parser_candidates --use_cuda True --batch_size 32 --num_epochs 300  --model_save_interval 1 --training_target all --eval_interval 10   
```

### Abalations
- The contribution of the Language Reasoner can be efaced by using the ground truth symbolic-program. Set the ```--training_target``` flag to ```concept_embeddings``` to train the visual and Action Modules using ground truth symbolic programs. That is,
```bash
jac-crun 0 scripts/train_single_step.py --dataset roboclevr --datadir <path_to_dataset>  --vocab_json <path_to_dataset>/vocab.json --instruction_transform program_parser_candidates --use_cuda True --batch_size 32 --num_epochs 300  --model_save_interval 1 --training_target concept_embeddings --eval_interval 10   
```
- Similarly, if the visual modules are pre-trained, the Language and Action Modules alone can be trained by setting the ```--training_target``` flag to ```non-visual```. Refer [model_new.py](nsrmp/model/model_new.py) to know more about training targets.

### Training baseline

Run the command below to train the baseline model on given dataset:  
`<baseline_type>`: Type of baseline used ('single': only supports single-step instructions, 'multi': supports instructions of arbitrary length)  

```bash
python3 scripts/train_baseline.py --datadir <path_to_dataset> --type <baseline_type>
```

[OPTIONAL] Arguments:  
**--use_cuda**: Use CUDA or not (default: True)  
**--batch_size**: Size of batch during training (default: 32)  
**--num_epochs**: Number of training epochs (default: 200)  
**--save_model**: Name with which to store the model dict (in `model_saves` directory) (default: None)  
**--load_model**: Name from which to load the model dict (in `model_saves` directory) (default: None)  
**--save_spliiter**: (only for `multi` type) Whether to store model's splitter dict or not (default: False)  
**--load_splitter**: (only for `multi` type) Whether to load model's spliiter dict or not (default: False)  
**--save_interval**: Model save interval (in epochs) (default: 5)  
**--use_iou_loss**: Use IoU loss or not (during training) (default: False)  
**--train_splitter**: (only for `multi` type) Train model's splitter or not (default: False)  
**--num_steps**: Range of lengths of instructions. For e.g., `--num_steps a b` filters examples where the number of steps in execution are > a and <= b (default: 0 2)  
**--num_objects**: Range of number of objects in scene. For e.g., `--num_objects a b` filters examples where the number of objects in scene are > a and <= b (default: 0 5)  
**--language_complexity**: Filter complexity of natural language instructions. For e.g., `simple` and `complex` (default: None)  
**--remove_relational**: Filter out examples involving spatial reasoning (default: False)  
**--only_relational**: Filter examples involving spatial reasoning (default: False)  
**--wandb**: Use WandB for logging training losses and model parameters (default: False)  

Training of baseline model on the given dataset is achieved by the following pipeline:
- Train baseline on examples consisting of single-step commands and upto 5 objects without IoU loss
```bash
python3 scripts/train_baseline.py --datadir <path_to_dataset> --type single --save_model <single_step_path> --num_steps 0 1
```
- Fine-tune trained model on the same dataset with IoU loss
```bash
python3 scripts/train_baseline.py --datadir <path_to_dataset> --type single --save_model <single_step_path> --num_steps 0 1 --load_model <single_step_path> --use_iou_loss True
```
- Using model trained on single-step commands, train model (with splitter) on examples consisting of upto two-step commands and upto 5 objects with IoU loss
```bash
python3 scripts/train_baseline.py --datadir <path_to_dataset> --type multi --save_model <multi_step_path> --load_model <single_step_path> --use_iou_loss True --train_splitter True
```
- Freeze model's splitter, and fine-tune the model further
```bash
python3 scripts/train_baseline.py --datadir <path_to_dataset> --type multi --save_model <multi_step_path> --load_model <multi_step_path> --use_iou_loss True
```

### Training image-reconstructor

TODO

## Data Generation

The break-up of the dataset used is defined in [curriculum.json](./data_generation/curriculum.json). It can be modified to generate examples of any particular kind.

Each entry in categories has the following paramaters:
- **type**: *any* (cube/lego/dice) or *cube* (only cube)
- **num_objects**: number of objects in scene
- **steps**: number of steps in instruction
- **relational**: *true* or *false*, whether it contains relational attributes to refer to objects (e.g. the block which is to the left of yellow cube)
- **language**: *simple* or *complex*
- **count**: number of examples to be generated for this category. Note that *count/train_count_downscale* examples are generated for train set, and similarly for val and test set

After setting up curriculum.json, run
```
cd data_generation
./construct_dataset.sh
```

## Object Detector Integration

TODO

## Acknowledgements

This work uses and adapts code from the following open-source projects

#### NSCL 
Repo: [https://github.com/vacancy/NSCL-PyTorch-Release](https://github.com/vacancy/NSCL-PyTorch-Release)        
License: [MIT](https://github.com/vacancy/NSCL-PyTorch-Release/blob/master/LICENSE)

#### Cliport (adapted for baseline)
Repo: [https://github.com/cliport/cliport](https://github.com/cliport/cliport)  
License: [Apache](https://github.com/cliport/cliport/blob/master/LICENSE)

## Citation

```bibtex
@inproceedings{Kalithasan2023NSRM,
	title={{Learning Neuro-symbolic Programs for Language Guided Robot Manipulation}},
	author={Kalithasan, Namasivayam and Singh, Himanshu and Bindal, Vishal and Tuli, Arnav and Agrawal, Vishwajeet and Jain, Rahul and Singla, Parag and Paul, Rohan},
	booktitle={IEEE International Conference on Robotics and Automation},
	year={2023}
}
```


