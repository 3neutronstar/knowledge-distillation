# knowledge-distillation
Knowledge Distillation

### How To Use
- Run the basic model
```shell script
    python run.py train --model [model]
``` 

- Run Knowledge Self Knowledge Distillation(cs-kd)
```shell script
    python run.py train --model [model] --custom_loss [self-kd name]
``` 
- Run Knowledge Offline Distillation
```shell script
    python run.py train_offkd --model [model] --pretrained_model [teacher model]
``` 


### Performance
- Cifar100, (Temperature: 2.0 in SoftTarget)

|Type|Teacher|Student|Best Eval Accuracy(%)|
|:---:|:---:|---:|---:|
|Baseline|None|ConvNet|56.31|
|SoftTarget(KL)|ResNet20|ConvNet|**61.21**|
|DeepMutualLearning|None|ConvNet|**56.52**|

- Cifar10, (Temperature: 2.0 in SoftTarget)

|Type|Teacher|Student|Best Eval Accuracy(%)|
|:---:|:---:|---:|---:|
|Baseline|None|ConvNet|83.52|
|Baseline|None|ResNet20|90.86|
|DeepMutualLearning|None|ConvNet|83.30|
|DeepMutualLearning|None|ResNet20|91.14|

- MNIST

|Type|Teacher|Student|Best Eval Accuracy(%)|
|:---:|:---:|---:|---:|
|Baseline|None|LeNet5|91.13|
|DeepMutualLearning|None|LeNet5|91.14|
