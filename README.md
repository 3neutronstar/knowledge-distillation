# knowledge-distillation
Knowledge Distillation

### How To Use
- Run the basic model
```shell script
    python run.py train --model [model]
``` 

- Run Knowledge Offline Distillation
```shell script
    python run.py train_offkd --model [model] --pretrained_model [teacher model]
``` 


### Performance
- Temperature: 2.0
|Type|Teacher|Student|Best Eval Accuracy(%)|
|:---:|:---:|---:|---:|
|Baseline|None|ConvNet|56.22|
|SoftTarget(KL)|ResNet20|ConvNet|**61.21**|
|DeepMutualLearning|ConvNet|ConvNet|**--**|