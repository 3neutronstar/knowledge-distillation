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
|Type|Teacher|Student|Best Eval Accuracy(%)|
|:---:|:---:|---:|---:|
|Baseline|None|ConvNet|56.22|
|SoftTarget(KL)|VGG16|ConvNet|**61.21**|
|SoftTarget(MSE)|VGG16|ConvNet|**----**|
