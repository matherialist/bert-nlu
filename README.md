# Dialog System NLU
Tensorflow and Keras Implementation of the state of the art researches in Dialog System NLU. 
Tested on Tensorflow version 1.15.0


## Implemented Papers

- [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
    
### BERT / ALBERT for Joint Intent Classification and Slot Filling

![Joint BERT](img/joint_bert.PNG?)

#### Supported data format:
- Data format as in the paper `Slot-Gated Modeling for Joint Slot Filling and Intent Prediction` (Goo et al):
	- Consists of 3 files:
		- `seq.in` file contains text samples (utterances)
		- `seq.out` file contains tags corresponding to samples from `seq.in`
		- `label` file contains intent labels corresponding to samples from `seq.in`

#### Datasets included in the repo:
- Snips Dataset (`Snips voice platform: an embedded spoken language understanding system for private- by-design voice interfaces` )(Coucke et al., 2018), which is collected from the Snips personal voice assistant. 
	- The training, development and test sets contain 13,084, 700 and 700 utterances, respectively. 
	- There are 72 slot labels and 7 intent types for the training set.

#### Training the model with SICK-like data:
4 models implemented `joint_bert` and `joint_bert_crf` each supports `bert` and `albert`
##### Required Parameters:
- ```--train``` or ```-t``` Path to training data in Goo et al format.
- ```--val``` or ```-v``` Path to validation data in Goo et al format.
- ```--save``` or ```-s``` Folder path to save the trained model.
##### Optional Parameters:
- ```--epochs``` or ```-e``` Number of epochs.
- ```--batch``` or ```-bs``` Batch size.
- ```--type``` or ```-tp``` to choose between `bert` and `albert`. Default is `bert`

```
python train_joint_bert.py --train=data/SH/train --val=data/SH/valid --save=saved_models/joint_bert_model --epochs=5 --batch=64 --type=bert
```

```
python train_joint_bert.py --train=data/SH/train --val=data/SH/valid --save=saved_models/joint_albert_model --epochs=5 --batch=64 --type=albert
```

```
python train_joint_bert_crf.py --train=data/SH/train --val=data/SH/valid --save=saved_models/joint_bert_crf_model --epochs=5 --batch=32 --type=bert
```

```
python train_joint_bert_crf.py --train=data/SH/train --val=data/SH/valid --save=saved_models/joint_albert_crf_model --epochs=5 --batch=32 --type=albert
```


#### Evaluating the Joint BERT / ALBERT NLU model:
##### Required Parameters:
- ```--model``` or ```-m``` Path to joint BERT / ALBERT NLU model.
- ```--data``` or ```-d``` Path to data in Goo et al format.
##### Optional Parameters:
- ```--batch``` or ```-bs``` Batch size.
- ```--type``` or ```-tp``` to choose between `bert` and `albert`. Default is `bert`


```
python eval_joint_bert.py --model=saved_models/joint_bert_model --data=data/SH/test --batch=128 --type=bert
```

```
python eval_joint_bert.py --model=saved_models/joint_albert_model --data=data/SH/test --batch=128 --type=albert
```

```
python eval_joint_bert_crf.py --model=saved_models/joint_bert_crf_model --data=data/SH/test --batch=128 --type=bert
```

```
python eval_joint_bert_crf.py --model=saved_models/joint_albert_crf_model --data=data/SH/test --batch=128 --type=albert
```


#### Running a basic REST service for the Joint BERT / ALBERT NLU model:
##### Required Parameters:
- ```--model``` or ```-m``` Path to joint BERT / ALBERT NLU model.
##### Optional Parameters:
- ```--type``` or ```-tp``` to choose between `bert` and `albert`. Default is `bert`


```
python bert_nlu_basic_api.py --model=saved_models/joint_albert_model --type=albert
```

##### Sample request:
- POST
- Payload: 
```
{
	"utterance": "make me a reservation in south carolina"
}
```

##### Sample Response:
```
{
	"intent": {
		"confidence": "0.9888",
		"name": "BookRestaurant"
	}, 
	"slots": [
	{
		"slot": "state",
		"value": "south carolina",
		"start": 5,
		"end": 6
	}
	]
}
```

