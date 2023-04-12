jac-run scripts/train.py --use_symbolic_programs=True  --save_model_to_file=model.pth --epochs=200 

jac-run scripts/train.py --use_symbolic_programs=True --load_model_from_file=model.pth --freeze_concept_embeddings=True --epochs=200 


# Training new implementation

## Train Concepts

```bash
python3 scripts/train_new.py --dataset roboclevr --datadir ../data --train_scenes_json ../data/scenes_train_concept.json --train_instructions_json ../data/instructions_train_concept.json --vocab_json ../data/vocab.json  --use_cuda True --batch_size 32 --num_epochs 200  --save_model_to_file model_action_concept.pth --model_save_interval 10 --training_target concept_embeddings --wandb True --load_model_from_file model_new3.pth
```

## Train parser alone after training concepts
```bash
jac-crun 0 scripts/train_new.py --dataset roboclevr --datadir ../data --train_scenes_json ../data/scenes_train_parser.json --train_instructions_json ../data/instructions_train_parser.json --vocab_json ../data/vocab.json --instruction_transform program_parser_candidates --use_cuda True --batch_size 32 --num_epochs 50  --save_model_to_file model_parser.pth --model_save_interval 2 --training_target parser  --load_model_from_file model_action_concept.pth --wandb True --eval_interval 2

```

## Train all
```bash
jac-crun 0 scripts/train_new.py --dataset roboclevr --datadir ../data --train_scenes_json ../data/scenes_train_parser.json --train_instructions_json ../data/instructions_train_parser.json --vocab_json ../data/vocab.json --instruction_transform program_parser_candidates --use_cuda True --batch_size 32 --num_epochs 100  --save_model_to_file model_end_to_end.pth --model_save_interval 5 --training_target all   --wandb True --eval_interval 10
```

# Testing/Evaluating the model
## Testing concept/action 
```bash
python3 scripts/eval.py --dataset roboclevr --datadir ../data --train_scenes_json ../data/scenes_train_concept.json --train_instructions_json ../data/instructions_train_concept.json --vocab_json ../data/vocab.json  --use_cuda True --batch_size 32 --load_model_from_file model_action_concept.pth --training_target concept_embeddings
```



jac-crun 0 scripts/eval.py --dataset roboclevr --datadir ../data --train_scenes_json ../data/scenes_train_concept.json --train_instructions_json ../data/instructions_train_concept.json --vocab_json ../data/vocab.json  --use_cuda False --batch_size 32 --load_model_from_file model_action_concept.pth --training_target concept_embeddings