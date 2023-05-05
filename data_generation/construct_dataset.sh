python construct_dataset.py

python move_files.py --root_dir .

python get_scenes_json.py --dataset_dir train --split train --out_path scenes-train.json
python get_instructions_json.py --dataset_dir train --split train --out_path instructions-train.json

python get_scenes_json.py --dataset_dir test --split test --out_path scenes-test.json
python get_instructions_json.py --dataset_dir test --split test --out_path instructions-test.json

python get_scenes_json.py --dataset_dir val --split val --out_path scenes-val.json
python get_instructions_json.py --dataset_dir val --split val --out_path instructions-val.json
