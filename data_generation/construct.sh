#Change the class inherited by the construct.py file to ensure the type of command to be used, can be automated in future. 
#num_samples and data_dir can be taken from user-- change done
#usage: ./construct.sh num_samples data_dir
#Size of the dataset

num_samples=$1
data_dir=$2
for((i=0; i<$num_samples; i++))
do
    echo Generating sample no $i 
    python3 construct.py --dataset_dir $data_dir --template_file ./panda/construct/templates/RelationalDoubleStep.json --metadata_file ./panda/construct/metadata.json 
    echo $'\n'
done

# python3 get_scenes_json.py --dataset_dir $data_dir  --out_path $data_dir/scenes.json
# python3 get_instructions_json.py --dataset_dir $data_dir  --out_path $data_dir/instructions.json
