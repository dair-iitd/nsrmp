from py import process
from construct import construct_main
import json
import configs
import random
import os
import subprocess
import tempfile
import json

template_file = {
    (1, False): "SingleStep.json",
    (1, True): "RelationalSingleStep.json",
    (2, False): "DoubleStep.json",
    (2, True): "RelationalDoubleStep.json",
    (6, False) :  "SixStep.json"
}
template_file_prefix = './panda/construct/templates/'
metadata_file = './panda/construct/metadata.json'

with open('curriculum.json', 'r') as f:
    data = json.load(f)

processes = []

for dataset in ['train']:
    count_downscale = data[dataset + '_count_downscale']

    for c, category in enumerate(data['categories']):
        f = tempfile.TemporaryFile()
        dir_name = 'tmp_' + dataset + "-" + str(c)
        num_examples = category['count'] // count_downscale
        command = f'python construct.py --template_file {os.path.join(template_file_prefix, template_file[(category["steps"], category["relational"])])} --metadata_file {metadata_file} --dataset_dir {dir_name} --type {category["type"]} --max_objects {category["num_objects"]} --language {category["language"]} --num_examples {num_examples}'
        p = subprocess.Popen(command.split(), stdout=f)
        processes.append((p,f))

logfile = open('log.txt', 'wb')

for p, f in processes:
    p.wait()
    f.seek(0)
    logfile.write(f.read())
    f.close()
