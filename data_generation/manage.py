import os
import cv2
import json
import numpy as np
from PIL import Image

def reorder_samples(dataset_directory):
	samples = os.listdir(dataset_directory)
	samples.sort()
	total_sample = len(samples)
	for i in range(total_sample):
		sample_directory = os.path.join(dataset_directory, samples[i])
		new_directory = os.path.join(dataset_directory, "{0:0=4d}".format(i))
		os.rename(sample_directory, new_directory)

def remove_sample(s):
	os.system(f"rm {s} -r")
	print(".......Sample Deleted: ", s,".......")

def confirm_remove_samples(s):
	print("Delete sample ? [Y/n]")
	if input().lower() == 'y':
		remove_sample(s)

# def show_sample(s):
#     	print(s.split('/')[-1])
# 	from mpl.dataset.loader import get_transitions_folder
# 	from PIL import Image
# 	folders = get_transitions_folder(s)
# 	rgba_images = [np.array(Image.open(os.path.join(f, 'rgba.png'))) for f in folders]
# 	numpy_horizontal = np.hstack(tuple(rgba_images))
# 	cv2.imshow('grid', numpy_horizontal); cv2.waitKey(4000)
# 	confirm_remove_samples(s)

# def show_samples(dataset_directory):
# 	print("Enter The Starting Index: ")
# 	start = int(input())
# 	samples = os.listdir(dataset_directory); samples.sort()
# 	samples = [os.path.join(dataset_directory, s) for s in samples[start:]]
# 	for s in samples: show_sample(s)

# def check_sample(dataset_directory):
# 	print("Enter sample no.")
# 	s = os.path.join(dataset_directory, "{0:0=4d}".format(int(input())))
# 	show_sample(s)
# 	confirm_remove_samples(s)

def get_mask_unique(smpl_dir):
	masks = [os.path.join(smpl_dir, fldr, 'mask.png') for fldr in os.listdir(smpl_dir) if fldr[0] == 'S']
	get_obj_set = lambda mask: set(np.unique(np.array(Image.open(mask))))
	return [(mask, get_obj_set(mask)) for mask in masks]

def check_mask_correctness(smpl_dir):
	unique_answer = set([0, 3, 4, 5, 6, 7, 8])
	msk_obj_set = get_mask_unique(smpl_dir)
	for _, obj_set in msk_obj_set:
		if not obj_set <= unique_answer:
			print("Wrong Mask ", obj_set)
			remove_sample(smpl_dir)
			return False
	return True

def check_instruction_correctness(smpl_dir):
	with open(os.path.join(smpl_dir, 'demo.json'), 'r') as f:
		data = json.load(f)
	instruction = data['instruction'].split(' ')
	if 'CANCEL' in instruction:
		print("Wrong Instruction")
		remove_sample(smpl_dir)
		return False
	return True

def check_labels(dataset_directory):
	samples = os.listdir(dataset_directory)
	samples = [os.path.join(dataset_directory, s) for s in samples]
	samples.sort()
	unique_answer = set([0, 3, 4, 5, 6])
	print("Start Idx"); start_idx = input()
	start_idx = int(start_idx)
	samples = samples[start_idx:]
	for s in samples:
		msk_obj_set = get_mask_unique(s)
		for mask, obj_set in msk_obj_set:
			if obj_set != unique_answer:
				print(mask, obj_set, unique_answer)

def get_instructions(dataset_directory):
	samples = os.listdir(dataset_directory)
	samples = [os.path.join(dataset_directory, s) for s in samples]
	instructions = []
	for s in samples:
		f = open(os.path.join(s, 'demo.json'), 'r')
		data = json.load(f)
		instructions.append(data['instruction'])
		f.close()
	return instructions

def clean_last_run():
  import glob
  list_of_files = glob.glob('runs/train/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  if os.path.isdir(latest_file):
    print(latest_file)
    os.system(f'rm -r "{latest_file}"')
  list_of_files = glob.glob('runs/val/*')
  latest_file = max(list_of_files, key=os.path.getctime)
  if os.path.isdir(latest_file):
    print(latest_file)
    os.system(f'rm -r "{latest_file}"')

def clean_all_runs_model():
  print("Confirm [Y/n]")
  if input().lower() == "y":
    os.system('rm -r runs/');    os.system('mkdir runs')
    os.system('rm -r trained/'); os.system('mkdir trained')

