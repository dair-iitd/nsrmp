import torch

def get_split(data, inds, device):
	tdata = dict()
	for key in data:
		if data[key].__class__==torch.Tensor:
			tdata[key] = data[key][inds].to(device)
		elif data[key].__class__==list:
			tdata[key] = list(data[key][i] for i in inds)
		elif data[key].__class__==dict and not key.startswith('split'):
			tdata[key] = get_split(data[key], inds, device)     
	return tdata

def filter_nolego(data, device):
    x = data['has_lego']
    inds = [i for i in range(len(x)) if not x[i]]
    return get_split(data, inds, device)

def filter_step(data, device, step=1):
    x = data['gprogram']
    inds = [i for i in range(len(x)) if len(x[i])==step]
    return get_split(data, inds, device)
def filter_step_lego(data, device, step=1):
    x = data['gprogram']
    y = data['has_lego']
    inds = [i for i in range(len(x)) if len(x[i])==step and y[i]]
    return get_split(data, inds, device)

def filter_onedepth(data, device):
    def is_1depth(concepts):
        return len(concepts) < 3 
    inds = [i for i in range(len(data['objects'])) if is_1depth(data['instruction_concepts'][i])]
    return get_split(data, inds, device)
 
def filter_onestep_onedepth(data, device):
    data = filter_onedepth(data, device)
    return filter_step(data, device, step=1)
 

def add_symbolic_program(train_dict):
    symbolic_programs = []
    for i,(instruction_concepts,gprogram) in enumerate(zip(train_dict['instruction_concepts'],train_dict['gprogram'])):
        symbolic_programs.append(dict(o1=1,o2=0,o1p=[instruction_concepts[0]],o2p=[instruction_concepts[1]],lp1=0.0,lp2=0.0))
    train_dict['symbolic_programs']=symbolic_programs 
    return 
        