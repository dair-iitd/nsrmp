#!/usr/bin/env python
# -*-coding:utf-8 -*-
#File    :   definition.py
#Time    :   2022/06/10 21:02:27
#Author  :   Namasivayam K
#Contact :   namasivayam.k@cse.iitd.ac.in
#Desc    :   This fle is a part of NSRM-PyTorch Release

from datasets.definition import DefinitionBase, get_global_defn, set_global_defn
from datasets.roboclevr import instruction_transforms, program_transforms
from datasets.roboclevr.program_transforms import roboclevr_to_nsrm
from PIL import Image
import torch
import os.path as osp
from cached_property import  cached_property
from . import image_transforms as IT
import re

class NSRMDefinition(DefinitionBase):
    parameter_types = ['attribute_concept', 'relational_concept', 'action_concept']
    variable_types = ['object', 'object_set']
    return_types = ['world']

    attribute_concepts= {
        "color":["blue", "green", "red", "cyan", "yellow", "magenta", "white"],
        "type": ["cube", "lego", "dice"],
    } 
 
    relational_concepts = {
        "spatial_relations":["left","right", "top", "behind", "front"]
    }

    action_concepts = {
        'move': ['left', 'right', 'top']
    }

    operation_signatures = [
        ('move', ['action_concept'], ['object','object','world'], 'world' ),
        ('idle',[],[],'world'),
        ('scene', [], [], 'object_set'),
        ('filter', ['attribute_concept'], ['object_set'], 'object_set'),
        ('relate', ['relational_concept'], ['object'], 'object_set'),
        # ('intersect', [], ['object_set', 'object_set'], 'object_set'),
        # ('union', [], ['object_set', 'object_set'], 'object_set'),
    ]

    # action_signatures = [
    #     ('move', ['action_concept'], ['object','object','world'], 'world' ),
    #     ('idle',[],[],'world')
    # ]

    synonyms = {
        "object": ["thing", "object"],
        "sphere": ["sphere", "ball"],
        "cube": ["cube", "box"],
        "block": ["block","brick"],
        "dice": ["dice"],
        "large": ["large", "big"],
        "small": ["small", "tiny", "little"],
        "metal": ["metallic", "metal", "shiny"],
        "rubber": ["rubber", "matte"],
        "top": ["above","atop",'top'],
        "move": ["put", "place", "move"]
      }

    actions = ['move', 'idle']
    word2lemma = { w:k for k,list in synonyms.items() for w in list}
    con2ebd = {
        "attribute_concepts": "<CONCEPTS>",
        "relational_concepts": "<REL_CONCEPTS>",
        "action_concepts": "<ACT_CONCEPTS>"
    }
    ebd2con = {
         "<CONCEPTS>": "attribute_concepts",
        "<REL_CONCEPTS>" : "relational_concepts",
        "<ACT_CONCEPTS>": "action_concepts"
    }
    extra_embeddings = list(con2ebd.values())
    @cached_property
    def nr_actions(self):
        #need to change this later
        return len(self.action_concepts['move'])
           
    @cached_property
    def param_to_concept(self):
        return {p:p+'s' for p in self.parameter_types} 

    @cached_property
    def concept_groups(self):
        return ["attribute_concepts","relational_concepts", "action_concepts"]
  
    @cached_property
    def action_signatures_dict(self):
        return {A[0]: A[1:] for A in self.action_signatures}

    def program_to_nsrmseq(self, program):
            return roboclevr_to_nsrm(program)
    
    @cached_property
    def token2type(self):
        return {"C": 'attribute_concepts', "T":"attribute_concepts", "A":"action_concepts", "R":"relational_concepts" }

    def concept_tokens(self, sent_lexed):
        concept_types = []
        for w in sent_lexed:
            match = re.search('<(\w)(\d+)>', w)
            if match:
                char_token = match.group(1)
                concept_types.append(self.token2type[char_token])
            else:
                match = re.search('<\w*CONCEPTS>', w)
                if match:
                    char_token = match.string
                    concept_types.append(self.ebd2con[char_token])
        return concept_types
        

#MPLDefinition needs to be changed. But currently providing this class for helping the other implementations
class MPLDefinition(DefinitionBase):
        #inspired from definition of Clevr in Mao-et al  
        parameter_types = ['attribute_concept', 'relational_concept', 'attribute']
        variable_types = ['object', 'object_set']
        return_types = ['void']

        operation_signatures = [
        # Part 1: CLEVR dataset.
        ('scene', [], [], 'object_set'),
        ('filter', ['attribute_concept'], ['object_set'], 'object_set'),
        ('relate', ['relational_concept'],['object'], 'object_set'),
        ('move',['relational_concept'],['object','object','void'],'void'),
        ('idle',[],[],'void')
        ]
        ## to change ? 
        attribute_concepts = { 
			 'color': ['blue', 'green', 'red', 'cyan', 'yellow', 'magenta', 'white'],
		     'type': ['cube', 'lego', 'dice']
		}

        relational_concepts = {
            'spatial_relation':['left','right','top','behind','front']
        }


        synonyms = {
            "object": ["thing", "object", "objects", "things"],
            "tray": ["trays"],
            "sphere": ["sphere", "ball"],
            "cube": ["cube", "box"],
            "block": ["block","brick"],
            "dice": ["dice"],
            "large": ["large", "big"],
            "small": ["small", "tiny", "little"],
            "metal": ["metallic", "metal", "shiny"],
            "rubber": ["rubber", "matte"],
            "top": ["above","atop","top"],
            "move": ["put", "place", "move"]
        }

        actions = ['move', 'idle']


        word2lemma = {
            v: k for k, vs in synonyms.items() for v in vs
        }

        con2ebd = {
        "attribute_concepts": "<CONCEPTS>",
        "relational_concepts": "<REL_CONCEPTS>",
        "attributes":"<ATTRIBUTES>"
        }
        extra_embeddings = list(con2ebd.values())
        
        @cached_property
        def nr_actions(self):
            #need to change this later
            return 4

        @cached_property
        def param_to_concept(self):
            return {p:p+'s' for p in self.parameter_types} 

        @cached_property
        def concept_groups(self):
            return ["attribute_concepts", "relational_concepts"] 

        def program_to_nsrmseq(self, program):
            return roboclevr_to_nsrm(program)

class NSRMImageTransformV1(object):

    def __init__(self,image_shape, bbox_mode,demo_root):
        self.image_shape = image_shape
        self.bbox_mode = bbox_mode
        self.demo_root = demo_root
        from . import image_transforms as IT
        self.transform = IT.Compose([IT.ChangeBoxMode(self.bbox_mode), 
                            IT.DenormalizeBbox(),
                            IT.CropBbox(),
                            IT.Resize(self.image_shape),
                            IT.ToTensor()
                            ]
                    )

    def __call__(self,scene):
        image_path = osp.join(self.demo_root,scene['image_filename'])
        bboxes = [obj['bbox'] for obj in scene['objects']]
        img = Image.open(image_path).convert('RGB')
        objects = []
        for box in bboxes :
            o,b = self.transform(img,[box])
            objects.append(o)
        return torch.stack(objects), torch.tensor(IT.ChangeBoxMode(self.bbox_mode).__call__(None,bboxes)[1])


class NSRMImageTransformV2(object):
    def __init__(self,image_shape,bbox_mode,demo_root):
        self.image_shape = image_shape
        self.bbox_mode = bbox_mode
        self.demo_root = demo_root
        self.transform  =   IT.Compose([
                            IT.ChangeBoxMode(self.bbox_mode), 
                            IT.Resize(self.image_shape), 
                            IT.DenormalizeBbox(),
                            IT.ToTensor()
                            ])

    def __call__(self,scene):
        image_path = osp.join(self.demo_root,scene['image_filename'])
        bboxes = [obj['bbox'] for obj in scene['objects']]
        img = Image.open(image_path).convert('RGB')
        return self.transform(img,bboxes)

class MPLQuestionTransform(object):
    '''
    group_concepts: (List) The conepts types you want to group. For example, if you want to group attribute_concepts, then group_concepts = ['attribute_concepts']
    '''
    def __init__(self, transform, group_concepts:list):
        self.group_concepts = group_concepts
        self.transform = transform
    def __call__(self,sent):
        return self.transform(sent,self.group_concepts)

class NSRMQuestionTransform(object):
    '''
    group_concepts: (List) The conepts types you want to group. For example, if you want to group attribute_concepts, then group_concepts = ['attribute_concepts']
    '''
    def __init__(self, transform, group_concepts:list):
        self.group_concepts = group_concepts
        self.transform = transform
    def __call__(self,sent, sent_lexed):
        return self.transform(sent,sent_lexed, self.group_concepts)


def build_mpl_dataset(args,configs,demo_root,scenes_json,instruction_json, vocab_json = None):
    if get_global_defn() is None:
        set_global_defn(MPLDefinition())
    # image_transform = NSRMImageTransformV1(configs.data.object_feature_bbox_size, configs.data.bbox_mode, demo_root) 
    image_transform = NSRMImageTransformV2(configs.data.image_shape, configs.data.bbox_mode, demo_root)

    if args.instruction_transform == 'off':
        instruction_transform = None
    elif args.instruction_transform == 'basic':
        from .instruction_transforms import encode_sentence
        instruction_transform = MPLQuestionTransform(encode_sentence,group_concepts=configs.data.group_concepts)
    elif args.instruction_transform == 'program_parser_candidates':
        from .program_search import SearchCandidatePrograms
        instruction_transform = SearchCandidatePrograms(group_concepts=configs.data.group_concepts, transform = encode_sentence, template_filename='mpl_program_templates.json')
    else:
        raise ValueError("Unknown instruction transform")

    from datasets.dataset import MPLDataset
    dataset = MPLDataset(args, scenes_json, instruction_json, demo_root, image_transform= image_transform, instruction_transform= instruction_transform, vocab_json = vocab_json )
    return dataset


def build_nsrm_dataset(args,configs,demo_root,scenes_json,instruction_json, vocab_json = None, **kwargs):
   
    if get_global_defn() is None:
        set_global_defn(NSRMDefinition())
   
    image_transform = NSRMImageTransformV2(configs.data.image_shape, configs.data.bbox_mode, demo_root)
    ins_transform = kwargs.get('instruction_transform', args.instruction_transform)
    if ins_transform == 'off':
        instruction_transform = None
    elif ins_transform == 'basic':
        from .instruction_transforms import encode_using_lexed_sentence
        instruction_transform = NSRMQuestionTransform(encode_using_lexed_sentence, group_concepts=configs.data.group_concepts)
    elif ins_transform == 'program_parser_candidates':
        from .instruction_transforms import encode_using_lexed_sentence
        from .program_search import SearchCandidatePrograms
        instruction_transform = SearchCandidatePrograms(group_concepts = kwargs.get('group_concepts',configs.data.group_concepts), transform = encode_using_lexed_sentence, template_filename = 'nsrm_program_templates.json')
    else:
        raise ValueError("Unknown instruction transform")

    from datasets.dataset import NSRMDataset
    dataset = NSRMDataset(args, scenes_json, instruction_json, demo_root, image_transform= image_transform, instruction_transform= instruction_transform, vocab_json = vocab_json )
    return dataset

class SelfsupervisionInstructionTransform(object):
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, sent_raw, sent_lexed):
        return self.transform(sent_raw, sent_lexed)

def build_selfsupervision_dataset(args,configs, owner_dataset, model, max_steps, demo_root, create_target = False, target_dataset_name = None, **kwargs):
    if get_global_defn() is None:
        set_global_defn(NSRMDefinition())
    from .program_transforms import append_program_trees
    program_transform = append_program_trees
    from .instruction_transforms import join_single_step_sentences
    instruction_transform =  join_single_step_sentences

    from datasets.dataset import NSRMParserDataset
    dataset = NSRMParserDataset(owner_dataset, model,max_steps, demo_root, program_transform, instruction_transform, target_dataset_name = target_dataset_name, create_target = create_target, **kwargs)
    return dataset

    