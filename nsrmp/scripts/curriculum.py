import torch
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR

def get_curr(model):

    '''
    add optimizers to optimizers list corresponding to the lesson in the curriculum dict below it 
    '''
    optimizers = [Adam([
                {"params": model.resnet.parameters(),"name":'resnet', "lr": 1e-5},
                {"params": model.parser.parameters(),"name":'parser', "lr": 5e-4},
                {"params": model.visual.parameters(),"name": 'visual', "lr":5e-4},
                {"params" : model.executor.concept_embeddings.parameters(),'name':'concept_emb', "lr" : 1e-4 },
                {"params" : model.executor.action_sim.parameters(),'name':'action_sim', "lr" : 1e-4}
            ]),
            Adam([
                {"params" : model.executor.action_sim.parameters(),'name':'action_sim', "lr" : 9e-5}
            ]),
            Adam([
                {"params": model.resnet.parameters(),"name":'resnet', "lr": 1e-5},
                {"params": model.visual.parameters(),"name": 'visual', "lr":5e-5},
                {"params" : model.executor.concept_embeddings.parameters(),'name':'concept_emb', "lr" : 5e-5, },
            ]),
            Adam([
                {"params": model.resnet.parameters(),"name":'resnet', "lr": 1e-5},
                {"params": model.parser.parameters(),"name":'parser', "lr": 5e-5},         
                {"params": model.visual.parameters(),"name": 'visual', "lr":5e-5},
                {"params" : model.executor.concept_embeddings.parameters(),'name':'concept_emb', "lr" : 5e-5, },
            ]),
            Adam([
                {"params": model.parser.parameters(),"name":'parser', "lr": 5e-4},
            ]),
            Adam([
                {"params" : model.multi_step_builder.splitter.parameters(),'name':'splitter', "lr" : 1e-4, "weight_decay": 1e-4},
            ]),
            Adam([
                {"params": model.resnet.parameters(),"name":'resnet', "lr": 1e-5},
                {"params": model.parser.parameters(),"name":'parser', "lr": 1e-5, "weight_decay":1e-5},
                {"params": model.visual.relation_feature_extract.parameters(),"name": 'visual realtional conv2D', "lr":5e-5},
                {"params":model.visual.relation_feature_fuse.parameters(),"name":"visual relational fuse", "lr":5e-5},
                {"params": model.visual.relation_feature_fc.parameters(),"name":"visual relational fc", "lr":5e-5},
                {"params" : model.executor.concept_embeddings.embedding_relational_concepts.parameters(),'name':'relational concept_emb', "lr" : 5e-5},
            ]),
            Adam([
                {"params": model.resnet.parameters(),"name":'resnet', "lr": 1e-5},
                {"params": model.parser.parameters(),"name":'parser', "lr": 5e-5},
                {"params": model.visual.parameters(),"name": 'visual', "lr":5e-5},
                {"params" : model.executor.concept_embeddings.parameters(),'name':'concept_emb', "lr" : 5e-5 },
                {"params" : model.executor.action_sim.parameters(),'name':'action_sim', "lr" : 5e-5}
            ]),

            ]
    curriculum  =[
        ###################################################################################################################################################
        #Simple training
        {
            "lesson":{'num_epochs':20, 'num_step_range':(0,1), 'scene_size_range':(0,5), 'program_size_range':(0,10), 'language_complexity':'simple'},
            "optimizer" : optimizers[0],
            "scheduler": StepLR(optimizers[0], step_size=5, gamma=0.90),
            "unique_mode" : "softmax",
            'change_group_concepts': ['attribute_concepts'],
            "extras": {'baseline':True}
        },
        {
            "lesson":{'num_epochs':30, 'num_step_range':(0,1), 'scene_size_range':(0,5), 'program_size_range':(0,8), 'language_complexity':'simple'},
            "optimizer" : optimizers[1],
            "scheduler": StepLR(optimizers[1], step_size=2, gamma=0.99),
            "unique_mode" : "argmax",
            "freeze_modules": ['visual', 'parser','concept_embeddings'],
            'change_group_concepts': ['attribute_concepts'],
        },
        {
            "lesson":{'num_epochs':20, 'num_step_range':(0,1), 'scene_size_range':(0,5), 'program_size_range':(0,10)},
            "optimizer" : optimizers[4],
            "scheduler": StepLR(optimizers[4], step_size=2, gamma=0.90),
            "unique_mode" : "argmax",
            "reset_target" : True,
            "freeze_modules": ['visual','concept_embeddings','action_simulator'],
            'change_training_target':'parser',
            "extras": {'baseline':True}
        },
        {
            'lesson':{'num_epochs':10, 'num_step_range':(0,2), 'scene_size_range':(0,5),'remove_relational': False,'length':3072},
            'optimizer': optimizers[5],
            'scheduler': StepLR(optimizers[5], step_size=1, gamma=0.99 ),
            "freeze_modules": ['visual','concept_embeddings','action_simulator','resnet','parser'],
            'change_training_target':'splitter',
            'change_instruction_transform': 'basic',
            "unique_mode" : 'argmax'
        }
    ]

    return curriculum 