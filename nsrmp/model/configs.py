from helpers.utils.container import DOView 
import numpy as np

from helpers.logging.logger import get_logger
logger = get_logger(__file__)

class Config(DOView):
    def print_configs(self):
        print(self.__dict__)

configs = Config()

configs.data = DOView()
configs.model = DOView()
configs.model.parser = DOView()
configs.train = DOView()
configs.others = DOView()

######################################################################
#data related configs
configs.data.bbox_mode = 'yxhw'
configs.data.image_shape = [256,384] #[H,W]
configs.data.object_feature_bbox_size = [6,6]
configs.data.relational_feature_bbox_size = [12,12]
configs.data.group_concepts = []


######################################################################
#model related configs
#1)visual
configs.model.visual_feature_dim = [0,256,256]
configs.fixed_resnet = False
 
#2)parser
configs.model.parser.use_bert = False
configs.model.parser.hidden_dim = 64
configs.model.parser.gru_nlayers = 2
configs.model.parser.word_embedding_dim = 32
configs.model.parser.positional_embedding_dim = 32
configs.model.parser.use_splitter = True

#3) Splitter
configs.model.splitter_use_bert = False

configs.model.splitter_hidden_dim = 256
configs.model.splitter_gru_nlayers = 1
configs.model.splitter_word_embedding_dim = 256
configs.model.splitter_positional_embedding_dim = None
configs.model.program_max_step = 5

#4)concept emb and executor
configs.model.make_concept_dict = False
configs.model.emb_dim = {
    'attribute_concepts':20,
    'relational_concepts': 20,
    # 'action_concepts': 60
}

#5)losses 
configs.reward_offset = 1.75
configs.discount = 0.9

#######################################################################
#Training related configs





#####################################################################
#Others
