########################
### Baseline Configs ###
########################

# Helper import 
from helpers.utils.container import DOView 

class Config(DOView):
    def print_configs(self):
        print(self.__dict__)

configs = Config()
configs.data = DOView()
configs.model = DOView()

# Data related configs
configs.data.bbox_mode = 'yxhw'
configs.data.image_shape = [256,384]
configs.data.object_feature_bbox_size = [4,4]

# Model related configs
configs.model.fixed_resnet = True
configs.model.train_splitter = False
configs.model.visual_feature_dim = 256
configs.model.parser_hidden_dim = 512
configs.model.gru_nlayers = 2
configs.model.word_embedding_dim = 256
configs.model.positional_embedding_dim = 50
configs.model.use_iou_loss = True
configs.model.data_assoc = True
