import os
from NAS_Search import SearchDelegator
from random_generator import RandomGenerator
from default_configs.default_config import DefaultConfig



SEED = 111111


def make_dir(_directory):
    if not os.path.exists(_directory):
        os.makedirs(_directory)

def create_directories(data_name):
    make_dir('./output')
    make_dir(os.path.join(DefaultConfig.get_config('valid_pred_imgs'), data_name))
    make_dir(os.path.join(DefaultConfig.get_config('test_pred_imgs'), data_name))
    make_dir('./restore')
    make_dir(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), data_name))
    make_dir(os.path.join(DefaultConfig.get_config('models_architecture_save_dir'), data_name))
    make_dir(os.path.join(DefaultConfig.get_config('snapshots_save_dir'), data_name))



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
i = 0
data_names = ['CIKM', 'Shanghai']
create_directories(data_names[i])

RandomGenerator.setup(SEED)
search_delegator = SearchDelegator(data_names[i])
search_delegator.search()
