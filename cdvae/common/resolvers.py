from cdvae.common.utils import PROJECT_ROOT
from omegaconf import DictConfig, OmegaConf

def register_resolvers():
    OmegaConf.register_new_resolver("project_root", lambda: PROJECT_ROOT)
    OmegaConf.register_new_resolver("multiply", lambda x, y: x*y)
    