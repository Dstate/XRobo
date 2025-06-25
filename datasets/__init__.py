from .LiberoEngine import build_libero_engine
from .CalvinEngine import build_calvin_engine
from .SimplerBridgeEngine import build_simpler_bridge_engine
from .SimplerRT1Engine import build_simpler_rt1_engine
from ._JointDataLoader import build_joint_dataloader

ENGINE_REPO = {
    'build_libero_engine': build_libero_engine,
    'build_calvin_engine': build_calvin_engine,
    'build_simpler_bridge_engine': build_simpler_bridge_engine,
    'build_simpler_rt1_engine': build_simpler_rt1_engine,
    'build_joint_dataloader': build_joint_dataloader
}

def create_engine(engine_name, **kwargs):
    return ENGINE_REPO[engine_name](**kwargs)