from .LiberoEngine import build_libero_engine
from .CalvinEngine import build_calvin_engine
ENGINE_REPO = {
    'build_libero_engine': build_libero_engine,
    'build_calvin_engine': build_calvin_engine
}

def create_engine(engine_name, **kwargs):
    return ENGINE_REPO[engine_name](**kwargs)