from .LCBC import bc_policy_ddpm_res34_libero, bc_policy_res34_libero, bc_policy_ddpm_res34_calvin
MODEL_REPO = {
    'bc_policy_res34_libero': bc_policy_res34_libero,
    'bc_policy_ddpm_res34_libero': bc_policy_ddpm_res34_libero,
    'bc_policy_ddpm_res34_calvin': bc_policy_ddpm_res34_calvin
}

def create_model(model_name, **kwargs):
    return MODEL_REPO[model_name](**kwargs)