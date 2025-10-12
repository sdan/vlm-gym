def create_env(env_name, tokenizer, **env_kwargs):
    env_name = env_name.lower()
    if env_name == 'nlvr2' or env_name == 'truefalse':
        from vlmrl.envs.nlvr2 import NLVR2Env
        env = NLVR2Env(tokenizer, **env_kwargs)
    elif env_name in {'vision', 'vision_caption', 'caption'}:
        from vlmrl.envs.vision_caption import VisionCaptionEnv
        env = VisionCaptionEnv(tokenizer, **env_kwargs)
    elif env_name == 'osv5m' or env_name == 'geospot':
        from vlmrl.envs.osv5m import OSV5MEnv
        env = OSV5MEnv(tokenizer, **env_kwargs)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env
