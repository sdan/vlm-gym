def create_env(env_name, tokenizer):
    env_name = env_name.lower()
    if env_name == 'nlvr2':
        from vlmrl.envs.nlvr2 import NLVR2Env
        env = NLVR2Env(tokenizer)
    elif env_name in {'vision', 'vision_caption'}:
        from vlmrl.envs.vision_caption import VisionCaptionEnv
        env = VisionCaptionEnv(tokenizer)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env
