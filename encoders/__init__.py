from .encoder import DAN

def Encoder(model_args):
    name_enc_map = {
        'dan': DAN
    }
    return name_enc_map[model_args.encoder](model_args)

