import os


class Config:
    _base_dir = '' if 'adaptive_bitrate_streaming' in os.getcwd() else 'adaptive_bitrate_streaming/'
    baseline_model_paths = {
        'genet': _base_dir + 'data/all_models/genet/nn_model_ep_9900.ckpt',
        'udr_1': _base_dir + 'data/all_models/udr_1/nn_model_ep_57600.ckpt',
        'udr_2': _base_dir + 'data/all_models/udr_2/nn_model_ep_52400.ckpt',
        'udr_3': _base_dir + 'data/all_models/udr_3/nn_model_ep_58000.ckpt',
        'udr_real': _base_dir + 'data/all_models/udr_real/nn_model_ep_49000.ckpt',
    }
    
    trace_dirs = {
        'fcc-train': '/content/drive/MyDrive/abr-gym/netllm/train/fcc-train',
        'fcc-valid': '/content/drive/MyDrive/abr-gym/netllm/valid/fcc-valid',
        'fcc-test': '/content/drive/MyDrive/abr-gym/netllm/test/fcc-test'
    }

    video_size_dirs = {
        'video1': '/content/drive/MyDrive/abr-gym/netllm/video/',
        'video2': _base_dir + 'data/videos/video2_sizes/',
    }

    results_dir = '/content/abr-gym/DATASET/artifacts/tmp/'
    exp_pools_dir = '/content/drive/MyDrive/abr-gym/netllm/exp_pool.pkl'

    # plm special
    plm_types = ['gpt2', 'llama', 'llava', 't5-lm', 'opt', 'mistral']
    plm_sizes = ['xxs', 'xs', 'small', 'base', 'large', 'xl', 'xxl']  # note that the actual size of plm is dependent on the type of plm. 
                                                         # for example, for llama, 'base' is 7b, while for gpt2, 'base' is 340M. you can specify it yourself.
    plm_dir = '/content/drive/MyDrive/abr-gym/huggingface_models/Llama-2-7b-hf'
    plm_ft_dir = '/content/drive/MyDrive/abr-gym/try_llama/try_llama2_7b'
    plm_embed_sizes = {
        'gpt2': {
            'base': 1024,
            'small': 768,
            'large': 1280,
            'xl': 1600,
        },
        'llama': {
            'base': 4096,
        },
        't5-lm': {
            'base': 768,
            'small': 512,
            'large': 4096,
            'xl': 2048,
        },
        'llava': {
            'base': 4096,
        },
        'mistral': {
            'base': 4096,
        },
        'opt': {
            'large': 5120,
            'base': 4096,
            'small': 2560,
            'xs': 2048,
            'xxs': 512,
        },
    }
    plm_layer_sizes = {
        'gpt2': {
            'base': 24,
            'small': 12,
            'large': 36,
            'xl': 48
        },
        'llama': {
            'base': 32,
        },
        't5-lm': { 
            'base': 12,
            'small': 6,
            'large': 24,
            'xl': 24
        },
        'llava': {
            'base': 32,
        },
        'mistral': {
            'base': 32,
        },
        'opt': {
            'large': 40,
            'base': 32,
            'small': 32,
            'xs': 32,
            'xxs': 16,
        },
    }


cfg = Config()

