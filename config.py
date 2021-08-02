class Config:
    scale = 2
    exp_name = 'degug'
    exp_dir = f'{exp_name}_x{scale}'

    num_iters = 10000000
    batch_size = 1
    lr_patch_size = 96
    num_workers = 1

    n_resblocks = 32
    n_feats = 256
    rgb_range = 255
    n_colors = 3
    res_scale = 0.1

    data_dir = 'dataset'
    div2k = f'{data_dir}/DIV2K_train_HR/*.png'
    flickr = f'{data_dir}/Flickr2K/*.png'
    valid_imgs = f'{data_dir}/DIV2K_valid_HR/*.png'
