
class config:
    epochs = 30
    batch_size = 4
    # lr = 1e-4
    # lr = 0.1
    lr = 0.001
    # lr = 3e-4
    seed = 42
    mixup = 0
    cutmix = 0
    accumulation_steps = 1
    single_fold = 0
    folds = 5
    apex = False
    scale = False # doesnt work in kaggle kernals

cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet18',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },

    'raster_params': {
        'raster_size': [400, 400],
        # 'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'data/aerial_map/aerial_map.png',
        'semantic_map_key': 'data/semantic_map/semantic_map.pb',
        'dataset_meta_key': 'data/meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'data/train.zarr',
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 0
    },

    'val_data_loader': {
        'key': 'data/validate.zarr',
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'max_num_steps': 30000,
        'checkpoint_every_n_steps': 1000,
    }
}