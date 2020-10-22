
class config:
    epochs = 10
    batch_size = 4
    regression = False
    num_classes = 6 - 1
    IMAGE_PATH = 'data/train/'
    lr = 1e-4
    # lr = 3e-4
    N = 36
    sz = 256
    mean = [1.0-0.90949707, 1.0-0.8188697, 1.0-0.87795304],
    std = [0.36357649, 0.49984502, 0.40477625],
    seed = 69420
    mixup = 0
    cutmix = 0
    accumulation_steps = 1
    single_fold = 0
    apex = True