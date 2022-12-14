class Config:
    feature_size = 64
    feature_length = 2
    feature_extra_length = 1

    validate_start = 0
    validate_interval = 4

    bushy = True

    batch_size = 128
    epochs = 200
    eval_mode = 100

    beam_width = 4
    memory_size = 4000

    sql_timeout_limit = 4
    resample_weight_cap = (0.5, 2)
    #resample_mode = 'replace'
    resample_mode = 'augment'
    resample_amount = 0

    use_hint = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
