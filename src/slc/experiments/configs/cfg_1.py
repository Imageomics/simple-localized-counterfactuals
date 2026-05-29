from slc.pipelines.latent_edit import LatentEditPipelineConfigs

configs = LatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=0.0,
    lr=1,
    early_stop=True,
    early_stop_threshold=0.9,
    max_iter_steps=500,
    mask_threshold=0.2,
    use_gaussian_blur=True,
    gaussian_blur_sigma=1,
    peturb_strength=0.2,
)
