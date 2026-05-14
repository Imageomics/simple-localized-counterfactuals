from slc.pipelines import LatentEditPipelineConfigs, SAELatentEditPipelineConfigs

configs = LatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=0.1,
    lr=0.1,
    early_stop=True,
    early_stop_threshold=0.95,
    max_iter_steps=500,
    mask_threshold=0.5,
    use_gaussian_blur=True,
    gaussian_blur_sigma=3,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=2,
)

pixel_configs = LatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=1,
    lr=0.01,
    early_stop=True,
    early_stop_threshold=0.95,
    max_iter_steps=500,
    mask_threshold=0.5,
    use_gaussian_blur=True,
    gaussian_blur_sigma=3,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=0.3,
)

pixel_mask_configs = LatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=1,
    lr=0.1,
    early_stop=True,
    early_stop_threshold=0.95,
    max_iter_steps=500,
    mask_threshold=0.6,
    use_gaussian_blur=True,
    gaussian_blur_sigma=10,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=0.3,
)


celeba_configs = LatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=1.0,
    lr=0.1,
    early_stop=True,
    early_stop_threshold=0.6,
    max_iter_steps=500,
    mask_threshold=0.5,
    use_gaussian_blur=True,
    gaussian_blur_sigma=2,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=0.0,
    init_max_iter_steps=10,
)

sae_celeba_configs = SAELatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=0.1,
    sae_sparsity_lambda=0.1,
    lr=0.1,
    early_stop=True,
    early_stop_threshold=0.6,
    max_iter_steps=500,
    mask_threshold=0.5,
    use_gaussian_blur=True,
    gaussian_blur_sigma=2,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=0.0,
    init_max_iter_steps=10,
)

sae_configs = SAELatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=0.1,
    sae_sparsity_lambda=0.1,
    lr=0.1,
    early_stop=True,
    early_stop_threshold=0.95,
    max_iter_steps=500,
    mask_threshold=0.5,
    use_gaussian_blur=True,
    gaussian_blur_sigma=3,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=1,
)

cub_sae_configs = SAELatentEditPipelineConfigs(
    verbose=False,
    sparsity_lambda=0.1,
    sae_sparsity_lambda=0.1,
    lr=0.1,
    early_stop=True,
    early_stop_threshold=0.05,
    max_iter_steps=500,
    mask_threshold=0.7,
    use_gaussian_blur=True,
    gaussian_blur_sigma=4,
    aggregate_region_constraint=True,
    only_positive_gradients=True,
    class_loss_target="target",
    peturb_strength=0.1,
)
