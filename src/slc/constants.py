from enum import StrEnum


class DatasetNames(StrEnum):
    CUB = "CUB"
    NABIRDS = "NABIRDS"
    IMAGENET = "IMAGENET"
    FLOWERS = "FLOWERS"
    MNIST = "MNIST"
    CELEBA = "CELEBA"
    CELEBA_HQ = "CELEBA_HQ"
    ZEBRA_SORREL = "ZEBRA_SORREL"
    CHEETAH_COUGAR = "CHEETAH_COUGAR"
    EGYPTIAN_PERSIAN = "EGYPTIAN_PERSIAN"


class SDVAEPretrainedModels(StrEnum):
    SD15 = "runwayml/stable-diffusion-v1-5"
    OSTRIS = "ostris/vae-kl-f8-d16"


DATASET_ROOTS = {}


def get_imagenet_sae_model_path():
    return "/local/scratch/carlyn.1/rcvcf/saes/imagenet_sae.pt"


def get_celeba_sae_model_path():
    return "/local/scratch2/carlyn.1/rcvcf/saes/celeba_sae.pt"


def get_celeba_hq_sae_model_path():
    return "/local/scratch2/carlyn.1/rcvcf/saes/celeba_hq_sae.pt"


def get_cub_sae_model_path():
    return "/local/scratch/carlyn.1/rcvcf/saes/cub_dino_sae.pt"


def get_zebra_sorrel_label_map():
    return {
        0: 339,  # sorrel
        1: 340,  # zebra
    }


def get_cheetah_cougar_label_map():
    return {
        0: 293,  # cheetah
        1: 286,  # cougar
    }


def get_egyptian_persian_label_map():
    return {
        0: 285,  # egyptian cat
        1: 283,  # persian cat
    }
