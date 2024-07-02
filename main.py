from modal import App, Image, gpu, build, enter, web_endpoint, method, Mount
from typing import List, Annotated
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
import io

KANDINSKY_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
SAE_MODEL_ID = "gytdau/clip-sae-128"

app = App("image-feature-playground")

kandinsky_image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install(
        # Using fork of diffusers to support using DDIMInverseScheduler with KandinskyV22
        # See https://github.com/yondonfu/diffusers/commit/8eaa4e8d2adf95d6af695af06bcfd3615bb603fe
        "git+https://github.com/yondonfu/diffusers.git@kandinsky-ddim-inverse",
        "Pillow",
        "transformers",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

base_image = Image.debian_slim().pip_install("Pillow")

with kandinsky_image.imports():
    import torch
    import PIL.Image
    from huggingface_hub import snapshot_download
    from processing import (
        KandinskyV22FeatureSteeringPipeline,
        pipeline_preprocess_image,
        pipeline_generate_image,
    )
    from diffusers import (
        KandinskyV22Pipeline,
    )
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )
    from sae import load_sae
    from util import get_torch_device, dict_convert_tensors_to_lists

with base_image.imports():
    import PIL.Image


@app.cls(gpu=gpu.A10G(), image=kandinsky_image)
class Processor:
    @build()
    def download_models(self):
        ignore_patterns = ["prior"]
        snapshot_download(KANDINSKY_PRIOR_MODEL_ID, ignore_patterns=ignore_patterns)
        snapshot_download(KANDINSKY_MODEL_ID)
        snapshot_download(SAE_MODEL_ID)

    @enter()
    def enter(self):
        device = get_torch_device()
        dtype = torch.float16

        kandinsky = KandinskyV22Pipeline.from_pretrained(
            KANDINSKY_MODEL_ID, torch_dtype=dtype
        ).to(device)
        sae = load_sae(SAE_MODEL_ID, device=device, dtype=dtype)
        image_processor = CLIPImageProcessor.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID, subfolder="image_processor"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID,
            subfolder="image_encoder",
            torch_dtype=dtype,
        ).to(device)

        self.pipeline = KandinskyV22FeatureSteeringPipeline(
            pipeline=kandinsky,
            sae=sae,
            image_encoder=image_encoder,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )

    @method()
    def preprocess(
        self, image: PIL.Image, n_features: int = 3, icon_image: PIL.Image = None
    ):
        return dict_convert_tensors_to_lists(
            pipeline_preprocess_image(self.pipeline, image, n_features, icon_image)
        )

    @method()
    def generate(
        self,
        image_embeds: list,
        inv_latents: list,
        feature_idx: List[int],
        feature_val: List[float],
        height: int = 768,
        width: int = 768,
    ):
        device = get_torch_device()
        dtype = torch.float16

        return pipeline_generate_image(
            self.pipeline,
            torch.tensor(image_embeds).to(device=device, dtype=dtype),
            torch.tensor(inv_latents).to(device=device, dtype=dtype),
            feature_idx,
            feature_val,
            height,
            width,
        )


class CLIPFeature(BaseModel):
    idx: int
    activation: float
    # List of base64 data urls for icon images
    icon: List[str]


class ImagePreprocessResult(BaseModel):
    image_embeds: list
    inv_latents: list
    features: List[CLIPFeature]
    height: int
    width: int


@app.function(
    mounts=[
        Mount.from_local_dir(
            "./data",
            remote_path="/root/data",
        )
    ],
    image=base_image,
)
@web_endpoint(method="POST")
async def preprocess_image(
    image: Annotated[UploadFile, File()],
    n_features: Annotated[int, Form()] = 3,
    generate_icon: Annotated[bool, Form()] = True,
) -> ImagePreprocessResult:
    image_bytes = await image.read()

    image_file = PIL.Image.open(io.BytesIO(image_bytes))
    icon_image_file = None
    if generate_icon:
        icon_image_file = PIL.Image.open("/root/data/cube_icon.png")

    return Processor().preprocess.remote(image_file, n_features, icon_image_file)


class ImageGenerateParams(BaseModel):
    image_embeds: list
    inv_latents: list
    feature_idx: List[int]
    feature_val: List[float]
    height: int = 768
    width: int = 768


@app.function(image=base_image)
@web_endpoint(method="POST")
def generate_image(params: ImageGenerateParams):
    return Processor().generate.remote(**params.model_dump())
