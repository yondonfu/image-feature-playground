import torch
import PIL
import io
from processing import (
    KandinskyV22FeatureSteeringPipeline,
    pipeline_preprocess_image,
    pipeline_generate_image,
)
from diffusers import KandinskyV22Pipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from sae import load_sae
from util import get_torch_device, dict_convert_tensors_to_lists
from fastapi import FastAPI, Depends, UploadFile, File, Form
from contextlib import asynccontextmanager
from dependencies import get_pipeline
from typing import Annotated, List
from pydantic import BaseModel

KANDINSKY_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
SAE_MODEL_ID = "gytdau/clip-sae-128"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.pipeline = load_pipeline()
    yield


def load_pipeline():
    device = get_torch_device()
    dtype = torch.float16

    # Load models
    kandinsky = KandinskyV22Pipeline.from_pretrained(
        KANDINSKY_MODEL_ID, torch_dtype=dtype
    ).to(device)
    sae = load_sae(SAE_MODEL_ID, device=device, dtype=dtype)
    image_processor = CLIPImageProcessor.from_pretrained(
        KANDINSKY_PRIOR_MODEL_ID, subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        KANDINSKY_PRIOR_MODEL_ID, subfolder="image_encoder"
    ).to(device, dtype=dtype)

    return KandinskyV22FeatureSteeringPipeline(
        pipeline=kandinsky,
        sae=sae,
        image_encoder=image_encoder,
        image_processor=image_processor,
        device=device,
        dtype=dtype,
    )


app = FastAPI()


@app.post("/preprocess_image")
async def preprocess(
    image: Annotated[UploadFile, File()],
    n_features: Annotated[int, Form()] = 3,
    generate_icon: Annotated[bool, Form()] = True,
    pipeline=Depends(get_pipeline),
):
    image_bytes = await image.read()

    image_file = PIL.Image.open(io.BytesIO(image_bytes))
    icon_image_file = None
    if generate_icon:
        icon_image_file = PIL.Image.open("/root/data/cube_icon.png")

    return dict_convert_tensors_to_lists(
        pipeline_preprocess_image(pipeline, image_file, n_features, icon_image_file)
    )


class ImageGenerateParams(BaseModel):
    image_embeds: list
    inv_latents: list
    feature_idx: List[int]
    feature_val: List[float]
    height: int = 768
    width: int = 768


@app.post("/generate_image")
def generate(params: ImageGenerateParams, pipeline=Depends(get_pipeline)):
    device = get_torch_device()
    dtype = torch.float16

    return pipeline_generate_image(
        pipeline,
        torch.tensor(params.image_embeds).to(device=device, dtype=dtype),
        torch.tensor(params.inv_latents).to(device=device, dtype=dtype),
        params.feature_idx,
        params.feature_val,
        params.height,
        params.width,
    )
