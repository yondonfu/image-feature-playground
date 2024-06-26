from modal import App, Image, gpu, build, enter, web_endpoint, method, Mount
from typing import Union, List, Annotated
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
import base64
import io

KANDINSKY_PRIOR_MODEL_ID = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
SAE_MODEL_ID = "gytdau/clip-sae-128"

app = App("image-feature-playground")

clip_image = (
    Image.debian_slim()
    .pip_install(
        "Pillow", "transformers", "accelerate", "huggingface_hub", "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

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

with clip_image.imports():
    import torch
    import PIL.Image
    from huggingface_hub import snapshot_download
    from transformers import (
        CLIPImageProcessor,
        CLIPTokenizer,
        CLIPVisionModelWithProjection,
        CLIPTextModelWithProjection,
    )

with kandinsky_image.imports():
    import torch
    import numpy as np
    import PIL.Image
    import os
    import importlib
    from huggingface_hub import snapshot_download
    from diffusers import (
        KandinskyV22Pipeline,
        DDIMScheduler,
        DDIMInverseScheduler,
    )

with base_image.imports():
    import PIL.Image


def get_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@app.cls(gpu=gpu.A10G(), image=clip_image)
class CLIP:
    @build()
    def download_models(self):
        # We are working directly with CLIP embeddings and do not need the prior
        ignore_patterns = ["prior"]
        snapshot_download(KANDINSKY_PRIOR_MODEL_ID, ignore_patterns=ignore_patterns)

    @enter()
    def enter(self):
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID, subfolder="image_processor"
        )
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID, subfolder="tokenizer"
        )
        self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID, subfolder="image_encoder"
        ).to(get_torch_device())
        self.clip_text_encoder = CLIPTextModelWithProjection.from_pretrained(
            KANDINSKY_PRIOR_MODEL_ID, subfolder="text_encoder"
        ).to(get_torch_device())

    @method()
    def embed_image(self, images: Union[PIL.Image, List[PIL.Image]]):
        return self._embed_image(images).tolist()

    @method()
    def embed_zero(self):
        return self._embed_zero().tolist()

    def _embed_image(self, images: Union[PIL.Image, List[PIL.Image]]):
        inputs = (
            self.clip_image_processor(images, return_tensors="pt")
            .pixel_values[0]
            .unsqueeze(0)
            .to(get_torch_device())
        )
        return self.clip_image_encoder(inputs)["image_embeds"]

    def _embed_text(self, text: Union[str, List[str]]):
        inputs = self.clip_tokenizer(
            text,
            padding="max_length",
            max_length=self.clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = inputs.input_ids.to(get_torch_device())
        return self.clip_text_encoder(text_input_ids).text_embeds

    def _embed_zero(self, batch_size=1):
        zero_img = torch.zeros(
            1,
            3,
            self.clip_image_encoder.config.image_size,
            self.clip_image_encoder.config.image_size,
        ).to(device=get_torch_device(), dtype=self.clip_image_encoder.dtype)
        zero_image_emb = self.clip_image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb


# SAE related code from https://colab.research.google.com/drive/1m6IRqKaRUWRL-d2PC82nslb7kmAgjoKV?usp=sharing
@app.cls(gpu=gpu.A10G(), image=kandinsky_image)
class Kandinsky:
    @build()
    def download_models(self):
        snapshot_download(KANDINSKY_MODEL_ID)
        snapshot_download(SAE_MODEL_ID)

    @enter()
    def enter(self):
        self.kandinsky = KandinskyV22Pipeline.from_pretrained(
            KANDINSKY_MODEL_ID, torch_dtype=torch.float16
        ).to(get_torch_device())
        self.kandinsky_scheduler = DDIMScheduler.from_config(
            self.kandinsky.scheduler.config
        )
        self.kandinsky_invert_scheduler = DDIMInverseScheduler.from_config(
            self.kandinsky.scheduler.config
        )

        def load_sae():
            # The model should be cached already and this should just return the local file path
            sae_file_path = snapshot_download(SAE_MODEL_ID)
            model_path = os.path.join(sae_file_path, "model.py")
            weights_path = os.path.join(sae_file_path, "sparse_autoencoder_128.pth")

            spec = importlib.util.spec_from_file_location("model_module", model_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)

            SAE = getattr(model_module, "SparseAutoencoder")

            embedding_dim = 1280
            hidden_dim = embedding_dim * 128
            model = SAE(embedding_dim, hidden_dim).to(
                device=get_torch_device(), dtype=torch.float16
            )
            model.load_state_dict(
                torch.load(weights_path, map_location=get_torch_device())
            )
            model.eval()

            return model

        self.sae = load_sae()

    @method()
    def invert(
        self,
        image: PIL.Image,
        image_embeds: list,
        negative_image_embeds: list,
        height: int = 768,
        width: int = 768,
    ):
        # Copied from https://github.com/huggingface/diffusers/blob/5b51ad0052f587c55a7fa843fff9c6e7f3db0372/src/diffusers/pipelines/kandinsky2_2/pipeline_kandinsky2_2_controlnet_img2img.py#L111
        def prepare_image(pil_image: PIL.Image, w=768, h=768):
            pil_image = pil_image.resize(
                (w, h), resample=PIL.Image.BICUBIC, reducing_gap=1
            )
            arr = np.array(pil_image.convert("RGB"))
            arr = arr.astype(np.float32) / 127.5 - 1
            arr = np.transpose(arr, [2, 0, 1])
            image = torch.from_numpy(arr).unsqueeze(0)
            return image

        inputs = prepare_image(image, w=width, h=height).to(
            device=get_torch_device(), dtype=torch.float16
        )
        with torch.no_grad():
            image_latents = self.kandinsky.movq.encode(inputs)["latents"]

        image_embeds = torch.tensor(image_embeds).to(
            device=get_torch_device(), dtype=torch.float16
        )
        negative_image_embeds = torch.tensor(negative_image_embeds).to(
            device=get_torch_device(), dtype=torch.float16
        )

        # Switch to inverse scheduler to perform DDIM inversion
        self.kandinsky.scheduler = self.kandinsky_invert_scheduler

        inv_latents = self.kandinsky(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            latents=image_latents,
            output_type="latent",
            guidance_scale=1,
            num_inference_steps=40,
            height=height,
            width=height,
        )

        # Switch back to normal DDIM scheduler
        self.kandinsky.scheduler = self.kandinsky_scheduler

        return inv_latents.images[0].unsqueeze(0).tolist()

    @method()
    def get_features(self, image_embeds: list, n_features: int = 3):
        image_embeds = torch.tensor(image_embeds).to(
            get_torch_device(), dtype=torch.float16
        )

        features = self.sae.encode(image_embeds).squeeze(0)
        top_n = torch.argsort(features, dim=-1, descending=False)[-n_features:].flip(
            dims=[-1]
        )
        return [
            {"idx": idx.tolist(), "activation": features[idx].tolist()} for idx in top_n
        ]

    @method()
    def generate(
        self,
        image_embeds: list,
        negative_image_embeds: list,
        inv_latents: list,
        feature_idx: List[int],
        feature_val: List[float],
        height: int = 768,
        width: int = 768,
    ):
        image_embeds = torch.tensor(image_embeds).to(
            get_torch_device(), dtype=torch.float16
        )
        negative_image_embeds = torch.tensor(negative_image_embeds).to(
            get_torch_device(), dtype=torch.float16
        )
        inv_latents = torch.tensor(inv_latents).to(
            get_torch_device(), dtype=torch.float16
        )

        # Ensure we are using normal DDIM scheduler and not inverse scheduler
        self.kandinsky.scheduler = self.kandinsky_scheduler

        # Calculate reconstruction loss
        features = self.sae.encode(image_embeds).squeeze(0)
        decoded = self.sae.decode(features)
        recon_loss = image_embeds - decoded

        # Feature intervention
        for idx, val in zip(feature_idx, feature_val):
            features[idx] = val

        # Generate new image embedding
        new_image_embeds = self.sae.decode(features) + recon_loss

        image = self.kandinsky(
            image_embeds=new_image_embeds,
            negative_image_embeds=negative_image_embeds,
            latents=inv_latents,
            guidance_scale=1,
            num_inference_steps=20,
            height=height,
            width=width,
        ).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="png")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(
            "utf-8"
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
    image_embeds = CLIP().embed_image.remote(image_file)
    negative_image_embeds = CLIP().embed_zero.remote()
    # Get top activating `n_features` features
    features = Kandinsky().get_features.remote(image_embeds, n_features)
    inv_latents = Kandinsky().invert.remote(
        image_file, image_embeds, negative_image_embeds
    )

    if generate_icon:
        icon_image_file = PIL.Image.open("/root/data/cube_icon.png")
        icon_image_embeds = CLIP().embed_image.remote(icon_image_file)
        icon_inv_latents = Kandinsky().invert.remote(
            icon_image_file,
            icon_image_embeds,
            negative_image_embeds,
            height=256,
            width=256,
        )

        for feature in features:
            icon_images = []
            feature_idx = feature["idx"]
            feature_val = 0

            for _ in range(0, 6):
                icon_images.append(
                    Kandinsky().generate.remote(
                        icon_image_embeds,
                        negative_image_embeds,
                        icon_inv_latents,
                        [feature_idx],
                        [feature_val],
                        height=256,
                        width=256,
                    )
                )
                feature_val += 10

            feature["icon"] = icon_images

    return {
        "image_embeds": image_embeds,
        "inv_latents": inv_latents,
        "features": features,
    }


class ImageGenerateParams(BaseModel):
    image_embeds: list
    inv_latents: list
    feature_idx: List[int]
    feature_val: List[float]


@app.function(image=base_image)
@web_endpoint(method="POST")
def generate_image(params: ImageGenerateParams):
    negative_image_embeds = CLIP().embed_zero.remote()
    images = Kandinsky().generate.remote(
        **params.model_dump(), negative_image_embeds=negative_image_embeds
    )
    return {"images": images}
