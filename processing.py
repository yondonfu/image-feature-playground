import torch
import PIL
import numpy as np
from diffusers import DDIMScheduler, DDIMInverseScheduler
from typing import List, Union
from util import pil_image_to_b64


def pipeline_preprocess_image(
    pipeline,
    image: PIL.Image,
    n_features: int = 3,
    icon_image: PIL.Image = None,
):
    width, height = image.size
    image_embeds = pipeline.embed_image(image)
    negative_image_embeds = pipeline.embed_zero()
    # Get top activating `n_features` features
    features = pipeline.get_features(image_embeds, n_features)
    inv_latents = pipeline.invert(
        image, image_embeds, negative_image_embeds, height=height, width=width
    )

    if icon_image:
        icon_image_embeds = pipeline.embed_image(icon_image)
        icon_inv_latents = pipeline.invert(
            icon_image,
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
                image = pipeline.generate(
                    icon_image_embeds,
                    negative_image_embeds,
                    icon_inv_latents,
                    [feature_idx],
                    [feature_val],
                    height=256,
                    width=256,
                )
                icon_images.append(pil_image_to_b64(image))
                feature_val += 10

            feature["icon"] = icon_images

    return {
        "image_embeds": image_embeds,
        "inv_latents": inv_latents,
        "features": features,
        "height": height,
        "width": width,
    }


def pipeline_generate_image(
    pipeline,
    image_embeds: torch.Tensor,
    inv_latents: torch.Tensor,
    feature_idx: List[int],
    feature_val: List[float],
    height: int = 768,
    width: int = 768,
):
    negative_image_embeds = pipeline.embed_zero()
    images = pipeline.generate(
        image_embeds=image_embeds,
        inv_latents=inv_latents,
        feature_idx=feature_idx,
        feature_val=feature_val,
        height=height,
        width=width,
        negative_image_embeds=negative_image_embeds,
    )
    return {"images": pil_image_to_b64(images)}


class KandinskyV22FeatureSteeringPipeline:
    def __init__(self, pipeline, sae, image_encoder, image_processor, device, dtype):
        self.pipeline = pipeline
        self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.invert_scheduler = DDIMInverseScheduler.from_config(
            self.pipeline.scheduler.config
        )

        self.sae = sae

        self.image_encoder = image_encoder
        self.image_processor = image_processor

        self.device = device
        self.dtype = dtype

    def embed_image(self, images: Union[PIL.Image, List[PIL.Image]]) -> torch.Tensor:
        inputs = (
            self.image_processor(images, return_tensors="pt")
            .pixel_values[0]
            .unsqueeze(0)
            .to(self.device)
        )
        return self.image_encoder(inputs)["image_embeds"]

    def embed_zero(self, batch_size: int = 1) -> torch.Tensor:
        zero_img = torch.zeros(
            1,
            3,
            self.image_encoder.config.image_size,
            self.image_encoder.config.image_size,
        ).to(
            device=self.device,
            dtype=self.dtype,
        )
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        return zero_image_emb

    def invert(
        self,
        image: PIL.Image,
        image_embeds: torch.Tensor,
        negative_image_embeds: torch.Tensor,
        height: int = 768,
        width: int = 768,
    ) -> torch.Tensor:
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

        device = self.device

        inputs = prepare_image(image, w=width, h=height).to(
            device=device, dtype=image_embeds.dtype
        )
        with torch.no_grad():
            image_latents = self.pipeline.movq.encode(inputs)["latents"]

        # Switch to inverse scheduler to perform DDIM inversion
        self.pipeline.scheduler = self.invert_scheduler

        inv_latents = self.pipeline(
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
        self.pipeline.scheduler = self.scheduler

        return inv_latents.images[0].unsqueeze(0)

    def get_features(
        self, image_embeds: torch.Tensor, n_features: int = 3
    ) -> dict[int, float]:
        features = self.sae.encode(image_embeds).squeeze(0)
        top_n = torch.argsort(features, dim=-1, descending=False)[-n_features:].flip(
            dims=[-1]
        )
        return [
            {"idx": idx.tolist(), "activation": features[idx].tolist()} for idx in top_n
        ]

    def generate(
        self,
        image_embeds: torch.Tensor,
        negative_image_embeds: torch.Tensor,
        inv_latents: torch.Tensor,
        feature_idx: List[int],
        feature_val: List[float],
        height: int = 768,
        width: int = 768,
    ) -> PIL.Image:
        # Ensure we are using normal DDIM scheduler and not inverse scheduler
        self.pipeline.scheduler = self.scheduler

        # Calculate reconstruction loss
        features = self.sae.encode(image_embeds).squeeze(0)
        decoded = self.sae.decode(features)
        recon_loss = image_embeds - decoded

        # Feature intervention
        for idx, val in zip(feature_idx, feature_val):
            features[idx] = val

        # Generate new image embedding
        new_image_embeds = self.sae.decode(features) + recon_loss

        return self.pipeline(
            image_embeds=new_image_embeds,
            negative_image_embeds=negative_image_embeds,
            latents=inv_latents,
            guidance_scale=1,
            num_inference_steps=20,
            height=height,
            width=width,
        ).images[0]
