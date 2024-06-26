This repo contains code for experimenting with manipulating features extracted from models to make semantic edits to images. At the moment, the techniques used are largely described in [this post](https://www.lesswrong.com/posts/Quqekpvx8BGMMcaem/interpreting-and-steering-features-in-images).

The inference code is currently deployed using [Modal](https://modal.com/), but relevant parts can also be extracted to run elsewhere.

# Deploy to Modal

```
MODAL_IMAGE_BUILDER_VERSION=2024.04 modal deploy main.py
```

The `MODAL_IMAGE_BUILDER_VERSION=2024.04` env var is required while the `2024.04` release of the Modal image builder is still in preview.

# Endpoints

After deploying to Modal, you should have the following endpoints available.

The full URLs should be available in your Modal dashboard and should look like `https://foobar-preprocess-image.modal.run`.

The thinking for the workflow here is:

- Preprocess the image
  - Generate CLIP image embedding.
  - Generate "inverted latents" via DDIM inversion. These latents can be used to reconstruct the original image and to generate image variations that are closer to the original image's structure.
  - Extract CLIP features.
  - Optionally generate an icon for each feature that can be displayed in a UI.
- Use the CLIP image embedding and inverted latents for new image generation while manipulating the values of specific CLIP features of interest.

## preprocess_image

Parameters:

- `image`: Image to preprocess.
- `n_features` (optional): The number of top activating features to return. Defaults to 3.
- `generate_icon` (optional): Whether to return generated icon for each feature. Defaults to true. 

Response: 

```
{
  "image_embeds": [1, 2, 3, 4],
  "inv_latents": [1, 2, 3, 4],
  "features": [
    {
      "idx": 777,
      "activation": 12.4,
      "icon": [
        "data:image/png;base64,...",
        "data:image/png;base64,...",
      ]
    }
  ]
}
```

Example:

```
const formData = new FormData()
formData.append("image", image)

const res = await fetch(PREPROCESS_IMAGE_URL, {
  method: "POST",
  body: formData,
});
```

For more details see the `preprocess_image()` function in `main.py`.

## generate_image

Parameters:

- `image_embeds`: The `image_embeds` returned from preprocessing.
- `inv_latents`: The `inv_latents` returned from preprocessing.
- `feature_idx`: A list of indices of the features to set values for.
- `feature_val`: A list of values for each feature.

Response:

```
{
  "images": "data:image/png;base64,..."
}
```

Example:

```
const params = {
  image_embeds: ...
  inv_latents: ...
  feature_idx: ...
  feature_val: ...
};
const res = await fetch(GENERATE_IMAGE_URL, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(params),
});
```

For more details see the `generate_image()` function in `main.py`.