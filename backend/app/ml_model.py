import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

# Use a headless backend for servers (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

# Try optional deps
try:
    from lime import lime_image
    _HAS_LIME = True
except Exception:
    _HAS_LIME = False

try:
    from skimage.color import label2rgb
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# --------------------------- Config ---------------------------

CLASS_NAMES: Dict[str, List[str]] = {
    "onion": ["Alternaria_D", "Fusarium_D", "Healthy_leaf", "Virosis_D"],
    "tomato": [
        "Blossom_end_rot_d", "TomL_Bacterial_floundering_d", "TomL_Mite_d",
        "TomL_alternaria_mite_d", "TomL_fusarium_d", "TomL_healthy_leaf",
        "TomL_virosis_d", "alternaria_d", "exces_nitrogen_d", "healthy_fruit",
        "sunburn_d", "tomato_late_blight_d"
    ],
    "maize": [
        "Abiotic_DiseaseD", "CurvulariaD", "Healthy_leaf", "HelminthosporiosisD",
        "RustD", "StripeD", "VirosisD"
    ],
}

IMAGE_SIZE = 128
CHANNELS = 3
PATCH_SIZE = 32

# LIME samples can be tuned via env; default lower for speed
LIME_SAMPLES = int(os.getenv("LIME_SAMPLES", "600"))

# --------------------- ViT-like model ------------------------

def create_vit_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    patch_size: int = PATCH_SIZE,
    num_transformer_layers: int = 6,
    num_heads: int = 4,
    hidden_dim: int = 128,
    use_patches_layer: bool = True,
) -> tf.keras.Model:
    class Patches(layers.Layer):
        def __init__(self, patch_size: int):
            super().__init__()
            self.patch_size = patch_size
        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

    class RepeatClassToken(layers.Layer):
        def call(self, inputs):
            cls_token, x = inputs
            batch_size = tf.shape(x)[0]
            return tf.repeat(cls_token, repeats=batch_size, axis=0)

    class CustomMultiHeadAttention(layers.Layer):
        def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
            super().__init__(**kwargs)
            self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        def call(self, query, value):
            output, attn_weights = self.mha(query, value, return_attention_scores=True)
            return output, attn_weights

    inputs = layers.Input(shape=input_shape)
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    if use_patches_layer:
        patches = Patches(patch_size)(inputs)
        x = layers.Dense(hidden_dim)(patches)
    else:
        conv = layers.Conv2D(filters=hidden_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
        x = layers.Reshape((num_patches, hidden_dim))(conv)

    # Class token
    cls_token = layers.Dense(hidden_dim)(tf.zeros((1, 1, hidden_dim)))
    cls_token = RepeatClassToken()([cls_token, x])
    x = layers.Concatenate(axis=1)([cls_token, x])  # (B, 1+N, D)
    num_tokens = num_patches + 1

    # Positional embedding
    positions = tf.range(start=0, limit=num_tokens, delta=1)
    pos_embed = layers.Embedding(input_dim=num_tokens, output_dim=hidden_dim)(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    x = x + pos_embed

    last_attn_weights = None
    for _ in range(num_transformer_layers):
        # MHA
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_layer = CustomMultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim, dropout=0.1)
        attn_output, attn_weights = attn_layer(x_norm, x_norm)
        x = x + layers.Dropout(0.1)(attn_output)

        # FFN
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(hidden_dim * 4, activation="gelu")(x_norm)
        ffn = layers.Dense(hidden_dim, activation="gelu")(ffn)
        x = x + layers.Dropout(0.1)(ffn)
        last_attn_weights = attn_weights

    # CLS head
    x = x[:, 0, :]
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.last_attn_weights = last_attn_weights
    return model

def load_crop_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    weights_files = {
        "onion":  "vit_model.weights.h5",
        "tomato": "vitTomato_model.weights.h5",
        "maize":  "vitMaize_model.weights.h5",
    }
    models_dir = Path(__file__).parent / "models"
    for crop, weights_name in weights_files.items():
        mdl = create_vit_model(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            num_classes=len(CLASS_NAMES[crop]),
            patch_size=PATCH_SIZE,
            num_transformer_layers=6,
            num_heads=4,
            hidden_dim=128,
            use_patches_layer=True,
        )
        weights_path = models_dir / weights_name
        try:
            mdl.load_weights(str(weights_path))
            models[crop] = mdl
        except Exception as e:
            print(f"[warn] Could not load weights for {crop}: {e}")
            models[crop] = None  # ensure predict() can error clearly
    return models

# ---------------- Core prediction (OOD removed) ----------------

def _preprocess_for_model(img_path: str):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    arr = image.img_to_array(img)
    x = np.expand_dims(arr, axis=0) / 255.0
    return x, arr.astype(np.uint8)

def _confidence_metrics(probs: np.ndarray):
    max_prob = float(np.max(probs))
    sorted_probs = np.sort(probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2]) if probs.size >= 2 else float(sorted_probs[-1])
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return {"max_prob": max_prob, "margin": margin, "entropy": entropy}

def predict(models: Dict[str, Any], img_path: str, crop_type: str) -> Dict[str, Any]:
    crop_type = crop_type.lower()
    if crop_type not in CLASS_NAMES:
        raise ValueError(f"Invalid crop type: {crop_type}. Must be one of {list(CLASS_NAMES.keys())}")
    model = models.get(crop_type)
    if model is None:
        raise ValueError(f"No model loaded for crop type: {crop_type}")

    x, _ = _preprocess_for_model(img_path)
    probs = model.predict(x, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    scores = _confidence_metrics(probs)

    return {
        "class": CLASS_NAMES[crop_type][class_idx],
        "confidence": float(probs[class_idx]),
        "crop": crop_type,                  # keep lowercase; template capitalizes
        "scores": scores,                   # diagnostic metrics
    }

def topk_predictions(models: Dict[str, Any], img_path: str, crop_type: str, k: int = 3):
    crop_type = crop_type.lower()
    model = models.get(crop_type)
    if model is None:
        return []
    x, _ = _preprocess_for_model(img_path)
    probs = model.predict(x, verbose=0)[0]
    idxs = np.argsort(probs)[::-1][:k]
    return [{"class": CLASS_NAMES[crop_type][i], "confidence": float(probs[i])} for i in idxs]

# ---------------- Visual explanations ----------------

def lime_explain(models: Dict[str, Any], img_path: str, crop_type: str, save_path: str) -> str:
    if not _HAS_LIME:
        raise RuntimeError("LIME is not installed in this environment.")
    crop_type = crop_type.lower()
    model = models[crop_type]
    x, orig = _preprocess_for_model(img_path)

    probs = model.predict(x, verbose=0)[0]
    class_idx = int(np.argmax(probs))

    def predict_fn(batch_uint8):
        batch = batch_uint8.astype("float32") / 255.0
        return model.predict(batch, verbose=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image=orig,
        classifier_fn=predict_fn,
        top_labels=1,
        num_samples=LIME_SAMPLES,
        hide_color=0,
    )
    _, mask = explanation.get_image_and_mask(
        label=class_idx, positive_only=True, num_features=8, hide_rest=False
    )

    plt.figure(figsize=(4, 4))
    if _HAS_SKIMAGE:
        plt.imshow(label2rgb(mask, orig, alpha=0.6))
    else:
        # fallback: show mask as transparency
        plt.imshow(orig)
        plt.imshow(mask, cmap="jet", alpha=0.35)
    plt.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return os.path.basename(save_path)

def occlusion_heatmap(models: Dict[str, Any], img_path: str, crop_type: str, save_path: str, patch: int = PATCH_SIZE) -> str:
    crop_type = crop_type.lower()
    model = models[crop_type]
    x_batch, orig = _preprocess_for_model(img_path)
    base_probs = model.predict(x_batch, verbose=0)[0]
    target_idx = int(np.argmax(base_probs))

    H = W = IMAGE_SIZE
    heat = np.zeros((H, W), dtype=np.float32)
    baseline_val = 0.5  # gray

    for y0 in range(0, H, patch):
        for x0 in range(0, W, patch):
            x_occ = x_batch.copy()
            x_occ[:, y0:y0+patch, x0:x0+patch, :] = baseline_val
            p = model.predict(x_occ, verbose=0)[0][target_idx]
            heat[y0:y0+patch, x0:x0+patch] = base_probs[target_idx] - p

    # Normalize to [0,1]
    heat -= heat.min()
    if heat.max() > 0:
        heat /= heat.max()

    plt.figure(figsize=(4, 4))
    plt.imshow(orig)
    plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return os.path.basename(save_path)

def integrated_gradients_explain(
    models: Dict[str, Any],
    img_path: str,
    crop_type: str,
    save_path: str,
    steps: int = 32
) -> str:
    """
    Integrated Gradients over the input image.
    Saves an overlay heatmap to `save_path` and returns just the filename.
    """
    crop_type = crop_type.lower()
    if crop_type not in CLASS_NAMES:
        raise ValueError(f"Invalid crop type: {crop_type}")
    model = models.get(crop_type)
    if model is None:
        raise ValueError(f"No model loaded for crop type: {crop_type}")

    # x_batch in [0,1], orig is HxWx3 uint8
    x_batch, orig = _preprocess_for_model(img_path)

    # target class index
    probs = model.predict(x_batch, verbose=0)[0]
    target_idx = int(np.argmax(probs))

    baseline = tf.zeros_like(x_batch)  # black baseline
    alphas = tf.linspace(0.0, 1.0, steps)
    integrated_grads = tf.zeros_like(x_batch)

    for alpha in alphas:
        with tf.GradientTape() as tape:
            x_interp = baseline + alpha * (x_batch - baseline)
            tape.watch(x_interp)
            preds = model(x_interp, training=False)
            target = preds[:, target_idx]
        grads = tape.gradient(target, x_interp)
        integrated_grads += grads

    integrated_grads = (x_batch - baseline) * integrated_grads / tf.cast(steps, tf.float32)
    ig = tf.reduce_mean(tf.math.abs(integrated_grads), axis=-1)[0].numpy()  # HxW
    ig -= ig.min()
    if ig.max() > 0:
        ig /= ig.max()

    # overlay on original
    plt.figure(figsize=(4, 4))
    plt.imshow(orig)
    plt.imshow(ig, cmap="jet", alpha=0.45)
    plt.axis("off")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return os.path.basename(save_path)

def image_stats(img_path: str) -> Dict[str, Any]:
    with Image.open(img_path) as im:
        w, h = im.size
        arr = np.asarray(im.convert("RGB")).astype(np.float32) / 255.0
    return {
        "width": int(w),
        "height": int(h),
        "mean_rgb": [float(arr[..., i].mean()) for i in range(3)],
        "std_rgb": [float(arr[..., i].std()) for i in range(3)],
        "brightness_mean": float(arr.mean()),
    }
