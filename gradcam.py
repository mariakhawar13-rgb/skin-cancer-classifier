import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")

def make_gradcam_heatmap(model, img_array):
    """
    img_array: (1, H, W, 3) uint8 or float32 in 0-255
    returns: heatmap (H,W) normalized 0-1, pred_idx (int)
    """
    # Ensure float32
    img_input = tf.cast(img_array, tf.float32)

    last_conv = get_last_conv_layer_name(model)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    heatmap = np.dot(conv_outputs, pooled_grads)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap = heatmap / (heatmap.max() + 1e-8)
    else:
        heatmap = heatmap

    return heatmap, int(pred_index)

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """original_img: H,W,3 uint8 ; heatmap: H,W 0-1"""
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img.astype("uint8"), 1 - alpha, heatmap_color, alpha, 0)
    return overlay
