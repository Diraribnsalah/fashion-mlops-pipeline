import cv2
import gradio as gr
import numpy as np
import onnxruntime as ort

#  Load ONNX Session
try:
    session = ort.InferenceSession("fashion_cnn.onnx")
    input_name = session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


def preprocess_advanced(image_array):
    """Advanced preprocessing pipeline to align input images with FashionMNIST distribution."""
    if image_array is None:
        return None, None

    # A) Convert to Grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()

    # B) Detect Background Color & Invert if Light
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_color = np.median(border_pixels)

    if bg_color > 127:
        gray = cv2.bitwise_not(gray)
        bg_color = 255 - bg_color

    # C) Subtract Background Noise
    gray = cv2.subtract(gray, int(bg_color) + 15)

    # D) Contour Analysis & Masking (Isolate primary garment, remove text/watermarks)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        isolated = cv2.bitwise_and(gray, mask)

        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = isolated[y:y + h, x:x + w]
    else:
        cropped = gray

    # E) Contrast Stretching (Normalize garment brightness)
    if cropped.max() > cropped.min():
        cropped = cv2.normalize(cropped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # F) Proportional Scaling (Scale object to ~72% of 25x25 canvas)
    h_c, w_c = cropped.shape
    max_side = max(h_c, w_c)
    scale = 18.0 / max_side
    new_w = max(1, int(w_c * scale))
    new_h = max(1, int(h_c * scale))

    resized_obj = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # G) Center Object on 25x25 Canvas with Padding
    canvas = np.zeros((25, 25), dtype=np.uint8)
    start_x = (25 - new_w) // 2
    start_y = (25 - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_obj

    # H) Upscale Canvas to 250x250 for Clean UI Preview (Nearest Neighbor for sharp pixels)
    preview_display = cv2.resize(canvas, (250, 250), interpolation=cv2.INTER_NEAREST)

    # I) Tensor Normalization [-1, 1]
    img_np = canvas.astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    tensor = np.expand_dims(img_np, axis=(0, 1)).astype(np.float32)

    return tensor, preview_display


def predict_fashion(image):
    if image is None:
        return None, None

    tensor, preview_display = preprocess_advanced(image)
    outputs = session.run(None, {input_name: tensor})[0]

    # Softmax probabilities
    exp_outputs = np.exp(outputs - np.max(outputs))
    probs = (exp_outputs / exp_outputs.sum(axis=1, keepdims=True))[0]

    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(10)}
    return confidences, preview_display


# Custom CSS styling for modern layout
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
#header h1 { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 5px; }
#header p { font-size: 1rem; color: #64748B; }
.output-card { border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; background: #FFFFFF; }
"""

# Modern Dashboard Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css) as demo:
    gr.HTML(
        """
        <div id="header">
            <h1> Fashion-MNIST AI Classifier</h1>
            <p>Production MLOps Pipeline with OpenCV Preprocessing & ONNX Runtime Inference</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        # Left Panel: User Input
        with gr.Column(scale=1):
            with gr.Group():
                input_image = gr.Image(
                    label="Upload Image (JPG, PNG, GIF)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=320
                )
                submit_btn = gr.Button(" Run AI Inference", variant="primary", size="lg")

        # Right Panel: Structured Outputs
        with gr.Column(scale=1):
            with gr.Group():
                output_labels = gr.Label(
                    num_top_classes=3,
                    label=" Classification Confidence"
                )
                with gr.Accordion(" Preprocessed Model Input (25x25 View)", open=True):
                    processed_preview = gr.Image(
                        label="Exact tensor feed sent to ONNX model",
                        image_mode="L",
                        show_label=False,
                        height=220
                    )

    submit_btn.click(
        fn=predict_fashion,
        inputs=input_image,
        outputs=[output_labels, processed_preview]
    )

if __name__ == "__main__":
    demo.launch(share=True)
