"""Step 9: Gradio UI for exploring trained models against the held-out test set.

Loads every `*.pth` checkpoint in MODEL_DIR (produced by train.py), lets you
pick a model and a test image, and shows the prediction, confidence
breakdown, and rolling accuracy/precision/recall/F1 across your session.
"""
import glob
import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import IMAGE_DIRECTORIES, METADATA_PARQUET, MODEL_DIR
from dataset import find_image_path
from dataset import standard_transform as ui_transform
from model import BACKBONES, build_architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not os.path.exists(METADATA_PARQUET):
    raise FileNotFoundError(f"{METADATA_PARQUET} not found. Run the data pipeline first.")

df_metadata = pd.read_parquet(METADATA_PARQUET)
test_df = df_metadata[df_metadata["fold"] == -2].reset_index(drop=True)

class_names = (
    df_metadata[["dx", "dx_numeric"]]
    .drop_duplicates()
    .sort_values("dx_numeric")["dx"]
    .tolist()
)
idx_to_class = {i: name for i, name in enumerate(class_names)}
num_classes = len(class_names)

test_choices = [f"{i} | {row['image_id']} | true: {row['dx']}" for i, row in test_df.iterrows()]

MODEL_PATHS = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pth")))
if not MODEL_PATHS:
    raise FileNotFoundError(f"No .pth models found in {MODEL_DIR}. Run train.py first.")


def parse_model_info(path):
    filename = os.path.basename(path)
    name = filename.lower()

    # Longest key first so e.g. "efficientnet_b2" wins over a shorter partial match.
    backbone = next((b for b in sorted(BACKBONES, key=len, reverse=True) if b in name), "unknown")
    backbone_label = BACKBONES[backbone]["label"] if backbone in BACKBONES else "Unknown Backbone"

    if "unaugmented" in name:
        data_type = "Unaugmented"
    elif "augmented" in name:
        data_type = "Augmented"
    else:
        data_type = "Unknown"

    if "scratch" in name:
        strategy = "Scratch"
    elif "full_finetune" in name:
        strategy = "Full Fine-tune"
    elif "last_layer" in name:
        strategy = "Last Layer"
    else:
        strategy = "Unknown"

    return {
        "filename": filename,
        "path": path,
        "backbone": backbone,
        "backbone_label": backbone_label,
        "data_type": data_type,
        "strategy": strategy,
        "display_name": f"{backbone_label} | {data_type} | {strategy}",
    }


MODEL_INFOS = [parse_model_info(p) for p in MODEL_PATHS]
MODEL_LOOKUP = {m["display_name"]: m for m in MODEL_INFOS}
MODEL_NAMES = sorted(MODEL_LOOKUP.keys())


loaded_model = None
loaded_model_name = None


def load_selected_model(model_display_name):
    global loaded_model, loaded_model_name

    if loaded_model is None or loaded_model_name != model_display_name:
        info = MODEL_LOOKUP[model_display_name]
        model = build_architecture(info["backbone"], num_classes)

        state_dict = torch.load(info["path"], map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        model = model.to(device).eval()

        loaded_model = model
        loaded_model_name = model_display_name

    return loaded_model


def preprocess_image_from_row(row):
    img_path = find_image_path(f"{row['image_id']}.jpg", IMAGE_DIRECTORIES)
    img = Image.open(img_path).convert("RGB")
    tensor = ui_transform(img).unsqueeze(0).to(device)
    return img, tensor


history = {
    "image": [], "model": [], "backbone": [], "data_type": [], "strategy": [],
    "true_label": [], "pred_label": [], "confidence": [], "correct": [],
    "accuracy": [], "precision": [], "recall": [], "f1": [],
}


def update_metrics():
    y_true, y_pred = history["true_label"], history["pred_label"]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    history["accuracy"].append(acc)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)

    return acc, precision, recall, f1


def history_dataframe():
    return pd.DataFrame({
        key: history[key]
        for key in ("image", "model", "backbone", "data_type", "strategy", "true_label", "pred_label", "confidence", "correct")
    })


def plot_rolling_metrics():
    if not history["accuracy"]:
        return None

    x = np.arange(1, len(history["accuracy"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        ax.plot(x, history[key], marker="o", label=label)

    ax.set_xlabel("Predictions")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rolling metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_confidence(confidence_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(confidence_df["class"], confidence_df["confidence"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confidence")
    ax.set_title("Class confidence")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    return fig


def get_model_info_text(model_display_name):
    info = MODEL_LOOKUP[model_display_name]
    return f"""
<div class="model-card">
    <div class="model-title">{info['backbone_label']}</div>
    <div class="model-meta">
        <span>{info['data_type']}</span>
        <span>{info['strategy']}</span>
    </div>
    <div class="model-file">{info['filename']}</div>
</div>
"""


def predict_one(model_display_name, selected_item):
    model = load_selected_model(model_display_name)
    info = MODEL_LOOKUP[model_display_name]

    row_idx = int(selected_item.split(" | ")[0])
    row = test_df.iloc[row_idx]

    original_img, tensor = preprocess_image_from_row(row)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx])

    true_label = row["dx"]
    correct = pred_label == true_label

    confidence_df = pd.DataFrame({"class": class_names, "confidence": probs}).sort_values(
        "confidence", ascending=False
    )

    history["image"].append(row["image_id"])
    history["model"].append(info["filename"])
    history["backbone"].append(info["backbone_label"])
    history["data_type"].append(info["data_type"])
    history["strategy"].append(info["strategy"])
    history["true_label"].append(true_label)
    history["pred_label"].append(pred_label)
    history["confidence"].append(round(confidence, 4))
    history["correct"].append(correct)

    acc, precision, recall, f1 = update_metrics()

    correct_badge = "✅ Correct" if correct else "❌ Wrong"

    result_text = f"""
<div class="result-card">
    <div class="result-title">Prediction Result</div>

    <div class="result-grid">
        <div><b>Image ID</b><br>{row['image_id']}</div>
        <div><b>True label</b><br>{true_label}</div>
        <div><b>Predicted</b><br>{pred_label}</div>
        <div><b>Confidence</b><br>{confidence:.4f}</div>
    </div>

    <div class="badge">{correct_badge}</div>

    <hr>

    <div class="result-title small">Rolling Metrics</div>

    <div class="metric-grid">
        <div><b>Accuracy</b><br>{acc:.4f}</div>
        <div><b>Precision</b><br>{precision:.4f}</div>
        <div><b>Recall</b><br>{recall:.4f}</div>
        <div><b>F1</b><br>{f1:.4f}</div>
    </div>
</div>
"""

    return (
        original_img,
        get_model_info_text(model_display_name),
        result_text,
        confidence_df,
        plot_confidence(confidence_df),
        plot_rolling_metrics(),
        history_dataframe(),
    )


def run_all(model_display_name):
    outputs = None
    for item in test_choices:
        outputs = predict_one(model_display_name, item)
    return outputs


def reset_history():
    for key in history:
        history[key] = []

    return (
        None,
        "",
        "<div class='result-card'>History reset</div>",
        pd.DataFrame(columns=["class", "confidence"]),
        None,
        None,
        history_dataframe(),
    )


CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
}

#title-block {
    text-align: center;
    padding: 28px 18px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.16));
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 20px;
}

#title-block h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

#title-block p {
    font-size: 16px;
    opacity: 0.85;
}

.model-card, .result-card {
    border-radius: 22px;
    padding: 20px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
}

.model-title {
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 12px;
}

.model-meta {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.model-meta span {
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(59,130,246,0.22);
    border: 1px solid rgba(147,197,253,0.25);
    font-size: 13px;
}

.model-file {
    font-family: monospace;
    opacity: 0.75;
    word-break: break-all;
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 14px;
}

.result-title.small {
    font-size: 18px;
}

.result-grid, .metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.result-grid div, .metric-grid div {
    padding: 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
}

.badge {
    display: inline-block;
    margin-top: 15px;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(34,197,94,0.18);
    border: 1px solid rgba(74,222,128,0.24);
    font-weight: 700;
}

button {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

.dataframe, .plot-container {
    border-radius: 18px !important;
}
"""

DARK_MODE_JS = """
function () {
    document.documentElement.classList.add("dark");
    localStorage.setItem("theme", "dark");
}
"""


def build_ui():
    with gr.Blocks(js=DARK_MODE_JS, css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div id="title-block">
                <h1>Skin Lesion Model Arena</h1>
                <p>Compare PyTorch backbones, training strategies, confidence scores, and rolling metrics.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(choices=MODEL_NAMES, value=MODEL_NAMES[0], label="Model")
                image_dropdown = gr.Dropdown(choices=test_choices, value=test_choices[0], label="Test image")

                with gr.Row():
                    predict_btn = gr.Button("Predict selected image", variant="primary")
                    run_all_btn = gr.Button("Run all test images")
                    reset_btn = gr.Button("Reset metrics")

            with gr.Column(scale=1):
                model_info_output = gr.HTML(value=get_model_info_text(MODEL_NAMES[0]), label="Selected model")

        with gr.Row():
            with gr.Column(scale=1):
                image_output = gr.Image(label="Selected image", height=360)
            with gr.Column(scale=1):
                result_output = gr.HTML(label="Prediction result")

        with gr.Row():
            confidence_plot = gr.Plot(label="Confidence visualization")
            rolling_plot = gr.Plot(label="Rolling metrics")

        with gr.Row():
            confidence_table = gr.Dataframe(label="Confidence scores")
            history_table = gr.Dataframe(label="Prediction history")

        outputs = [
            image_output, model_info_output, result_output,
            confidence_table, confidence_plot, rolling_plot, history_table,
        ]

        model_dropdown.change(fn=get_model_info_text, inputs=[model_dropdown], outputs=[model_info_output])
        predict_btn.click(fn=predict_one, inputs=[model_dropdown, image_dropdown], outputs=outputs)
        run_all_btn.click(fn=run_all, inputs=[model_dropdown], outputs=outputs)
        reset_btn.click(fn=reset_history, inputs=[], outputs=outputs)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
