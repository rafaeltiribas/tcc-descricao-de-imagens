import os, sys, json, warnings, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eco2ai import Tracker

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# ==== Lê argumento ====
if len(sys.argv) < 2 or sys.argv[1].lower() not in ["pracegover", "test"]:
    print("Uso: python main.py <pracegover|test>")
    sys.exit(1)
dataset_choice = sys.argv[1].lower()

# ==== Caminhos ====
model_path  = "TucanoBR/ViTucano-1b5-v1"
dataset_json = f"./dataset/{dataset_choice}/{dataset_choice}.json"
images_dir   = f"./dataset/{dataset_choice}/images"
output_json  = f"./output/{dataset_choice}_descriptions.json"

prompt = (
    "Forneça uma descrição clara e detalhada desta imagem, incluindo cores, "
    "formas, objetos, pessoas e o contexto geral, para que uma pessoa cega "
    "possa entender a cena. "
    "Descreva as características fenotípicas das pessoas quando pertinente."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tracker = Tracker(
    project_name=f"tucano_captioning_{dataset_choice}",
    experiment_description=f"Caption generation for dataset {dataset_choice}",
    file_name=f"./output/{dataset_choice}_eco2ai.csv",
)
tracker.start()

model = AutoModelForCausalLM.from_pretrained(
    model_path, attn_implementation="eager", trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

with open(dataset_json, "r", encoding="utf-8") as f:
    data = json.load(f)

allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
image_paths = [
    os.path.join(images_dir, img["filename"])
    for img in data.get("images", [])
    if img.get("split") == "test" and
       os.path.splitext(img.get("filename",""))[1].lower() in allowed_ext
]

print(f"Total de imagens no split='test': {len(image_paths)}")

results = []
with torch.inference_mode():
    for i, path in enumerate(image_paths, 1):
        name = os.path.basename(path)
        try:
            output_text, _ = model.chat(
                prompt=prompt,
                image=path,
                tokenizer=tokenizer,
            )
            results.append({"image": name, "generated_caption": output_text.strip()})
            print(f"[{i}/{len(image_paths)}] OK: {name}")
        except Exception as e:
            results.append({"image": name, "generated_caption": None, "error": str(e)})
            print(f"[{i}/{len(image_paths)}] ERRO: {name} -> {e}")

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

tracker.stop()
print(f"Eco2AI report saved to {dataset_choice}_eco2ai.csv")
print(f"Total images processed: {len(results)}")
