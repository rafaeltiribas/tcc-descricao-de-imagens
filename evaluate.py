import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import evaluate

# =========================
# CONFIG
# =========================
INPUT_DIR = Path("./generations")
FILE_GLOB = "*.json"
PRED_COL = "generated_caption"
REF_COL = "generated_caption"
BERTSCORE_LANG = "pt"

OUTPUT_DIR = Path("./metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_JSON = OUTPUT_DIR / "summary_by_model.json"
SUMMARY_TXT  = OUTPUT_DIR / "summary_by_model.txt"

# =========================
# Métricas (evaluate)
# =========================
bertscore_metric = evaluate.load("bertscore")
meteor_metric    = evaluate.load("meteor")
bleu_metric      = evaluate.load("bleu")
rouge_metric     = evaluate.load("rouge")
cider_metric     = evaluate.load("cider")

SAMPLE_METRICS = [
    "BERTScore_precision",
    "BERTScore_recall",
    "BERTScore_f1",
    "METEOR",
    "BLEU",
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "ROUGE-Lsum",
    "CIDEr",
]

# =========================
# Utils
# =========================
def _safe_list_of_dicts(obj: Any) -> List[Dict[str, Any]]:
    """
    Aceita:
      - lista de dicts (cada item é uma instância)
      - dict com chave 'entries' que é lista de instâncias
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
        return obj["entries"]
    raise ValueError("JSON precisa ser uma lista de objetos ou um dict com chave 'entries' (lista).")

def _load_pairs(json_path: Path, pred_key: str, ref_key: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Retorna (preds, refs, rows_minimos)
    rows_minimos: apenas {pred_key, ref_key} por instância, preservando strings.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = _safe_list_of_dicts(data)

    preds, refs, rows_min = [], [], []
    for row in rows:
        pred = str(row.get(pred_key, "") or "")
        ref  = str(row.get(ref_key, "") or "")
        preds.append(pred)
        refs.append(ref)
        rows_min.append({pred_key: pred, ref_key: ref})
    return preds, refs, rows_min

def _avg(xs: List[float]) -> float:
    return float(sum(xs)/len(xs)) if xs else 0.0

# =========================
# Cálculo por arquivo (modelo)
# =========================
def evaluate_file_per_instance(json_path: Path) -> Dict[str, Any]:
    preds, refs, base_rows = _load_pairs(json_path, PRED_COL, REF_COL)
    if not preds or len(preds) != len(refs):
        raise ValueError(f"{json_path.name}: listas vazias ou de tamanhos diferentes.")

    N = len(preds)

    bert = bertscore_metric.compute(predictions=preds, references=refs, lang=BERTSCORE_LANG)
    bert_p_list = [float(x) for x in bert["precision"]]
    bert_r_list = [float(x) for x in bert["recall"]]
    bert_f_list = [float(x) for x in bert["f1"]]

    entries_out: List[Dict[str, Any]] = []

    m_METEOR, m_BLEU, m_CIDER = [], [], []
    m_R1, m_R2, m_RL, m_RLsum = [], [], [], []
    avg_len_pred_list = []

    for i, (pred, ref) in enumerate(zip(preds, refs)):
        meteor_res = meteor_metric.compute(predictions=[pred], references=[ref])
        bleu_res   = bleu_metric.compute(predictions=[pred], references=[[ref]])
        rouge_res  = rouge_metric.compute(predictions=[pred], references=[ref])
        cider_res  = cider_metric.compute(predictions=[pred], references=[[ref]])

        m_METEOR.append(float(meteor_res["meteor"]))
        m_BLEU.append(float(bleu_res["bleu"]))
        m_CIDER.append(float(cider_res["score"]))
        m_R1.append(float(rouge_res["rouge1"]))
        m_R2.append(float(rouge_res["rouge2"]))
        m_RL.append(float(rouge_res["rougeL"]))
        m_RLsum.append(float(rouge_res["rougeLsum"]))
        avg_len_pred_list.append(float(len(pred.split())))

        entry = {
            REF_COL: base_rows[i][REF_COL],
            PRED_COL: base_rows[i][PRED_COL],
            "metrics": {
                "BERTScore_precision": bert_p_list[i],
                "BERTScore_recall":    bert_r_list[i],
                "BERTScore_f1":        bert_f_list[i],
                "METEOR":              float(meteor_res["meteor"]),
                "BLEU":                float(bleu_res["bleu"]),
                "CIDEr":               float(cider_res["score"]),
                "ROUGE-1":             float(rouge_res["rouge1"]),
                "ROUGE-2":             float(rouge_res["rouge2"]),
                "ROUGE-L":             float(rouge_res["rougeL"]),
                "ROUGE-Lsum":          float(rouge_res["rougeLsum"]),
            }
        }
        entries_out.append(entry)

    overall_metrics = {
        "BERTScore_precision": _avg(bert_p_list),
        "BERTScore_recall":    _avg(bert_r_list),
        "BERTScore_f1":        _avg(bert_f_list),
        "METEOR":              _avg(m_METEOR),
        "BLEU":                _avg(m_BLEU),
        "CIDEr":               _avg(m_CIDER),
        "ROUGE-1":             _avg(m_R1),
        "ROUGE-2":             _avg(m_R2),
        "ROUGE-L":             _avg(m_RL),
        "ROUGE-Lsum":          _avg(m_RLsum),
        "AvgGeneratedLength":  _avg(avg_len_pred_list),
        "NumSamples":          int(N),
    }

    return {"overall_metrics": overall_metrics, "entries": entries_out}


def main():
    files = sorted(p for p in INPUT_DIR.glob(FILE_GLOB) if p.is_file())
    if not files:
        raise SystemExit(f"Nenhum arquivo encontrado em {INPUT_DIR} com padrão {FILE_GLOB}")

    summary_by_model = {}
    for f in files:
        try:
            result = evaluate_file_per_instance(f)
            summary_by_model[f.stem] = result["overall_metrics"]
            out_path = OUTPUT_DIR / f"{f.stem}_evaluated.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] {f.name} -> {out_path}")
        except Exception as e:
            print(f"[ERRO] {f.name}: {e}")

    # ===== Salva sumário global (JSON) =====
    SUMMARY_JSON.write_text(
        json.dumps(summary_by_model, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # ===== Salva sumário global (TXT) =====
    with SUMMARY_TXT.open("w", encoding="utf-8") as fp:
        fp.write("# Summary by model (means)\n\n")
        for model_name in sorted(summary_by_model.keys()):
            fp.write(f"## {model_name}\n")
            for k in sorted(summary_by_model[model_name].keys()):
                v = summary_by_model[model_name][k]
                if isinstance(v, (int, float)):
                    fp.write(f"  {k}: {v:.6f}\n")
                else:
                    fp.write(f"  {k}: {v}\n")
            fp.write("\n")

if __name__ == "__main__":
    main()
