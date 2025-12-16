import csv
import io
from typing import Dict, List, Sequence

from fastapi import HTTPException, UploadFile

from db import log_inference
from model import predict_text


def _read_csv_rows(file: UploadFile, required_columns: Sequence[str]) -> List[Dict]:
    """Читает CSV в память и проверяет наличие обязательных колонок."""
    if file is None:
        raise HTTPException(status_code=400, detail="файл не передан")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="требуется .csv файл")

    try:
        raw = file.file.read()
        text_stream = io.StringIO(raw.decode("utf-8-sig"))
        reader = csv.DictReader(text_stream)
    except Exception:
        raise HTTPException(status_code=400, detail="не удалось прочитать csv")

    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="csv не содержит заголовок колонок")

    missing = [col for col in required_columns if col not in reader.fieldnames]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"csv должен содержать колонки: {', '.join(required_columns)}",
        )

    rows = [row for row in reader]
    if not rows:
        raise HTTPException(status_code=400, detail="csv файл пустой")
    return rows


def _predict_single_text(text: str) -> Dict:
    """Запускает предсказание и логирует результат/ошибку."""
    if not text:
        raise HTTPException(status_code=400, detail="поле text не может быть пустым")
    try:
        result = predict_text(text)
    except RuntimeError:
        log_inference(
            text=text,
            label=None,
            probs=None,
            has_image=False,
            status="failed",
        )
        raise HTTPException(
            status_code=403, detail="модель не смогла обработать данные"
        )

    log_inference(
        text=text,
        label=result["label"],
        probs=result["probs"],
        has_image=False,
        status="ok",
    )
    return result


def _compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Подсчет accuracy/precision/recall в мультклассе (macro average)."""
    if len(y_true) != len(y_pred) or not y_true:
        raise HTTPException(status_code=400, detail="некорректные данные для метрик")

    labels = sorted({0, 1, 2})
    total = len(y_true)
    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / total

    precisions = []
    recalls = []
    for lbl in labels:
        tp = sum(int(p == lbl and t == lbl) for t, p in zip(y_true, y_pred))
        fp = sum(int(p == lbl and t != lbl) for t, p in zip(y_true, y_pred))
        fn = sum(int(p != lbl and t == lbl) for t, p in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
    }


def run_forward(text: str) -> Dict:
    """Обертка для одиночного предсказания."""
    return _predict_single_text(text)


def run_forward_batch(file: UploadFile) -> Dict:
    """Прогон батча текстов из CSV."""
    rows = _read_csv_rows(file, required_columns=["text"])

    predictions = []
    for idx, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        if not text:
            raise HTTPException(
                status_code=400, detail=f"пустое поле text в строке {idx + 1}"
            )
        result = _predict_single_text(text)
        predictions.append({"text": text, **result})

    return {"items": predictions}


def run_evaluate(file: UploadFile) -> Dict:
    """Прогон датасета с метками, возврат предсказаний и метрик."""
    rows = _read_csv_rows(file, required_columns=["text", "target"])

    y_true: List[int] = []
    y_pred: List[int] = []
    items: List[Dict] = []

    for idx, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        target_raw = row.get("target")

        if text == "":
            raise HTTPException(
                status_code=400, detail=f"пустое поле text в строке {idx + 1}"
            )

        try:
            target = int(target_raw)
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"target должен быть int в строке {idx + 1}"
            )

        if target not in {0, 1, 2}:
            raise HTTPException(
                status_code=400,
                detail=f"target должен быть в диапазоне [0,2] в строке {idx + 1}",
            )

        result = _predict_single_text(text)
        y_true.append(target)
        y_pred.append(result["label"])
        items.append({"text": text, "target": target, **result})

    metrics = _compute_metrics(y_true, y_pred)
    return {"metrics": metrics, "items": items}


