# Experiment results log

Raw outputs from running the notebook, kept for the final check against the assignment brief (accuracy / precision / recall on the **test** set, training time, trials over time).

---

## Trial 1 — Baseline CNN

_(Run after kernel restart; `test_ds` uses `class_names=class_names` so labels match train.)_

**Model summary:** 4,751,363 trainable params (Rescaling → 3×Conv/Pool → Flatten → Dense 512 → Dropout 0.2 → Dense 3).

**Training time:** 126.5 s (~2.1 min)

### Training log (10 epochs)

| Epoch | accuracy | loss | val_accuracy | val_loss |
|-------|----------|------|--------------|----------|
| 1 | 0.6951 | 0.6898 | 0.7682 | 0.5512 |
| 2 | 0.7904 | 0.5228 | 0.7904 | 0.5118 |
| 3 | 0.7984 | 0.4755 | 0.7950 | 0.4912 |
| 4 | 0.8196 | 0.4276 | 0.7987 | 0.4990 |
| 5 | 0.8307 | 0.3870 | 0.7978 | 0.5088 |
| 6 | 0.8549 | 0.3495 | 0.7969 | 0.5311 |
| 7 | 0.8785 | 0.2943 | 0.7876 | 0.5878 |
| 8 | 0.8872 | 0.2709 | 0.7775 | 0.6046 |
| 9 | 0.9101 | 0.2204 | 0.7525 | 0.7091 |
| 10 | 0.9255 | 0.1772 | 0.7682 | 0.8877 |

### Test set (held-out `dataset/chest_xray/test`)

- **Baseline test loss:** 0.8022  
- **Baseline test accuracy:** 0.7620 (**76.20%**)  
- **Evaluate run:** 14/14 steps (~438 images, batch size 32)

### Notes

- Last epoch **train** accuracy **92.55%** vs **val** **76.82%** and **val_loss** rising — clear **overfitting** in late epochs (good to mention in the report).
- **Test 76.2%** lines up with **val ~77–80%**; this is believable for the baseline.
- An earlier notebook run showed **~32% test accuracy**; that was **invalid** (test class order not locked to training). Do **not** use those numbers in the report.

---

## Trial 2 — (paste when ready)

_(empty)_

---

## Trial 3 — (paste when ready)

_(empty)_

---

## Trial 4 — (paste when ready)

_(empty)_

---

## Final checklist vs brief

- [ ] Test metrics reported for the model you consider “best”
- [ ] Per-class precision, recall, F1 (from classification report)
- [ ] Training time(s) recorded
- [ ] Discussion of balance, overfitting, augmentation, transfer learning, GradCAM
- [x] Train/val/test label alignment checked (`class_names` on test loader); Trial 1 test ~76% matches val trend

---

## Debugging note (Trial 1 val vs test)

If you see high **val** but random-looking **test** accuracy, run these **right after** building `train_ds` / `test_ds` (or in a scratch cell):

```python
print("train class names:", train_ds.class_names)
print("test class names:", test_ds.class_names)
```

They must be the **same names in the same order**. Also confirm the notebook’s working directory is the project folder (`dataset/chest_xray/...` exists). Optionally compare `baseline_model.evaluate(val_ds)` to the last epoch’s `val_accuracy` — they should be close.
