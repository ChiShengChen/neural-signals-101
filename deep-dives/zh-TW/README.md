# Deep-dives — 進階天花板 🏔️

[English](../README.md) · **繁體中文**

這些是給已經做完主線(`notebooks/00`–`15`)、想要真正深度(數學、推導、邊角案例)讀者的
**選修支線**。和主線章節不同,它們**不受「5 分鐘 CPU」承諾限制**,而且預設你已經熟悉對應
的章節。

每份 deep-dive 結尾都有一個 **「⚠️ 更隱晦的陷阱」**——一個該主題特有、不那麼明顯的自我
欺騙方式——讓整份教學的北極星(「別騙自己」)一路貫穿到最深的內容,而不是只停在第 12 章。

| Deep-dive | 內容 | 對應主線 |
|---|---|---|
| [CSP 幾何](csp_geometry.ipynb) | CSP 是廣義特徵值問題;白化 + 旋轉的觀點;推導 | 第 03、07 章 |
| [Riemann 於小資料](riemann_small_data.ipynb) | SPD 流形、幾何平均 vs 歐氏平均(「膨脹效應」)、為何共變異在低 N 勝出 | 第 07 章 |
| [統計嚴謹度](stats_rigor.ipynb) | 為何 naive CV t-test 會過度拒絕;Nadeau-Bengio 修正檢定;nested CV;多重比較 | 第 11 章 |
| [chance level 與信賴區間](chance_level_ci.ipynb) | 二項檢定、Wilson/Clopper-Pearson 區間、Müller-Putz「高於 chance」門檻 | 第 11 章 |
| [Benchmark 過擬合](benchmark_overfitting.md) | 整個領域如何對單一 benchmark 過擬合——群體規模的洩漏(純文字) | 第 12 章 |
| [是腦還是假影?](artifact_confounds.ipynb) | 解碼的混淆——EOG/EMG 假扮成「BCI」;誠實檢查(洩漏的生理孿生) | 第 03、05、12 章 |
| [遷移學習與領域適應](domain_adaptation.ipynb) | Euclidean/Riemannian 對齊、校準試驗、微調——救回跨受試者/session 的準確率 | 第 07、12 章 |
| [Filter-bank CSP(FBCSP)](fbcsp.ipynb) | 多頻帶 CSP + 特徵選擇;經典競賽冠軍級的 MI baseline | 第 06、07 章 |
| [解讀模型與資料增強](interpretability_augmentation.ipynb) | EEGNet 梯度 saliency + 小資料的 EEG data augmentation | 第 09 章 |
| [真實 P300 與 SSVEP(MOABB)](real_p300_ssvep.ipynb) | 真實 ERP/SSVEP 資料集(對比第 10 章的模擬 demo) | 第 10、12 章 |
| [ICA 與 ASR 內部機制](ica_asr_internals.ipynb) | 盲源分離的數學、雞尾酒會 demo、ASR 的子空間重建 | 第 05 章 |

## 建置與執行

這裡的 `.ipynb` 由 `deep-dives/zh-TW/_src/*.py`(jupytext percent 格式)產生,和主線
notebook 一樣:

```bash
make notebooks                                   # 建置主線 + deep-dives(英文 + 繁中)
python scripts/run_all_notebooks.py              # 只跑主線(CI smoke 目標)
# 直接打開一份 deep-dive:
jupyter notebook deep-dives/zh-TW/csp_geometry.ipynb
```

> deep-dives 刻意**不納入 CI smoke 測試**(較重、屬選修)。每一份在撰寫時都已確認可在 CPU
> 執行;若你修改了某份,請在本機跑一次確認。
>
> 註:繁中 deep-dive 與英文版**程式碼完全相同**,只翻譯教學文字與註解。
