# 跨個體 NIRS 預測專案

本專案包含論文使用的兩階段流程：第一階段從 DTOF 學習個體描述子，第二階段以 DRS 為主要輸入，結合個體資訊預測 HC／tHb 與 StO₂。主要實驗使用 32 維 DTOF 描述子與 gated residual 融合；其他融合方式與控制實驗也保留在設定檔中。

以下指令均在專案根目錄（`pyproject.toml` 所在位置）執行。完整 154-fold LOSO 訓練需要對應資料與足夠運算資源；初次閱讀可先查看 notebook 與既有結果，不必重新訓練。

## 目錄導覽

| 目錄 | 用途 |
| --- | --- |
| `configs/` | 第一、二階段及控制實驗的 YAML 設定。 |
| `data/` | 原始資料與資料放置說明；大型或私人資料不納入版本控制。 |
| `src/subject_nirs/stage1/` | DTOF 表徵學習、匯出與第一階段分析。 |
| `src/subject_nirs/stage2/` | DRS 預測、特徵融合與第二階段結果分析。 |
| `scripts/` | 訓練、匯出、續跑與基本檢查的執行入口。 |
| `notebooks/` | 依論文分析順序整理的圖表與結果檢視。 |
| `artifacts/` | 模型權重及預測輸出；由訓練程序建立。 |
| `cache/` | 可重建的前處理快取。 |
| `results/tables/`、`results/figures/` | 分析表格與圖檔。 |
| `tests/` | 快速測試。 |

分析程式以研究問題分工；notebook 主要負責串接分析與展示圖表。圖表的共用繪製邏輯盡量放在 `src/`，避免同一張圖在多處各自維護。

## 安裝與確認

需要 Python 3.10 以上版本及適合本機的 PyTorch 環境。可使用既有 Conda 環境，或先建立虛擬環境，再安裝專案：

```bash
python -m pip install -e ".[notebook,test]"
python scripts/smoke_test.py
pytest -q
```

資料位置與必要檔案見 [`data/README.md`](data/README.md)。資料、模型權重與分析結果不會由安裝指令產生。

## 使用順序

1. 準備 `data/` 中的資料，並檢查 `configs/` 的輸入／輸出路徑。
2. 執行第一階段 LOSO 訓練與 DTOF 特徵匯出。若已有第一階段權重與 `subject_features_all_op.npz`，可直接沿用。
3. 訓練第二階段 DRS-only baseline；Fine-tuning 融合實驗會使用對應 fold 的 baseline 權重。
4. 執行所需融合或控制實驗，再依序閱讀 `notebooks/` 中的分析。

常用入口如下。這些 LOSO 指令預設涵蓋全部 154 個 fold，不適合作為快速測試：

```bash
bash scripts/run_stage1_loso.sh
bash scripts/run_stage2_loso.sh configs/stage2/baseline/hc.yaml
bash scripts/run_stage2_loso.sh configs/stage2/baseline/sto2.yaml
bash scripts/run_stage2_loso.sh configs/stage2/residual/finetune/dtof_hc.yaml
bash scripts/run_stage2_loso.sh configs/stage2/residual/finetune/dtof_sto2.yaml
```

單一 fold 可改用 `scripts/train_stage1.py` 或 `scripts/train_stage2.py`；其他融合方式、訓練策略與目標參數可由 `configs/stage2/` 選擇。查看尚未完成的第二階段工作，可先執行：

```bash
python scripts/run_remaining_stage2.py --dry-run
```

## 結果分析

`notebooks/00`–`03` 檢視 DTOF 個體差異、表徵檢索、OP 穩健性／CKA 與結構探針；`04`–`08` 檢視 baseline、融合比較、失敗個體、未見資料及描述子控制實驗。Notebook 內有各圖的說明與對應參數，建議依編號閱讀。

CKA 的 mean、跨 fold SD、單參數 OP transition 與 conditional 圖由同一繪圖模組產生。已有 `results/tables/stage1/cka_relat_cons_32/` 的分析摘要時，只重畫圖、不重新執行模型推論，可使用：

```bash
python -m subject_nirs.stage1.cka_plotting --analysis_dir results/tables/stage1/cka_relat_cons_32
```

若使用自訂輸出位置，請將 `--analysis_dir` 換成含有 `full_cka_summary.h5` 與 transition 摘要表的目錄。
