# 詞彙表與縮寫索引

[English](../GLOSSARY.md) · **繁體中文**

本教學用到的所有術語與縮寫，集中收錄於此。各詞彙也會在 notebook 第一次
出現時說明——本頁供快速查閱。依主題大致分組；請用瀏覽器的搜尋功能（Ctrl/Cmd-F）。

## 訊號與神經科學
- **EEG**（Electroencephalography，腦電圖）— 透過貼在頭皮上的電極記錄大腦的電活動。價格便宜、非侵入式，但訊雜比偏低。
- **MEG**（Magnetoencephalography，腦磁圖）— 記錄相同神經電流產生的磁場；需要磁屏蔽室。
- **ECoG**（Electrocorticography，皮質腦電圖）— 在手術中將電極直接放置在大腦皮質表面；訊號品質高，但屬侵入式。
- **LFP**（Local Field Potential，局部場電位）— 由插入組織的電極測得的小群神經元電位。
- **Spike**（動作電位）— 單一神經元的動作電位（約 1 毫秒的電脈衝）。
- **fNIRS**（functional Near-Infrared Spectroscopy，功能性近紅外光譜）— 透過近紅外光穿透頭皮測量血液含氧量（作為神經活動的緩慢代理指標）。
- **EMG**（Electromyography，肌電圖）— 肌肉的電活動；是常見的 EEG 雜訊來源。
- **EOG**（Electrooculography，眼電圖）— 眼球運動／眨眼產生的電活動；是 EEG 的主要雜訊。
- **ECG / EKG**（Electrocardiography，心電圖）— 心跳訊號；可能汙染 EEG 資料。
- **PSP**（Post-Synaptic Potential，突觸後電位）— 神經元的緩慢電位變化；數千個排列整齊的細胞的 PSP 加總後，就產生了 EEG 訊號。
- **Pyramidal neuron**（錐體神經元）— 大腦皮質中細長型的細胞；其同步化的 PSP 產生了頭皮 EEG 的主要訊號。
- **Volume conduction**（體積傳導）— 腦部訊號源透過腦組織、顱骨和頭皮擴散／模糊的現象，導致相鄰電極之間產生相關性。
- **µV（microvolt，微伏）** — 百萬分之一伏特；EEG 訊號的典型量級。
- **10-20 system**（10-20 電極系統）— 命名與放置頭皮電極的標準方案（例如 Fz、Cz、C3、C4、Oz）。字母代表腦區，奇數代表左側，偶數代表右側。
- **Montage**（電極配置）— 從通道名稱到頭部 3D 電極位置的對應圖。
- **Reference**（參考電極）— EEG 記錄的是電壓*差*；參考點是你要減去的那個點（例如平均參考 = 所有電極的平均值）。
- **SNR**（Signal-to-Noise Ratio，訊雜比）— 訊號相對於雜訊的強度。

## 腦波節律與反應
- **Delta／Theta／Alpha／Beta／Gamma** — EEG 標準頻率帶（約 1–4 / 4–8 / 8–13 / 13–30 / 30–45 Hz）。
- **Mu rhythm**（Mu 節律）— 感覺運動皮質上約 8–12 Hz 的節律；在做出動作時受到抑制。
- **ERD / ERS**（Event-Related De-/Synchronization，事件相關去同步化／再同步化）— 由事件引起的頻帶功率下降／反彈（例如手部動作想像時，對側 mu／beta 頻帶的 ERD）。
- **ERP**（Event-Related Potential，事件相關電位）— 大腦對某事件的平均電壓反應，透過對多次試驗取平均值後顯現。
- **P300** — 在罕見且受到注意的刺激出現後約 300 毫秒出現的正向 ERP 波峰。
- **Oddball paradigm**（奇球典範）— 一種實驗設計，將目標刺激設計為罕見，使其誘發 P300。
- **SSVEP**（Steady-State Visual Evoked Potential，穩態視覺誘發電位）— 視覺皮質以閃爍刺激的頻率進行振盪（「頻率標記」）。
- **Sleep stages**（睡眠期）— 清醒、N1、N2、N3（深眠）、REM；每個睡眠期都有特徵性節律（例如 N2 的睡眠紡錘波／K 複合波，N3 的 delta 波）。
- **Seizure**（癲癇發作）— 高度同步化、高振幅的節律性神經放電。

## 訊號處理（DSP）
- **DSP**（Digital Signal Processing，數位訊號處理）— 以數值方式處理取樣後的訊號。
- **Sampling rate / `sfreq`**（取樣率）— 每秒記錄的樣本數（Hz）。
- **Nyquist frequency**（奈奎斯特頻率）— 取樣率的一半；可表示的最高頻率。
- **Aliasing**（混疊）— 訊號取樣不足時，過高的頻率偽裝成較低頻率的現象。
- **Quantization**（量化）— 將連續電壓四捨五入為離散的數位等級。
- **Filter**（濾波器）— 保留或去除特定頻率範圍。**低通 / 高通 / 帶通 / 陷波（帶拒）**。
- **FIR / IIR**（Finite / Infinite Impulse Response，有限／無限脈衝響應）— 兩類濾波器；FIR 穩定且具線性相位，IIR 計算量較小但可能失真相位。
- **Zero-phase filtering**（零相位濾波）— 將濾波器正向與反向各套用一次（`filtfilt`）以消除相位失真。
- **FFT**（Fast Fourier Transform，快速傅立葉轉換）— 快速計算訊號頻率成分的演算法。
- **PSD**（Power Spectral Density，功率頻譜密度）— 功率在各頻率上的分布。
- **Welch's method**（Welch 法）— 對重疊窗口的頻譜取平均，以得到穩定的 PSD 估計。
- **STFT**（Short-Time Fourier Transform，短時傅立葉轉換）/ **Spectrogram**（頻譜圖）— 隨時間變化的頻率成分。
- **Wavelet**（小波）— 使用頻率相關窗口長度的時頻分析方法。
- **Band power**（頻帶功率）— 某頻帶內的訊號能量（常用特徵）。
- **Epoch / trial**（試驗段）— 從連續錄製中截取出的短段且附有標記的訊號。
- **Baseline correction**（基線校正）— 減去事件前的平均值，使各試驗段從零點開始。
- **ICA**（Independent Component Analysis，獨立成分分析）— 將錄製訊號分解為獨立來源，以便去除雜訊（眨眼、心跳）。
- **ASR**（Artifact Subspace Reconstruction，雜訊子空間重建）— 從乾淨子空間重建受汙染的訊號段；本教學使用簡化的振幅截波替代方案。

## 特徵與模型
- **Feature**（特徵）— 用來摘要一筆試驗供模型使用的數值（或向量）。
- **Connectivity**（連結性）— 各通道間的關係：**coherence**（頻域相關性）、**PLV**（Phase-Locking Value，相位鎖定值，0–1）。
- **CSP**（Common Spatial Patterns，共同空間模式）— 學習使兩類別之間變異數差距最大化的通道混合方式（空間濾波器）；運動想像的主力方法。
- **Covariance matrix**（共變異數矩陣）— 一筆試驗中每對通道之間的共變程度。
- **Riemannian / tangent space**（黎曼流形／切線空間）— 在共變異數矩陣的曲面幾何上操作，並投影到平面空間供一般分類器使用；是一個強力的基準方法。
- **LDA / SVM / RF** — Linear Discriminant Analysis（線性判別分析）/ Support Vector Machine（支援向量機）/ Random Forest（隨機森林）：傳統分類器。
- **Pipeline** — 一個 sklearn 物件，將前處理步驟與模型串接，確保每一步只在訓練資料上擬合（防止洩漏）。
- **EEGNet / ShallowConvNet / DeepConvNet** — 專為 EEG 設計的卷積神經網路。
- **LSTM**（Long Short-Term Memory，長短期記憶）— 處理序列的遞迴神經網路。
- **Transformer**（轉換器）— 以自注意力機制為基礎的神經網路。
- **Self-supervised / foundation model**（自監督學習／基礎模型）— 在大量未標記資料上預訓練，再針對小型標記任務進行微調。

## 機器學習基礎
- **Features (X) / labels (y)**（特徵／標記）— 輸入資料與預測目標。
- **Train / validation / test**（訓練集／驗證集／測試集）— 分別用於擬合模型／調整超參數／最終一次性誠實評估。**測試集是神聖的**——只在最後用一次。
- **Overfitting**（過擬合）— 模型記住訓練資料的雜訊；訓練分數高，測試分數差。
- **Decision boundary**（決策邊界）— 分類器切換預測結果的分隔面。
- **Cross-validation (CV)**（交叉驗證）— 反覆切分訓練集與測試集，以更穩定地估計模型效能。
- **Chance level**（隨機基準）— 最簡單基準線的分數（**不**總是 1/n_classes——對於不平衡資料，是多數類別的比例）。
- **Accuracy / Balanced accuracy / F1 / ROC-AUC** — 分類評估指標；在類別不平衡時，balanced accuracy 和 F1 比 accuracy 更公平。
- **Confusion matrix**（混淆矩陣）— 真實類別與預測類別的交叉表。
- **mean ± std**（平均值 ± 標準差）— 某指標在多折／多種子上的平均值與分散程度；務必一起回報。
- **Confidence interval (CI)**（信賴區間）— 合理包含真實值的範圍。
- **Paired test**（配對檢定，t 檢定／Wilcoxon）— 在宣稱某一模型優於另一模型之前，逐折比較兩個模型。

## 評估誠信（本教學的核心）
- **Leakage**（資料洩漏）— 測試集的資訊滲入訓練過程（直接洩漏，或透過對全體資料擬合的轉換器間接洩漏）；會虛高分數。
- **Random-shuffle split**（隨機打亂切分）— 隨機切分有時間相關性的時間序列；會洩漏相鄰樣本。**本 repo 禁止使用。**
- **Block / trial-aware split**（區塊／試驗感知切分）— 保持整段試驗／區塊在切分的同一側（`make_block_split`）。
- **Subject-dependent vs subject-independent**（受試者相依 vs 跨受試者）— 在相同 vs 新的受試者上測試。
- **LOSO**（Leave-One-Subject-Out，留一受試者出法）— 將某位受試者的全部資料保留作測試集；最誠實的對外發表指標。
- **Domain / distribution shift**（領域／分布偏移）— 訓練與測試資料來自不同條件（不同次錄製、不同日期、不同設備）；效能因此下降。
- **Non-stationarity**（非穩態性）— 訊號統計特性隨時間改變。

## BCI 與資料集
- **BCI**（Brain-Computer Interface，腦機介面）— 將腦部訊號轉換為指令的系統。
- **Motor imagery (MI)**（運動想像）— 想像做出某個動作以控制 BCI。
- **Brain-to-text / speech neuroprosthesis**（腦文字轉換／語音神經義肢）— 解碼意圖中的言語，通常來自侵入式植入物。
- **Neuro-rights**（神經權利）— 保護心理隱私與自主性的提案性權利。
- **Dual-use**（雙重用途）— 可用於有益或有害目的的技術。
- **BCI Competition IV 2a** — 標準的 9 位受試者運動想像 EEG 資料集（本教學的主打資料）。
- **PhysioNet / Sleep-EDF / MNE sample** — 本教學使用的其他公開資料集。
- **MOABB**（Mother of All BCI Benchmarks）— 提供標準化 BCI 資料集的函式庫。
- **MNE** — Python 標準 EEG/MEG 分析函式庫。
- **LSL**（Lab Streaming Layer，實驗室串流層）— 串流即時訊號的協定；**BrainFlow** / **MNE-LSL** 是消費級 EEG（OpenBCI、Muse、Emotiv）常用的工具。

## 本 Repo 專用
- **`neuro101`** — 本教學可匯入的輔助程式包（`src/neuro101/`）。
- **`NEURO101_SMOKE=1`** — 環境變數，設定後只載入最小資料切片（供 CI 使用）。
- **`NEURO101_DATA`** — 環境變數，用來覆寫資料集快取目錄。
- **Smoke mode**（煙霧測試模式）— 使用次取樣資料快速執行，讓 notebook 在幾秒內跑完。
