# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiShengChen/neural-signals-101/blob/main/notebooks/zh-TW/02_ml_from_zero.ipynb)
#
# > **在 Google Colab 上執行？** 請先執行下一個儲存格——它會安裝所有套件並
# > 下載輔助工具包。**在本地端執行（執行 `make setup` 之後）？** 下一個
# > 儲存格不會做任何事；直接執行並繼續即可。

# %%
# --- Colab 啟動程序：僅在 Colab 環境中安裝相依套件與 neuro101 套件 ---
import sys, os
if "google.colab" in sys.modules:
    !pip install -q "mne==1.8.0" "moabb==1.2.0" "braindecode==0.8.1" "pyriemann==0.7" "scikit-learn==1.5.2"
    if not os.path.exists("neural-signals-101"):
        !git clone -q https://github.com/ChiShengChen/neural-signals-101
    sys.path.insert(0, os.path.abspath("neural-signals-101/src"))
    print("Colab 設定完成——請繼續進入下方章節。")

# %% [markdown]
# # 第二章 — 從零開始的機器學習（ML from Zero）
#
# ## 學習目標
# 1. 理解什麼是**特徵（features）**（X）和**標籤（labels）**（y），並在
#    二維散點圖上視覺化呈現。
# 2. 將資料分割為**訓練集（train）/ 驗證集（validation）/ 測試集（test）**，
#    並確切了解每個部分的用途。
# 3. 辨識**過擬合（overfitting）**：為何訓練準確率上升至 1.0 是警告訊號，
#    而非成功的象徵。
# 4. 閱讀**決策邊界（decision boundary）**圖，並憑直覺辨別欠擬合、良好擬合
#    和過擬合的模型。
# 5. 理解為何偷看測試集來選擇模型會**虛報分數**，以及如何避免這種情況。
#
# > **先備知識：** 無——這是從零開始的起點。
# > **難度：** ★★☆☆☆
# > **執行時間：** 約 1 分鐘（玩具資料，使用 CPU）。

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

# 本章後續使用的其他 sklearn 匯入
from sklearn.datasets import make_moons, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1 — 特徵與標籤：X 和 y 是什麼
#
# 每個監督式機器學習（supervised machine-learning）流程都從兩個陣列開始：
#
# | 符號 | 形狀 | 白話意義 |
# |---|---|---|
# | **X** | `(n_samples, n_features)` | 每個樣本的*測量值*——模型**看到**的東西 |
# | **y** | `(n_samples,)` | 每個樣本的*正確答案*——模型必須**預測**的東西 |
#
# **樣本（sample）** 是你測量的一件事（此處為：一個資料點）。
# **特徵（feature）** 是描述該樣本的一個數字（此處為：它在平面上的 x 和 y 座標）。
# **標籤（label）** 是該樣本所屬的類別（此處為：0 = 藍色，1 = 橘色）。
#
# 我們使用 `sklearn.datasets.make_moons` 來生成一個玩具二分類問題——
# 兩個交錯的半月形狀，容易視覺化但*並非*線性可分
#（一條直線無法完美地將它們分開）。

# %%
# ----- 生成玩具資料 -----
X, y = make_moons(n_samples=500, noise=0.30, random_state=0)

print(f"X 形狀 : {X.shape}   <- (n_samples, n_features)")
print(f"y 形狀 : {y.shape}   <- (n_samples,)")
print(f"類別 : {np.unique(y)}   <- 0 = 月亮-A，1 = 月亮-B")

fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", alpha=0.6, edgecolors="k",
                     linewidths=0.4, s=30)
ax.set_xlabel("特徵 0  (X[:, 0])")
ax.set_ylabel("特徵 1  (X[:, 1])")
ax.set_title("我們的玩具資料集——兩個月亮\n"
             "顏色 = 標籤 y（紅=1，藍=0）")
plt.colorbar(scatter, ax=ax, label="y（類別標籤）")
plt.tight_layout()
plt.show()

# %% [markdown]
# 注意藍色和紅色是交纏在一起的——沒有任何一條直線能完美地將它們分開。
# 這正是重點：真實資料很少是線性可分的，我們需要一個足夠靈活、
# 能捕捉曲線邊界的模型。

# %% [markdown]
# ## 2 — 訓練集 / 驗證集 / 測試集分割
#
# 在碰模型之前，我們先將資料集分割成**三個不重疊**的部分：
#
# | 部分 | 典型大小 | 用途 |
# |---|---|---|
# | **訓練集（Train）** | 60–70% | 模型從這裡*學習* |
# | **驗證集（Validation）** | 15–20% | 在這裡調整超參數（hyperparameters）；可以隨意查看 |
# | **測試集（Test）** | 15–20% | **神聖不可侵犯**——僅在最後階段使用一次 |
#
# ### 為什麼測試集神聖不可侵犯？
#
# 每次你查看測試集並做出決定（例如「複雜度 7 比複雜度 5 好」），
# 你都在*隱性地*對它進行擬合。多次偷看之後，測試準確率就不再告訴你
# 模型在真正未見過的資料上的表現——它告訴你的是你的搜索有多好。
# 你將在第 5 節看到具體的示範。
#
# > **關於資料洗牌的注意事項——在第 12 章之前請務必閱讀。**
# > 以下我們在分割前用 `train_test_split` 洗牌資料。這在*這裡*是正確的，
# > 因為 `make_moons` 生成的是**獨立同分佈（i.i.d.，independently and
# > identically distributed）**的隨機點——每個點都獨立於其他所有點。
# > EEG 錄音是**時間序列（time series）**：附近的樣本是相關的，
# > 因此洗牌會將未來的資訊洩漏到訓練集中，並給出過度樂觀的結果。
# > 第 12 章介紹正確的時間序列資料分割方式。

# %%
# ----- 三向分割 -----
# 第一步：切出測試集（20%）
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y
)
# 第二步：將剩餘部分分為訓練集（80% 的 75% = 60%）+ 驗證集（20%）
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=0, stratify=y_trainval
)

print(f"總樣本數     : {len(X)}")
print(f"  訓練集     : {len(X_train)}  ({len(X_train)/len(X):.0%})")
print(f"  驗證集     : {len(X_val)}   ({len(X_val)/len(X):.0%})")
print(f"  測試集（神聖）: {len(X_test)}  ({len(X_test)/len(X):.0%})")

# %% [markdown]
# ## 3 — 過擬合（Overfitting）
#
# **決策樹（Decision Tree）**將特徵空間分割成矩形區域；
# 樹越深，分割越多，邊界越複雜。
#
# `max_depth=1` 只能做一次分割——它會**欠擬合（underfit）**（太簡單）。
# `max_depth=15` 可以記住每個訓練點——它會**過擬合（overfit）**（太複雜，
# 邊界扭曲，無法泛化）。
#
# ### 執行下一個儲存格之前——先做預測！
#
# 當 `max_depth` 從 1 增加到 15 時：
# - 你預期**訓練準確率**會發生什麼變化？
# - 你預期**驗證準確率**會發生什麼變化？
#
# 先寫下你的猜測（或只是想一想），然後再執行儲存格。

# %%
depths = list(range(1, 16))
train_accs, val_accs = [], []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, clf.predict(X_train)))
    val_accs.append(accuracy_score(y_val, clf.predict(X_val)))

best_val_depth = depths[np.argmax(val_accs)]
print(f"驗證準確率最佳深度 : {best_val_depth}")
print(f"  最佳深度的訓練準確率 : {train_accs[best_val_depth-1]:.3f}")
print(f"  最佳深度的驗證準確率 : {max(val_accs):.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(depths, train_accs, "o-", color="steelblue", label="訓練準確率")
ax.plot(depths, val_accs,   "s-", color="tomato",    label="驗證準確率")
ax.axvline(best_val_depth, color="gray", linestyle="--", alpha=0.7,
           label=f"最佳驗證深度 = {best_val_depth}")
ax.set_xlabel("樹的 max_depth（模型複雜度 →）")
ax.set_ylabel("準確率")
ax.set_title("過擬合 U 形差距曲線\n"
             "訓練持續攀升；驗證先升後降")
ax.legend()
ax.set_ylim(0.5, 1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# **你應該看到的結果：**
#
# - **訓練準確率**隨著 `max_depth` 增大而穩定攀升至 1.0。
#   在深度 15 時，樹已記住每個訓練點——訓練準確率達到 100%。
# - **驗證準確率**在某個適中的深度達到峰值，然後隨著樹開始記住訓練雜訊而*下降*。
# - 兩條曲線之間的差距就是**過擬合差距**。差距越大，表示你的模型越無法
#   泛化到新資料，這是一個警告訊號。
#
# 「正確的」深度是**驗證**曲線達到峰值的那個——我們絕不使用測試集來做這個選擇。

# %% [markdown]
# ## 4 — 決策邊界視覺化（Decision Boundary Visualisation）
#
# **決策邊界**是模型將預測從一個類別切換到另一個類別的線（或曲線）。
# 我們透過預測細密網格上每個點的類別來繪製它，然後以顏色填充網格
#（這稱為 **meshgrid contourf** 圖）。
#
# 我們將並排查看三種深度：
#
# | 深度 | 預期行為 |
# |---|---|
# | 1 | **欠擬合（Underfit）**——一條水平或垂直切割；遺漏了大部分結構 |
# | best_val | **良好擬合（Good fit）**——大致跟隨兩個月亮的曲線邊界 |
# | 15 | **過擬合（Overfit）**——鋸齒狀、記憶化的孤島，擬合了訓練資料中的*雜訊* |
#
# ### 執行之前：預測邊界的形狀
#
# 對於深度 1、`best_val_depth` 和深度 15——你預期彩色區域的形狀是什麼？
# 它們看起來會是平滑的還是鋸齒狀的？

# %%
def plot_boundary(ax, clf, X_plot, y_plot, title):
    """繪製帶有訓練點疊加的 meshgrid 決策邊界。"""
    x_min, x_max = X_plot[:, 0].min() - 0.4, X_plot[:, 0].max() + 0.4
    y_min, y_max = X_plot[:, 1].min() - 0.4, X_plot[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="bwr", levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap="bwr",
               edgecolors="k", linewidths=0.4, s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, depth, label in zip(
    axes,
    [1, best_val_depth, 15],
    ["欠擬合 (depth=1)", f"良好擬合 (depth={best_val_depth})", "過擬合 (depth=15)"]
):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_train, y_train)
    plot_boundary(ax, clf, X_trainval, y_trainval, label)

fig.suptitle("決策邊界——欠擬合 / 良好擬合 / 過擬合", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **你應該看到的結果：**
#
# - **左邊（depth=1）：**一條單一的水平或垂直條帶——模型對曲線月亮毫無概念。
#   許多點都是錯誤的。
# - **中間（depth=best）：**一條平滑的曲線邊界，跟隨月亮走向而不會太扭曲。
#   在嘈雜的重疊區域附近有些錯誤——這是正常且預期中的。
# - **右邊（depth=15）：**鋸齒狀、充滿孤島的區域，緊密包裹著各個訓練點。
#   在*新的*資料上，那些孤島大多是錯誤的。

# %% [markdown]
# ## 5 — 重點結論：偷看測試集會虛報你的分數
#
# 假設我們很不耐煩，跳過了驗證集。我們在每個深度訓練一棵樹，
# 每次都在**測試集**上評估它，然後選擇最大化測試準確率的深度。
# 我們會回報什麼分數？

# %%
# ----- 「作弊」情境：透過偷看測試集來選擇深度 -----
test_accs_all = []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=0)
    clf.fit(X_train, y_train)            # 僅在訓練集上訓練
    test_accs_all.append(accuracy_score(y_test, clf.predict(X_test)))

best_cheat_depth = depths[np.argmax(test_accs_all)]
cheat_score      = max(test_accs_all)

# ----- 誠實情境：透過驗證集選擇深度，僅回報一次測試結果 -----
honest_clf = DecisionTreeClassifier(max_depth=best_val_depth, random_state=0)
honest_clf.fit(X_train, y_train)
honest_score = accuracy_score(y_test, honest_clf.predict(X_test))

print("=" * 50)
print(f"作弊  — 透過最大化測試準確率選擇 depth={best_cheat_depth}")
print(f"  回報的測試準確率 : {cheat_score:.3f}  <- 過度樂觀，請勿信任")
print()
print(f"誠實  — 透過驗證集選擇 depth={best_val_depth}，僅回報一次測試結果")
print(f"  回報的測試準確率 : {honest_score:.3f}  <- 可信賴的")
print("=" * 50)
print()
print("差異（樂觀偏差）:", round(cheat_score - honest_score, 3))

# %% [markdown]
# **誠實**的數字更低。這個差距稱為**樂觀偏差（optimism bias）**：
# 每次你查看測試集並用它來做決定，你就消耗了一點它的獨立性。
# 經過足夠多次偷看之後，測試集實際上已成為另一個訓練集。
#
# **規則：**
# - 使用**驗證集**來調整超參數（樹的深度、鄰居數量、正則化強度……）。
# - 在最後**一次**評估**測試集**，並回報那個數字。
#   然後停止。不要再回去試圖擠出更多。
#
# 如果你的資料非常少，可以在訓練集+驗證集上使用 **k 折交叉驗證（k-fold
# cross-validation）**，而不是單次分割——但測試集無論如何都必須保持神聖不可侵犯。

# %% [markdown]
# ## ✅ 概念確認（Concept Check）
#
# 繼續之前請先回答以下問題：
#
# 1. 一個模型達到 99% 的訓練準確率和 72% 的測試準確率。這是好還是壞？
#    這種情況的專業術語是什麼？
# 2. 你用不同的超參數訓練了五個模型，在測試集上分別評估每個模型，
#    並選擇最好的那個。為什麼回報的測試準確率會有誤導性？
# 3. **驗證集**的目的是什麼？它與測試集有何不同？
# 4. 在本章中我們在分割前洗牌了資料。為什麼對 EEG 時間序列資料洗牌是*錯誤*的？
#    （提示：思考「i.i.d.」的含義。）
#
# **答案：**
#
# 1. 不好——這是**過擬合（overfitting）**。模型記住了訓練資料，但無法泛化。
#    訓練和測試準確率之間 27 個百分點的差距就是過擬合差距。
# 2. 你實際上將測試集用作第二個驗證集。每次比較都是一次靠運氣選出看起來最好結果的機會，
#    因此獲勝的數字存在樂觀偏差。
# 3. 驗證集用於在模型開發過程中調整超參數；你可以多次查看它。
#    測試集在最後階段才被使用，且只使用一次，因此其準確率是對未來表現的無偏估計。
# 4. 時間點 t 和 t+1 的 EEG 樣本是相關的（非獨立同分佈）。洗牌破壞了這個結構，
#    讓模型在訓練期間「看到」未來的時間點，並產生過度樂觀的準確率。
#    第 12 章介紹時間序列安全的分割策略。

# %% [markdown]
# ## ⚠️ 警告——常見錯誤 / 為什麼這樣做是錯的
#
# - **透過查看測試集來選擇你的模型。** 即使「只看一眼」也算。
#   每一眼都讓測試集影響你的選擇，虛報你的準確率。
#   所有調整決策都應使用驗證集。
#
# - **完全沒有保留測試集。** 只回報訓練準確率（甚至只回報完整資料集的交叉驗證準確率）
#   對泛化性無法提供任何資訊。永遠保留一個你從未碰過的測試集。
#
# - **洗牌時間序列資料。** 對於本章中的獨立同分佈玩具資料集，洗牌是沒問題的。
#   但對於 EEG（或任何時間序列），它會將未來的資訊洩漏到訓練集中。
#   第 12 章展示了正確的方法：群組感知或時間有序的分割。
#
# - **閱讀一個準確率數字而不與基準比較。** 如果你的資料集有 90% 的 A 類樣本，
#   一個總是預測 A 的模型可以在零知識的情況下獲得 90% 的準確率。
#   永遠與**多數類別基準（majority-class baseline）**（或隨機機率）進行比較。
#
# - **忘記做分層（stratify）。** 在 `train_test_split` 中不加 `stratify=y` 的話，
#   隨機分割可能將幾乎所有某個類別的資料放在訓練集，而測試集幾乎沒有
#   （這在小型或不平衡資料集中是個問題）。分類任務中永遠傳入 `stratify=y`。

# %% [markdown]
# **下一章：** 第三章——*看得見的數學（Math You Can See）*——建立幾何直覺
# （點積、距離、投影），這是本教學中每個模型的基礎，搭配互動式圖表，
# 不需要繁重的代數。
