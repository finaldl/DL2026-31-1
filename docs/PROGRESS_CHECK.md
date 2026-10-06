# 🔍 Báo cáo tiến độ & Checklist Double-Check

> Người soạn: Manager (qua Claude Code) — **cập nhật 07/10/2026**, sau khi kiểm tra các push: Thu (`thu`: `8762440`, `3aec714`), Đức (`leanhduc`: `d754f89`, kNN `knn_k=5`), Thiết (`thiet`: `d908fb5`, `cb09506`) và Zam (`zam-audit`: `a7973cd`, audit fine-tune).
> Mục đích: ghi lại những gì đã kiểm tra, kết quả, và những việc **còn mở** trước khi đưa số liệu lên slide.
> Đối chiếu với: [`PROJECT_PLAN.md`](PROJECT_PLAN.md), [`project_analysis.md`](project_analysis.md), [`../README.md`](../README.md).

---

## 🔒 EXPERIMENT FREEZE ĐÃ CHỐT (T3 06/10)

- ✅ **ResNet-50 weights**: Thu đã chuyển `src/features.py` về `IMAGENET1K_V1`, trích xuất lại 4 file `resnet50_*.pt`, chạy lại frozen grid, build lại master table và plot — xong lúc 21:26, trước khi hết ngày freeze. Frozen và fine-tune giờ **cùng weights**.
- ✅ **`knn_k`**: theo phương án (a) — Đức chạy lại kNN với **`knn_k=5`** trên feature V1 (`d754f89`, 07/10 00:02). Lần đầu (22:48 06/10, `af699c1`) dùng nhầm feature ResNet V2 → đã chạy lại, kiểm tra theo từng dòng đạt (mục 4.2). Bản `knn_k=20` giữ trong `results/frozen_grid_v2.csv` làm đối chứng. Linear Probe vẫn là phương pháp frozen chính.
- 🔒 Từ giờ **không chạy thí nghiệm mới**. **Nguồn số duy nhất cho slide = `results/master_table_wide_v2_knn5.csv`** (kNN `knn_k=5` + Linear Probe V1 + fine-tune), plot ở `plots/eval_v2_knn5/`.

---

## 🎯 TÓM TẮT: AI CẦN LÀM GÌ

| Ai | Việc cần làm | Hạn | Chi tiết |
|---|---|---|---|
| **Thiết** | (Tùy chọn, thẩm mỹ) Tiêu đề plot → "Accuracy vs label budget", nhãn "KNN (k=5)" → "kNN (5 neighbours)", thống nhất màu với `finetune_vs_frozen.png` | 🟢 Thấp | Mục 2.2 |
| **Thiết** | (Tùy chọn) Chỉnh nhỏ `scripts/plot_finetune_vs_frozen.py`: đọc `master_table_v2_knn5.csv`, tiêu đề ghi "LP: 5 seeds · Fine-tune: 3 seeds", ghi chú trục y bắt đầu từ 75% | 🟢 Thấp | Mục 2.2 |
| **Mỹ** | Đưa draft slide lý thuyết #4–#8 + Q&A cheat sheet lên repo/drive chung | 🔴 **Trễ** (hạn CN 4/10) | Mục 1 |
| **Đức** | Slide kết quả #10–#13, lấy số từ **`master_table_wide_v2_knn5.csv`**, plot từ `plots/eval_v2_knn5/`. Không dùng `KET_QUA_DOC_BAO.md` (legacy) hay các file `_knn5` của commit `af699c1` (ResNet V2) | 🟡 T5 08/10 | Mục 3 |
| **Managers** | Viết slide #14–#16 (phân tích, limitations, kết luận) theo số V1 — xem bảng câu chuyện ở mục 3 | 🟡 T5 08/10 | Mục 3 |
| **Managers** | Merge `zam-audit` (kết quả fine-tune đã sửa bug), đưa bảng phụ lên backup slide; sửa câu "fine-tune là lower bound" nếu đã có trên slide | 🔴 Sáng T5 08/10 | Mục 2.1 |
| **Đức** (không bắt buộc) | Nếu còn nhớ: cho biết `splits/*.json` được sinh bằng code nào. Không còn chặn việc gì — splits đã được coi là dữ liệu cố định, kiểm tra bằng `scripts/verify_splits.py` | 🟢 Thấp | Mục 2.4 |
| **Thiết (+ Thu)** | Reproducibility check trên máy khác — dùng `features.zip` mới trên Drive, so với sai số ~0.1 điểm (mục 2.5) | 🟡 T6 09/10 | Mục 2.5 |
| **Managers + Đức** | Thêm vào slide limitations: bug head Stage 2, fine-tune và frozen dùng ảnh k-shot khác nhau ở k ≤ 25 (so sánh theo mean) | 🟡 T5 08/10 | Mục 2.1, 2.4 |

**Đã xong, không cần nhắc nữa**: splits 70 file (verify độc lập Pets + EuroSAT), feature 8 file `.pt` (ResNet bản V1), code eval + 10 test, frozen grid v2 (280/280, chạy lại 06/10), kNN `knn_k=5` (280/280, 07/10), fine-tune grid sạch (24/24), master table gộp (64 dòng), memory benchmark, **thống nhất ResNet-50 weights V1**, **chốt `knn_k=5`**, **plot fine-tune vs frozen**, **audit fine-tune + sửa `run_finetune.sh`** (Zam), **quyết định về bug fine-tune** (phương án b).

---

## 1. Trạng thái tổng quan theo Milestone

| Milestone | Deadline | Trạng thái (07/10) | Ghi chú |
|---|---|---|---|
| Features cached + kNN verified ≥2 người | CN 27/9 | 🟢 Xong 29/9 | ResNet trích xuất lại bằng V1 ngày 06/10 |
| Frozen grid hoàn thành | T3 29/9 | 🟢 Xong | Chạy lại 06/10 với feature ResNet V1; kNN `knn_k=5` chạy lại 07/10 |
| Fine-tune loop + recipe khóa | T4 30/9 | 🟢 Xong | Xem lưu ý recipe ở mục 2.1 |
| Fine-tune grid 24 runs | CN 4/10 | 🟢 Xong sớm (29/9) | |
| Master table mọi kết quả | CN 4/10 | 🟢 Xong | Bản chính thức `master_table_v2_knn5.csv` (07/10), 64 dòng |
| Biểu đồ fine-tune vs frozen | CN 4/10 | 🟢 Xong 07/10 (trễ 3 ngày) | `plots/finetune_vs_frozen.png` (Thiết, `d908fb5`) |
| Draft slides lý thuyết (Mỹ) | CN 4/10 | 🔴 **Trễ** — chưa thấy | Managers hỏi trực tiếp |
| 🔒 Experiment Freeze | T3 6/10 | 🟢 **Đã chốt** | Xem đầu file |
| Slide hoàn thành | T5 8/10 | ⏳ Còn 2 ngày | Chưa có thư mục `slides/` |
| Reproducibility check | T6 9/10 | 🟡 Có rủi ro | `run_finetune.sh` đã sửa (Zam); `splits/*.json` không tái tạo được từ `src/sampler.py` (mục 2.4) |
| 2 buổi diễn tập | T7–CN 10–11/10 | ⏳ | |

---

## 2. ⚠️ Việc còn mở

### 🔴 2.1 Audit fine-tuning — XONG (Zam, `a7973cd`), phát hiện bug

Zam đã audit xong: tài liệu giải thích phần fine-tune + Q&A (Zam giữ bản riêng; đã đưa lên `zam-audit` rồi xóa ở `e145e93`), hình `plots/finetune_loss_curves.png` (backup slide B2, sinh bằng `scripts/plot_finetune_losses.py` từ log TensorBoard `runs/`). Loss giảm đều ở cả 2 stage, không NaN, không phân kỳ.

Pets, Top-1 %, fine-tune 3 seed, LP 5 seed, cả hai cùng ResNet V1:

| k | FT ResNet-50 | LP ResNet-50 | Δ | FT DINOv2 | LP DINOv2 | Δ |
|---|---|---|---|---|---|---|
| 5 | 79.4 ± 2.8 | 85.6 | −6.1 | 81.9 ± 1.3 | 88.3 | −6.4 |
| 10 | 81.6 ± 0.7 | 89.2 | −7.5 | 88.4 ± 0.7 | 91.5 | −3.1 |
| 25 | 86.8 ± 1.0 | 92.4 | −5.6 | 91.3 ± 0.2 | 93.9 | −2.6 |
| all | 92.3 ± 0.6 | 94.3 | −1.9 | 95.2 ± 0.4 | 96.2 | −1.0 |

**🐛 Bug (Manager đã đối chiếu code, xác nhận)**: trong `src/finetune.py`, `build_model()` trả head params dưới dạng **generator** (`model.fc.parameters()` / `model.head.parameters()`). `opt1 = AdamW(head_params, ...)` ở Stage 1 đọc hết generator → ở Stage 2, `{'params': head_params}` là **nhóm rỗng** (PyTorch không báo lỗi vì nhóm backbone không rỗng). Nghĩa là trong 30 epoch Stage 2 **chỉ backbone được train**, head đứng yên ở giá trị sau 10 epoch Stage 1. Ban đầu nghĩ số chính thức là *lower bound* — kết quả bản sửa bug (bên dưới) cho thấy **không phải**: sửa xong lại thấp hơn.

Các phát hiện khác của Zam (đã sửa trong `src/finetune_fixed.py`, **chưa chạy**, ghi ra `results/finetune_grid_fixed.csv` + `runs_fixed/`, không đụng số chính thức):
- Chỉ seed việc chọn ảnh; khởi tạo head, thứ tự batch, augmentation không seed → chạy lại lệch cỡ std (1–3 điểm ở k=5).
- Stage 1 gọi `model.train()` → BatchNorm của ResNet vẫn cập nhật running stats dù backbone "đóng băng".
- Dùng ảnh k-shot khác với frozen grid (mục 2.4).
- Recipe giữ nguyên (không tune) — đúng tinh thần freeze.

**✅ Managers đã quyết (07/10): phương án (b)** — số chính thức giữ nguyên, chạy `src/finetune_fixed.py` làm kết quả phụ.

**✅ Kết quả bản đã sửa bug (Zam, `5356825`, 07/10 04:28) — kiểm tra đạt**: 24/24 dòng, không trùng key, cùng bộ key với bản gốc; log không có `generating this split` (đã đọc đúng `splits/*.json`); loss giảm đều; exit code 0 (dòng `Traceback … triton` là cảnh báo vô hại khi import torch trên Windows).

| Backbone | k | Bản gốc | Bản sửa bug | LP | Sửa bug − gốc | Sửa bug − LP |
|---|---|---|---|---|---|---|
| ResNet-50 | 5 | 79.4 ± 2.8 | 71.5 ± 1.0 | 85.6 | −8.0 | −14.1 |
| ResNet-50 | 10 | 81.6 ± 0.7 | 76.1 ± 3.3 | 89.2 | −5.5 | −13.1 |
| ResNet-50 | 25 | 86.8 ± 1.0 | 82.3 ± 1.5 | 92.4 | −4.5 | −10.1 |
| ResNet-50 | all | 92.3 ± 0.6 | 90.7 ± 0.5 | 94.3 | −1.6 | −3.5 |
| DINOv2 | 5 | 81.9 ± 1.3 | 81.3 ± 3.6 | 88.3 | −0.5 | −6.9 |
| DINOv2 | 10 | 88.4 ± 0.7 | 85.4 ± 1.1 | 91.5 | −3.0 | −6.1 |
| DINOv2 | 25 | 91.3 ± 0.2 | 89.0 ± 0.4 | 93.9 | −2.3 | −4.9 |
| DINOv2 | all | 95.2 ± 0.4 | 94.0 ± 0.6 | 96.2 | −1.1 | −2.2 |

**Kết luận**: sửa bug **không** làm fine-tune tốt lên — bản sửa thấp hơn bản gốc ở mọi k. Vậy số chính thức **không phải lower bound** (bỏ cụm này khỏi slide/report); kết luận RQ2 (FT < LP, DINOv2 > ResNet) đứng vững dù có hay không có bug. Bản sửa đổi 4 thứ cùng lúc (head train ở Stage 2, dùng `splits/*.json`, seed đầy đủ, BatchNorm đóng băng ở Stage 1) nên không quy được mức giảm cho một thay đổi; ở k=all ảnh giống hệt, nên mức giảm ở đó đến từ thay đổi khi train. Cách đọc hợp lý: head tiếp tục train với LR 1e-3 thêm 30 epoch trên ít ảnh → overfit; bug vô tình đóng vai trò regularization. Đây là câu trả lời Q&A tốt.

Lưu ý recipe: `config/finetune_recipe.yaml` trùng y hệt bộ ứng viên trong plan; không có tập validation. Trên slide nói **"recipe cố định, chọn trước, không tune theo k"**.

### 🟢 2.2 Biểu đồ

- `plots/eval_v2_knn5/frozen_curves_pets.png` (bản chính thức, Đức vẽ 07/10; `plots/eval_v2/` là bản `knn_k=20`) **đã có** 2 đường fine-tune, nhưng chưa dùng được trên slide: 6 đường trong một hình, fine-tune DINOv2 dùng màu xanh giống ResNet (không có style riêng trong `make_plots.py`), tiêu đề "Frozen-feature evaluation".
- ✅ **`plots/finetune_vs_frozen.png`** (Thiết, `d908fb5`, 07/10 00:08) + script `scripts/plot_finetune_vs_frozen.py` — **dùng cho slide RQ2**. Đã kiểm tra: chỉ LP vs fine-tune trên Pets, k ∈ {5, 10, 25, all}; ResNet xanh / DINOv2 đỏ (cùng màu với `eval_v2_knn5`), nét liền = LP, nét đứt = fine-tune, có error bar, DPI 200. Cả 16 điểm (mean + std) khớp bảng; Pets không lệch ô nào giữa `master_table_v2.csv` (script đang đọc) và `master_table_v2_knn5.csv`. Chưa chạy lại được script ở máy Manager (thiếu pandas/matplotlib).
- **Vẽ lại 2 plot `eval_v2_knn5/` (Thiết đề xuất 07/10, đã duyệt)**: hai file này do `scripts/make_plots.py` sinh ra, nên muốn đổi hình thì **sửa script rồi chạy lại**, không chép đè PNG vẽ bằng code khác — nếu không, lần chạy `run_frozen.sh`/`make_plots.py` kế tiếp (VD repro check 09/10) sẽ ghi đè lại bản cũ, và hình trên slide không tái tạo được. Cần sửa: thêm `STYLE` cho `finetune` (cùng màu backbone, marker tam giác, nét chấm-gạch), tiêu đề không còn ghi "Frozen-feature", nhãn chú thích dễ đọc. Giữ nguyên tên file và tham số `--input/--output-dir`.
- Góp ý nhỏ (tùy chọn): đổi `INPUT` sang `master_table_v2_knn5.csv` cho đúng quy ước một nguồn số (hình không đổi); tiêu đề "3-5 seeds" → "LP: 5 seeds · Fine-tune: 3 seeds"; trục y bắt đầu từ 75% làm khoảng cách trông lớn hơn → nói rõ khi trình bày.
- EuroSAT: `plots/eval_v2_knn5/frozen_curves_eurosat.png` dùng được thay cho `eurosat_comparison.png`.

### 🟢 2.3 `scripts/run_finetune.sh` — ĐÃ SỬA (Zam, `a7973cd`)

Giờ: `set -e`, xóa `results/finetune_grid.csv` cũ, chạy `python src/finetune.py --grid` từ repo root, in "Completed" **sau** khi chạy xong. Đúng như yêu cầu.

### 🟡 2.4 Fine-tune và frozen dùng ảnh k-shot KHÁC NHAU; `splits/*.json` không tái tạo được

Zam kiểm tra (có `data/`): ảnh k-shot mà `finetune.py` tự sinh bằng `make_kshot_splits` **chỉ trùng ~3%** với `splits/pets_seed*_k*.json` ở k=5/10/25 (≈ mức trùng ngẫu nhiên); k=all giống hệt. Báo cáo 01/10 đoán "nên trùng" — **sai**. Lý do: `splits/*.json` được commit cùng pipeline eval của Đức (`87ed900`), **không** sinh từ `src/sampler.py`; Zam thử `random` và nhiều generator numpy đều không tái tạo được → nhiều khả năng từ một notebook chưa commit. Manager chưa tự kiểm tra con số 3% được (không có `data/`); Zam có script kiểm tra trong tài liệu riêng.

Hệ quả:
- Không có leakage, không sai: cả hai đều là cách chọn ngẫu nhiên hợp lệ (đúng k ảnh/lớp, từ trainval) — splits JSON đã được verify độc lập (mục 4.3).
- So sánh FT vs LP ở k ≤ 25 là so **mean** của các bộ ảnh ngẫu nhiên khác nhau, không phải so theo cặp seed. Ghi một dòng trên slide limitations.
- Repro check: `splits/` không sinh lại được từ code → coi là **dữ liệu cố định đã commit**, kiểm tra bằng `python3 scripts/verify_splits.py` (07/10: đạt cả 70 file; thêm `--features-dir features` để kiểm số ảnh mỗi lớp — chưa chạy vì máy Manager không có `features/`).

### 🟡 2.5 Reproducibility: Linear Probe không khớp tuyệt đối giữa các máy

Khi so frozen grid trước/sau lần chạy lại của Thu: 136/140 dòng DINOv2 giống hệt, **4 dòng LP lệch đúng 1 ảnh test** (Pets k=25 seed 0: 93.30 → 93.23; EuroSAT k=1 seed 0–2: −0.04 điểm). Feature DINOv2 không đổi → nguyên nhân là solver lbfgs/BLAS trên máy khác. Khi repro check: so với sai số **~0.1 điểm**, không đòi khớp bitwise. Lần chạy `knn_k=5` của Đức trên máy thứ ba cho thấy đúng hiện tượng này (5 dòng LP ResNet + 2 dòng LP DINOv2 lệch ≤ 1 ảnh test EuroSAT).

⚠️ `scripts/run_frozen_grid.py` **ghi nối tiếp** vào file `--output` và **bỏ qua** các `run_id` đã có; `run_id` băm từ cấu hình, không từ feature. Khi chạy lại trên feature mới phải **xóa file output trước**, nếu không log ra `new=0` và giữ nguyên số cũ.

### 🟢 2.6 Ghi chú nhỏ

- `features.zip` trên Drive (Thu, 06/10, 276 MB) thay cho 8 file `.pt` cũ (đã chuyển vào thùng rác). Dung lượng hợp lý (~308 MB float32 chưa nén); Manager chưa giải nén kiểm tra nội dung. Thu giữ bản V2 cũ ở local (`features_old_v2/`, `results_old_v2weights/` — đã thêm vào `.gitignore`).
- `results/finetune_benchmark.log`: peak VRAM 21.7 GB trên RTX 4060 (vượt VRAM thật → tràn shared memory Windows). Không trích lên slide.
- Pets test = 20% ngẫu nhiên (stratified, seed 42) từ HF `pcuenq/oxford-pets` (1,478 ảnh), **không** phải official test split (3,669 ảnh) → limitations.
- `pytest`, `tensorboard` chưa có trong `requirements.txt`.
- `results/KET_QUA_DOC_BAO.md`, `frozen_grid.csv`, `master_table.csv`, `plots/frozen_curves_*.png` là bản **legacy** (distance-weighted kNN + ResNet V2). Không dùng cho slide.

---

## 3. Số liệu chính thức sau freeze + câu chuyện cho slide

**Pets — Linear Probe** (5 seed):

| k | 1 | 2 | 5 | 10 | 25 | 50 | all |
|---|---|---|---|---|---|---|---|
| ResNet-50 (V1) | 71.2 | 78.7 | 85.6 | 89.2 | 92.4 | 93.4 | 94.3 |
| DINOv2 | **72.6** | **81.0** | **88.3** | **91.5** | **93.9** | **95.1** | **96.2** |

**EuroSAT — Linear Probe** (5 seed):

| k | 1 | 2 | 5 | 10 | 25 | 50 | all |
|---|---|---|---|---|---|---|---|
| ResNet-50 (V1) | **57.8** | **70.2** | **77.1** | **80.7** | 84.5 | 86.9 | 93.3 |
| DINOv2 | 54.7 | 64.5 | 74.4 | 79.8 | **86.1** | **88.7** | **95.2** |

**kNN (`knn_k=5`, chính thức)** (5 seed):

| Pets | 1 | 2 | 5 | 10 | 25 | 50 | all |
|---|---|---|---|---|---|---|---|
| ResNet-50 (V1) | **14.9** | **59.1** | **83.9** | 87.2 | 90.0 | 91.6 | 93.1 |
| DINOv2 | 13.8 | 58.3 | 83.7 | **88.1** | **92.0** | **93.2** | **94.7** |

| EuroSAT | 1 | 2 | 5 | 10 | 25 | 50 | all |
|---|---|---|---|---|---|---|---|
| ResNet-50 (V1) | **29.7** | **51.9** | **72.1** | **77.7** | **82.2** | **84.4** | 91.5 |
| DINOv2 | 24.7 | 44.2 | 64.8 | 72.2 | 80.4 | 84.0 | **92.7** |

Tác dụng của `knn_k=5` so với `knn_k=20` (Pets ResNet): k=1 3.0 → 14.9, k=2 9.8 → 59.1, k=5 60.7 → 83.9; từ k=10 hai bản gần nhau (±1 điểm). k=1 vẫn thấp vì 5 hàng xóm trên pool 1 ảnh/lớp vẫn nhiễu. Trên slide nói: **`knn_k=5` chọn theo kích thước pool nhỏ nhất (10–37 ảnh), không tune theo accuracy test**.

| RQ | Dự kiến trong plan | Thực tế (V1) |
|---|---|---|
| RQ1 | DINOv2 frozen > ResNet frozen khi ít nhãn | ✅ **DINOv2 thắng ở mọi k** trên Pets (+1.4 đến +2.7 điểm LP) |
| RQ2 | Fine-tune vượt frozen khi k ≥ 25 | ❌ **FT < LP ở mọi k**, cả 2 backbone; khoảng cách thu hẹp khi tăng nhãn (ResNet −6.1 → −1.9; DINOv2 −6.4 → −1.0) |
| RQ2 | DINOv2 frozen ≈/> ResNet fine-tuned ở k ≤ 10 | ✅ Đúng ở **mọi** k (88.3 vs 79.4 ở k=5; 96.2 vs 92.3 ở k=all) |
| RQ3 | SSL tổng quát hơn → thắng trên EuroSAT | ⚠️ **Đường cong cắt nhau** (LP): ResNet thắng ở k ≤ 10, DINOv2 thắng từ k ≥ 25 (+1.6 đến +1.9). kNN: ResNet dẫn tới k=50, DINOv2 chỉ dẫn ở k=all |
| — | kNN Pets | Hai backbone ngang nhau ở k ≤ 5, DINOv2 dẫn từ k=10 → phiên bản yếu hơn của RQ1 |

So với trước khi đổi V1: hiện tượng "ResNet thắng ở k ≤ 2 trên Pets" **biến mất** (đó là do weights V2); kết luận RQ2 giờ chắc chắn hơn vì frozen và fine-tune cùng điểm xuất phát.

Q&A cần chuẩn bị thêm: bug head Stage 2 (tự nêu trước, kèm kết quả bản đã sửa: thấp hơn ở mọi k → kết luận không đổi); vì sao dùng `knn_k=5` thay vì 20 (`knn_k=20` > ½ pool ở k=1/k=2 → gần random; 5 chọn theo pool nhỏ nhất, không theo test); vì sao kNN k=1 vẫn thấp; vì sao fine-tune thua LP; vì sao EuroSAT ResNet thắng ở ít nhãn; test set Pets là gì; vì sao đổi weights ResNet giữa chừng (để frozen và fine-tune cùng điểm xuất phát, đúng plan).

---

## 4. Đã tự kiểm tra (đóng)

### 4.1 Features (`features/*.pt`, 8 file, gitignored)

Shape/NaN đã kiểm 29/9: Pets 5,912 / 1,478; EuroSAT 21,600 / 2,700; ResNet 2048-d, DINOv2 384-d; không NaN. `src/features.py`: `Resize(256) → CenterCrop(224) → Normalize(ImageNet)`, không augment, ResNet **`IMAGENET1K_V1`** (từ `8762440`).

**Kiểm tra lần chạy lại của Thu (06/10)**, so `frozen_grid_v2.csv` cũ và mới theo từng dòng:
- 280/280 dòng, cùng key, cùng cột, cùng tham số (`knn_k=20`, uniform, C=1.0).
- ResNet: 128/140 dòng đổi số (12 dòng không đổi gồm 10 dòng EuroSAT kNN k=1/2 luôn = 11.74%) → đúng là feature ResNet đã đổi.
- DINOv2: 136/140 giống hệt, 4 dòng LP lệch 1 ảnh test (mục 2.5) → feature DINOv2 không đổi, đúng như commit mô tả.
- `master_table_v2.csv` 64 dòng, vẫn đủ 8 dòng fine-tune.

### 4.2 kNN `knn_k=5` của Đức (`d754f89`, kiểm tra 07/10)

So `frozen_grid_v2_knn5.csv` với bản chính thức V1 trên `main` và với bản `_knn5` lỗi (`af699c1`) theo từng dòng:
- Log `expected=280, new=280` → đã xóa file cũ, chạy lại đủ. Tham số: `knn_k=5`, uniform, C=1.0, l2. Không sửa code (`src/`, `scripts/`, `config/`, `tests/` không đổi).
- **Feature ResNet là V1**: LP ResNet Pets k=1 = 71.16, k=all = 94.25; 65/70 dòng LP ResNet khớp tuyệt đối bản V1, 5 dòng lệch ≤ 1 ảnh test EuroSAT (mục 2.5).
- kNN ResNet: 69/70 dòng khác bản `af699c1` → đã chạy lại thật trên feature mới. kNN DINOv2: 70/70 giống `af699c1` (feature DINOv2 không đổi) ✅.
- `master_table_v2_knn5.csv`: 64 dòng (56 frozen + 8 fine-tune); tự tính lại mean/std 56 dòng frozen từ raw → **khớp 100%**.
- So với `master_table_v2.csv`, LP chỉ lệch 2 ô EuroSAT (DINOv2 k=1 54.64 → 54.66; ResNet k=all 93.26 → 93.30).
- Branch `leanhduc` đã merge `main` (`1a49d6b`) → merge fast-forward vào `main` được.

### 4.3 Splits (`splits/*.json`, 70 file) — verify độc lập (01/10)

| Dataset | Kích thước k=1→all | Nested ∀ seed 0–4 | Trùng index | Max index | Khác nhau giữa seed |
|---|---|---|---|---|---|
| Pets | 37, 74, 185, 370, 925, 1850, 5912 | ✅ | Không | 5911 | ✅ |
| EuroSAT | 10, 20, 50, 100, 250, 500, 21600 | ✅ | Không | 21599 | ✅ |

Không có test leakage theo thiết kế (tensor test là file riêng).

### 4.4 Fine-tune grid

`finetune_grid.csv`: 24/24 dòng, 0 trùng key. Đối chiếu tay: ResNet k=5 = (79.70, 82.07, 76.52) → 79.43 ± 2.78 ✅.

### 4.5 Test

`pytest tests/` → 10/10 pass (29–30/9). Code eval không đổi kể từ đó (Thu chỉ đổi `features.py` + kết quả, Đức chỉ thêm kết quả). Chạy lại 07/10 trên `main` `76efe23` (sau mọi merge, venv riêng theo `requirements.txt`): **10/10 pass**.

---

## 5. Lịch sử xử lý

| Ngày | Sự kiện | Kết quả |
|---|---|---|
| 26/9 | Đức chạy frozen grid legacy (distance-weighted kNN, ResNet V2) | 280/280, kNN k nhỏ cao bất thường |
| 27/9 | Zam memory benchmark | `results/finetune_benchmark.log` |
| 29/9 | Pipeline v2 + 10 test (`87ed900`); sửa loader đọc feature tách file (`22fb6d0`) | Full grid v2 280/280 |
| 29/9 | `finetune_grid.csv` trùng key → thay bằng bản chạy lại sạch (`6f9d78a`) | Master table 64 dòng |
| 30/9 | Thu sync `src/features.py` (`93a94da`) | Provenance feature xác nhận |
| 01/10 | Rà soát `main` | Verify splits EuroSAT; phát hiện ResNet V1/V2, `run_finetune.sh` hỏng, plot thiếu fine-tune |
| 02–06/10 (chiều) | Không có commit mới | 2 hạng mục CN 4/10 trễ |
| 06/10 18:46 | Commit README + PROGRESS_CHECK (`8047e04`) | |
| 06/10 20:24–21:26 | Thu: ResNet → V1 (`8762440`), trích xuất lại + chạy lại frozen grid + plot (`3aec714`), upload `features.zip` lên Drive | ✅ Kiểm tra theo từng dòng, đạt |
| 06/10 22:48 | Đức: kNN `knn_k=5` (`af699c1`, upload qua web) | ❌ ResNet dùng nhầm feature V2 (LP ResNet khớp bản V2 70/70) |
| 06/10 23:35 – 07/10 00:02 | Đức: merge `main` vào `leanhduc` (`1a49d6b`), chạy lại `knn_k=5` trên feature V1 (`d754f89`) | ✅ Kiểm tra theo từng dòng, đạt. Freeze chốt hoàn toàn |
| 07/10 00:08 | Thiết: `plots/finetune_vs_frozen.png` + `scripts/plot_finetune_vs_frozen.py` (`d908fb5`) | ✅ Số liệu 16/16 điểm khớp bảng, đạt yêu cầu mục 2.2 |
| 07/10 | Manager merge `leanhduc` + `thiet` → `main`, commit docs (`5ac1e5a`) | ✅ |
| 07/10 01:10 | Thiết: `cb09506` — vẽ lại `plots/eval_v2/frozen_curves_pets.png` | ❌ Chưa merge: không commit `make_plots.py` (style fine-tune chỉ có ở máy Thiết), vẽ trên bảng `knn_k=20` vào `eval_v2/` thay vì `eval_v2_knn5/`, thiếu EuroSAT, tiêu đề/chú thích chưa đổi |
| 07/10 01:10 | Zam (`zam-audit`, `a7973cd`): audit fine-tune, `docs/ZAM_FINETUNE_EXPLAINED.md`, loss curves, sửa `run_finetune.sh`, `src/finetune_fixed.py` | ✅ Đạt. Phát hiện bug head Stage 2 (Manager xác nhận trên code) và split fine-tune ≠ `splits/*.json`. Chờ merge + quyết định của Managers |
| 07/10 01:14–01:21 | Thiết: `72b57dc`, `f5b57c1` — commit `make_plots.py` có style fine-tune (đổi sang màu Okabe-Ito), vẽ lại `plots/eval_v2/` | 🟡 Tiến bộ nhưng chưa merge: branch chưa lấy `main` mới (không có file `knn_k=5`), vẫn vẽ bảng `knn_k=20`, tiêu đề chưa đổi, màu lệch `finetune_vs_frozen.png` |
| 07/10 01:17 | Zam xóa `docs/ZAM_FINETUNE_EXPLAINED.md` khỏi `zam-audit` (`e145e93`) | Tài liệu giữ riêng; docs bỏ tham chiếu |
| 07/10 | Managers chọn phương án (b): Zam chạy `finetune_fixed.py` ngay | ⏳ Đang chạy |
| 07/10 | Manager merge `zam-audit` → `main` (`20bdae1`), commit docs (`76efe23`) | ✅ |
| 07/10 | Manager chạy `pytest tests/` trên `76efe23` | ✅ 10/10 pass |
| 07/10 | Mỹ gửi `related_works.md` (ViT, DINOv2) | 🟡 Dùng được cho slide #5–#7; cần sửa 1 lỗi (nguồn gốc ViT-S/14), bổ sung distillation, kiểm tra Pets trong LVD-142M, thêm ResNet / DINO v1 / LP-FT / dataset |
| 07/10 | Manager: thêm `scripts/verify_splits.py` (70/70 file đạt, bắt được lỗi khi thử splits hỏng); soạn bản nháp Results & Limitations cho report | ✅ |
| 07/10 03:10–03:12 | Thiết: merge `main` vào `thiet`, `63531ca` (vẽ lại `eval_v2_knn5/`), `44e5528` (`scripts/make_plots_knn5.py`) | ❌ Không merge: 2 plot `eval_v2_knn5/` vẽ từ bảng `knn_k=20` (Pets kNN k=2 ≈ 10%, EuroSAT k=1/2 = 11.74%) nhưng chú thích "KNN (k=5)" và đè plot chính thức; script mới mặc định `--input master_table_v2.csv`; script trùng `make_plots.py`; tiêu đề chưa đổi |
| 07/10 03:15 | Zam chưa push kết quả `finetune_fixed.py` → Manager chuẩn bị chạy dự phòng trên máy WSL (RTX 4060 8 GB) | Không cần: Zam dậy, đang chạy, dự kiến xong ~1 giờ nữa |
| 07/10 03:26 | Thiết: `42ddbfa` (vẽ lại `eval_v2_knn5/`), `845c1d1` (mặc định `--input master_table_v2_knn5.csv`) | ✅ Số liệu đúng `knn_k=5` (Pets kNN k=2 ≈ 59%, EuroSAT k=1 ≈ 30/25%); merge được. Còn góp ý thẩm mỹ: tiêu đề, nhãn "KNN (k=5)", màu lệch `finetune_vs_frozen.png`, script trùng |
| 07/10 | Mỹ gửi `related_works.md` bản 2 | 🟡 Dùng được cho slide/report sau khi sửa 4 chỗ nói quá/sai và tự xác nhận Table 15 (Pets trong nguồn LVD-142M) |
| 07/10 04:28 | Zam (`zam-audit`, `5356825`): chạy `finetune_fixed.py` 24 run | ✅ Đạt kiểm tra. Bản sửa bug thấp hơn bản gốc ở mọi k (−0.5 đến −8.0) → số chính thức không phải lower bound, RQ2 giữ nguyên. Report + README cập nhật |
| 07/10 | Manager tạo `report/` (LaTeX, chỉ lưu ở máy Manager, không push; mỗi section một file, hình đọc từ `plots/`, hỗ trợ tên tiếng Việt qua T5 + lmodern) | ✅ Build sạch, 8 trang, 0 lỗi / 0 overfull |
| 07/10 | Gộp trang bìa USTH vào `report/main.tex` (môn Deep Learning, đề tài Label-Efficient Image Classification…); logo chuyển vào `report/figures/usth_logo.png` | ✅ Build sạch, 9 trang |
| 07/10 | Report (local): thêm mục lục (link tới từng chương, có References), List of Figures, List of Tables; phụ lục soạn sẵn ở `sections/appendix.tex` nhưng tắt | ✅ Build sạch, 11 trang |
| 07/10 | Manager hoàn thiện `docs/related_works.md` thay Mỹ, đối chiếu paper gốc | ✅ DINOv2 (TMLR 01/2024): Table 15 có *Oxford-IIIT Pet / trainval* làm nguồn truy hồi (1M ảnh), EuroSAT không có (Table 15, 18); chỉ khử trùng lặp với test/val chính thức → test set tự chia của nhóm có thể có ảnh gần trùng. ViT-g 1.1B, model nhỏ distill từ ViT-g. LP-FT: FT tốt hơn LP ~2% in-distribution, kém ~7% OOD. Cập nhật report LaTeX, report Claude Docs, README |
| 07/10 | Report (local): rà soát cuối (8 chỗ sửa: phân tích theo cặp seed, các yếu tố không đồng nhất DINOv2/ResNet, cách chọn knn_k, abstract kèm điều kiện…), thêm hình pipeline (TikZ), bật phụ lục A–E với bảng đóng góp suy ra từ git | ✅ Build sạch, 15 trang. Nhóm cần xác nhận: bảng đóng góp, bìa, yêu cầu của môn, 3 đoạn `% DRAFT` |

---

## 6. Checklist

```
[x] Splits nested + không trùng + không leak — Pets & EuroSAT, seed 0–4
[x] src/, scripts/, tests/ đầy đủ — pytest 10/10 pass
[x] Full frozen grid v2 → 280/280 dòng
[x] finetune_grid.csv sạch 24/24, 0 trùng key
[x] Master table gộp frozen + fine-tune — 64 dòng
[x] QUYẾT ĐỊNH: ResNet-50 weights → V1 cho cả 2 phase (Thu, 06/10)
[x] QUYẾT ĐỊNH: knn_k=5 (Đức chạy lại trên V1, 07/10); knn_k=20 giữ làm đối chứng
[x] Kiểm tra lần chạy lại của Thu theo từng dòng
[x] Merge branch thu → main
[x] Kiểm tra lần chạy knn_k=5 của Đức theo từng dòng
[x] Merge branch leanhduc + thiet → main (`5ac1e5a`, 07/10)
[x] Vẽ lại plots/eval_v2_knn5/ đúng số knn_k=5, style fine-tune (Thiết, `845c1d1`, 07/10)
[x] plots/finetune_vs_frozen.png slide-ready (Thiết, 07/10)
[x] Mỹ: ghi chú research ViT + DINOv2 (related_works.md, nhận 07/10)
[x] Mỹ: related_works.md bản 2 — đã sửa ViT-S, distillation, thêm ResNet / DINO v1 / LP-FT / datasets (07/10)
[x] related_works.md bản cuối → `docs/related_works.md` (Manager hoàn thiện thay Mỹ, 07/10): sửa 4 chỗ, thêm lại position embedding / inductive bias, danh sách tài liệu tham khảo; **Table 15 đã xác nhận** từ paper DINOv2 (TMLR)
[ ] Draft slide lý thuyết + Q&A (Mỹ) — trễ từ 04/10
[ ] Slide kết quả #10–#13 + phân tích #14–#16 theo master_table_wide_v2_knn5.csv — 08/10
[x] Khung report LaTeX `report/` (07/10, **chỉ lưu ở máy Manager, không push**): build sạch, có sẵn Results / Limitations / Method / Setup / Related work / Abstract / Conclusion
[x] Report: nháp cho 3 `[TODO]` cuối (động lực + đóng góp ở intro, phân tích EuroSAT ở discussion, future work), đánh dấu `% DRAFT` (07/10)
[ ] Report: nhóm đọc lại các đoạn `% DRAFT` (`grep -rn DRAFT report/sections/`) — trước 8h 08/10
[x] Audit fine-tune + loss curves (Zam, 07/10)
[x] QUYẾT ĐỊNH: bug fine-tune → giữ số chính thức + nêu limitation, chạy finetune_fixed.py làm kết quả phụ (07/10)
[x] Zam: results/finetune_grid_fixed.csv 24/24 dòng (`5356825`, 07/10 04:28) — kiểm tra đạt; bảng phụ trong report + README
[x] Merge branch zam-audit (kết quả fixed) → main (`feaab47`, 07/10)
[x] Sửa scripts/run_finetune.sh (Zam, 07/10)
[x] Merge branch zam-audit → main (`20bdae1`, 07/10)
[x] Chạy lại pytest trên main `76efe23` → 10/10 pass (07/10; Python 3.12, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.3)
[ ] Kiểm tra nội dung features.zip (8 file, đúng shape) — làm trong repro check
[x] Stretch goals (ViT-S supervised, CLIP) — bỏ theo Experiment Freeze
[ ] Reproducibility check, sai số ~0.1 điểm (mục 2.5) — 09/10
[x] Verify split fine-tune vs splits/*.json → KHÁC (~3% trùng ở k ≤ 25), Zam kiểm tra (mục 2.4)
[x] scripts/verify_splits.py — 70/70 file đạt (07/10); splits coi là dữ liệu cố định
[ ] (Không bắt buộc) Đức: nguồn gốc code sinh splits/*.json
[ ] Repro check chạy thêm `verify_splits.py --features-dir features` (kiểm số ảnh mỗi lớp)
[x] Bản nháp phần Results & Limitations của report (Claude Docs, 07/10) — chờ thêm kết quả fine-tune đã sửa bug
[x] Bổ sung pytest, tensorboard vào requirements.txt (07/10)
```
