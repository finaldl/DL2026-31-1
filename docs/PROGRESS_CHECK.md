# 🔍 Báo cáo tiến độ & Checklist Double-Check

> Người soạn: Manager (qua Claude Code) — **cập nhật tối 06/10/2026**, sau khi Thu push branch `thu` (`8762440`, `3aec714`).
> Mục đích: ghi lại những gì đã kiểm tra, kết quả, và những việc **còn mở** trước khi đưa số liệu lên slide.
> Đối chiếu với: [`PROJECT_PLAN.md`](PROJECT_PLAN.md), [`project_analysis.md`](project_analysis.md), [`../README.md`](../README.md).

---

## 🔒 EXPERIMENT FREEZE ĐÃ CHỐT (T3 06/10)

- ✅ **ResNet-50 weights**: Thu đã chuyển `src/features.py` về `IMAGENET1K_V1`, trích xuất lại 4 file `resnet50_*.pt`, chạy lại frozen grid, build lại master table và plot — xong lúc 21:26, trước khi hết ngày freeze. Frozen và fine-tune giờ **cùng weights**.
- ✅ **`knn_k`**: không chạy lại → giữ `knn_k=20`, **ghi là limitation**, slide chỉ trình bày kNN từ k ≥ 5; Linear Probe là phương pháp frozen chính.
- 🔒 Từ giờ **không chạy thí nghiệm mới**. Số liệu chính thức = `results/master_table_wide_v2.csv` trên branch `thu` (sau khi merge vào `main`).

---

## 🎯 TÓM TẮT: AI CẦN LÀM GÌ

| Ai | Việc cần làm | Hạn | Chi tiết |
|---|---|---|---|
| **Manager (người giữ repo)** | Merge `thu` → `main` (fast-forward) + commit README/PROGRESS_CHECK đã cập nhật số V1 | 🔴 Ngay | — |
| **Thiết** | Làm `plots/finetune_vs_frozen.png` riêng, slide-ready (plot Pets hiện tại đã có đường fine-tune nhưng 6 đường chồng nhau, màu fine-tune DINOv2 trùng màu ResNet, tiêu đề vẫn ghi "Frozen-feature evaluation") | 🔴 **Trễ** (hạn CN 4/10) | Mục 2.2 |
| **Mỹ** | Đưa draft slide lý thuyết #4–#8 + Q&A cheat sheet lên repo/drive chung | 🔴 **Trễ** (hạn CN 4/10) | Mục 1 |
| **Đức** | Slide kết quả #10–#13, lấy số từ `master_table_wide_v2.csv` **bản mới (V1)**. Không dùng `KET_QUA_DOC_BAO.md` (legacy, V2) | 🟡 T5 08/10 | Mục 3 |
| **Managers** | Viết slide #14–#16 (phân tích, limitations, kết luận) theo số V1 — xem bảng câu chuyện ở mục 3 | 🟡 T5 08/10 | Mục 3 |
| **Zam** | Audit fine-tune (TensorBoard `runs/`), chuẩn bị câu trả lời "vì sao fine-tune thua Linear Probe" | 🟡 T5 08/10 | Mục 2.1 |
| **Zam** | Sửa `scripts/run_finetune.sh` | 🟡 T6 09/10 | Mục 2.3 |
| **Thiết (+ Thu)** | Reproducibility check trên máy khác — dùng `features.zip` mới trên Drive, so với sai số ~0.1 điểm (mục 2.5) | 🟡 T6 09/10 | Mục 2.5 |
| **Thu hoặc Zam** | Xác nhận split lúc fine-tune == `splits/pets_seed*_k*.json` | 🟢 Thấp | Mục 2.4 |

**Đã xong, không cần nhắc nữa**: splits 70 file (verify độc lập Pets + EuroSAT), feature 8 file `.pt` (ResNet bản V1), code eval + 10 test, frozen grid v2 (280/280, chạy lại 06/10), fine-tune grid sạch (24/24), master table gộp (64 dòng), memory benchmark, **thống nhất ResNet-50 weights V1**.

---

## 1. Trạng thái tổng quan theo Milestone

| Milestone | Deadline | Trạng thái (06/10 tối) | Ghi chú |
|---|---|---|---|
| Features cached + kNN verified ≥2 người | CN 27/9 | 🟢 Xong 29/9 | ResNet trích xuất lại bằng V1 ngày 06/10 |
| Frozen grid hoàn thành | T3 29/9 | 🟢 Xong | Chạy lại 06/10 với feature ResNet V1 |
| Fine-tune loop + recipe khóa | T4 30/9 | 🟢 Xong | Xem lưu ý recipe ở mục 2.1 |
| Fine-tune grid 24 runs | CN 4/10 | 🟢 Xong sớm (29/9) | |
| Master table mọi kết quả | CN 4/10 | 🟢 Xong | Build lại 06/10, 64 dòng |
| Biểu đồ fine-tune vs frozen | CN 4/10 | 🟡 Một phần | Pets plot đã có đường fine-tune; thiếu bản riêng slide-ready |
| Draft slides lý thuyết (Mỹ) | CN 4/10 | 🔴 **Trễ** — chưa thấy | Managers hỏi trực tiếp |
| 🔒 Experiment Freeze | T3 6/10 | 🟢 **Đã chốt** | Xem đầu file |
| Slide hoàn thành | T5 8/10 | ⏳ Còn 2 ngày | Chưa có thư mục `slides/` |
| Reproducibility check | T6 9/10 | 🟡 Có rủi ro | `run_finetune.sh` vẫn hỏng |
| 2 buổi diễn tập | T7–CN 10–11/10 | ⏳ | |

---

## 2. ⚠️ Việc còn mở

### 🟡 2.1 Audit fine-tuning (để có câu trả lời Q&A)

Pets, Top-1 %, fine-tune 3 seed, LP 5 seed, **cả hai cùng ResNet V1**:

| k | FT ResNet-50 | LP ResNet-50 | Δ | FT DINOv2 | LP DINOv2 | Δ |
|---|---|---|---|---|---|---|
| 5 | 79.4 ± 2.8 | 85.6 | −6.1 | 81.9 ± 1.3 | 88.3 | −6.4 |
| 10 | 81.6 ± 0.7 | 89.2 | −7.5 | 88.4 ± 0.7 | 91.5 | −3.1 |
| 25 | 86.8 ± 1.0 | 92.4 | −5.6 | 91.3 ± 0.2 | 93.9 | −2.6 |
| all | 92.3 ± 0.6 | 94.3 | −1.9 | 95.2 ± 0.4 | 96.2 | −1.0 |

Fine-tune thua LP ở mọi k, kể cả khi đã cùng weights → đây là kết quả thật của setup này, không còn do lệch V1/V2. Giả thuyết cần chuẩn bị để trả lời:
- Stage 1 (10 epoch train head trên backbone đóng băng) ≈ một linear probe yếu hơn (ít epoch, có augment) → Stage 2 với ít ảnh dễ overfit / phá feature tốt sẵn có.
- `RandomResizedCrop(224)` mặc định `scale=(0.08, 1)` khá mạnh với ít dữ liệu.
- Recipe chưa từng được tune: `config/finetune_recipe.yaml` **trùng y hệt** bộ ứng viên trong plan; `finetune.py` không có tập validation. Trên slide nói **"recipe cố định, chọn trước, không tune theo k"**.

Gợi ý cho Zam: mở TensorBoard `runs/` xem loss Stage 1/Stage 2 giảm đều, không NaN — chụp 1 hình cho backup slide B2.

### 🟡 2.2 Biểu đồ

- `plots/eval_v2/frozen_curves_pets.png` (bản Thu vẽ lại 06/10) **đã có** 2 đường fine-tune, nhưng chưa dùng được trên slide: 6 đường trong một hình, fine-tune DINOv2 dùng màu xanh giống ResNet (không có style riêng trong `make_plots.py`), tiêu đề "Frozen-feature evaluation".
- Cần `plots/finetune_vs_frozen.png` riêng: chỉ LP vs fine-tune, k ∈ {5, 10, 25, all}, mỗi backbone 1 màu cố định, nét liền = LP, nét đứt = fine-tune. Không phải thí nghiệm mới → không vướng freeze.
- EuroSAT: `plots/eval_v2/frozen_curves_eurosat.png` dùng được thay cho `eurosat_comparison.png`.

### 🟡 2.3 `scripts/run_finetune.sh` hỏng

In "Completed" **trước** khi chạy, gọi `python finetune.py` sai thư mục (file ở `src/`), và chạy thêm 2 run lẻ trước `--grid` (gây trùng key trong CSV). Sửa tối thiểu:

```bash
rm -f results/finetune_grid.csv
python src/finetune.py --grid
```

### 🟢 2.4 Fine-tune không đọc `splits/*.json`

`finetune.py` gọi `make_kshot_splits(...)` lúc chạy thay vì đọc file JSON. Cùng thuật toán + cùng seed nên **nên** trùng, nhưng chưa verify (máy Manager không có `data/`).

### 🟡 2.5 Reproducibility: Linear Probe không khớp tuyệt đối giữa các máy

Khi so frozen grid trước/sau lần chạy lại của Thu: 136/140 dòng DINOv2 giống hệt, **4 dòng LP lệch đúng 1 ảnh test** (Pets k=25 seed 0: 93.30 → 93.23; EuroSAT k=1 seed 0–2: −0.04 điểm). Feature DINOv2 không đổi → nguyên nhân là solver lbfgs/BLAS trên máy khác. Khi repro check: so với sai số **~0.1 điểm**, không đòi khớp bitwise.

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
| DINOv2 | 54.6 | 64.5 | 74.4 | 79.8 | **86.1** | **88.7** | **95.2** |

**kNN (v2, `knn_k=20`)** — chỉ trình bày từ k ≥ 5. Pets ResNet 60.7 / 85.9 / 90.7 / 91.5 / 93.6; DINOv2 58.1 / 82.7 / 90.1 / 92.0 / 94.4 (k = 5 / 10 / 25 / 50 / all).

| RQ | Dự kiến trong plan | Thực tế (V1) |
|---|---|---|
| RQ1 | DINOv2 frozen > ResNet frozen khi ít nhãn | ✅ **DINOv2 thắng ở mọi k** trên Pets (+1.4 đến +2.7 điểm LP) |
| RQ2 | Fine-tune vượt frozen khi k ≥ 25 | ❌ **FT < LP ở mọi k**, cả 2 backbone; khoảng cách thu hẹp khi tăng nhãn (ResNet −6.1 → −1.9; DINOv2 −6.4 → −1.0) |
| RQ2 | DINOv2 frozen ≈/> ResNet fine-tuned ở k ≤ 10 | ✅ Đúng ở **mọi** k (88.3 vs 79.4 ở k=5; 96.2 vs 92.3 ở k=all) |
| RQ3 | SSL tổng quát hơn → thắng trên EuroSAT | ⚠️ **Đường cong cắt nhau**: ResNet thắng ở k ≤ 10, DINOv2 thắng từ k ≥ 25 (+1.6 đến +1.9) |

So với trước khi đổi V1: hiện tượng "ResNet thắng ở k ≤ 2 trên Pets" **biến mất** (đó là do weights V2); kết luận RQ2 giờ chắc chắn hơn vì frozen và fine-tune cùng điểm xuất phát.

Q&A cần chuẩn bị thêm: vì sao kNN k=1/k=2 gần random (`knn_k=20` > ½ pool); vì sao fine-tune thua LP; vì sao EuroSAT ResNet thắng ở ít nhãn; test set Pets là gì; vì sao đổi weights ResNet giữa chừng (để frozen và fine-tune cùng điểm xuất phát, đúng plan).

---

## 4. Đã tự kiểm tra (đóng)

### 4.1 Features (`features/*.pt`, 8 file, gitignored)

Shape/NaN đã kiểm 29/9: Pets 5,912 / 1,478; EuroSAT 21,600 / 2,700; ResNet 2048-d, DINOv2 384-d; không NaN. `src/features.py`: `Resize(256) → CenterCrop(224) → Normalize(ImageNet)`, không augment, ResNet **`IMAGENET1K_V1`** (từ `8762440`).

**Kiểm tra lần chạy lại của Thu (06/10)**, so `frozen_grid_v2.csv` cũ và mới theo từng dòng:
- 280/280 dòng, cùng key, cùng cột, cùng tham số (`knn_k=20`, uniform, C=1.0).
- ResNet: 128/140 dòng đổi số (12 dòng không đổi gồm 10 dòng EuroSAT kNN k=1/2 luôn = 11.74%) → đúng là feature ResNet đã đổi.
- DINOv2: 136/140 giống hệt, 4 dòng LP lệch 1 ảnh test (mục 2.5) → feature DINOv2 không đổi, đúng như commit mô tả.
- `master_table_v2.csv` 64 dòng, vẫn đủ 8 dòng fine-tune.

### 4.2 Splits (`splits/*.json`, 70 file) — verify độc lập (01/10)

| Dataset | Kích thước k=1→all | Nested ∀ seed 0–4 | Trùng index | Max index | Khác nhau giữa seed |
|---|---|---|---|---|---|
| Pets | 37, 74, 185, 370, 925, 1850, 5912 | ✅ | Không | 5911 | ✅ |
| EuroSAT | 10, 20, 50, 100, 250, 500, 21600 | ✅ | Không | 21599 | ✅ |

Không có test leakage theo thiết kế (tensor test là file riêng).

### 4.3 Fine-tune grid

`finetune_grid.csv`: 24/24 dòng, 0 trùng key. Đối chiếu tay: ResNet k=5 = (79.70, 82.07, 76.52) → 79.43 ± 2.78 ✅.

### 4.4 Test

`pytest tests/` → 10/10 pass (29–30/9). Code eval không đổi kể từ đó (Thu chỉ đổi `features.py` + kết quả). Môi trường Manager chưa cài `pytest` nên không chạy lại 06/10.

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
| 06/10 20:24–21:26 | Thu: ResNet → V1 (`8762440`), trích xuất lại + chạy lại frozen grid + plot (`3aec714`), upload `features.zip` lên Drive | ✅ Kiểm tra theo từng dòng, đạt. Freeze chốt |

---

## 6. Checklist

```
[x] Splits nested + không trùng + không leak — Pets & EuroSAT, seed 0–4
[x] src/, scripts/, tests/ đầy đủ — pytest 10/10 pass
[x] Full frozen grid v2 → 280/280 dòng
[x] finetune_grid.csv sạch 24/24, 0 trùng key
[x] Master table gộp frozen + fine-tune — 64 dòng
[x] QUYẾT ĐỊNH: ResNet-50 weights → V1 cho cả 2 phase (Thu, 06/10)
[x] QUYẾT ĐỊNH: knn_k giữ 20, ghi limitation, kNN chỉ từ k ≥ 5 (freeze 06/10)
[x] Kiểm tra lần chạy lại của Thu theo từng dòng
[ ] Merge branch thu → main
[ ] plots/finetune_vs_frozen.png slide-ready (mục 2.2) — trễ từ 04/10
[ ] Draft slide lý thuyết + Q&A (Mỹ) — trễ từ 04/10
[ ] Slide kết quả #10–#13 + phân tích #14–#16 theo số V1 — 08/10
[ ] Audit fine-tune / câu trả lời Q&A (mục 2.1) — 08/10
[ ] Sửa scripts/run_finetune.sh (mục 2.3) — 09/10
[ ] Reproducibility check, sai số ~0.1 điểm (mục 2.5) — 09/10
[ ] Verify make_kshot_splits lúc fine-tune == splits/pets_*.json (mục 2.4)
[ ] Bổ sung pytest, tensorboard vào requirements.txt
```
