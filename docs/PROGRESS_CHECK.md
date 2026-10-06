# 🔍 Báo cáo tiến độ & Checklist Double-Check

> Người soạn: Manager (qua Claude Code) — **cập nhật 06/10/2026**, trạng thái branch `main` tại commit `f70bee7` (không có commit mới nào trên mọi branch kể từ 30/09).
> Mục đích: ghi lại những gì đã kiểm tra, kết quả, và những việc **còn mở** trước khi đưa số liệu lên slide.
> Đối chiếu với: [`PROJECT_PLAN.md`](PROJECT_PLAN.md), [`project_analysis.md`](project_analysis.md), [`../README.md`](../README.md).

---

## 🚨 HÔM NAY (T3 06/10) LÀ EXPERIMENT FREEZE

Thí nghiệm đã xong từ 29–30/9, nhưng **6 ngày qua repo không có thay đổi nào** và toàn bộ việc mở ngày 01/10 vẫn còn nguyên. Hai mốc CN 4/10 đã trễ (biểu đồ fine-tune vs frozen, draft slide lý thuyết). Hai quyết định thiết kế (mục 2.1, 2.2) **phải chốt trong hôm nay**: sau mốc freeze không chạy lại gì nữa — nếu chưa làm kịp thì mặc định **giữ nguyên số liệu hiện tại và ghi vào slide limitations**.

---

## 🎯 TÓM TẮT: AI CẦN LÀM GÌ (đọc mục này là đủ để đi nhắc việc)

| Ai | Việc cần làm | Hạn | Chi tiết |
|---|---|---|---|
| **Managers (+ Thu)** | Chốt ResNet-50 weights: frozen dùng `IMAGENET1K_V2`, fine-tune dùng `IMAGENET1K_V1`. Nếu chọn sửa: Thu trích xuất lại 4 file `resnet50_*.pt` bằng V1, Đức chạy lại `run_frozen.sh` (~2 phút). Nếu không kịp: giữ nguyên, ghi limitation | 🔴 **Hôm nay 06/10** | Mục 2.1 |
| **Managers + Đức** | Chốt `knn_k`: chạy lại frozen grid với `--knn-k` co giãn (~2 phút, cần `features/`) hoặc giữ 20, ghi limitation và chỉ trình bày kNN từ k ≥ 5 | 🔴 **Hôm nay 06/10** | Mục 2.2 |
| **Thiết** | `python3 scripts/make_plots.py` để plot Pets có đường fine-tune + làm `plots/finetune_vs_frozen.png` (không phải thí nghiệm mới, không vướng freeze) | 🔴 **Trễ** (hạn CN 4/10) | Mục 2.5 |
| **Mỹ** | Đưa draft slide lý thuyết #4–#8 + Q&A cheat sheet lên repo/drive chung | 🔴 **Trễ** (hạn CN 4/10) | Mục 1 |
| **Đức** | Slide kết quả #10–#13, lấy số từ `master_table_wide_v2.csv` (không dùng `KET_QUA_DOC_BAO.md` — bản legacy) | 🟡 T5 08/10 | Mục 3.4 |
| **Managers** | Viết lại slide #14–#16 (phân tích, limitations, kết luận) theo số thực tế — 2/3 insight dự kiến trong plan **không khớp** | 🟡 T5 08/10 | Mục 2.3 |
| **Zam** | Audit fine-tune (TensorBoard `runs/`, giải thích k=5 cao + FT < LP) để có câu trả lời Q&A | 🟡 T5 08/10 | Mục 2.3 |
| **Zam** | Sửa `scripts/run_finetune.sh` | 🟡 T6 09/10 | Mục 2.4 |
| **Thiết** | Reproducibility check trên máy khác | 🟡 T6 09/10 | Mục 1 |
| **Thu hoặc Zam** | Xác nhận split lúc fine-tune == `splits/pets_seed*_k*.json` | 🟢 Thấp | Mục 2.6 |
| **Manager (người giữ repo)** | Commit README + `docs/PROGRESS_CHECK.md` (đang nằm local, chưa push) để cả nhóm thấy danh sách này | 🟡 Hôm nay | — |

**Đã xong, không cần nhắc nữa**: splits 70 file (verify độc lập cả Pets **và EuroSAT**, 5 seed), feature 8 file `.pt` + `src/features.py`, code eval + 10 test, pipeline v2 đọc được feature Thu, frozen grid v2 (280/280), fine-tune grid sạch (24/24, 0 trùng key), master table gộp (64 dòng), memory benchmark. Tất cả branch thành viên (`leanhduc`, `my`, `thiet`, `thu`, `viet`, `zam`, `zang`) đã nằm trọn trong `main` — không còn code nào chưa merge.

---

## 1. Trạng thái tổng quan theo Milestone

| Milestone | Deadline | Trạng thái (06/10) | Ghi chú |
|---|---|---|---|
| Features cached + kNN verified ≥2 người | CN 27/9 | 🟢 Xong 29/9 (trễ ~2 ngày) | LP legacy = LP v2 tuyệt đối → cùng nguồn feature. Cross-check kNN độc lập của Thiết: chưa thấy báo cáo |
| Frozen grid hoàn thành | T3 29/9 | 🟢 Xong đúng hạn | `results/frozen_grid_v2.csv` (280/280) |
| Fine-tune loop + recipe khóa | T4 30/9 | 🟢 Xong | Xem lưu ý recipe ở mục 2.3 |
| Fine-tune grid 24 runs | CN 4/10 | 🟢 Xong sớm (29/9) | `results/finetune_grid.csv` (24/24) |
| Master table mọi kết quả | CN 4/10 | 🟢 Xong sớm (29/9) | `master_table_v2.csv` (64 dòng) + `master_table_wide_v2.csv` |
| Biểu đồ fine-tune vs frozen | CN 4/10 | 🔴 **Trễ 2 ngày** | Mục 2.5 |
| Draft slides lý thuyết (Mỹ) | CN 4/10 | 🔴 **Trễ 2 ngày** — chưa thấy trong repo | Managers hỏi trực tiếp |
| 🔒 Experiment Freeze | T3 6/10 | ⚠️ **Hôm nay** | Mục 2.1, 2.2 chưa chốt |
| Slide hoàn thành | T5 8/10 | ⏳ Còn 2 ngày | Chưa có thư mục `slides/` |
| Reproducibility check | T6 9/10 | 🟡 Có rủi ro | Phase 1: `run_frozen.sh` chạy được. Phase 2: `run_finetune.sh` hỏng (mục 2.4) |
| 2 buổi diễn tập | T7–CN 10–11/10 | ⏳ | |

---

## 2. ⚠️ Việc còn mở (ưu tiên cao → thấp)

### 🔴 2.1 ResNet-50 dùng 2 bộ weights khác nhau giữa Phase 1 và Phase 2 — CHỐT HÔM NAY

| Nơi | Weights |
|---|---|
| `PROJECT_PLAN.md` mục 1.3 (thiết kế) | `IMAGENET1K_V1` |
| `src/features.py` → frozen kNN / LP | `ResNet50_Weights.IMAGENET1K_V2` |
| `src/finetune.py` → fine-tune | `ResNet50_Weights.IMAGENET1K_V1` |

V2 được train với recipe mạnh hơn (ImageNet top-1 ~80.9% so với ~76.1% của V1). Hệ quả:
- So sánh **"ResNet frozen vs ResNet fine-tune"** (RQ2) đang so 2 mô hình khởi đầu khác nhau → không công bằng.
- Có thể góp phần giải thích ResNet frozen ở k=1 cao bất thường (LP 76.3%) so với dự đoán 20–30%.
- DINOv2 không bị ảnh hưởng (cùng `dinov2_vits14` ở cả 2 phase).

> Báo cáo 30/09 từng ghi V2 là "đúng như thiết kế" — **đó là nhận định sai**, plan ghi V1.

**Phương án** (theo thứ tự đề xuất):
1. Sửa `features.py` về `IMAGENET1K_V1`, trích xuất lại 4 file `resnet50_*.pt`, chạy lại `scripts/run_frozen.sh` + build master table có fine-tune. Chi phí: vài phút GPU + ~2 phút CPU. **Chỉ khả thi nếu Thu làm trong hôm nay.**
2. Giữ nguyên số liệu, ghi rõ trên slide limitations: "ResNet frozen dùng V2, fine-tune dùng V1; so sánh frozen vs fine-tune của ResNet chỉ mang tính tham khảo". Kết luận chính về DINOv2 (LP > FT, DINOv2 LP > ResNet FT) không bị ảnh hưởng.

### 🔴 2.2 kNN suy biến ở k=1/k=2 — CHỐT HÔM NAY

`knn_k=20` cố định, uniform vote. Khi `n_train` nhỏ, 20 hàng xóm chiếm phần lớn (hoặc toàn bộ) pool:

| Dataset | k=1 (`n_train`) | kNN v2 | k=2 (`n_train`) | kNN v2 |
|---|---|---|---|---|
| Pets | 37 | ResNet 2.72%, DINOv2 3.19% | 74 | 8.54%, 9.66% |
| EuroSAT | 10 | **11.74%** (cả 2 backbone, std 0) | 20 | **11.74%** (cả 2 backbone, std 0) |

EuroSAT ra đúng một con số hằng 11.74% vì `n_train ≤ 20` → mọi ảnh test bỏ phiếu trên **toàn bộ** pool, mỗi lớp đúng 1–2 phiếu → luôn ra cùng một nhãn. Đây là lỗi thiết kế protocol, không phải bug code. Code đã có sẵn `--knn-k`.

**Phương án**: (a) chạy lại frozen grid với `knn_k` nhỏ hơn (VD `--knn-k 5`), ~2 phút, chỉ cần máy có `features/`; hoặc (b) giữ nguyên, ghi limitation và chỉ trình bày kNN từ k ≥ 5. Linear Probe không bị ảnh hưởng — là phương pháp frozen chính trên slide.

### 🟡 2.3 Audit fine-tuning + câu chuyện slide

Số liệu (Pets, Top-1 %, 3 seed):

| k | FT ResNet-50 | LP ResNet-50 | FT DINOv2 | LP DINOv2 | Dự đoán FT ResNet / DINOv2 |
|---|---|---|---|---|---|
| 5 | 79.4 ± 2.8 | 87.7 | 81.9 ± 1.3 | 88.3 | 45–60 / 55–70 |
| 10 | 81.6 ± 0.7 | 89.8 | 88.4 ± 0.7 | 91.5 | 65–75 / 72–82 |
| 25 | 86.8 ± 1.0 | 91.5 | 91.3 ± 0.2 | 93.9 | 80–87 / 85–90 |
| all | 92.3 ± 0.6 | 93.4 | 95.2 ± 0.4 | 96.2 | 90–93 / 92–95 |

Hai điểm cần audit (Zam, để có câu trả lời Q&A):
1. **k=5 cao hơn dự đoán ~20 điểm** ở cả 2 backbone. Giả thuyết: Stage 1 (10 epoch train head trên backbone đóng băng) đã đưa model lên gần mức Linear Probe; Pets có nhiều giống chó/mèo trùng lớp ImageNet. Chưa phải bằng chứng bug.
2. **Fine-tune thua Linear Probe ở mọi k** (−6 đến −8 điểm ở k=5, còn ~−1 ở k=all). Có thể do: (a) ResNet weights V1 vs V2 (mục 2.1); (b) `RandomResizedCrop(224)` mặc định `scale=(0.08, 1)` khá mạnh cho ít dữ liệu; (c) recipe chưa từng được tune.

Gợi ý kiểm tra: mở TensorBoard `runs/` xem loss Stage 1/Stage 2 giảm đều, không NaN.

Lưu ý về recipe: `config/finetune_recipe.yaml` **trùng y hệt** bộ ứng viên trong plan mục 1.5; `finetune.py` không có tập validation → không có dấu vết bước "tune tại k=10/25". Trên slide nói **"recipe cố định, chọn trước, không tune theo k"**.

**Đối chiếu câu chuyện dự kiến (cho slide #14–#16):**

| Dự kiến trong plan | Thực tế |
|---|---|
| DINOv2 frozen ≈/> ResNet fine-tuned ở k ≤ 10 | ✅ Đúng ở **mọi** k (88.3 vs 79.4 ở k=5; 96.2 vs 92.3 ở k=all) |
| Fine-tune vượt frozen khi k ≥ 25 | ❌ FT < LP ở mọi k, khoảng cách thu hẹp dần |
| SSL tổng quát hơn (EuroSAT) | ❌ EuroSAT hai backbone chênh ≤ 4 điểm, ResNet nhỉnh hơn ở k ≤ 10 |
| Lợi thế DINOv2 lớn nhất ở k nhỏ | ❌ Ngược lại: Pets LP — ResNet thắng ở k ≤ 2, DINOv2 thắng từ k=5, cách biệt tăng theo k |

### 🟡 2.4 `scripts/run_finetune.sh` hỏng

Nội dung hiện tại: in "Phase 2 Execution Completed Successfully!" **trước** khi chạy, gọi `python finetune.py` (file nằm ở `src/`, lỗi khi chạy từ repo root), và chạy thêm 2 run lẻ trước `--grid` (ghi thêm 2 dòng thừa vào `finetune_grid.csv` → lại gây trùng key). Sửa tối thiểu:

```bash
rm -f results/finetune_grid.csv
python src/finetune.py --grid
```

### 🔴 2.5 Biểu đồ còn thiếu (trễ hạn CN 4/10)

- `plots/finetune_vs_frozen.png` — chưa có.
- `plots/eval_v2/frozen_curves_*.png` vẽ ở commit `22fb6d0`, **trước** khi merge fine-tune (`6f9d78a`). `make_plots.py` đã hỗ trợ mọi method trong master table → chỉ cần chạy `python3 scripts/make_plots.py` là Pets plot có thêm 2 đường fine-tune. Không cần `features/` hay GPU, ai trong nhóm cũng chạy được.
- EuroSAT: `plots/eval_v2/frozen_curves_eurosat.png` đủ dùng thay cho `eurosat_comparison.png` trong plan.
- Nếu Managers chốt chạy lại ở mục 2.1/2.2 → vẽ lại plot **sau** khi chạy lại.

### 🟢 2.6 Fine-tune không đọc `splits/*.json`

`finetune.py` gọi `make_kshot_splits(trainval_hf, k_values=[k], seed=seed)` lúc chạy thay vì đọc file JSON. Vì thuật toán xáo trộn theo từng lớp với `random.Random(seed)` và chỉ lấy `idxs[:k]`, kết quả **nên** trùng với file JSON. Chưa verify được (máy Manager không có `data/`). Ai có data chạy 1 dòng so sánh là đóng mục này.

### 🟢 2.7 Ghi chú nhỏ

- `results/finetune_benchmark.log`: peak VRAM 21.7 GB trên RTX 4060 và chỉ ~16 img/s — vượt VRAM thật của card, nhiều khả năng do Windows tràn sang shared memory. Không ảnh hưởng kết quả (grid chạy batch 32), không nên trích số này lên slide.
- Pets test = 20% ngẫu nhiên (stratified, seed 42) từ split `train` của HF `pcuenq/oxford-pets` (1,478 ảnh), **không** phải official test split (3,669 ảnh). Ghi trong limitations.
- `pytest`, `tensorboard` chưa có trong `requirements.txt`.
- `results/KET_QUA_DOC_BAO.md` mô tả bản **legacy** (distance-weighted kNN). Làm slide lấy số từ `master_table_wide_v2.csv`.

---

## 3. Đã tự kiểm tra (đóng)

### 3.1 Features của Thu (`features/*.pt`, 8 file, gitignored)

| File | Shape | NaN | Số lớp |
|---|---|---|---|
| `dinov2_pets_trainval.pt` | (5912, 384) | Không | 37 |
| `dinov2_pets_test.pt` | (1478, 384) | Không | 37 |
| `dinov2_eurosat_train.pt` | (21600, 384) | Không | 10 |
| `dinov2_eurosat_test.pt` | (2700, 384) | Không | 10 |
| `resnet50_pets_trainval.pt` | (5912, 2048) | Không | 37 |
| `resnet50_pets_test.pt` | (1478, 2048) | Không | 37 |
| `resnet50_eurosat_train.pt` | (21600, 2048) | Không | 10 |
| `resnet50_eurosat_test.pt` | (2700, 2048) | Không | 10 |

`src/features.py` (Thu sync từ Colab 30/9): `Resize(256) → CenterCrop(224) → Normalize(ImageNet)`, không augment, output đúng tên + format `{"features", "labels"}` mà loader v2 kỳ vọng. ✅ Provenance xác nhận. (Riêng weights ResNet — xem mục 2.1.)

### 3.2 Splits (`splits/*.json`, 70 file) — verify độc lập bằng script riêng (01/10)

| Dataset | Kích thước k=1→all | Nested ∀ seed 0–4 | Trùng index | Max index | Khác nhau giữa seed |
|---|---|---|---|---|---|
| Pets | 37, 74, 185, 370, 925, 1850, 5912 | ✅ | Không | 5911 | ✅ |
| EuroSAT | 10, 20, 50, 100, 250, 500, 21600 | ✅ | Không | 21599 | ✅ |

Index nằm gọn trong train(val) (tensor test là file riêng) → **không có test leakage theo thiết kế**.

### 3.3 Frozen grid

- Legacy `frozen_grid.csv` và v2 `frozen_grid_v2.csv`: mỗi bản đủ 280/280 tổ hợp, không trùng, không NaN; mean/std tự tính lại khớp 100% với master table.
- **Linear Probe legacy = v2 tuyệt đối** (VD LP · DINOv2 · Pets · k=all: 96.21 ở cả 2) → hai bộ feature cùng nguồn. kNN khác nhau chỉ do distance-weighted (legacy) vs uniform (v2).

### 3.4 Fine-tune grid + master table

- `finetune_grid.csv`: 24/24 dòng, 0 trùng key (bản chạy lại sạch của Zam, commit `6f9d78a`).
- `master_table_v2.csv`: 56 dòng frozen + 8 dòng fine-tune = 64. Đối chiếu tay: ResNet k=5 = (79.70, 82.07, 76.52) → 79.43 ± 2.78 ✅ khớp.
- Bảng số cho slide: `results/master_table_wide_v2.csv` (đã có cả Pets, EuroSAT, frozen và fine-tune).

### 3.5 Test

`pytest tests/` → 10/10 pass (chạy 29–30/9 sau mỗi lần pull). Code eval không đổi kể từ lần pass cuối. Môi trường Manager hiện chưa cài `pytest` nên các lần cập nhật 01/10 và 06/10 không chạy lại.

---

## 4. Lịch sử xử lý (tóm tắt, để tra cứu)

| Ngày | Sự kiện | Kết quả |
|---|---|---|
| 26/9 | Đức chạy frozen grid legacy (distance-weighted kNN), viết `KET_QUA_DOC_BAO.md` | 280/280, số kNN k nhỏ cao bất thường |
| 27/9 | Zam memory benchmark | `results/finetune_benchmark.log` |
| 29/9 | Đức push `leanhduc` (`87ed900`): pipeline v2 + 10 test | Pipeline v2 **lỗi** khi đọc feature tách file của Thu |
| 29/9 | Sửa loader: thêm `load_split_features()` + fallback trong `find_feature_cache()` (`22fb6d0`) | Full grid v2 280/280; LP khớp legacy tuyệt đối |
| 29/9 | Phát hiện `finetune_grid.csv` trùng key `resnet50/pets/k=5/seed=0` (78.28 vs 78.21) | Thay bằng bản chạy lại sạch của Zam (`6f9d78a`), build master table 64 dòng |
| 29/9 | README viết lại toàn bộ (`4cd5a3e`) | |
| 30/9 | Thu sync `src/features.py` từ Colab (`93a94da`, merge `f70bee7`) | Provenance feature xác nhận |
| 01/10 | Rà soát lại toàn bộ `main` | Verify splits EuroSAT ✅; phát hiện ResNet V1/V2 (2.1), `run_finetune.sh` hỏng (2.4), plot chưa có fine-tune (2.5), fine-tune không đọc JSON splits (2.6). README + docs cập nhật (chưa commit) |
| 02–06/10 | Không có commit mới trên bất kỳ branch nào | Mốc CN 4/10 trễ 2 hạng mục (plot, draft slide); 2 quyết định thiết kế dồn đến ngày freeze |
| 06/10 | Rà soát lại, viết lại báo cáo này | Thêm hạn chót cho từng việc, phương án mặc định nếu không kịp chạy lại trước freeze |

---

## 5. Checklist cho người double-check

```
[x] Splits nested + không trùng + không leak — Pets, seed 0–4
[x] Splits nested + không trùng + không leak — EuroSAT, seed 0–4 (01/10)
[x] src/, scripts/, tests/ đầy đủ — pytest 10/10 pass
[x] Loader v2 đọc được feature tách file của Thu
[x] Full frozen grid v2 → 280/280 dòng
[x] LP v2 = LP legacy → cùng nguồn feature
[x] finetune_grid.csv sạch 24/24, 0 trùng key
[x] Master table gộp frozen + fine-tune — 64 dòng
[x] src/features.py có nội dung, khớp format feature đã gửi
[x] Tất cả branch thành viên đã nằm trong main
[ ] QUYẾT ĐỊNH: ResNet-50 weights V1 vs V2 (mục 2.1) — HÔM NAY 06/10
[ ] QUYẾT ĐỊNH: co giãn knn_k hay ghi limitation (mục 2.2) — HÔM NAY 06/10
[ ] plots: chạy lại make_plots.py + finetune_vs_frozen.png (mục 2.5) — trễ từ 04/10
[ ] Draft slide lý thuyết + Q&A (Mỹ) — trễ từ 04/10
[ ] Slide kết quả #10–#13 + phân tích #14–#16 theo số thực tế — 08/10
[ ] Audit fine-tune: k=5 cao + FT < LP ở mọi k (mục 2.3) — 08/10
[ ] Sửa scripts/run_finetune.sh (mục 2.4) — 09/10
[ ] Reproducibility check trên máy khác — 09/10
[ ] Verify make_kshot_splits lúc fine-tune == splits/pets_*.json (mục 2.6)
[ ] Bổ sung pytest, tensorboard vào requirements.txt
[ ] Commit + push README và docs/PROGRESS_CHECK.md
```
