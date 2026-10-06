# Related Works: Vision Transformers, DINOv2, and Baselines

Research summary of the papers behind our comparison of the self-supervised DINOv2 ViT-S/14 and the supervised ResNet-50 under limited labels.

Original notes: Mỹ (v1, v2). Final revision 2026-10-07: review corrections applied and the DINOv2 data-source claims checked against the paper (TMLR version, Tables 15 and 18).

---

## 1. ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
**Dosovitskiy et al., ICLR 2021** ([arXiv:2010.11929](https://arxiv.org/abs/2010.11929))

### Overview
Before this paper, attention in computer vision was usually combined with CNNs. ViT shows that a pure Transformer applied directly to a sequence of image patches can match CNNs on image classification.

### Key ideas
* **Images as words (patches):** the image is split into fixed-size patches; each patch is linearly embedded into a vector, like a word token in NLP.
* **Position embeddings:** Transformers have no notion of order, so learnable position embeddings are added to the patch embeddings to keep spatial information.
* **Standard Transformer encoder:** the patch sequence goes through multi-head self-attention layers. A `[class]` token is prepended, and its output state is used for classification.
* **Inductive bias vs. data scale:** CNNs build in locality and translation equivariance; ViT has much less image-specific inductive bias. On mid-sized datasets (ImageNet without strong regularisation) ViT trails comparable ResNets by a few points; pretrained on 14M–300M images, it overtakes them. This is why ViTs depend on large-scale pretraining.
* **Model sizes:** the ViT paper defines ViT-B/L/H with patch sizes 16 and 32. The small **ViT-S** variant comes from DeiT (Touvron et al., 2021), and the **patch size 14** is DINOv2's choice.

---

## 2. DINO: Emerging Properties in Self-Supervised Vision Transformers
**Caron et al., ICCV 2021** ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294))

* Self-distillation with no labels: a student network matches the output of a momentum teacher on different augmented crops of the same image.
* Frozen DINO features are strong k-NN classifiers (78.3% top-1 on ImageNet). DINO's k-NN protocol uses a **weighted** vote over 20 nearest neighbours (cosine similarity, temperature 0.07). Our kNN is **adapted from** this protocol: uniform vote over 5 neighbours, because our smallest training pools have only 10–37 images.

---

## 3. DINOv2: Learning Robust Visual Features without Supervision
**Oquab et al., TMLR 2024** ([arXiv:2304.07193](https://arxiv.org/abs/2304.07193))

### Overview
DINOv2 scales self-supervised learning to produce general-purpose visual features that work out of the box, without fine-tuning, across many domains and tasks.

### Key ideas
* **Training objective:** combines an image-level objective (DINO) with a patch-level masked-image-modelling objective (iBOT).
* **Distillation:** the ViT-g/14 model has **1.1B parameters**; smaller models, including the **ViT-S/14** we use (~22M parameters), are **distilled from ViT-g** rather than trained from scratch. Relevant when comparing it with the ~25M-parameter ResNet-50: model size is matched, but DINOv2-S inherits knowledge from a much larger teacher.
* **Data curation (LVD-142M):** 142M images retrieved from a pool of 1.2B uncurated web images by choosing images close to those of curated datasets (self-supervised embeddings + clustering).
* **Findings:** frozen DINOv2 features are competitive with or better than weakly supervised models (e.g. OpenCLIP) on many classification and dense tasks.

### Pretraining overlap with our datasets (checked in the paper)
* **Oxford-IIIT Pets is a retrieval source** of LVD-142M: Table 15 lists *Oxford-IIIT Pet / trainval*, 3,680 images, cluster-based retrieval, **1,000,000 images** included in LVD-142M. Table 18 marks Oxford Pets as used for "retrieving pretraining data" and for evaluation.
* **EuroSAT is not used**: it appears in neither Table 15 nor Table 18.
* The paper removes near-duplicates of the **test/validation** images of its benchmarks from the pretraining data. Our Pets test set is a random 20% of all 7,390 images, so it also contains images from the official *trainval* split, which were retrieval queries and were not de-duplicated. Near-duplicates of some of our test images in LVD-142M cannot be ruled out.
* This is consistent with our results: DINOv2 beats ResNet-50 at every label budget on Pets, while on EuroSAT ResNet-50 leads with 10 or fewer images per class. It is a plausible contributing factor, not a proven cause; ResNet-50's ImageNet-1k pretraining also contains many dog and cat breeds.

---

## 4. Other baselines and methods

* **ResNet** — He et al., CVPR 2016 ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)). Residual connections address the **degradation problem**: deeper plain networks reach *higher training error*. (Vanishing gradients are mostly handled by normalised initialisation and BatchNorm, as the paper notes.) ImageNet-pretrained ResNet-50 is our supervised CNN baseline.
* **DeiT** — Touvron et al., ICML 2021 ([arXiv:2012.12877](https://arxiv.org/abs/2012.12877)). Data-efficient training of ViTs on ImageNet-1k; introduces the ViT-S (small) configuration.
* **LP-FT** — Kumar et al., ICLR 2022, *"Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"* ([arXiv:2202.10054](https://arxiv.org/abs/2202.10054)). Across their benchmarks, full fine-tuning is on average about 2% **better** than linear probing in-distribution but about 7% **worse** out-of-distribution, because it distorts good pretrained features. Their fix, LP-FT (train the linear head first, then fine-tune everything), beats both. Our two-stage recipe follows LP-FT in spirit.
* **Oxford-IIIT Pets** — Parkhi et al., CVPR 2012 ("Cats and Dogs"). 37 cat and dog breeds.
* **EuroSAT** — Helber et al., IEEE JSTARS 2019. Sentinel-2 satellite images, 10 land-use / land-cover classes.

---

## Relevance to the project

Our project compares **DINOv2 ViT-S/14** with **ResNet-50** under low-label (k-shot) budgets on Pets and EuroSAT.

* **RQ1 (frozen features):** DINOv2 beats ResNet-50 on Pets at every budget with a Linear Probe, clearly from k = 5 (at k = 1–2 the lead is within one standard deviation). This **partially** confirms DINOv2's claim of strong frozen features; part of the Pets advantage may come from LVD-142M having been retrieved with Pets as a source.
* **RQ2 (fine-tuning):** our fine-tuning stays below the Linear Probe at every budget, even in-distribution. LP-FT reports the opposite in-distribution on average (fine-tuning about 2% better), so our result is **not** explained by that paper. It is consistent with their feature-distortion argument in a very low-label regime with a fixed, untuned recipe. Our implementation also had a bug that kept the head frozen in Stage 2; a bug-fixed rerun scored lower, not higher, so the ordering holds either way.
* **RQ3 (cross-domain):** on EuroSAT, which DINOv2 never saw during pretraining data curation, the curves cross: ResNet-50 leads with 10 or fewer images per class, DINOv2 from 25 on.

---

## References

1. A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
2. H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, H. Jégou. *Training Data-Efficient Image Transformers & Distillation through Attention.* ICML 2021.
3. M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, A. Joulin. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.
4. M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. *DINOv2: Learning Robust Visual Features without Supervision.* Transactions on Machine Learning Research (TMLR), 2024.
5. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
6. A. Kumar, A. Raghunathan, R. Jones, T. Ma, P. Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022.
7. O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.* CVPR 2012.
8. P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.
