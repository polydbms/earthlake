Their pretrained data and benchmark info (if available) will be added soon!


# REMSA Phase 1: Model Recommendations with TerraTorch Doability

## Summary Table

| # | Model | TerraTorch Support | Doability | Effort | Recommendation |
|---|-------|-------------------|-----------|--------|----------------|
| 1 | Prithvi | ✅ Native | ⭐⭐⭐⭐⭐ | Trivial | **Must Include** |
| 2 | SatMAE | ✅ Native | ⭐⭐⭐⭐⭐ | Trivial | **Must Include** |
| 3 | Scale-MAE | ✅ Native | ⭐⭐⭐⭐⭐ | Trivial | **Must Include** |
| 4 | Satlas | ✅ Via TorchGeo | ⭐⭐⭐⭐⭐ | Trivial | **Must Include** |
| 5 | SSL4EO-S12 | ✅ Via TorchGeo | ⭐⭐⭐⭐⭐ | Trivial | **Must Include** |
| 6 | Clay | ✅ Native | ⭐⭐⭐⭐⭐ | Trivial | **Recommended** |
| 7 | DOFA | ✅ Via TorchGeo | ⭐⭐⭐⭐⭐ | Trivial | **Recommended** |
| 8 | CROMA | ❌ Not supported | ⭐⭐⭐⭐ | Low | **Recommended** |
| 9 | GFM | ⚠️ Partial (timm) | ⭐⭐⭐ | Medium | Optional |
| 10 | Cross-Scale MAE | ❌ Not supported | ⭐⭐⭐ | Medium | Optional |
| 11 | RemoteCLIP | ❌ Not supported | ⭐⭐⭐ | Medium | Optional |
| 12 | SkySense++ | ❌ Not supported | ⭐⭐ | High | Phase 2 |
| 13 | RVSA | ❌ Not supported | ⭐⭐ | High | Phase 2 |
| 14 | SpectralGPT | ❌ Not supported | ⭐⭐ | High | Phase 2 |
| 15 | RingMo | ❌ Not supported | ⭐ | Very High | Not Recommended |

---

## Detailed Descriptions

### Tier 1: Native TerraTorch Support (Trivial - Just Write Config)

---

#### 1. Prithvi (IBM-NASA)
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ First-class native support |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Write a YAML config file, run `terratorch fit/test` |
| **Effort estimate** | 30 minutes to get running |
| **Description** | The flagship model in TerraTorch. Multiple versions available (100M, 300M, 600M). Supports temporal sequences. Excellent documentation with example configs for flood mapping, burn scars, crop classification. IBM actively maintains both Prithvi and TerraTorch, so integration is seamless. |
| **Modality** | Multispectral (HLS: Landsat + Sentinel-2, 6 bands) |
| **Tasks** | Segmentation, Classification, Regression |

---

#### 2. SatMAE
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ First-class native support |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Write a YAML config file, specify `backbone: satmae_vit_base` |
| **Effort estimate** | 30 minutes to get running |
| **Description** | Temporal and spectral MAE from Stanford. TerraTorch has dedicated SatMAE model factory. Handles both temporal (fMoW) and multispectral (Sentinel) variants. Well-cited NeurIPS 2022 paper means community familiarity. |
| **Modality** | RGB temporal or Multispectral (Sentinel-2) |
| **Tasks** | Classification, Segmentation |

---

#### 3. Scale-MAE
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ First-class native support |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Write a YAML config file, specify `backbone: scalemae_vit_large` |
| **Effort estimate** | 30 minutes to get running |
| **Description** | Scale-aware MAE from Berkeley AI Research. Key differentiator: handles multi-resolution imagery gracefully. TerraTorch integration means you just swap the backbone name in your config. Good for tasks where ground sampling distance varies. |
| **Modality** | RGB |
| **Tasks** | Classification, Segmentation |

---

#### 4. Satlas (Allen AI)
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ Supported via TorchGeo integration |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Use TorchGeo's Satlas weights with TerraTorch |
| **Effort estimate** | 1 hour (slightly more config work than native models) |
| **Description** | Allen AI's foundation model trained on 302M labels across 137 categories. TorchGeo provides the pretrained weights, TerraTorch provides the training framework. Supports multiple sensors (Sentinel-1, Sentinel-2, Landsat, NAIP aerial). Swin-v2 and ResNet backbones available. |
| **Modality** | Multi-sensor (S1, S2, Landsat, Aerial) |
| **Tasks** | Multi-task (segmentation, detection, classification, regression) |

---

#### 5. SSL4EO-S12
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ Supported via TorchGeo integration |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Use TorchGeo's SSL4EO weights with TerraTorch |
| **Effort estimate** | 1 hour |
| **Description** | Self-supervised models (MoCo, DINO, MAE, data2vec) trained on large-scale Sentinel-1/2 dataset. Multiple SSL methods available - good for comparing pretraining strategies. TorchGeo provides ResNet weights pretrained with different SSL methods. Multi-modal (SAR + optical) and multi-temporal. |
| **Modality** | Sentinel-1 (SAR) + Sentinel-2 (optical), multi-temporal |
| **Tasks** | Classification, Segmentation, Change Detection |

---

#### 6. Clay
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ First-class native support |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Write a YAML config file |
| **Effort estimate** | 30-60 minutes |
| **Description** | Open-source foundation model from Clay Foundation. Recently added to TerraTorch with native support. Designed for global-scale Earth observation. Good community momentum and active development. Worth including for model diversity. |
| **Modality** | Multispectral |
| **Tasks** | Segmentation, Classification |

---

#### 7. DOFA
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ✅ Supported via TorchGeo integration |
| **Doability** | ⭐⭐⭐⭐⭐ Trivial |
| **What you need to do** | Use TorchGeo's DOFA weights with TerraTorch |
| **Effort estimate** | 1 hour |
| **Description** | Dynamic One-For-All model - a sensor-agnostic foundation model that adapts to different input configurations. Interesting architectural approach. TorchGeo integration means straightforward usage. Good for testing generalization across sensors. |
| **Modality** | Sensor-agnostic (adapts to input) |
| **Tasks** | Classification, Segmentation |

---

### Tier 2: Not in TerraTorch, But Easy to Add (Low-Medium Effort)

---

#### 8. CROMA
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐⭐⭐⭐ Low effort |
| **What you need to do** | Write standalone Python script (~100 lines) |
| **Effort estimate** | 2-4 hours |
| **Description** | **Best non-TerraTorch option.** Author explicitly designed it to be "noob-friendly" with a single `use_croma.py` file. Weights on HuggingFace. The only multi-modal (SAR + optical joint) model that's easy to use. Pre-processed benchmark datasets provided by authors. Contrastive + MAE approach. Worth the small extra effort for SAR capability. |
| **Modality** | Multi-modal: Sentinel-1 (SAR) + Sentinel-2 (optical) joint encoding |
| **Tasks** | Classification, Segmentation |
| **Code needed** | Load model, load data via TorchGeo, write eval loop |

---

#### 9. GFM (Geospatial Foundation Model)
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ⚠️ Partial - Swin backbone available via timm |
| **Doability** | ⭐⭐⭐ Medium effort |
| **What you need to do** | Load Swin backbone via timm, manually load GFM weights |
| **Effort estimate** | 4-6 hours |
| **Description** | Amazon's continual pretraining approach. Uses Swin Transformer backbone which is available in timm/TerraTorch. The challenge is loading their specific GeoPile-pretrained weights correctly. Config files for 7 downstream tasks are provided in their repo. SimMIM-based, so architecture is standard. |
| **Modality** | RGB/Optical |
| **Tasks** | Classification, Segmentation, Change Detection, Super-Resolution |
| **Code needed** | Weight loading adapter, verify input normalization |

---

#### 10. Cross-Scale MAE
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐⭐⭐ Medium effort |
| **What you need to do** | Write standalone script or adapt their codebase |
| **Effort estimate** | 4-6 hours |
| **Description** | NeurIPS 2023 paper on multi-scale exploitation. Very similar architecture to SatMAE (both are MAE-based ViTs), so code patterns are familiar. Pretrained weights available with checksums. Their repo has train/finetune/linprobe shell scripts. Could potentially adapt TerraTorch's SatMAE factory with modifications. |
| **Modality** | RGB |
| **Tasks** | Classification, Segmentation |
| **Code needed** | Standalone eval script or backbone adapter |

---

#### 11. RemoteCLIP
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐⭐⭐ Medium effort |
| **What you need to do** | Write standalone script using OpenCLIP |
| **Effort estimate** | 4-8 hours |
| **Description** | Vision-language model for RS. Uses OpenCLIP format which is a well-documented standard. Zero-shot evaluation is different from standard classification (need to handle text prompts). Authors provide Jupyter notebook demo. Good for VQA and retrieval tasks if those are in scope for REMSA. Different evaluation paradigm than other models. |
| **Modality** | RGB + Text |
| **Tasks** | Zero-shot classification, Image-text retrieval, Object counting |
| **Code needed** | OpenCLIP loading, text prompt handling, zero-shot eval logic |

---

### Tier 3: Significant Effort Required (Consider for Phase 2)

---

#### 12. SkySense++
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐⭐ High effort |
| **What you need to do** | Adapt their complex codebase, handle multi-modal inputs |
| **Effort estimate** | 1-2 days |
| **Description** | State-of-the-art multi-modal model (Nature Machine Intelligence 2025). Very powerful but complex architecture with separate encoders for HR RGB, Sentinel-2, and Sentinel-1. Code recently released but documentation is still maturing. Non-commercial license restriction on v1 weights. Would require significant adaptation work. |
| **Modality** | Multi-modal: HR RGB + Sentinel-2 + Sentinel-1 |
| **Tasks** | Classification, Segmentation, Detection, Change Detection |
| **Why defer** | Complex architecture, license concerns, immature documentation |

---

#### 13. RVSA (ViTAE Remote Sensing)
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐⭐ High effort |
| **What you need to do** | Set up MMSegmentation/OBBDetection frameworks |
| **Effort estimate** | 1-2 days |
| **Description** | Strong model with Rotated Varied-Size Window Attention. The problem: it's deeply integrated with MMSegmentation and OBBDetection frameworks, which are different ecosystems from TerraTorch/Lightning. Would require either framework setup or significant code extraction. Multiple pretrained variants available. |
| **Modality** | RGB |
| **Tasks** | Classification, Segmentation, Object Detection |
| **Why defer** | Framework dependency (MMSeg/OBBDet), architectural complexity |

---

#### 14. SpectralGPT
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐⭐ High effort |
| **What you need to do** | Adapt spectral-specific architecture |
| **Effort estimate** | 1-2 days |
| **Description** | Specialized for hyperspectral imagery with spectral sequence modeling. Interesting if hyperspectral is important for REMSA users. However, requires handling variable spectral bands and their specific preprocessing. Less documentation than other models. Niche use case. |
| **Modality** | Hyperspectral |
| **Tasks** | Classification, Segmentation |
| **Why defer** | Specialized modality, less documentation, niche application |

---

#### 15. RingMo
| Aspect | Details |
|--------|---------|
| **TerraTorch Support** | ❌ Not integrated |
| **Doability** | ⭐ Very high effort |
| **What you need to do** | Significant reverse engineering |
| **Effort estimate** | 2+ days |
| **Description** | Huawei's RS foundation model. Powerful results reported but limited code/weight availability outside Huawei's ecosystem. Multiple variants (RingMo, RingMo-Sense, RingMo-Aerial, RingMo-Agent) with unclear public access. Not recommended for Phase 1 due to accessibility issues. |
| **Modality** | Various |
| **Tasks** | Various |
| **Why skip** | Limited public availability, ecosystem lock-in |

---

## Recommended Phase 1 Selection

### Must Include (5 models) - All TerraTorch Native
| Model | Key Differentiator |
|-------|-------------------|
| **Prithvi** | Flagship, temporal, best documented |
| **SatMAE** | Temporal + spectral, widely cited |
| **Scale-MAE** | Multi-resolution handling |
| **Satlas** | Multi-sensor, 137 task categories |
| **SSL4EO-S12** | Multi-modal (SAR+optical), SSL comparison |

### Strongly Recommended (+2-3 models)
| Model | Key Differentiator |
|-------|-------------------|
| **Clay** | TerraTorch native, growing community |
| **DOFA** | Sensor-agnostic architecture |
| **CROMA** | Best SAR+optical joint model (worth extra effort) |

---
