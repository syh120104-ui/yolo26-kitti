# 읽을 논문과 코드 대응표

YOLO26을 이해하는 데 필요한 순서대로 정리했어요. "읽을 부분"만 먼저 보고, 해당 코드를 열어 같이 보면 빨라요.
v1부터의 전체 발전 과정은 별도 문서(YOLOv1 → YOLO26 발전 과정: 문제와 해결 방안)에 있어요.

| 순서 | 논문 | 읽을 부분 | 이 저장소에서 대응하는 코드 |
| --- | --- | --- | --- |
| 1 | [YOLO26 (arXiv 2606.03748)](https://arxiv.org/abs/2606.03748) | Sec. 3 전체, Table 2 (ablation), Eq. 2–3 (ProgLoss), Eq. 5 (STAL), Fig. S1 (구조), Table S2·S4·S7 (하이퍼파라미터) | 전체 |
| 2 | [YOLOv10 (arXiv 2405.14458)](https://arxiv.org/abs/2405.14458) | Sec. 3.1 Consistent Dual Assignments | `model.Detect26`, `loss.YOLO26Loss` |
| 3 | [TOOD (arXiv 2108.07755)](https://arxiv.org/abs/2108.07755) | Sec. 3.2 Task Alignment Learning | `assigner.TaskAlignedAssigner` |
| 4 | [Muon (Keller Jordan)](https://kellerjordan.github.io/posts/muon/) · [Kimi K2 / Moonlight (arXiv 2502.16982)](https://arxiv.org/abs/2502.16982) | Newton–Schulz 직교화, 대규모 학습에서의 Muon | `optim.MuSGD` |
| 5 | [GFL / DFL (arXiv 2006.04388)](https://arxiv.org/abs/2006.04388) | Distribution Focal Loss — YOLO26이 왜 이걸 뺐는지 이해하려면 | `model.DFL`, `loss.BranchLoss._dfl` |
| 6 | [CIoU / DIoU (arXiv 1911.08287)](https://arxiv.org/abs/1911.08287) | IoU 계열 loss | `utils.bbox_iou` |
| 6.5 | [BoTNet (arXiv 2101.11605)](https://arxiv.org/abs/2101.11605) | Sec. 3 (BoT block), Fig. 4 (relative position MHSA) | `nn.RelPosMHSA`, `nn.BoTBottleneck`, `nn.C3k2BoT` |
| 7 | [YOLOv12 (arXiv 2502.12524)](https://arxiv.org/abs/2502.12524) | Area Attention | (확장 아이디어) C2PSA 대신 A2 적용 |
| 8 | [YOLOv13 (arXiv 2506.17733)](https://arxiv.org/abs/2506.17733) | HyperACE, FullPAD | (확장 아이디어) |
| 9 | [YOLO26 리뷰 (Sapkota et al., arXiv 2509.25164)](https://arxiv.org/abs/2509.25164) | YOLOv8–v13와의 엣지 벤치마크 비교 | – |
| 10 | [KITTI (CVPR 2012)](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) | 2D object benchmark 정의, 난이도 기준 | `tools/prepare_kitti.py` |

## 참고 GitHub 저장소

- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26 공식 구현 (`ultralytics/cfg/models/26/`, `ultralytics/nn/modules/head.py`, `ultralytics/utils/loss.py`, `ultralytics/utils/tal.py`, `ultralytics/optim/muon.py`)
- [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) — dual assignment의 원조 구현
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — Muon optimizer 원본

## 논문을 읽으면서 해 볼 만한 확장 실험

1. **P2 head 추가** (stride 4): KITTI 원거리 보행자에게 효과가 있는지 보기 — `model.py`에 레벨 하나 추가
2. **STAL의 s_ref를 8 / 16 / 32로 바꾸기**: 논문 Table 6을 KITTI에서 재현 — `assign.stal_sref=8`
3. **ProgLoss 스케줄 비교**: (0.8→0.1) vs (1.0→0.1) vs 고정 — 논문 Table 5 재현
4. **Area Attention(v12)으로 C2PSA 교체**: attention 종류에 따른 차이
5. **이전 MiniYOLO + FPN + D-IoU 실험과 비교**: v1식 grid 회귀 → YOLO26까지 성능이 얼마나 벌어지는지


## 작은 물체 검출 (보행자 개선 실험)

| 논문 | 핵심 | 이 저장소에서 |
| --- | --- | --- |
| Wang et al., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*, arXiv 2110.13389 · [code](https://github.com/jwwangchn/NWD) | 박스를 2D 가우시안 N([cx,cy], diag(w²/4,h²/4))로 보고 W₂² = ‖[cx,cy,w/2,h/2]ₐ − [cx,cy,w/2,h/2]ᵦ‖², NWD = exp(−√W₂²/C). 작은 물체에서 IoU가 위치 어긋남에 너무 민감한 문제를 해결. assignment·NMS·loss에 적용 | `yolo26/utils.py bbox_nwd`, `assigner.py _metrics`, `loss.py BranchLoss` (`nwd_ratio`) |
| Xu et al., *NWD-RKA*, ISPRS J. 2022, arXiv 2206.13996 | NWD + 순위 기반 정답 배정 | TAL이 이미 top-k 순위 배정이라 같은 구조 |
| Li et al., *MST-YOLO*, Sensors 24(22):7347, 2024 | 자율주행 소형 물체: YOLOv8 neck에 P2 검출층 + SSFF·TFE + MLCA | `model.p2=true` (P2 검출층만 적용) |
| Wang & Zhao, *SOD-YOLO*, arXiv 2507.12727 | YOLOv8 + ASF 융합 + P2 + Soft-NMS (VisDrone) | 같은 P2 구조 |
| Chen et al., *SDD-YOLO*, arXiv 2603.25218 | P2 head + YOLO26의 DFL-free·NMS-free 구조 | YOLO26 + P2 결합 참고 |
