# 구현 상세 설명 (YOLO26-KITTI)

[Ultralytics YOLO26 논문](https://arxiv.org/abs/2606.03748)(Jocher et al., 2026)의 구조와 학습 기법을 PyTorch로 직접 구현한 저장소예요.
`ultralytics` 패키지를 쓰지 않고, 논문의 각 요소가 어느 파일·함수에 있는지 바로 찾을 수 있게 작게 나눠 두었어요.

| 논문 요소 | 이 저장소 위치 | 한 줄 설명 |
| --- | --- | --- |
| YOLO11식 backbone/neck (Conv–C3k2–SPPF–C2PSA, PAN) | `yolo26/nn.py`, `yolo26/model.py` | 논문 Fig. S1 레이어 구성. 파라미터 수가 논문 Table 7과 일치해요(n 2.4M / 5.4 GFLOPs) |
| DFL 제거 (reg_max=1) + L1 loss | `Detect26`, `BranchLoss` | ltrb 거리 4개를 바로 회귀해요. `model.reg_max=16`으로 바꾸면 YOLO11식 DFL로 돌아가요 |
| Dual head, NMS-free 추론 | `Detect26.forward / predict` | one-to-many(top-k 10, NMS)와 one-to-one(top-k 7→1, NMS 없음) |
| ProgLoss | `loss.progloss_alpha` | o2m 가중치를 epoch마다 0.8 → 0.1로 선형 이동 (논문 Eq. 2–3) |
| STAL | `assigner.TaskAlignedAssigner._stal_boxes` | 한 변 < 8px인 GT는 후보 선택 때만 16px로 키워요 (논문 Eq. 5) |
| MuSGD | `yolo26/optim.py` | Muon(Newton–Schulz 직교화) + SGD 가중합, 1D 파라미터는 SGD만 써요 |
| TAL (TOOD) | `yolo26/assigner.py` | score^0.5 × IoU^6 정렬 지표로 top-k |
| Mosaic, affine, flip, 밝기 증강 | `yolo26/data.py` | KITTI 비율(약 3.3:1)에 맞춘 직사각형 입력 |
| COCO식 mAP | `yolo26/metrics.py` | mAP50, mAP50-95 (101-point) |

## 폴더 구조

```
yolo26-kitti/
├── configs/kitti_yolo26n.yaml   # 모든 하이퍼파라미터 (CLI에서 section.key=value로 덮어쓰기)
├── yolo26/
│   ├── nn.py          # Conv, C3k2, SPPF, C2PSA, PSABlock ...
│   ├── model.py       # YOLO26 + Detect26 head (+ 디코딩 / NMS-free 후처리)
│   ├── assigner.py    # TAL + STAL
│   ├── loss.py        # CIoU + BCE + L1(or DFL), ProgLoss
│   ├── optim.py       # MuSGD, 파라미터 그룹
│   ├── data.py        # YOLO 포맷 데이터셋, Mosaic, 증강
│   ├── metrics.py     # P, R, mAP
│   ├── config.py, utils.py
├── tools/
│   ├── prepare_kitti.py   # KITTI label_2 -> YOLO 포맷 + 부분집합 추출 + train/val 분할
│   ├── plot_results.py    # 여러 실험의 학습 곡선 비교 그림
│   ├── analyze_errors.py  # 혼동행렬, 오류 유형, 틀린 이미지
│   └── compare_runs.py    # 실험 비교 markdown 표 (baseline vs BoTNet)
├── train.py  val.py  predict.py
└── docs/papers.md         # 읽을 논문 목록과 코드 대응표
```

## 1. 설치 (VSCode)

```bash
# VSCode 터미널에서
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # GPU(CUDA 12.4) 예시
pip install -r requirements.txt
python -m yolo26.model      # 스케일별 파라미터 수가 논문 Table 7과 같은지 확인
```

PyTorch 설치 명령은 GPU·CUDA 버전마다 달라요. https://pytorch.org/get-started/locally/ 에서 고르세요.
VSCode 오른쪽 아래에서 인터프리터를 `.venv`로 선택하세요.

## 2. 데이터 준비 (Google Drive의 KITTI)

1. Drive에 있는 KITTI 폴더를 PC로 가져와요. Google Drive 데스크톱 앱으로 동기화하거나, 폴더를 zip으로 내려받아 풀면 돼요.
   폴더 안에 `image_2`(이미지)와 `label_2`(라벨) 폴더가 있으면 위치는 상관없어요.
2. 변환과 부분집합 추출:

```bash
python tools/prepare_kitti.py --kitti-root "D:/GoogleDrive/kitti" --out datasets/kitti_yolo --num-images 3000
# Van을 Car에 합치려면:            --merge Van:Car
# 클래스를 더 쓰려면:              --classes Car Van Truck Pedestrian Cyclist
```

> **용량을 줄인 버전 주의**: 이미지를 리사이즈해서 용량을 줄였다면 KITTI 라벨(원본 픽셀 좌표)과 맞지 않아요.
> 이 경우 스크립트가 경고를 띄워요. 이미지 개수만 줄인 거라면 문제없어요.
> Grayscale 변환은 학습할 때 자동으로 해요(`model.in_channels: 1`). 미리 변환하지 않아도 돼요.

### 몇 장을 쓸까? (추천)

KITTI 2D 학습셋은 라벨이 있는 이미지가 7,481장이에요. 클래스는 Car가 압도적으로 많고 Cyclist가 가장 적어요.

| 목적 | 이미지 수 (train / val) | 근거 |
| --- | --- | --- |
| 코드 동작 확인 | 300 (240 / 60) | 5–10 epoch이면 끝나요. 숫자는 의미 없어요 |
| **과제·ablation 본 실험 (추천)** | **3,000 (2,400 / 600)** | 학습셋의 약 40%예요. 가장 적은 Cyclist도 수백 개 들어가서 클래스별 AP가 흔들리지 않아요. GPU 1장으로 실험 1회가 1–2시간 안팎이라 4–6개 ablation을 돌릴 수 있어요 |
| 최종 성능 | 7,481 전체 (6,000 / 1,481) | ablation으로 고른 설정을 마지막에 한 번만 돌려요 |

- 데이터를 1,500장 이하로 줄이면 Cyclist·Pedestrian 학습 예시가 너무 적어져요. 그러면 ablation 차이(보통 0.2–1 AP)가 랜덤 시드에 따른 흔들림보다 작아질 수 있어요.
- KITTI는 주행 영상에서 뽑은 프레임이라, 무작위로 나누면 같은 주행의 비슷한 장면이 train과 val에 섞여 val 점수가 높게 나와요. 더 엄밀하게 하려면 주행(sequence) 단위로 나눈 3712/3769 분할 목록을 구해서 `--train-list`, `--val-list`로 넣으세요.

## Colab에서 실행하기 (GPU가 없을 때 추천)

`colab/YOLO26_KITTI_colab.ipynb`를 Colab에서 열고 셀을 위에서부터 실행하세요.
데이터 준비, 학습, 평가, 결과 그림까지 한 번에 돼요. 학습 결과는 드라이브 `yolo26_runs/`에 epoch마다 저장돼요.
연결이 끊겨도 같은 셀을 다시 실행하면 `last.pt`에서 이어서 학습해요.

## 3. 학습

```bash
python train.py --cfg configs/kitti_yolo26n.yaml --name yolo26n_full
python train.py --resume runs/train/yolo26n_full/weights/last.pt     # 중간에 멈춘 학습 이어 하기
```

- 결과는 `runs/train/<name>/`에 저장돼요: `weights/best.pt`, `last.pt`, `results.csv`, `config.yaml`.
- 기본 설정: YOLO26-n, grayscale, 입력 256×832, 150 epoch, batch 16, MuSGD(lr 0.01, momentum 0.94).
- GPU 메모리가 부족하면 `train.batch=8`을 쓰세요. Windows에서 DataLoader 오류가 나면 `train.workers=0`을 쓰세요.
- 사전학습 가중치 없이 처음부터 학습해요. 그래서 논문의 COCO 수치와 직접 비교하면 안 돼요.

## 4. 평가와 추론

```bash
python val.py --weights runs/train/yolo26n_full/weights/best.pt          # NMS-free (one-to-one)
python val.py --weights runs/train/yolo26n_full/weights/best.pt --nms    # one-to-many + NMS
python predict.py --weights runs/train/yolo26n_full/weights/best.pt --source datasets/kitti_yolo/images/val --max 20
```

학습이 끝나면 `train.py`가 두 방식을 모두 출력해요. 학습이 짧으면 one-to-one(NMS-free) 점수가 one-to-many + NMS보다 꽤 낮게 나와요.
one-to-one head는 GT당 예측 1개만 정답으로 받아서 신호가 적고, ProgLoss가 후반에야 가중치를 몰아주기 때문이에요.
충분히 학습하면 차이가 줄어요(논문 COCO 기준 0.8 AP 이내). 두 숫자를 같이 보고하면 dual head를 분석하기에 좋아요.

## 5. 오답 분석 — 무엇을 어떻게 틀렸는지

```bash
python tools/analyze_errors.py --weights runs/train/yolo26n/weights/best.pt          # NMS-free head
python tools/analyze_errors.py --weights runs/train/yolo26n/weights/best.pt --nms    # one-to-many + NMS
```

`runs/train/<이름>/analysis/`에 아래 파일이 생겨요.

| 파일 | 내용 |
| --- | --- |
| `confusion_matrix.png` | 혼동행렬 (행 = 예측, 열 = 정답, 마지막 행·열 = 배경). 열마다 합이 1이에요 |
| `error_summary.txt` | 클래스별 오류 유형 개수(정답, 놓침, 배경 오검출, 클래스 혼동, 박스 부정확, 중복)와 물체 높이별 recall(KITTI 난이도 기준 25px, 40px) |
| `errors/*.png` | 틀린 개수가 많은 순서로 정렬한 val 이미지 (원본 컬러 위에 그림) |
| `errors_top.png` | 가장 많이 틀린 8장을 한 장으로 모은 그림 |

그림 색: 초록 얇은 박스 = 정답, 파랑 = 맞힌 예측, 빨강 = 배경 오검출·중복, 보라 = 클래스 혼동, 주황 굵은 박스 = 놓친 물체, 노랑 = 박스가 느슨함(IoU 0.1–0.5).

## 6. Baseline vs BoTNet

BoTNet(Srinivas et al., CVPR 2021)은 ResNet 마지막 단계 bottleneck의 3×3 conv를 relative-position multi-head self-attention으로 바꾼 구조예요.
여기서는 YOLO26 backbone 마지막 단계(P5, `b8`)의 bottleneck만 BoT block으로 바꾸고, 나머지는 baseline과 똑같이 둬요. 그래서 차이는 BoTNet 하나에서만 나와요.

```bash
python train.py --name yolo26n                                  # baseline
python train.py --name yolo26n_botnet model.botnet=true          # BoTNet
python tools/compare_runs.py runs/train/yolo26n runs/train/yolo26n_botnet   # markdown 비교표 -> runs/compare.md
python tools/plot_results.py runs/train/yolo26n runs/train/yolo26n_botnet   # 학습 곡선
python tools/analyze_errors.py --weights runs/train/yolo26n_botnet/weights/best.pt
```

- 구현: `yolo26/nn.py`의 `RelPosMHSA`(content-content + content-position 로짓, 논문 Fig. 4), `BoTBottleneck`, `C3k2BoT`
- 머리 수 4개(논문과 같음). relative position 표는 입력 크기(256×832 → P5 8×26)에 맞춰 만들고, 크기가 다르면 보간해요.
- n 스케일에서 BoTNet 버전은 파라미터가 약 0.1M 적어요(2.49M → 2.39M). C3k 블록보다 MHSA가 가벼워서예요. 보고서에 함께 적어 두면 좋아요.
- 이미 P5에 C2PSA attention이 있으니, "attention을 하나 더 넣었을 때"가 아니라 "P5의 conv bottleneck을 attention으로 바꿨을 때"의 비교라는 점을 같이 쓰세요.

## 7. 구현 노트 — 논문·공식 코드와 다른 점

- **처음부터 학습**: 논문은 Objects365로 사전학습한 뒤 COCO로 fine-tune해요. 여기서는 KITTI만으로 학습해요.
- **neck 마지막 `C3k2(attn=True)`**: 논문 그림은 attention을 켠다는 것까지만 보여 줘요. 여기서는 Bottleneck 뒤에 PSABlock을 붙였어요.
- **one-to-one head 입력 detach**: YOLOv10 공식 구현을 따랐어요(`model.o2o_detach`). YOLO26 공식 코드와 다를 수 있어 설정으로 바꿀 수 있게 했어요.
- **STAL 규칙**: 논문 Eq. 5 그대로(`d < 8`이면 `16`으로 교체) 구현했어요. 논문에도 이 규칙이 불연속(7px → 16px, 8px → 8px)이라는 점이 언급돼 있어요.
- **MuSGD**: Ultralytics `muon.py` 문서에 설명된 동작(NS 5단계, 계수, 가중합, SGD 부분에만 weight decay)을 다시 작성했어요.
- **평가 지표**: COCO식 mAP예요. KITTI 공식 벤치마크(Easy/Moderate/Hard, Car IoU 0.7)와는 기준이 달라요.

## 참고 문헌과 코드

- Srinivas et al., *Bottleneck Transformers for Visual Recognition*, CVPR 2021 — [논문](https://arxiv.org/abs/2101.11605)
- Jocher et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748, 2026 — [논문](https://arxiv.org/abs/2606.03748), [코드](https://github.com/ultralytics/ultralytics)
- Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024 — [논문](https://arxiv.org/abs/2405.14458), [코드](https://github.com/THU-MIG/yolov10)
- Feng et al., *TOOD: Task-aligned One-stage Object Detection*, ICCV 2021 — [논문](https://arxiv.org/abs/2108.07755)
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks*, 2024 — [코드](https://github.com/KellerJordan/Muon)
- Geiger et al., *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012 — [데이터](https://www.cvlibs.net/datasets/kitti/eval_object.php)

전체 읽기 목록과 코드 대응은 [`papers.md`](papers.md)에 있어요.

**라이선스 메모**: 모듈 구조와 MuSGD 동작은 Ultralytics 코드(AGPL-3.0)를 참고해 다시 작성했어요. 저장소를 공개할 때는 AGPL-3.0으로 두고 위 출처를 남기는 걸 권장해요. KITTI 데이터는 CC BY-NC-SA 3.0이라 저장소에 이미지를 올리면 안 돼요(`.gitignore`에서 `datasets/`를 제외해 뒀어요).
