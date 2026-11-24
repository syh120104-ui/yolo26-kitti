# 실험 결과

학습이 끝나면 드라이브 `yolo26_runs/` 에서 아래 파일을 이 폴더로 복사한다.

| 파일 | 가져올 곳 |
| --- | --- |
| `compare.md`, `compare.png` | `yolo26_runs/compare.md`, `yolo26_runs/compare.png` |
| `results_baseline.csv` | `yolo26_runs/train/yolo26n/results.csv` |
| `results_botnet.csv` | `yolo26_runs/train/yolo26n_botnet/results.csv` |
| `confusion_matrix_baseline.png` | `yolo26_runs/train/yolo26n/analysis/confusion_matrix.png` |
| `confusion_matrix_botnet.png` | `yolo26_runs/train/yolo26n_botnet/analysis/confusion_matrix.png` |
| `error_summary_baseline.txt` | `yolo26_runs/train/yolo26n/analysis/error_summary.txt` |
| `error_summary_botnet.txt` | `yolo26_runs/train/yolo26n_botnet/analysis/error_summary.txt` |
| `errors_top_baseline.png` | `yolo26_runs/train/yolo26n/analysis/errors_top.png` |
| `errors_top_botnet.png` | `yolo26_runs/train/yolo26n_botnet/analysis/errors_top.png` |

가중치 파일(`.pt`)은 올리지 않는다.
