| run | inference | params (M) | GFLOPs | mAP50 | mAP50-95 | Car AP50-95 | Pedestrian AP50-95 | Cyclist AP50-95 | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | NMS-free | 2.49 | 2.88 | 0.465 | 0.350 | 0.612 | 0.159 | 0.278 | 3.4 |
| yolo26n | NMS | 2.49 | 2.88 | 0.709 | 0.495 | 0.732 | 0.294 | 0.458 | 3.8 |
| yolo26n_botnet | NMS-free | 2.39 | 2.87 | 0.469 | 0.349 | 0.624 | 0.157 | 0.266 | 3.2 |
| yolo26n_botnet | NMS | 2.39 | 2.87 | 0.705 | 0.496 | 0.732 | 0.293 | 0.464 | 3.7 |

(params count both heads during training; the deployed model keeps one)
