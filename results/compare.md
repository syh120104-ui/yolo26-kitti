| run | inference | params (M) | GFLOPs | mAP50 | mAP50-95 | Car AP50-95 | Pedestrian AP50-95 | Cyclist AP50-95 | ms/img |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | NMS-free | 2.49 | 2.88 | 0.466 | 0.350 | 0.613 | 0.159 | 0.278 | 1.6 |
| yolo26n | NMS | 2.49 | 2.88 | 0.710 | 0.495 | 0.732 | 0.294 | 0.458 | 2.0 |
| yolo26n_botnet | NMS-free | 2.39 | 2.87 | 0.469 | 0.349 | 0.624 | 0.157 | 0.266 | 1.4 |
| yolo26n_botnet | NMS | 2.39 | 2.87 | 0.704 | 0.496 | 0.732 | 0.293 | 0.463 | 1.9 |

(params count both heads during training; the deployed model keeps one)
