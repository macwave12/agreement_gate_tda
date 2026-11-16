### Table 1. Design test sets, 10 stream orders (mean ± std)

|  | method | DTD | Flowers | Pets | EuroSAT | Aircraft | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Acc. | CLIP | 44.39 | 67.32 | 88.20 | 42.06 | 23.85 | 53.16 |
|  | TDA | 47.10 ± 0.34 | 70.28 ± 0.40 | 88.70 ± 0.17 | 54.01 ± 0.88 | 24.06 ± 0.25 | 56.83 ± 0.22 |
|  | agreement gate | 47.16 ± 0.18 | 69.92 ± 0.33 | 88.71 ± 0.20 | 53.87 ± 0.90 | 24.15 ± 0.07 | 56.76 ± 0.20 |
| ECE | CLIP | 8.27 | 2.63 | 4.37 | 7.16 | 5.44 | 5.57 |
|  | TDA | 17.36 ± 0.43 | 9.56 ± 0.49 | 2.49 ± 0.23 | 6.61 ± 0.75 | 18.32 ± 0.50 | 10.87 ± 0.31 |
|  | agreement gate | 7.38 ± 0.29 | 3.21 ± 0.46 | 4.51 ± 0.19 | 3.84 ± 0.38 | 8.38 ± 0.23 | 5.47 ± 0.18 |

### Table 2. Design test sets, 3 stream orders, mean of the five sets; changes are against TDA

| method | Acc. | ECE | ΔAcc. | ΔECE | ECE wins |
| --- | --- | --- | --- | --- | --- |
| CLIP | 53.16 | 5.57 | -3.73 | -5.25 | 3/3 |
| TDA | 56.89 | 10.82 | -- | -- | -- |
| agreement gate | 56.80 | 5.53 | -0.10 | -5.29 | 3/3 |
| read-first TDA | 56.63 | 8.71 | -0.27 | -2.11 | 3/3 |
| TDA + temperature (fit on val) | 56.89 | 7.03 | +0.00 | -3.79 | 3/3 |
| TDA + ZAEC | 56.89 | 6.02 | +0.00 | -4.80 | 3/3 |
| gate + ZAEC | 56.80 | 5.36 | -0.10 | -5.46 | 3/3 |
| DMN-weak | 56.97 | 13.94 | +0.08 | +3.12 | 0/3 |
| ECALP | 58.34 | 32.28 | +1.44 | +21.45 | 0/3 |

### Table 3. Held-out sets, 3 stream orders: accuracy, ECE and confidence − accuracy gap

| tier | dataset | acc. CLIP | acc. TDA | acc. gate | ECE CLIP | ECE TDA | ECE gate | gap CLIP | gap TDA | gap gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| near | Caltech101 | 93.27 | 93.46 | 93.27 | 5.03 | 3.46 | 5.26 | -5.0 | -0.6 | -5.2 |
|  | Food101 | 83.64 | 84.04 | 84.13 | 2.20 | 2.42 | 2.49 | -0.6 | +2.2 | -0.8 |
|  | Cars | 63.70 | 64.82 | 64.62 | 3.04 | 4.54 | 3.09 | -2.8 | +4.5 | -2.6 |
|  | *mean (3)* | 80.20 | 80.77 | 80.67 | 3.42 | 3.47 | 3.61 | -2.8 | +2.0 | -2.8 |
| far | GTSRB | 35.25 | 38.16 | 38.24 | 10.43 | 7.90 | 8.14 | +0.2 | +7.4 | +0.2 |
|  | ImageNet-R | 73.98 | 75.30 | 75.35 | 3.54 | 1.51 | 3.65 | -3.5 | -1.1 | -3.6 |
|  | ImageNet-Sketch | 46.09 | 48.75 | 48.88 | 4.88 | 11.99 | 4.58 | +4.9 | +12.0 | +4.6 |
|  | FER2013 | 45.17 | 48.99 | 49.47 | 2.91 | 5.68 | 3.90 | +0.4 | +4.7 | -0.7 |
|  | MNIST | 19.53 | 29.50 | 29.58 | 26.85 | 24.03 | 16.68 | +26.9 | +24.0 | +16.6 |
|  | PCam | 57.47 | 61.02 | 61.02 | 15.54 | 24.96 | 15.28 | +14.7 | +24.7 | +11.3 |
|  | Rendered SST2 | 57.44 | 57.40 | 57.40 | 19.13 | 20.29 | 18.53 | +19.0 | +20.3 | +18.5 |
|  | SVHN | 25.77 | 31.88 | 31.97 | 7.72 | 11.01 | 7.24 | +7.5 | +11.0 | +7.1 |
|  | *mean (8)* | 45.09 | 48.88 | 48.99 | 11.37 | 13.42 | 9.75 | +8.8 | +12.9 | +6.8 |

### Table 4. Cache ablation, 3 stream orders: accuracy / ECE (far: without ImageNet-Sketch)

| accuracy / ECE | design (test) | near | far |
| --- | --- | --- | --- |
| CLIP | 53.2 / 5.6 | 80.2 / 3.4 | 44.9 / 12.3 |
| + negative cache (b) | 54.9 / 4.5 | 80.7 / 3.9 | 47.5 / 10.1 |
| b + positive cache on agreeing samples | 55.0 / 9.9 | 80.7 / 3.2 | 47.4 / 13.0 |
| b + positive cache on disagreeing samples (gate) | 56.8 / 5.5 | 80.7 / 3.6 | 49.0 / 10.5 |
| b + positive cache on as many random samples | 56.0 / 7.1 | 80.7 / 2.7 | 48.3 / 11.6 |
| b + positive cache on all samples (TDA) | 56.9 / 10.8 | 80.8 / 3.5 | 48.9 / 13.6 |
| samples the gate opens on (disagreeing) | 45% | 25% | 52% |

### Appendix A. TDA's per-dataset alpha and beta against the single setting (design test, 3 orders)

| method | acc. per-dataset | acc. single | ECE per-dataset | ECE single |
| --- | --- | --- | --- | --- |
| CLIP | 53.16 | 53.16 | 5.57 | 5.57 |
| TDA | 56.65 | 56.89 | 11.02 | 10.83 |
| agreement gate | 56.58 | 56.81 | 6.15 | 5.50 |

### Appendix B. ECE (%) of every method of Table 2 on each design test set

| ECE (%) | DTD | Flowers | Pets | EuroSAT | Aircraft |
| --- | --- | --- | --- | --- | --- |
| CLIP | 8.27 | 2.63 | 4.37 | 7.16 | 5.44 |
| TDA | 17.23 | 9.62 | 2.60 | 6.40 | 18.27 |
| agreement gate | 7.52 | 3.23 | 4.61 | 3.74 | 8.56 |
| read-first TDA | 12.62 | 7.91 | 2.81 | 6.22 | 13.99 |
| TDA + temperature (fit on val) | 3.58 | 5.51 | 11.34 | 9.90 | 4.83 |
| TDA + ZAEC | 8.07 | 2.67 | 4.93 | 5.65 | 8.78 |
| gate + ZAEC | 5.79 | 3.22 | 5.60 | 6.68 | 5.50 |
| DMN-weak | 25.59 | 7.19 | 4.44 | 4.30 | 28.18 |
| ECALP | 39.37 | 26.65 | 10.23 | 24.03 | 61.11 |
