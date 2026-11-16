
# Agreement-Gated TDA

Code for "Agreement-Gated TDA: the cache that echoes CLIP, and a fix that adds no parameters" (`paper.pdf`), built
on the VLM-TTA benchmark below. TDA's positive cache is filled with CLIP's own predictions, so on many samples it
votes for the class CLIP already chose: such a vote cannot change the decision and only adds confidence. The
agreement gate uses the positive cache only where its top class differs from CLIP's.

What this fork adds to the benchmark:

- `online_method/tda_gate.py`: the agreement gate, on TDA's own cache functions.
- `main.py`: extracts CLIP features, runs the streams, and writes the paper's Figs. 2-5 and all of its tables.
- `prepare_data.py`: downloads the 16 evaluation sets and cuts the paper's fixed subsets.
- `data/datautils.py`, `data/wds_classnames.json`: the seven held-out sets taken from the CLIP_benchmark mirrors.
- Three small fixes so the benchmark runs elsewhere: TDA's cache functions follow the features' device, the
  few-shot split files are read from the dataset folder, and the CLIP download folder comes from `CLIP_CACHE_ROOT`.

### Reproducing the paper

```
pip install -r requirements.txt          # after installing torch and torchvision for your CUDA version
python prepare_data.py --data /path/to/data_root
python main.py --data /path/to/data_root
```

`main.py` writes to `results/`:

| paper | file |
|---|---|
| Fig. 2 | `paper_selfvote_val.png`, `paper_agreement_val.png` |
| Fig. 3 | `paper_where_all.png` |
| Fig. 4, Fig. 5 | `no_info_break_even.png`, `no_info_scores.png` |
| Tables 1-4, Appendix A and B | `tables.md` |

Each figure comes with the numbers behind it as a CSV. Fig. 1 of the paper is the overview figure of the TDA
paper and is not generated.

Feature extraction needs a GPU and runs once (`features/`). The streams of CLIP, TDA and the gate run on CPU from
the cached features; DMN-weak and ECALP (Table 2) run on the GPU if there is one. Every stream is kept in
`results/runs/`, so a second run only redraws. `--datasets` restricts the run to some of the 16 sets; the
temperature baseline of Table 2 is fitted on the val streams of the design sets that are included.

Features extracted on another GPU or software version differ in their last bits, which moves the numbers
slightly: `paper.pdf` was built from features extracted on an RTX 5060 with the versions in `requirements.txt`.
ECALP is the one method that is sensitive to this beyond the second decimal.

Layout of the data root that `prepare_data.py` creates:

| sets | folders | format |
|---|---|---|
| DTD, Flowers, Pets, EuroSAT, Aircraft, Caltech101, Food101 | `dtd`, `oxford_flowers`, `oxford_pets`, `eurosat`, `fgvc_aircraft`, `caltech-101`, `food-101` | [CoOp](https://github.com/KaiyangZhou/CoOp) layout, with each `split_zhou_*.json` in its dataset folder |
| Cars, GTSRB, FER2013, PCam, SVHN, MNIST, Rendered SST2 | `stanford-cars-wds`, `gtsrb`, `fer2013`, `pcam`, `svhn`, `mnist`, `rendered_sst2` | `images/<class index, four digits>/<image>`, unpacked from the `clip-benchmark/wds_*` test shards on Hugging Face |
| ImageNet-R, ImageNet-Sketch | `imagenet-rendition/imagenet-r`, `imagenet-sketch/images` | class folders named by wnid |

The paper uses a fixed random subset of the larger test sets: 10,000 images of Food101 and 3,000 each of FER2013,
PCam, SVHN and MNIST.

# VLM-TTA: A Benchmark of Test-Time Adaptation for Vision-Language Models

## 

## [**[NeurIPS-2025] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models**](https://arxiv.org/abs/2506.24000)

### Framework:  
<img src="figs/ttavlm.png" width="600"/>

### Prerequisites:

### Dataset:

Please follow [CoOp](https://github.com/KaiyangZhou/CoOp) and manually download the require datasets.

Replace /path/to/dataset with your dataset folder root in below commands and check the path of json file in fewshot_datasets.py.

Replace your_cache_path with your cache folder root in some files (`clip/constants.py`, `instance_tta.py`, `online_tta.py`).

### Run:

1. ##### Episodic Test-Time Adaptation on CLIP

	```python
    python instance_tta.py --data /path/to/dataset --test_sets I -a ViT-B/16 -b 64 --gpu $gpu --ctx_init a_photo_of_a -p 50 --output_dir 'episodic_results/ckps/tpt' --algorithm tpt
    ```

    The algorithm can be replaced by a series of episodic methods supported by this benchmark: `clipzs`, `tpt`, `ctpt`, `rlcf`, `mta`, `zero`, `ttl`, `tps`, `rtpt`.

2. ##### Online Test-Time Adaptation on CLIP

    ```python
    python online_tta.py --data /path/to/dataset --test_sets I -a ViT-B/16 -b 1 --gpu $gpu --ctx_init a_photo_of_a -p 50 --output_dir 'online_results/ckps/tda' --algorithm tda
    ```

    The algorithm can be replaced by a series of online methods supported by this benchmark: `clipzs`, `tda`, `dmn_weak`, `dmn`, `onzeta`, `boostadapter`, `dpe`, `ecalp`, `dyanprompt`.


### Citation:

If you find our survey and repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{sheng2025illusion,
  title={The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models},
  author={Sheng, Lijun and Liang, Jian and He, Ran and Wang, Zilei and Tan, Tieniu},
  booktitle={Proc. NeurIPS},
  year={2025}
}
```

