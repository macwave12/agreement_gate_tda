"""Agreement-gated TDA: use TDA's positive cache only where its top class differs from CLIP's.

Built on TDA's own cache functions (online_method/tda.py). With `gate=False` this is TDA, by default
under its single ImageNet-val setting (alpha = 2, beta = 5), used on every dataset; `read_first=True`
reads the caches before the current sample is written to them (no self-vote).
"""
import math

import torch

from online_method.tda import Entropy, compute_cache_logits, update_cache

ALPHA, BETA = 2.0, 5.0
NEG_ALPHA, NEG_BETA = 0.117, 1.0


def clip_logits(image_features, text_features, logit_scale):
    """CLIP's logits in fp16, as TDA computes them under CUDA autocast. On CPU (cached features)
    the same rounding is done by hand: fp16 inputs, fp32 product, fp16 result."""
    if image_features.is_cuda:
        with torch.cuda.amp.autocast():
            return logit_scale * image_features @ text_features.t()
    scaled = (logit_scale * image_features).half().float()
    return (scaled @ text_features.t().half().float()).half()


class TDAGate():
    def __init__(self, model, device, gate=True, read_first=False, keep_trace=False, alpha=ALPHA, beta=BETA):
        self.model = model
        self.device = device
        self.gate = gate
        self.alpha, self.beta = alpha, beta
        self.read_first = read_first
        self.keep_trace = keep_trace

    def prepare_model_and_optimization(self, args):
        self.model.eval()
        self.pos_cache = {}
        self.neg_cache = {}
        self.trace = []  # per sample: CLIP logits z, b = z - negative cache, positive cache c, self-vote

    def pre_adaptation(self):
        pass

    def adaptation_process(self, image, images, args):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                image_features, text_features, logit_scale = self.model.forward_features(images)
            logits = clip_logits(image_features, text_features, logit_scale)
            softmax0 = logits.float().softmax(dim=-1)

        ent0 = Entropy(softmax0)
        pred0 = torch.max(logits, 1)[1].item()
        num_classes = logits.size(1)

        def write():
            update_cache(self.pos_cache, pred0, [image_features, ent0], 3)
            if 0.2 < ent0 / math.log2(num_classes) and ent0 / math.log2(num_classes) < 0.5:
                update_cache(self.neg_cache, pred0, [image_features, ent0, softmax0], 2, True)

        def read():
            neg = compute_cache_logits(image_features, self.neg_cache, NEG_ALPHA, NEG_BETA,
                                       text_features.unsqueeze(0), (0.03, 1.0)) if len(self.neg_cache) > 0 else 0.0
            has_cache = len(self.pos_cache) > 0
            cache = compute_cache_logits(image_features, self.pos_cache, 1.0, self.beta,
                                         text_features.unsqueeze(0)) if has_cache else torch.zeros_like(logits)
            base = logits.clone()
            base -= neg
            return neg, base.float(), cache.float(), has_cache

        if self.read_first:
            neg, base, cache, has_cache = read()
            write()
            self_vote = False
        else:  # TDA's order: the sample is written first and can vote for itself
            write()
            neg, base, cache, has_cache = read()
            self_vote = any(item[0] is image_features for item in self.pos_cache[pred0])

        gate_open = has_cache and int(cache.reshape(-1).argmax()) != pred0
        if self.gate:
            output = base + self.alpha * cache if gate_open else base
        else:  # TDA's own arithmetic: both caches are added to its fp16 logits in place
            output = logits.clone()
            output += self.alpha * cache
            output -= neg
        if self.keep_trace:
            self.trace.append((logits.float().cpu(), base.cpu(), cache.cpu(), self_vote))

        return {'output': output}
