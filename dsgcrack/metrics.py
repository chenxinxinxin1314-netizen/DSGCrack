"""Two explicit, separate protocols: historical paper metrics and sigmoid metrics."""
import numpy as np


def divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=np.asarray(b) != 0)


def legacy_image(logits):
    value = np.asarray(logits)
    if not np.isfinite(value).all():
        raise FloatingPointError("Nonfinite prediction logits")
    peak = value.max()
    score = np.zeros_like(value) if peak == 0 else 255*(value/peak)
    return np.rint(np.clip(score, 0, 255)).astype(np.uint8)


class LegacyMetrics:
    """Exact histogram implementation of the historical component-ablation evaluator."""
    def __init__(self):
        self.rows = []
        self.positive = np.arange(256)[None, :]/255. > np.arange(0., 1., .01)[:, None]

    def update(self, score, label):
        score, label = np.asarray(score), np.asarray(label, dtype=bool)
        if score.shape != label.shape or score.dtype != np.uint8:
            raise ValueError("Expected same-shape uint8 legacy scores and binary labels")
        hp = np.bincount(score[label].ravel(), minlength=256)
        hn = np.bincount(score[~label].ravel(), minlength=256)
        tp, fp = self.positive@hp, self.positive@hn
        self.rows.append(np.stack([tp, fp, hp.sum()-tp, hn.sum()-fp]))

    def result(self):
        if not self.rows:
            raise ValueError("Cannot evaluate an empty split")
        tp, fp, fn, tn = np.stack(self.rows).transpose(1, 0, 2)
        image_f1 = divide(2*tp, 2*tp+fp+fn)
        image_miou = .5*(divide(tp, tp+fp+fn)+divide(tn, tn+fp+fn))
        T, P, N = tp.sum(0), fp.sum(0), fn.sum(0)
        return {"images": len(self.rows), "mIoU": float(image_miou.mean(0).max()),
                "ODS": float(image_f1.mean(0).max()), "OIS": float(image_f1.max(1).mean()),
                "F1": float(divide(2*T, 2*T+P+N)[0]),
                "Precision": float(np.divide(T, T+P, out=np.ones_like(T, dtype=np.float64),
                                             where=T+P != 0)[0]),
                "Recall": float(divide(T, T+N)[0])}


class ProbabilityMetrics:
    """Raw sigmoid probabilities; fixed-threshold metrics and a separate global-F1 sweep."""
    def __init__(self, threshold=.5):
        if not 0 <= threshold <= 1:
            raise ValueError("Probability threshold must be within [0,1]")
        self.threshold = threshold
        self.thresholds = np.arange(101, dtype=np.float64)/100.
        self.rows, self.fixed = [], []

    def update(self, probability, label):
        p, y = np.asarray(probability), np.asarray(label, dtype=bool)
        if p.shape != y.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError("Invalid probability or label")
        bins = np.searchsorted(self.thresholds, p, side="left")
        hp = np.bincount(bins[y].ravel(), minlength=102)
        hn = np.bincount(bins[~y].ravel(), minlength=102)
        tp, fp = hp[::-1].cumsum()[::-1][1:], hn[::-1].cumsum()[::-1][1:]
        self.rows.append(np.stack([tp, fp, hp.sum()-tp, hn.sum()-fp]))
        pred = p > self.threshold
        self.fixed.append([(pred & y).sum(), (pred & ~y).sum(),
                           (~pred & y).sum(), (~pred & ~y).sum()])

    def best_threshold(self):
        tp, fp, fn, _ = np.stack(self.rows).sum(0)
        curve = divide(2*tp, 2*tp+fp+fn)
        candidates = np.flatnonzero(curve == curve.max())
        index = candidates[np.argmin(abs(self.thresholds[candidates]-.5))]
        return float(self.thresholds[index])

    def result(self):
        tp, fp, fn, tn = np.asarray(self.fixed).T
        T, P, N = tp.sum(), fp.sum(), fn.sum()
        a, b, c, _ = np.stack(self.rows).transpose(1, 0, 2)
        return {"images": len(self.rows), "threshold": float(self.threshold),
                "mIoU": float((.5*(divide(tp, tp+fp+fn)+divide(tn, tn+fp+fn))).mean()),
                "F1": float(divide(2*T, 2*T+P+N)),
                "Precision": float(T/(T+P)) if T+P else 1.,
                "Recall": float(divide(T, T+N)),
                "ODS": float(divide(2*a.sum(0), 2*a.sum(0)+b.sum(0)+c.sum(0)).max()),
                "OIS": float(divide(2*a, 2*a+b+c).max(1).mean())}
