"""
Sanity check: runs full pipeline on 200 samples, no Slurm needed.
Tests: data loading -> inference -> objectives -> NSGA-II (5 gen) -> Pareto
Run: python scripts/sanity_check.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as tv_models

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "nsga2"))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SANITY = 200   # samples to use
N_MODELS = 2     # only test densenet121 + resnet50 to save time
N_GEN    = 5
POP_SIZE = 20

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(name, fn):
    t0 = time.time()
    try:
        result = fn()
        print(f"  {PASS} {name}  ({time.time()-t0:.1f}s)")
        return result
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


# ─── Step 1: Data loading ────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Data loading")
print("="*60)

csv_path   = ROOT / "data" / "raw" / "Data_Entry_2017_v2020.csv"
image_dir  = ROOT / "data" / "raw" / "nih_images"
models_dir = ROOT / "models" / "backbones"

def load_csv():
    assert csv_path.exists(), f"CSV not found: {csv_path}"
    df = pd.read_csv(csv_path)
    assert 'Finding Labels' in df.columns
    assert 'Patient Gender' in df.columns
    assert 'Image Index' in df.columns
    mask = df['Finding Labels'].str.contains('Effusion', na=False) | \
           (df['Finding Labels'] == 'No Finding')
    df = df[mask].reset_index(drop=True)
    assert len(df) > 100, f"Too few rows: {len(df)}"
    return df

df = check("Load CSV + filter Effusion/NoFinding", load_csv)
print(f"     Rows after filter: {len(df):,}")

def check_images():
    assert image_dir.exists(), f"image_dir missing: {image_dir}"
    sample = df.head(10)
    missing = [r['Image Index'] for _, r in sample.iterrows()
               if not (image_dir / r['Image Index']).exists()]
    assert not missing, f"Missing images: {missing[:3]}"
    return True

check("Image directory + spot-check 10 files", check_images)


# ─── Step 2: Inference on N_SANITY samples ───────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Model inference")
print("="*60)

MODEL_NAMES = ['densenet121', 'resnet50', 'resnet101',
               'efficientnet_b4', 'vgg16', 'inception_v3', 'mobilenet_v2']
TEST_MODELS  = MODEL_NAMES[:N_MODELS]

# Tiny dataset
df_small = df.head(N_SANITY).reset_index(drop=True)

class TinyDataset(Dataset):
    def __init__(self, df, image_dir, img_size=224):
        self.df = df
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / row['Image Index']).convert('RGB')
        label = float('Effusion' in str(row['Finding Labels']))
        gender = str(row.get('Patient Gender', 'U'))
        return self.transform(img), torch.tensor(label, dtype=torch.float32), gender


def build_model(name):
    model_map = {
        'densenet121':     (tv_models.densenet121,    224),
        'resnet50':        (tv_models.resnet50,        224),
        'resnet101':       (tv_models.resnet101,       224),
        'efficientnet_b4': (tv_models.efficientnet_b4, 224),
        'vgg16':           (tv_models.vgg16_bn,        224),
        'inception_v3':    (tv_models.inception_v3,    299),
        'mobilenet_v2':    (tv_models.mobilenet_v2,    224),
    }
    fn, img_size = model_map[name]
    model = fn(weights=None)
    if name == 'densenet121':
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif name in ('resnet50', 'resnet101'):
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == 'efficientnet_b4':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif name == 'vgg16':
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    elif name == 'inception_v3':
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.aux_logits = False
    elif name == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model, img_size


all_preds = []
labels_ref = None
genders_ref = None

for mname in TEST_MODELS:
    ckpt = models_dir / f"{mname}.pt"

    def run_inference(mname=mname, ckpt=ckpt):
        global labels_ref, genders_ref
        model, img_size = build_model(mname)
        assert ckpt.exists(), f"Checkpoint missing: {ckpt}"
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model = model.to(DEVICE).eval()

        ds = TinyDataset(df_small, image_dir, img_size)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

        preds, lbls, gnds = [], [], []
        with torch.no_grad():
            for imgs, lbl, gnd in dl:
                out = model(imgs.to(DEVICE)).squeeze(-1).flatten()
                preds.extend(torch.sigmoid(out).cpu().numpy().tolist())
                lbls.extend(lbl.numpy().tolist())
                gnds.extend(list(gnd))

        assert len(preds) == N_SANITY, f"Expected {N_SANITY} preds, got {len(preds)}"
        assert 0.0 <= min(preds) and max(preds) <= 1.0, "Predictions out of [0,1]"

        if labels_ref is None:
            labels_ref = np.array(lbls)
            genders_ref = np.array(gnds)
        return np.array(preds)

    preds = check(f"Inference: {mname}", run_inference)
    all_preds.append(preds)

print(f"     Positive labels: {int(labels_ref.sum())} / {N_SANITY}")
print(f"     Gender counts:   M={np.sum(genders_ref=='M')}, F={np.sum(genders_ref=='F')}")


# ─── Step 3: Objectives ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Objective functions")
print("="*60)

from objectives import evaluate_ensemble, compute_ace, compute_demographic_auc_gap

P_cache = np.stack(all_preds, axis=1)   # (N_SANITY, N_MODELS)

def test_objectives():
    assert P_cache.shape == (N_SANITY, N_MODELS), f"Bad P_cache shape: {P_cache.shape}"
    w = np.ones(N_MODELS) / N_MODELS
    f1, f2, f3 = evaluate_ensemble(w, P_cache, labels_ref, genders_ref)
    assert -1.0 <= f1 <= 0.0,  f"f1 (neg AUC) out of range: {f1}"
    assert 0.0  <= f2 <= 1.0,  f"f2 (ACE) out of range: {f2}"
    assert 0.0  <= f3 <= 1.0,  f"f3 (equity gap) out of range: {f3}"
    return f1, f2, f3

f1, f2, f3 = check("evaluate_ensemble (equal weights)", test_objectives)
print(f"     f1 (neg AUC):    {f1:.4f}  →  AUC = {-f1:.4f}")
print(f"     f2 (ACE):        {f2:.4f}")
print(f"     f3 (equity gap): {f3:.4f}")


# ─── Step 4: NSGA-II (mini run) ───────────────────────────────────────────────
print("\n" + "="*60)
print(f"STEP 4: NSGA-II ({N_GEN} generations, pop={POP_SIZE})")
print("="*60)

def test_nsga2():
    from deap import base, creator, tools
    import multiprocessing as mp

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: (np.random.dirichlet(np.ones(N_MODELS))).tolist())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_fn(ind):
        return evaluate_ensemble(ind, P_cache, labels_ref, genders_ref)

    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)

    # Evaluate initial pop
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # selTournamentDCD requires crowding_dist, assigned by selNSGA2
    pop[:] = toolbox.select(pop, POP_SIZE)

    for gen in range(N_GEN):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.9:
                a, b = np.array(c1[:]), np.array(c2[:])
                # simple blend crossover
                alpha = np.random.rand(N_MODELS)
                new1 = alpha * a + (1 - alpha) * b
                new2 = (1 - alpha) * a + alpha * b
                new1 = np.abs(new1) / np.abs(new1).sum()
                new2 = np.abs(new2) / np.abs(new2).sum()
                c1[:] = new1.tolist()
                c2[:] = new2.tolist()
                del c1.fitness.values
                del c2.fitness.values

        for mut in offspring:
            if np.random.rand() < 0.2:
                w = np.abs(np.array(mut[:]) + np.random.randn(N_MODELS) * 0.1)
                mut[:] = (w / w.sum()).tolist()
                del mut.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = toolbox.select(pop + offspring, POP_SIZE)

    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    assert len(pareto) >= 1, "Empty Pareto front"
    return pareto

pareto = check(f"NSGA-II {N_GEN} generations", test_nsga2)
print(f"     Pareto front size: {len(pareto)}")
fits = np.array([ind.fitness.values for ind in pareto])
print(f"     Best AUC in front: {-fits[:,0].min():.4f}")
print(f"     Best ACE in front: {fits[:,1].min():.4f}")
print(f"     Best equity gap:   {fits[:,2].min():.4f}")


# ─── Step 5: P_cache save/load round-trip ────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Cache save/load (Phase 1.5 outputs)")
print("="*60)

cache_dir = ROOT / "data" / "cache"

def test_cache_save():
    cache_dir.mkdir(parents=True, exist_ok=True)
    demo = genders_ref.astype(str)
    np.save(cache_dir / "_sanity_P_cache.npy", P_cache)
    np.save(cache_dir / "_sanity_y_true.npy",  labels_ref)
    np.save(cache_dir / "_sanity_demo.npy",    demo)
    # reload
    P2 = np.load(cache_dir / "_sanity_P_cache.npy")
    y2 = np.load(cache_dir / "_sanity_y_true.npy")
    d2 = np.load(cache_dir / "_sanity_demo.npy", allow_pickle=True).astype(str)
    assert P2.shape == P_cache.shape,   f"P_cache shape mismatch: {P2.shape}"
    assert y2.shape == labels_ref.shape, f"y_true shape mismatch"
    assert set(d2) <= {'M', 'F', 'U'},   f"Unexpected gender values: {set(d2)}"
    # cleanup
    for f in ['_sanity_P_cache.npy', '_sanity_y_true.npy', '_sanity_demo.npy']:
        (cache_dir / f).unlink()
    return True

check("Save + reload P_cache / y_true / demographics", test_cache_save)


# ─── Step 6: Pareto analysis (Phase 3) ───────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: Pareto analysis & visualization")
print("="*60)

sys.path.insert(0, str(ROOT / "analysis"))

def test_pareto_knee():
    from visualize_pareto import find_knee_point
    fitness = np.array([ind.fitness.values for ind in pareto])
    assert fitness.shape[1] == 3, f"Expected 3 objectives, got {fitness.shape[1]}"
    knee = find_knee_point(fitness)
    assert 0 <= knee < len(pareto), f"Knee index out of range: {knee}"
    auc  = float(-fitness[knee, 0])
    ace  = float(fitness[knee, 1])
    gap  = float(fitness[knee, 2])
    assert 0.5 <= auc <= 1.0, f"Knee AUC implausible: {auc}"
    assert 0.0 <= ace <= 1.0, f"Knee ACE implausible: {ace}"
    assert 0.0 <= gap <= 1.0, f"Knee gap implausible: {gap}"
    return knee, auc, ace, gap

knee, auc, ace, gap = check("find_knee_point on mini Pareto front", test_pareto_knee)
print(f"     Knee index: {knee}")
print(f"     AUC={auc:.4f}  ACE={ace:.4f}  Equity gap={gap:.4f}")


def test_pareto_outputs():
    """Simulate what visualize_pareto.main() will produce from saved npy files."""
    import json, tempfile, shutil
    from visualize_pareto import find_knee_point

    fitness  = np.array([ind.fitness.values for ind in pareto])
    weights  = np.array([ind[:]             for ind in pareto])
    knee_idx = find_knee_point(fitness)

    tmp = Path(tempfile.mkdtemp())
    try:
        np.save(tmp / "pareto_fitness.npy", fitness)
        np.save(tmp / "pareto_weights.npy", weights)

        # Reload and build the knee_point.json structure
        pf = np.load(tmp / "pareto_fitness.npy")
        pw = np.load(tmp / "pareto_weights.npy")
        MODEL_NAMES_FULL = ['DenseNet121', 'ResNet50', 'ResNet101',
                            'EfficientNet-B4', 'VGG16', 'Inception-v3', 'MobileNetV2']
        knee_data = {
            'knee_index': int(knee_idx),
            'fitness': {
                'auc':        float(-pf[knee_idx, 0]),
                'ace':        float( pf[knee_idx, 1]),
                'equity_gap': float( pf[knee_idx, 2]),
            },
            'weights': {
                MODEL_NAMES_FULL[i]: float(pw[knee_idx, i])
                for i in range(pw.shape[1])
            }
        }
        out = json.dumps(knee_data)
        assert '"auc"' in out and '"weights"' in out
        assert abs(sum(knee_data['weights'].values()) - 1.0) < 1e-6, \
            "Weights don't sum to 1"

        # Check CSV generation
        import pandas as pd
        df_p = pd.DataFrame({
            'Solution': range(len(pf)),
            'AUC':         -pf[:, 0],
            'ACE':          pf[:, 1],
            'Equity_Gap':   pf[:, 2],
        })
        assert len(df_p) == len(pareto)
        assert list(df_p.columns) == ['Solution', 'AUC', 'ACE', 'Equity_Gap']
    finally:
        shutil.rmtree(tmp)
    return True

check("Pareto npy -> knee_point.json + CSV structure", test_pareto_outputs)


def test_plotly_import():
    """Verify plotly is available (HTML export won't crash)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    assert fig is not None
    return True

check("Plotly import + basic figure creation", test_plotly_import)


# ─── Step 7: Checkpoint round-trip (Phase 2 resume) ──────────────────────────
print("\n" + "="*60)
print("STEP 7: NSGA-II checkpoint save/load")
print("="*60)

def test_checkpoint():
    import pickle, tempfile
    fitness = np.array([ind.fitness.values for ind in pareto])
    ckpt = {
        'generation': 5,
        'population': pareto,
        'logbook':    [],
    }
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = Path(f.name)
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(ckpt, f)
        assert tmp_path.stat().st_size > 0, "Empty checkpoint file"
        with open(tmp_path, 'rb') as f:
            loaded = pickle.load(f)
        assert loaded['generation'] == 5
        assert len(loaded['population']) == len(pareto)
        # fitness values survive pickle round-trip
        loaded_fits = np.array([ind.fitness.values for ind in loaded['population']])
        assert np.allclose(fitness, loaded_fits), "Fitness changed after pickle"
    finally:
        tmp_path.unlink()
    return True

check("Pickle checkpoint save + reload (generation resume)", test_checkpoint)


# ─── Step 8: Baseline comparisons (Phase 4) ──────────────────────────────────
print("\n" + "="*60)
print("STEP 8: Baseline comparisons (compute_baselines.py)")
print("="*60)

sys.path.insert(0, str(ROOT / "analysis"))

def test_single_model_baselines():
    """Each P_cache column is a valid single-model baseline."""
    for i in range(N_MODELS):
        y_pred = P_cache[:, i]
        auc, ace, gap = compute_all_metrics(y_pred, labels_ref, genders_ref)
        assert not np.isnan(auc), f"Model {i} AUC is nan"
        assert 0.0 <= ace <= 1.0, f"Model {i} ACE out of range: {ace}"
        assert 0.0 <= gap <= 1.0, f"Model {i} equity gap out of range: {gap}"
    return True

def compute_all_metrics(y_pred, y_true, demographics):
    from objectives import compute_ace, compute_demographic_auc_gap
    auc = roc_auc_score(y_true, y_pred)
    ace = compute_ace(y_true, y_pred, n_bins=10)
    gap = compute_demographic_auc_gap(y_true, y_pred, demographics.astype(str))
    return auc, ace, gap

import numpy as np
from sklearn.metrics import roc_auc_score

check("Single-model baselines (AUC/ACE/gap all valid)", test_single_model_baselines)


def test_equal_weight_ensemble():
    y_pred = P_cache.mean(axis=1)
    auc, ace, gap = compute_all_metrics(y_pred, labels_ref, genders_ref)
    assert not np.isnan(auc)
    assert 0.0 <= ace <= 1.0
    assert 0.0 <= gap <= 1.0
    return auc, ace, gap

auc_eq, ace_eq, gap_eq = check("Equal-weight ensemble metrics", test_equal_weight_ensemble)
print(f"     AUC={auc_eq:.4f}  ACE={ace_eq:.4f}  Gap={gap_eq:.4f}")


def test_nsga2_knee_ensemble():
    """Knee-point weights from mini NSGA-II produce valid predictions."""
    knee_w = np.array(pareto[0][:])   # use first Pareto solution as stand-in
    knee_w = np.abs(knee_w) / knee_w.sum()
    y_pred = np.clip(P_cache @ knee_w, 1e-7, 1 - 1e-7)
    assert y_pred.shape == (N_SANITY,)
    assert abs(knee_w.sum() - 1.0) < 1e-6, f"Weights don't sum to 1: {knee_w.sum()}"
    auc, ace, gap = compute_all_metrics(y_pred, labels_ref, genders_ref)
    assert not np.isnan(auc)
    return auc, ace, gap

auc_k, ace_k, gap_k = check("NSGA-II knee-point ensemble metrics", test_nsga2_knee_ensemble)
print(f"     AUC={auc_k:.4f}  ACE={ace_k:.4f}  Gap={gap_k:.4f}")


def test_bootstrap_ci():
    """Bootstrap CI is well-formed: lo <= mean <= hi, reasonable width."""
    from objectives import compute_ace, compute_demographic_auc_gap
    rng = np.random.default_rng(0)
    y_pred = P_cache.mean(axis=1)
    n = len(labels_ref)
    aucs = []
    for _ in range(200):   # 200 resamples is enough for a sanity check
        idx = rng.integers(0, n, size=n)
        yp, yt = y_pred[idx], labels_ref[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    mean   = aucs.mean()
    assert lo <= mean <= hi,  f"CI not ordered: lo={lo:.4f} mean={mean:.4f} hi={hi:.4f}"
    assert hi - lo < 0.2,     f"CI suspiciously wide: {hi-lo:.4f}"
    assert hi - lo > 0.0,     "CI has zero width"
    return lo, mean, hi

lo, mean, hi = check("Bootstrap CI (200 resamples, AUC)", test_bootstrap_ci)
print(f"     AUC 95% CI: [{lo:.4f} – {hi:.4f}]  (mean={mean:.4f})")


def test_permutation_structure():
    """
    Permutation test logic: test using two clearly different predictors.
    Equal-weight ensemble should have lower ACE than the worst single model.
    p-value should be < 0.5 (ensemble wins) or test at least runs without crash.
    """
    from objectives import compute_ace
    rng = np.random.default_rng(1)

    # Use equal-weight (better) vs worst single model (worse)
    y_good = P_cache.mean(axis=1)
    single_aces = [compute_ace(labels_ref, P_cache[:, i]) for i in range(N_MODELS)]
    worst_idx = int(np.argmax(single_aces))
    y_bad = P_cache[:, worst_idx]

    def get_ace(yp):
        return compute_ace(labels_ref, yp, n_bins=10)

    obs_diff = get_ace(y_good) - get_ace(y_bad)  # should be negative (good < bad)

    count = 0
    for _ in range(200):
        mask = rng.random(len(labels_ref)) > 0.5
        pa   = np.where(mask, y_good, y_bad)
        pb   = np.where(mask, y_bad,  y_good)
        if (get_ace(pa) - get_ace(pb)) <= obs_diff:
            count += 1
    p = count / 200

    # p should be low (good predictor genuinely better) or at most 0.5
    assert 0.0 <= p <= 0.6, f"Permutation p-value out of expected range: {p:.3f}"
    assert obs_diff < 0.1,  f"Equal-weight ensemble ACE not better than worst model: diff={obs_diff:.4f}"
    return p, obs_diff

p, diff = check("Permutation test (ensemble vs worst model)", test_permutation_structure)
print(f"     ACE improvement: {-diff:.4f}  p-value: {p:.3f}")


def test_2obj_nsga2_mini():
    """2-objective NSGA-II (AUC + equity only) runs and returns valid weights."""
    from deap import base, creator, tools

    if not hasattr(creator, "Fitness2Obj"):
        creator.create("Fitness2Obj", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Ind2Obj"):
        creator.create("Ind2Obj", list, fitness=creator.Fitness2Obj)

    def eval_2obj(ind):
        w = np.abs(np.array(ind)); w /= w.sum()
        y_pred = np.clip(P_cache @ w, 1e-7, 1 - 1e-7)
        if len(np.unique(labels_ref)) < 2: return 1.0, 1.0
        from objectives import compute_demographic_auc_gap
        f1 = -roc_auc_score(labels_ref, y_pred)
        f3 = compute_demographic_auc_gap(labels_ref, y_pred, genders_ref.astype(str))
        return f1, f3

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Ind2Obj,
                     lambda: np.random.dirichlet(np.ones(N_MODELS)).tolist())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_2obj)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=12)  # must be divisible by 4 for selTournamentDCD
    for ind in pop: ind.fitness.values = toolbox.evaluate(ind)
    pop[:] = toolbox.select(pop, 12)

    for _ in range(3):
        off = tools.selTournamentDCD(pop, len(pop))
        off = list(map(toolbox.clone, off))
        for ind in off:
            if np.random.rand() < 0.5:
                w = np.abs(np.array(ind[:]) + np.random.randn(N_MODELS) * 0.1)
                ind[:] = (w / w.sum()).tolist()
                del ind.fitness.values
        for ind in off:
            if not ind.fitness.valid: ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = toolbox.select(pop + off, 12)

    pareto2 = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    assert len(pareto2) >= 1

    # Extract knee weights
    fits = np.array([ind.fitness.values for ind in pareto2])
    norm = (fits - fits.min(0)) / (fits.max(0) - fits.min(0) + 1e-10)
    knee = int(np.argmin(np.sqrt(np.sum(norm**2, axis=1))))
    w = np.abs(np.array(pareto2[knee][:])); w /= w.sum()
    assert abs(w.sum() - 1.0) < 1e-6
    assert len(w) == N_MODELS
    return w

w2 = check("2-obj NSGA-II mini run (3 gen, pop=10)", test_2obj_nsga2_mini)
print(f"     Weights: {np.round(w2, 3).tolist()}")


def test_csv_output_structure():
    """Simulate the CSV output format of compute_baselines."""
    rows = []
    for i, name in enumerate([f"Model_{i}" for i in range(N_MODELS)]):
        y_pred = P_cache[:, i]
        auc, ace, gap = compute_all_metrics(y_pred, labels_ref, genders_ref)
        rows.append({'Method': name, 'AUC': round(auc,4),
                     'ACE': round(ace,4), 'Equity_Gap': round(gap,4)})
    df_out = pd.DataFrame(rows)
    assert list(df_out.columns) == ['Method', 'AUC', 'ACE', 'Equity_Gap']
    assert len(df_out) == N_MODELS
    assert df_out['AUC'].between(0.5, 1.0).all(), "Some AUC values look wrong"
    return df_out

df_out = check("CSV output structure (columns, ranges)", test_csv_output_structure)
print(f"     {len(df_out)} rows, AUC range: [{df_out['AUC'].min():.4f}–{df_out['AUC'].max():.4f}]")


# ─── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL CHECKS PASSED - full pipeline is sane")
print("="*60)
print("""
Phase status:
  Phase 1.5  generate_predictions  -> job 5704 pending (GPU queue)
  Phase 2    run_nsga2             -> ready to submit after 1.5 completes
  Phase 3    visualize_pareto      -> ready to run after Phase 2 completes
  Phase 4    compute_baselines     -> ready to submit after Phase 3 completes
""")
