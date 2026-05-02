# train / test / webcam for asl finger-spelling images (see readme)

import os
import sys


def _env_truthy(name: str) -> bool:
    # is this env var "yes"? (1 / true / yes)
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _apply_argv_before_tf_import() -> None:
    # --cpu-only has to run before tensorflow wakes up; drop it from argv so argparse stays happy
    if "--cpu-only" not in sys.argv:
        return
    os.environ["ASL_TF_CPU_ONLY"] = "1"
    sys.argv = [a for a in sys.argv if a != "--cpu-only"]


def _darwin_reexec_for_tf_stability() -> None:
    # mac only: restart once with safer env vars so `import tensorflow` does not crash weirdly
    # normal path = still fast (gpu ok). "slow" path = ASL_TF_STABLE=1 or --cpu-only (cpu-ish, safer)
    # turn whole thing off with ASL_TF_MACOS_FIX=0
    if __name__ != "__main__":
        return
    if sys.platform != "darwin":
        return
    fix = os.environ.get("ASL_TF_MACOS_FIX", "1").strip().lower()
    if fix in ("0", "false", "no"):
        return
    if os.environ.get("_ASL_TF_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["_ASL_TF_REEXEC"] = "1"
    env["GRPC_POLL_STRATEGY"] = "poll"
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    slow_threading = _env_truthy("ASL_TF_STABLE") or _env_truthy("ASL_TF_CPU_ONLY")
    if slow_threading:
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["TF_NUM_INTEROP_THREADS"] = "1"
        env["TF_NUM_INTRAOP_THREADS"] = "1"
        env["TF_DISABLE_XNNPACK"] = "1"
    else:
        env.pop("TF_DISABLE_XNNPACK", None)

    if _env_truthy("ASL_TF_CPU_ONLY"):
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    script = os.path.abspath(__file__)
    argv = [sys.executable, script, *sys.argv[1:]]
    os.execve(sys.executable, argv, env)


_apply_argv_before_tf_import()
_darwin_reexec_for_tf_stability()

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


# where the clean cut-out hands live (train/ and test/ folders, one folder per letter or digit)
DEFAULT_DATA_DIR = Path("data/asl_processed")
DEFAULT_RAW_DIR = Path("data/asl_dataset")
DEFAULT_MODEL_DIR = Path("models/asl_model")


def _nonempty_str(flag: str):
    # Safety-check for linebreaks
    def _type(s: str) -> str:
        v = (s or "").strip()
        if not v:
            raise argparse.ArgumentTypeError(
                f"{flag} is empty. In zsh, a line break can leave `--data-dir` or `--model-dir` "
                "without a value; keep the path on the same line or use `\\` continuation."
            )
        return v

    return _type


@dataclass(frozen=True)
class ModelSpec:
    name: str
    image_size: int


# image size each model expects (width = height)
MODEL_SPECS: dict[str, ModelSpec] = {
    "efficientnetb0": ModelSpec(name="efficientnetb0", image_size=224),
    "mobilenetv2": ModelSpec(name="mobilenetv2", image_size=224),
}


def set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)


def configure_tensorflow_runtime() -> None:
    # cpu-only = hide gpus. otherwise let gpu memory grow instead of grabbing everything at once
    if _env_truthy("ASL_TF_CPU_ONLY"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    else:
        try:
            for dev in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(dev, True)
        except Exception:
            pass
    try:
        inter = int(os.environ.get("TF_NUM_INTEROP_THREADS", "0"))
        intra = int(os.environ.get("TF_NUM_INTRAOP_THREADS", "0"))
        if inter > 0:
            tf.config.threading.set_inter_op_parallelism_threads(inter)
        if intra > 0:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
    except Exception:
        pass


def log_compute_devices() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"TensorFlow GPU devices (Metal/CUDA): {[d.name for d in gpus]}")
    else:
        print("TensorFlow: no GPU visible — training on CPU (install `tensorflow-metal` on Apple Silicon for Metal).")


def maybe_set_mixed_precision(use_mixed_float16: bool) -> None:
    # optional speed trick on gpu (half precision math); skip if no gpu
    if not use_mixed_float16:
        return
    if not tf.config.list_physical_devices("GPU"):
        print("--mixed-float16 ignored: no GPU.")
        return
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Using global mixed_float16 policy (faster on GPU; head stays float32).")


def make_train_val_datasets(
    train_dir: Path,
    image_size: int,
    batch_size: int,
    seed: int,
    val_split: float,
    cache_in_memory: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    # Hold out part of train/ for callbacks; keep data/.../test/ untouched for honest final metrics
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )
    class_names = list(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )

    autotune = tf.data.AUTOTUNE
    if cache_in_memory:
        train_ds = train_ds.cache().prefetch(autotune)
        val_ds = val_ds.cache().prefetch(autotune)
    else:
        train_ds = train_ds.prefetch(autotune)
        val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def make_test_dataset(
    test_dir: Path,
    image_size: int,
    batch_size: int,
    cache_in_memory: bool = False,
) -> tf.data.Dataset:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    if cache_in_memory:
        return test_ds.cache().prefetch(autotune)
    return test_ds.prefetch(autotune)


def make_raw_split_datasets(
    raw_dir: Path,
    image_size: int,
    batch_size: int,
    seed: int,
    val_split: float,
    cache_in_memory: bool = False,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    # messy real-world photos: one folder per class, keras carves out a val slice for you
    raw_train = tf.keras.utils.image_dataset_from_directory(
        raw_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )
    raw_val = tf.keras.utils.image_dataset_from_directory(
        raw_dir,
        labels="inferred",
        label_mode="int",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )

    class_names = list(raw_train.class_names)
    autotune = tf.data.AUTOTUNE
    if cache_in_memory:
        raw_train = raw_train.cache().prefetch(autotune)
        raw_val = raw_val.cache().prefetch(autotune)
    else:
        raw_train = raw_train.prefetch(autotune)
        raw_val = raw_val.prefetch(autotune)
    return raw_train, raw_val, class_names


def build_model(
    arch: str,
    num_classes: int,
    image_size: int,
    dropout: float,
) -> tf.keras.Model:
    # image_dataset_from_directory yields float RGB in [0, 255]; applications include their own rescaling
    # no RandomFlip: horizontal mirror can swap visually similar letters in finger-spelling
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.RandomRotation(0.08)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    if arch == "efficientnetb0":
        core = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=x
        )
    elif arch == "mobilenetv2":
        core = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_tensor=x
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")

    core.trainable = False
    backbone = tf.keras.Model(inputs, core.output, name="backbone")
    backbone.trainable = False
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # last layer outputs class scores (probabilities after softmax)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def _configure_backbone_finetune(model: tf.keras.Model, freeze_first_n: int) -> None:
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_first_n


def compile_model(model: tf.keras.Model, lr: float) -> None:
    # adam is the default optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )


def save_artifacts(model_dir: Path, class_names: list[str], image_size: int, arch: str) -> None:
    # save little sidecar files so the notebook / webcam script know label order + input size
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    (model_dir / "meta.json").write_text(
        json.dumps(
            {
                "arch": arch,
                "image_size": image_size,
                "num_classes": len(class_names),
                "inputs_are_0_255": True,
            },
            indent=2,
        )
    )


def _keras_bundle_path(model_dir: Path) -> Path:
    return model_dir / "model.keras"


def load_artifacts(model_dir: Path) -> tuple[tf.keras.Model, list[str], dict]:
    model_dir = Path(model_dir)
    bundle = _keras_bundle_path(model_dir)
    if bundle.is_file():
        model = tf.keras.models.load_model(bundle)
    else:
        # Older layouts: whole directory or single file was passed to load_model
        model = tf.keras.models.load_model(model_dir)
    class_names = json.loads((model_dir / "class_names.json").read_text())
    meta = json.loads((model_dir / "meta.json").read_text())
    return model, class_names, meta


def train(args: argparse.Namespace) -> None:
    spec = MODEL_SPECS[args.arch]
    set_seed(args.seed)
    log_compute_devices()
    maybe_set_mixed_precision(args.mixed_float16)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.val_split < 1.0:
        raise ValueError("--val-split must be strictly between 0 and 1.")

    train_dir = Path(args.data_dir) / "train"
    test_dir = Path(args.data_dir) / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/test directories under {args.data_dir}. "
            f"Found train={train_dir.exists()} test={test_dir.exists()}"
        )

    train_ds, val_ds, class_names = make_train_val_datasets(
        train_dir=train_dir,
        image_size=spec.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
        cache_in_memory=args.cache_in_memory,
    )
    test_ds = make_test_dataset(
        test_dir=test_dir,
        image_size=spec.image_size,
        batch_size=args.batch_size,
        cache_in_memory=args.cache_in_memory,
    )

    model = build_model(
        arch=args.arch,
        num_classes=len(class_names),
        image_size=spec.image_size,
        dropout=args.dropout,
    )
    compile_model(model, lr=args.lr)

    # save best model, auto lower learning rate if stuck, stop early if nothing improves
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "ckpt.keras"),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            mode="max",
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.reduce_lr_min,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", mode="max", patience=5, restore_best_weights=True
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    phase_histories = [history]

    if args.fine_tune:
        # let more of the pretrained net learn, but gently (lower learning rate)
        _configure_backbone_finetune(model, args.fine_tune_at)

        compile_model(model, lr=args.fine_tune_lr)
        h_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        phase_histories.append(h_ft)

    if args.raw_finetune:
        # bonus round on messier photos so the model survives real life a bit better
        raw_dir = Path(args.raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw dir not found: {raw_dir}")

        raw_train, raw_val, raw_class_names = make_raw_split_datasets(
            raw_dir=raw_dir,
            image_size=spec.image_size,
            batch_size=args.raw_batch_size,
            seed=args.seed,
            val_split=args.raw_val_split,
            cache_in_memory=args.cache_in_memory,
        )
        if raw_class_names != class_names:
            raise RuntimeError(
                "Class order mismatch between processed and raw datasets.\n"
                f"processed={class_names}\nraw={raw_class_names}"
            )

        _configure_backbone_finetune(model, args.raw_finetune_at)

        compile_model(model, lr=args.raw_lr)
        h_raw = model.fit(
            raw_train,
            validation_data=raw_val,
            epochs=args.raw_epochs,
            callbacks=callbacks,
        )
        phase_histories.append(h_raw)

    save_artifacts(model_dir, class_names, spec.image_size, args.arch)
    model.save(_keras_bundle_path(model_dir))

    best_vals = [
        float(np.max(h.history.get("val_acc", [float("nan")]))) for h in phase_histories
    ]
    best_overall = max(best_vals) if best_vals else float("nan")
    print(f"Saved model to: {model_dir}")
    print(f"Best val_acc (per phase): {[f'{v:.4f}' for v in best_vals]}")
    print(f"Best val_acc (overall): {best_overall:.4f}")
    test_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    print(f"Held-out test: {test_metrics}")


def evaluate(args: argparse.Namespace) -> None:
    model, class_names, meta = load_artifacts(Path(args.model_dir))
    image_size = int(meta["image_size"])

    test_dir = Path(args.data_dir) / "test"
    test_ds = make_test_dataset(
        test_dir=test_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        cache_in_memory=False,
    )

    results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    print(json.dumps({"metrics": metrics, "num_classes": len(class_names)}, indent=2))


def _center_square_crop(img: np.ndarray) -> np.ndarray:
    # dumb but ok fallback: grab the middle square of the frame
    h, w = img.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0 : y0 + s, x0 : x0 + s]


def _try_mediapipe_hand_crop(bgr: np.ndarray) -> np.ndarray | None:
    # try to auto-crop just the hand; returns none if mediapipe does not see a hand
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None

    rgb = bgr[:, :, ::-1]
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0].landmark
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        h, w = bgr.shape[:2]
        pad = 0.25
        x1 = max(0.0, x1 - pad)
        y1 = max(0.0, y1 - pad)
        x2 = min(1.0, x2 + pad)
        y2 = min(1.0, y2 + pad)
        xa, xb = int(x1 * w), int(x2 * w)
        ya, yb = int(y1 * h), int(y2 * h)
        if xb <= xa or yb <= ya:
            return None
        return bgr[ya:yb, xa:xb]


def realtime(args: argparse.Namespace) -> None:
    import cv2

    model, class_names, meta = load_artifacts(Path(args.model_dir))
    image_size = int(meta["image_size"])

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    last_pred = None
    last_ts = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop = None
        if args.use_mediapipe:
            crop = _try_mediapipe_hand_crop(frame)
        if crop is None:
            crop = _center_square_crop(frame)

        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        # BGR -> RGB; match training pixel scale ([0,255] float for new models, /255 for older checkpoints)
        rgb = crop[:, :, ::-1].astype(np.float32)
        if meta.get("inputs_are_0_255"):
            x = rgb
        else:
            x = rgb / 255.0
        x = np.expand_dims(x, axis=0)

        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = class_names[idx]
        conf = float(probs[idx])

        # do not repaint the text every single frame (looks smoother)
        now = time.time()
        if now - last_ts >= 0.05:
            last_pred = (label, conf)
            last_ts = now

        disp = frame.copy()
        if last_pred is not None:
            text = f"{last_pred[0]}  ({last_pred[1]*100:.1f}%)"
            cv2.putText(
                disp,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("realtime asl", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASL training + evaluation + realtime inference")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train a classifier on processed images")
    p_train.add_argument("--data-dir", type=_nonempty_str("--data-dir"), default=str(DEFAULT_DATA_DIR))
    p_train.add_argument("--raw-dir", type=_nonempty_str("--raw-dir"), default=str(DEFAULT_RAW_DIR))
    p_train.add_argument("--model-dir", type=_nonempty_str("--model-dir"), default=str(DEFAULT_MODEL_DIR))
    p_train.add_argument("--arch", type=str, choices=sorted(MODEL_SPECS.keys()), default="efficientnetb0")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-step batch size (GPU/Metal can often use 64+; lower if OOM).",
    )
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=2,
        help="Epochs with no val_acc improvement before ReduceLROnPlateau cuts LR.",
    )
    p_train.add_argument("--reduce-lr-factor", type=float, default=0.5)
    p_train.add_argument("--reduce-lr-min", type=float, default=1e-7)
    p_train.add_argument("--dropout", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of train/ images used for validation (callbacks); test/ stays held out.",
    )
    p_train.add_argument("--fine-tune", action="store_true")
    p_train.add_argument("--fine-tune-at", type=int, default=200)
    p_train.add_argument("--fine-tune-epochs", type=int, default=8)
    p_train.add_argument("--fine-tune-lr", type=float, default=1e-5)
    p_train.add_argument(
        "--raw-finetune",
        action="store_true",
        help="Option A: fine-tune on raw images after processed training",
    )
    p_train.add_argument("--raw-epochs", type=int, default=5)
    p_train.add_argument("--raw-batch-size", type=int, default=64)
    p_train.add_argument(
        "--cache-in-memory",
        action="store_true",
        help="Cache train/val/test tensors in RAM after epoch 1 (faster later epochs; needs lots of RAM).",
    )
    p_train.add_argument(
        "--mixed-float16",
        action="store_true",
        help="Use mixed precision on GPU (Metal/CUDA) for faster training.",
    )
    p_train.add_argument("--raw-lr", type=float, default=5e-5)
    p_train.add_argument("--raw-val-split", type=float, default=0.2)
    p_train.add_argument("--raw-finetune-at", type=int, default=200)
    p_train.set_defaults(func=train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model on the test split")
    p_eval.add_argument("--data-dir", type=_nonempty_str("--data-dir"), default=str(DEFAULT_DATA_DIR))
    p_eval.add_argument("--model-dir", type=_nonempty_str("--model-dir"), default=str(DEFAULT_MODEL_DIR))
    p_eval.add_argument("--batch-size", type=int, default=64)
    p_eval.set_defaults(func=evaluate)

    p_rt = sub.add_parser("realtime", help="Run realtime webcam inference")
    p_rt.add_argument("--model-dir", type=_nonempty_str("--model-dir"), default=str(DEFAULT_MODEL_DIR))
    p_rt.add_argument("--camera", type=int, default=0)
    p_rt.add_argument("--use-mediapipe", action="store_true")
    p_rt.set_defaults(func=realtime)

    return p


def main() -> None:
    configure_tensorflow_runtime()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    # 2 = less tensorflow spam in the terminal
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
