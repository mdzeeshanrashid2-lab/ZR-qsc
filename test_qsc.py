"""
Pure test script for ASR Semantic Communication with Emotion Classification (TCN Version).
Loads best trained model and evaluates across different SNR levels.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

from qsc_model import ASR_model

# =================== Args ===================
def parse_args():
    parser = argparse.ArgumentParser(description="Test TCN ASR + Emotion Classification")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame_size", type=float, default=0.025)
    parser.add_argument("--stride_size", type=float, default=0.010)
    parser.add_argument("--num_mels", type=int, default=64)
    parser.add_argument("--num_frame", type=int, default=198)
    parser.add_argument("--num_tcn_layers", type=int, default=8)
    parser.add_argument("--tcn_filters", type=int, default=128)
    parser.add_argument("--tcn_kernel_size", type=int, default=7)
    parser.add_argument("--dropout_rate", type=float, default=0.25)
    parser.add_argument("--num_channel_units", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_records", type=str,default="/mnt/DATA/EE24M311/DeepSC-R/NEW_TCN_MODEL/Shikha/Full_MSP_records_Frame198/testset.tfrecords")
    parser.add_argument("--logdir", type=str,default="/mnt/DATA/EE24M311/New_MSP/old_model/logs_model2_new/logs_model2_test")
    parser.add_argument("--best_model", type=str,default="/mnt/DATA/EE24M311/New_MSP/old_model/logs_model2_new/best_asr_model_SNR15.ckpt")
    return parser.parse_args()

# =================== TFRecord Dataset ===================
def map_function(example, num_frame, num_mels):
    feature_map = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    spec_flat = tf.io.decode_raw(parsed_example["audio"], out_type=tf.float32)
    spec = tf.reshape(spec_flat, (num_frame, num_mels))
    label = tf.cast(parsed_example["label"], tf.int32)
    return spec, label

def make_dataset(path, batch_size, num_frame, num_mels):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(lambda x: map_function(x, num_frame, num_mels),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# =================== MAIN ===================
if __name__ == "__main__":
    args = parse_args()

    # ✅ FIX: ensure log directory exists
    os.makedirs(args.logdir, exist_ok=True)

    # GPU setup
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load dataset
    testset = make_dataset(args.test_records, args.batch_size,
                           args.num_frame, args.num_mels)

    # Initialize model
    num_emotions = 8
    asr_model_net = ASR_model(args, num_classes=num_emotions)

    # ✅ FIX: load weights directly from checkpoint path
    asr_model_net.load_weights(args.best_model)
    print(f"\n✅ Loaded trained model from {args.best_model}")

    # =================== Select Channel ===================
    print("\nSelect channel type for testing:")
    choice = input("Enter choice (1=AWGN, 2=Rayleigh, 3=Rician): ").strip()

    # ✅ FIX: use scalars instead of lists
    if choice == "1":
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [1.0], [0.0], [0.0]
    elif choice == "2":
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [0.0], [1.0], [0.0]
    elif choice == "3":
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [0.0], [0.0], [1.0]

    else:
        print("⚠️ Invalid choice. Defaulting to AWGN.")
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [1.0], [0.0], [0.0]

    # =================== SNR Evaluation ===================
    # ✅ FIX: more realistic SNR sweep (edit if needed)
    snr_values = [-10, -5, 0, 5, 10, 15, 20, 30]

    emotion_labels = ["Neutral", "Happy", "Angry", "Sad",
                      "Fear", "Disgust", "Surprise", "Contempt"]

    metrics_df = pd.DataFrame(columns=[
        "SNR", "Accuracy", "Weighted Accuracy",
        "Weighted Precision", "Weighted Recall", "Weighted F1",
        "Macro Precision", "Macro Recall", "Macro F1", "Micro F1"
    ])

    for snr in snr_values:
        print(f"\n=== Confusion Matrix & Metrics @ {snr} dB ===")

        # noise std
        snr_linear = 10 ** (snr / 10)
        std_test = tf.constant(np.sqrt(1 / (2 * snr_linear)), dtype=tf.float32)

        y_true, y_pred = [], []

        for features, labels in testset:
            probs = asr_model_net(
                features,
                AWGNtest_flag, Rayleightest_flag, Riciantest_flag,
                std=[std_test],
                training=False
            )

            #✅ Optional sanity check (runs once)
            # print("Output shape:", probs.shape)

            preds = tf.argmax(probs, axis=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.numpy().tolist())

        # ================= Confusion Matrix =================
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix @ {snr} dB")
        plt.savefig(os.path.join(args.logdir, f"cm_{snr}dB.png"))
        plt.close()

        # Normalized CM
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), cm_sum,
                            out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm,
                                           display_labels=emotion_labels)
        disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f")
        plt.title(f"Normalized Confusion Matrix @ {snr} dB")
        plt.savefig(os.path.join(args.logdir, f"cm_norm_{snr}dB.png"))
        plt.close()

        # ================= Metrics =================
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        weighted_acc = recall_weighted

        print(f"Accuracy        : {accuracy:.4f}")
        print(f"Weighted Acc    : {weighted_acc:.4f}")
        print(f"Weighted Prec   : {precision_weighted:.4f}")
        print(f"Weighted Recall : {recall_weighted:.4f}")
        print(f"Weighted F1     : {f1_weighted:.4f}")
        print(f"Macro Prec      : {precision_macro:.4f}")
        print(f"Macro Recall    : {recall_macro:.4f}")
        print(f"Macro F1        : {f1_macro:.4f}")
        print(f"Micro F1        : {f1_micro:.4f}")

        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            "SNR": snr,
            "Accuracy": accuracy,
            "Weighted Accuracy": weighted_acc,
            "Weighted Precision": precision_weighted,
            "Weighted Recall": recall_weighted,
            "Weighted F1": f1_weighted,
            "Macro Precision": precision_macro,
            "Macro Recall": recall_macro,
            "Macro F1": f1_macro,
            "Micro F1": f1_micro,
        }])], ignore_index=True)

    # Save metrics
    metrics_csv_path = os.path.join(args.logdir, "snr_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n📊 All SNR metrics saved to {metrics_csv_path}")
