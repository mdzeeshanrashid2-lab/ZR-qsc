from __future__ import absolute_import, division, print_function
import os, time, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)


from qsc_model import ASR_model  # TCN-based model

# GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

############### define global parameters ####
def parse_args():
    parser = argparse.ArgumentParser(description="ASR Semantic Communication with Emotion Classification (TCN Version)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame_size", type=float, default=0.025)
    parser.add_argument("--stride_size", type=float, default=0.010) #Changed by Anirban
    parser.add_argument("--window_size", type=int, default= 31920 ) #Ignore this
    parser.add_argument("--num_tcn_layers", type=int, default=6) 
    parser.add_argument("--tcn_filters", type=int, default=128)  
    parser.add_argument("--tcn_kernel_size", type=int, default=7)  # Previous  7
    parser.add_argument("--dropout_rate", type=float, default=0.25)  # Previous 0.25, keep it 0.25 decided by Shikha and Anirban
    parser.add_argument("--num_channel_units", type=int, default=40) #Here we have 8 classes, so any number above 8 is fine
    parser.add_argument("--batch_size", type=int, default=64) # we made changes from 64 to 8
    parser.add_argument("--num_epochs", type=int, default=100) #Changed by Anirban from 30 to 100, early stopping at 15
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
    "--train_records",
    type=str,
    default=r"D:\MSP\TFRecords\trainset.tfrecords"
    )

    parser.add_argument(
    "--valid_records",
    type=str,
    default=r"D:\MSP\TFRecords\validset.tfrecords"
    )

    parser.add_argument(
    "--test_records",
    type=str,
    default=r"D:\MSP\TFRecords\testset.tfrecords"
    )
    parser.add_argument(
    "--logdir",
    type=str,
    default=r"D:\zrqsc\logs_qsc"
    )
    parser.add_argument("--num_mels", type=int, default=64)
    parser.add_argument("--num_frame", type=int, default=198)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    frame_length = int(args.sr * args.frame_size)
    stride_length = int(args.sr * args.stride_size)
    
    window_size = args.num_frame * stride_length + frame_length - stride_length
    
    num_frame = (window_size - frame_length) // stride_length + 1
    if num_frame != args.num_frame:
        print(f"[Warning] computed num_frame={num_frame} != args.num_frame={args.num_frame}. Using computed value.")
        args.num_frame = num_frame

    print(f"Using window_size={window_size}, frame_length={frame_length}, stride_length={stride_length}, num_frame={args.num_frame}, num_mels={args.num_mels}")

    num_emotions = 8

    # Initialize TCN model
    asr_model_net = ASR_model(args, num_classes=8)
    #classifier = tf.keras.layers.Dense(num_emotions, activation="softmax", name="emotion_classifier")
    # Build model
    # =====================================================
    # Build model & print summary (FULL ASR MODEL)
    # =====================================================
    dummy_input = tf.zeros([1, args.num_frame, args.num_mels])

    # Dummy channel settings
    AWGN_flag = [0.0]
    Rayleigh_flag = [1.0]
    Rician_flag = [0.0]
    std = [tf.constant(0.1, dtype=tf.float32)]

    # Build model
    asr_model_net(dummy_input, AWGN_flag, Rayleigh_flag, Rician_flag, std)

    print("\n🔎 ASR Model Summary\n")
    asr_model_net.summary()

    # Parameter counts
    trainable = np.sum([np.prod(v.shape) for v in asr_model_net.trainable_variables])
    non_trainable = np.sum([np.prod(v.shape) for v in asr_model_net.non_trainable_variables])

    print(f"\nTrainable params: {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")


    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()  # no label_smoothing

    # Channel selection
    print("Select channel type for training:")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        AWGN_flag, Rayleigh_flag, Rician_flag = [1.0], [0.0], [0.0]
    elif choice == "2":
        AWGN_flag, Rayleigh_flag, Rician_flag = [0.0], [1.0], [0.0]
    elif choice == "3":
        AWGN_flag, Rayleigh_flag, Rician_flag = [0.0], [0.0], [1.0]
    else:
        AWGN_flag, Rayleigh_flag, Rician_flag = [1.0], [0.0], [0.0]

    print("Select channel type for testing:")
    choice = input("Enter choice (1/2/3): ").strip()
    if choice == "1":
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [1.0], [0.0], [0.0]
    elif choice == "2":
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [0.0], [1.0], [0.0]
    elif choice == "3":
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [0.0], [0.0], [1.0]
    else:
        AWGNtest_flag, Rayleightest_flag, Riciantest_flag = [1.0], [0.0], [0.0]

    ############### Dataset parsing ###############
    AUTOTUNE = tf.data.AUTOTUNE

    def map_function(example):
        feature_map = {
            "audio": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example, features=feature_map)
        spec_flat = tf.io.decode_raw(parsed_example["audio"], out_type=tf.float32)
        spec = tf.reshape(spec_flat, (args.num_frame, args.num_mels))
        spec = tf.cast(spec, tf.float32)
        label = tf.cast(parsed_example["label"], tf.int32)  # Fix: int32
        return spec, label

    def make_dataset(path, batch_size, shuffle=False):
        ds = tf.data.TFRecordDataset(path)
        ds = ds.map(map_function, num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    trainset = make_dataset(args.train_records, args.batch_size, shuffle=True)
    validset = make_dataset(args.valid_records, args.batch_size)
    testset  = make_dataset(args.test_records, args.batch_size)

# DEBUG MODE   #we add it cut karenge baad mei
    #trainset = trainset.take(10)
   # validset = validset.take(2)
    #testset  = testset.take(2)

    ############### Training functions ###############
    @tf.function
    def train_step(features, labels, std):
        with tf.GradientTape() as tape:
            logits = asr_model_net(features, AWGN_flag, Rayleigh_flag, Rician_flag, std=[std], training=True)
            #pooled = tf.reduce_mean(latent, axis=1)
            #logits = classifier(pooled)
            loss_value = loss_object(labels, logits)
        grads = tape.gradient(loss_value, asr_model_net.trainable_variables )
        optimizer.apply_gradients(zip(grads, asr_model_net.trainable_variables ))

        # accuracy (labels are integers)
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.argmax(logits, axis=-1), tf.int32), labels), tf.float32)
        )

        return loss_value, acc

    @tf.function
    def valid_step(features, labels, std):
        logits = asr_model_net(features, AWGN_flag, Rayleigh_flag, Rician_flag, std=[std], training=False)
        #pooled = tf.reduce_mean(latent, axis=1)
        #logits = classifier(pooled)
        #print('labels: ', labels[0], 'logits: ',logits[0])
        loss_value = loss_object(labels, logits)

        acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(tf.argmax(logits, axis=-1), tf.int32), labels), tf.float32))
        
        return loss_value, acc

    ############### Training loop with gradual SNR ###############
    snr_schedule = [15] #Do with high SNR now by Anirban
    snr_epochs   = [200] # currently we  r working with 5 instead of 200  200 karna h
    train_loss_all, valid_loss_all = [], []
    train_acc_all, valid_acc_all = [], []
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    lr_factor = 0.5
    min_lr = 1e-6

    for snr_input, n_epoch_stage in zip(snr_schedule, snr_epochs):
        snr = 10 ** (snr_input / 10)
        std_train = tf.constant(np.sqrt(1 / (2*snr)), dtype=tf.float32)
        std_valid = std_train

        for epoch in range(n_epoch_stage):
            start = time.time()
            train_losses, train_accuracies = [], []
            train_steps = sum(1 for _ in trainset)
            tqdm_train = tqdm(trainset.repeat(1), total=train_steps,
                              desc=f"SNR {snr_input}dB Epoch {epoch+1}/{n_epoch_stage} - Training", ncols=100)
            for features, labels in tqdm_train:
                # print("Shape:",features.shape)
                loss, acc = train_step(features, labels, std_train)
                train_accuracies.append(float(acc))
                train_losses.append(loss)

            train_loss = np.mean(train_losses)
            train_acc = np.mean(train_accuracies)
            train_loss_all.append(train_loss)
            train_acc_all.append(train_acc)
            
            tqdm.write(f"SNR {snr_input}dB Epoch {epoch+1}: train_loss={train_loss:.6f},train_acc={train_acc:.4f}, time={(time.time()-start):.2f}s")

            # VALIDATION
            valid_losses, valid_accuracies = [], []
            valid_steps = sum(1 for _ in validset)
            tqdm_valid = tqdm(validset.repeat(1), total=valid_steps,
                              desc=f"SNR {snr_input}dB Epoch {epoch+1}/{n_epoch_stage} - Validation", ncols=100)
            for features, labels in tqdm_valid:
                loss, acc = valid_step(features, labels, std_valid)
                loss, acc = float(loss), float(acc)
                valid_losses.append(loss)
                valid_accuracies.append(acc)
            valid_loss = np.mean(valid_losses)
            valid_acc = np.mean(valid_accuracies)
            valid_loss_all.append(valid_loss)
            valid_acc_all.append(valid_acc)


            tqdm.write(f"SNR {snr_input}dB Epoch {epoch+1}: valid_loss={valid_loss:.6f}, valid_acc={valid_acc:.4f}, time={(time.time()-start):.2f}s")

            # Save best model changed by Anirban for dynamic learning rate 
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                patience_counter = 0
                tqdm.write(f"Epoch {epoch+1}: New best model saved (val_loss={valid_loss:.6f})")
                asr_model_net.save_weights(os.path.join(args.logdir, f"best_asr_model_SNR{snr_input}.ckpt"))
                # asr_model_net.save(os.path.join(args.logdir, "best_asr_model.h5"))
                
            else:
                patience_counter += 1
                current_lr = float(tf.keras.backend.get_value(optimizer.lr))
                tqdm.write(f"No improvement. Patience {patience_counter}/{patience}")
    
                if patience_counter >= patience:
                     tqdm.write("Early stopping triggered!")
                     break
                elif current_lr > min_lr:
                # Reduce LR if not already too small
                        new_lr = max(round(current_lr * lr_factor,7), min_lr)
                        tf.keras.backend.set_value(optimizer.lr, new_lr)
                        tqdm.write(f"Reducing learning rate to {new_lr:.2e}")
                        # patience_counter = 0  # Reset patience after LR change
                    
                       

            """ if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                patience_counter = 0
                os.makedirs(args.logdir, exist_ok=True)
                asr_model_net.save_weights(os.path.join(args.logdir, f"best_asr_model_SNR{snr_input}.ckpt"))
                tqdm.write(f"Epoch {epoch+1}: New best model saved (val_loss={valid_loss:.6f})")
            else:
                patience_counter += 1
                tqdm.write(f"No improvement. Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    tqdm.write("Early stopping triggered!")
                    break  """
 
    
      # Save loss and accuracy curves
    os.makedirs(args.logdir, exist_ok=True)

    # Save .mat files
    sio.savemat(os.path.join(args.logdir, "train_loss.mat"), {"train_loss": np.array(train_loss_all, dtype=np.float32)})
    sio.savemat(os.path.join(args.logdir, "valid_loss.mat"), {"valid_loss": np.array(valid_loss_all, dtype=np.float32)})
    sio.savemat(os.path.join(args.logdir, "train_acc.mat"),  {"train_acc":  np.array(train_acc_all, dtype=np.float32)})
    sio.savemat(os.path.join(args.logdir, "valid_acc.mat"),  {"valid_acc":  np.array(valid_acc_all, dtype=np.float32)})

    # Plot
    epochs = range(1, len(train_loss_all) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # ---- Loss subplot ----h_power = tf.abs(h) ** 2                     # float32
    # denom = tf.cast(h_power + eps, tf.complex64) # complex64
    axs[0].plot(epochs, train_loss_all, label="Training Loss", marker='o')
    axs[0].plot(epochs, valid_loss_all, label="Validation Loss", marker='s')
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].legend()
    axs[0].grid(True)

    # ---- Accuracy subplot ----
    axs[1].plot(epochs, train_acc_all, label="Training Accuracy", marker='o')
    axs[1].plot(epochs, valid_acc_all, label="Validation Accuracy", marker='s')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.logdir, "loss_acc_curves.png"))
    #plt.show()


    # ###############    FINAL TEST AFTER TRAINING    ###############
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

    # Load best weights
asr_model_net.load_weights(os.path.join(args.logdir, "best_asr_model_SNR15.ckpt"))

print("\n✅ Best model weights loaded for final testing")

    # Reduce test batch size to avoid memory issues
testset_small = testset.unbatch().batch(64)  # small batch size changed by Anirban from 8 to 32

    # Define SNR values for evaluation
snr_values = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

#  FOR MSP TESTING
emotion_labels = ["Neutral", "Happy", "Angry", "Sad", "Fear", "Disgust", "Surprise", "Contempt"]

# FOR RAVDESS mapping
#emotion_labels = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]


    # Prepare dataframe to store metrics
metrics_df = pd.DataFrame(columns=["SNR", "Accuracy", "Precision", "Recall", "F1"])

for snr in snr_values:
    print(f"\n=== Confusion Matrix & Metrics @ {snr} dB ===")

        # Convert SNR → std deviation
    std_test = tf.constant(
        np.sqrt(1 / (2*pow(10, (snr/10)))),
            dtype=tf.float32
        )

    y_true, y_pred = [], []

    for features, labels in testset_small:
            # features: [batch, num_frame, num_mels]
        logits = asr_model_net(features,
                                   AWGNtest_flag, Rayleightest_flag, Riciantest_flag,
                                   std=[std_test], training=False)
        #pooled = tf.reduce_mean(logits, axis=1)
        #preds = tf.argmax(classifier(logits), axis=1)
        preds = tf.argmax(logits, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.numpy().tolist())

        # Print first 10 predictions for sanity check
    print("Sample predictions:", y_pred[:10])
    print("Sample true labels:", y_true[:10])

        # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix @ {snr} dB")
    plt.savefig(os.path.join(args.logdir, f"cm_{snr}dB.png"))
    plt.close()

        # Normalized confusion matrix
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=emotion_labels)
    disp_norm.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title(f"Normalized Confusion Matrix @ {snr} dB")
    plt.savefig(os.path.join(args.logdir, f"cm_norm_{snr}dB.png"))
    plt.close()

        # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

        # Save metrics
    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            "SNR": snr,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }])], ignore_index=True)

    # Save all metrics to CSV
metrics_csv_path = os.path.join(args.logdir, "snr_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nAll SNR metrics saved to {metrics_csv_path}")

