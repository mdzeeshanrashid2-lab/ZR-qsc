import tensorflow as tf
import math
import numpy as np
import pennylane as qml

# =========================================================
# Channel generators (kept)
# =========================================================

def channel_Rayleigh(shape):
    sigma = math.sqrt(1/2)
    return tf.random.normal(shape, 0.0, sigma , dtype=tf.float32)

def channel_Rician(shape, K):
    mean = math.sqrt(K / (K + 1))
    std = math.sqrt(1 / (K + 1))
    return tf.random.normal(shape, mean, std, dtype=tf.float32)


# =========================================================
# STABLE CHANNEL LAYER (FIXED DTYPES)
# =========================================================

def channel_layer(x_norm, AWGN_flag, Rayleigh_flag, Rician_flag, std):
    """
    Realistic wireless channel 

    x_norm : [B, T, 2]  → IQ symbols
    std    : noise std derived from SNR
    """

    eps = tf.constant(1e-6, dtype=tf.float32)
    B = tf.shape(x_norm)[0]
    T = tf.shape(x_norm)[1]

    # ================= IQ → Complex =================
    x_complex = tf.complex(x_norm[..., 0], x_norm[..., 1])  # complex64

    # ================= Channel Fading =================
    if Rayleigh_flag[0] == 1:
        # Rayleigh fading: complex Gaussian
        h_real = tf.random.normal([B, T], stddev=tf.sqrt(0.5))
        h_imag = tf.random.normal([B, T], stddev=tf.sqrt(0.5))
        h = tf.complex(h_real, h_imag)

    elif Rician_flag[0] == 1:
        # Rician fading: LOS + scattered
        K = 2.0  # Rician factor
        mean = tf.sqrt(K / (K + 1.0))
        std_h = tf.sqrt(1.0 / (K + 1.0))
        h_real = tf.random.normal([B, T], mean=mean, stddev=std_h)
        h_imag = tf.random.normal([B, T], stddev=std_h)
        h = tf.complex(h_real, h_imag)

    else:
        # AWGN channel → no fading
        h = tf.complex(tf.ones([B, T]), tf.zeros([B, T]))

    # ================= Additive Noise (ALWAYS present) =================
    # Noise exists for AWGN, Rayleigh, and Rician
    n_real = tf.random.normal([B, T], stddev=std[0])
    n_imag = tf.random.normal([B, T], stddev=std[0])
    noise = tf.complex(n_real, n_imag)

    # ================= Channel Output =================
    y = h * x_complex + noise
    
    
    # ---------------- Equalization (dtype-safe) ----------------
    h_power = tf.abs(h) ** 2                     # float32
    denom = tf.cast(h_power + eps, tf.complex64) # complex64
    x_hat = y* tf.math.conj(h) / denom

    # Back to IQ
    x_hat = tf.stack(
        [tf.math.real(x_hat), tf.math.imag(x_hat)],
        axis=-1
    )

    # IMPORTANT: stop gradients through channel
    return x_hat
#-----------global quantum setup------------
n_qubits = 8

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(n_qubits)
    )

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(n_qubits)
    ]


# =========================================================
# SE Residual Block (stable)
# =========================================================

class SEResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, kernel_size, dilation_rate, dropout_rate):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv1D(
            n_filters, kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            use_bias=False
        )
        self.ln1 = tf.keras.layers.LayerNormalization()

        self.conv2 = tf.keras.layers.Conv1D(
            n_filters, kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu',
            use_bias=False
        )
        self.ln2 = tf.keras.layers.LayerNormalization()

        self.dropout = tf.keras.layers.SpatialDropout1D(dropout_rate)

        reduced = max(n_filters // 8, 1)
        self.se_reduce = tf.keras.layers.Conv1D(reduced, 1, activation='relu')
        self.se_expand = tf.keras.layers.Conv1D(n_filters, 1, activation='sigmoid')

        self.res_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.conv1.filters:
            self.res_conv = tf.keras.layers.Conv1D(self.conv1.filters, 1)
        super().build(input_shape)

    def call(self, x, training=None):
        y = self.conv1(x)
        y = self.ln1(y)
        y = self.dropout(y, training=training)

        y = self.conv2(y)
        y = self.ln2(y)

        se = tf.reduce_mean(y, axis=1, keepdims=True)
        se = self.se_reduce(se)
        se = self.se_expand(se)
        y = y * se

        if self.res_conv is not None:
            x = self.res_conv(x)

        return x + y


# =========================================================
# ASR MODEL (FULLY FIXED)
# =========================================================

class ASR_model(tf.keras.Model):

    def __init__(self, args, num_classes):
        super().__init__()

        self.num_channel_units = args.num_channel_units
        assert self.num_channel_units % 2 == 0

        self.tcn_layers = [
            SEResidualBlock(
                args.tcn_filters,
                args.tcn_kernel_size,
                2**i,
                args.dropout_rate
            )
            for i in range(args.num_tcn_layers)
        ]

        self.tcn_final_conv = tf.keras.layers.Conv1D(args.tcn_filters, 1)
        self.tcn_ln = tf.keras.layers.LayerNormalization()

        self.fc1 = tf.keras.layers.Dense(args.num_channel_units, activation="relu")

        self.enc1 = tf.keras.layers.Dense(args.num_channel_units, activation="relu")
        self.enc2 = tf.keras.layers.Dense(args.num_channel_units)
        
        # Quantum Semantic Block

        self.quantum_projection = tf.keras.layers.Dense(8)

        weight_shapes = {
        "weights": (4, 8, 3)
        }

        self.quantum_layer = qml.qnn.KerasLayer(
            quantum_circuit,
            weight_shapes,
            output_dim=8
        )

        self.quantum_expand = tf.keras.layers.Dense(
            args.num_channel_units
        )

        self.dec1 = tf.keras.layers.Dense(args.num_channel_units, activation="relu")
        self.dec2 = tf.keras.layers.Dense(args.num_channel_units, activation="relu")

        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, features_inputs,
             AWGN_flag, Rayleigh_flag, Rician_flag, std,
             training=None):

        B = tf.shape(features_inputs)[0]
        x = features_inputs

        # TCN
        for block in self.tcn_layers:
            x = block(x, training=training)

        x = self.tcn_final_conv(x)
        x = self.tcn_ln(x)

        x = self.fc1(x)

        enc = self.enc1(x)
        enc = self.enc2(enc)

     # ==================================
     # Quantum Semantic Encoder
     # ==================================

        shape_enc = tf.shape(enc)

        flat = tf.reshape(
            enc,
            [-1, self.num_channel_units]
        )

        flat = self.quantum_projection(flat) 
        print("Quantum input shape =", flat.shape)    #11111111

        #flat = self.quantum_layer(flat)

        flat = self.quantum_expand(flat)

        enc = tf.reshape(
            flat,
            shape_enc
        )

        # IQ reshape
        x = tf.reshape(enc, [B, -1, 2])

        # -------- POWER NORMALIZATION (FIXED) --------
        power = tf.reduce_mean(tf.reduce_sum(x**2, axis=-1), axis=1, keepdims=True)
        power = tf.reshape(power, [-1, 1, 1])
        x_norm = x / tf.sqrt(power + 1e-6)

        # Channel
        x_hat = channel_layer(
            x_norm,
            AWGN_flag,
            Rayleigh_flag,
            Rician_flag,
            std
        )

        # Decoder
        x_hat = tf.reshape(x_hat, [B, -1, self.num_channel_units])
        dec = self.dec1(x_hat)
        dec = self.dec2(dec)

        pooled = tf.reduce_mean(dec, axis=1)
        return self.classifier(pooled)