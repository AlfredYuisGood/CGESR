# This file includes the demonstration of GRUs, attention score, softmask, and back-door adjustment

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Define the GRU-based item representation function
def gru_item_representation(x):
    gru = layers.GRU(units=d, return_sequences=True)
    z = layers.Dense(units=d, activation="sigmoid")
    r = layers.Dense(units=d, activation="sigmoid")
    h = layers.Dense(units=d, activation="tanh")
    h_tilde = layers.Dense(units=d, activation="tanh")

    h_t_minus_1 = tf.zeros(shape=(d,))

    for t in range(T):
        z_t = z(x[:, t]) + tf.matmul(h_t_minus_1, U_z)
        r_t = r(x[:, t]) + tf.matmul(h_t_minus_1, U_r)
        h_t = (1 - z_t) * h_t_minus_1 + z_t * h(x[:, t]) + tf.matmul(r_t * h_t_minus_1, U)

        h_t_minus_1 = h_t

    return h_t

# Define the attention score calculation function
def calculate_attention_scores(h_i, h_j):
    mlp = layers.Dense(units=1)

    alpha_c_i = tf.nn.softmax(mlp(h_i))
    alpha_t_i = tf.nn.softmax(mlp(h_i))
    beta_c_i_j = tf.nn.softmax(mlp(tf.concat([h_i, h_j], axis=-1)))
    beta_t_i_j = tf.nn.softmax(mlp(tf.concat([h_i, h_j], axis=-1)))

    return alpha_c_i, alpha_t_i, beta_c_i_j, beta_t_i_j

# Define the back-door adjustment function
def backdoor_adjustment(E_G_i, E_G_e):
    z_G_prime = phi(E_G_i * E_G_e)

    return z_G_prime

# Set the dimensionality and time steps
d = 128
T = 10

# Load the saved session preference from Part 1
loaded_session_preference = np.load("session_preference.npy")

# Replace placeholder X with loaded_session_preference
X = loaded_session_preference


# Generate some sample data
X = np.random.randn(100, T, d)
A = np.random.randint(2, size=(100, 100))
y = np.random.randint(2, size=(100,))

# Build the model architecture
input_sequence = layers.Input(shape=(T, d))
X_i = gru_item_representation(input_sequence)
X_e = gru_item_representation(input_sequence)

alpha_c_i, alpha_t_i, beta_c_i_j, beta_t_i_j = calculate_attention_scores(X_i, X_i)

M_a = tf.multiply(A, beta_c_i_j)
M_x = tf.multiply(X, alpha_c_i)
M_a_bar = 1 - M_a
M_x_bar = 1 - M_x

G_i = [M_a, M_x]
G_e = [M_a_bar, M_x_bar]

E_G_i = readout(G_i)
E_G_e = readout(G_e)

z_G_prime = backdoor_adjustment(E_G_i, E_G_e)

# Define the loss functions
loss_supervised = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, E_G_i))
loss_uniform = tf.keras.losses.kullback_leibler_divergence(y_uniform, E_G_e)
loss_causal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z_G_prime))

# Define the overall loss
loss = loss_supervised + loss_uniform + loss_causal

# Create the optimizer
optimizer = tf.keras.optimizers.Adam()

# Perform the training loop
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        logits = model(X)
        loss_value = loss(logits, y)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss_value}")



# Evaluate the model
logits = model(X_test)
predicted_labels = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, y_test), dtype=tf.float32))
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("causal_distillation_model")

# Load the saved model
loaded_model = tf.keras.models.load_model("causal_distillation_model")




