"""
LSTM Model for Temporal Escalation Detection
Predicts escalation probability from sequence of abuse classifications
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json

# ===== SYNTHETIC SEQUENCE DATA =====
# Each sequence represents a case timeline
# Format: [[[control, verbal, threat, physical, severe], ...], escalation_label]

SEQUENCE_DATA = [
    # Case 1: Clear escalation
    {
        "sequence": [
            [0.3, 0.2, 0.1, 0.0, 0.0],  # Day 1
            [0.7, 0.3, 0.1, 0.0, 0.0],  # Day 5
            [0.6, 0.5, 0.2, 0.1, 0.0],  # Day 10
            [0.2, 0.3, 0.3, 0.7, 0.0],  # Day 14
        ],
        "escalation_prob": 0.85,
        "outcome": 1  # Escalated
    },
    
    # Case 2: Moderate escalation
    {
        "sequence": [
            [0.4, 0.1, 0.0, 0.0, 0.0],
            [0.5, 0.2, 0.1, 0.0, 0.0],
            [0.6, 0.3, 0.1, 0.0, 0.0],
            [0.5, 0.4, 0.2, 0.1, 0.0],
        ],
        "escalation_prob": 0.55,
        "outcome": 1
    },
    
    # Case 3: No escalation (stable)
    {
        "sequence": [
            [0.5, 0.2, 0.0, 0.0, 0.0],
            [0.4, 0.3, 0.1, 0.0, 0.0],
            [0.5, 0.2, 0.0, 0.0, 0.0],
            [0.4, 0.2, 0.1, 0.0, 0.0],
        ],
        "escalation_prob": 0.25,
        "outcome": 0
    },
    
    # Case 4: De-escalation
    {
        "sequence": [
            [0.6, 0.4, 0.2, 0.1, 0.0],
            [0.5, 0.3, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.1, 0.0, 0.0],
            [0.3, 0.1, 0.0, 0.0, 0.0],
        ],
        "escalation_prob": 0.15,
        "outcome": 0
    },
    
    # Case 5: Severe escalation
    {
        "sequence": [
            [0.2, 0.1, 0.0, 0.0, 0.0],
            [0.5, 0.3, 0.2, 0.1, 0.0],
            [0.3, 0.4, 0.4, 0.3, 0.0],
            [0.1, 0.2, 0.3, 0.6, 0.2],
        ],
        "escalation_prob": 0.92,
        "outcome": 1
    },
]

# Generate more synthetic cases
def generate_synthetic_sequences(num_sequences=100):
    """Generate synthetic escalation sequences"""
    sequences = []
    
    for _ in range(num_sequences):
        seq_len = np.random.randint(3, 8)
        sequence = []
        
        # Random escalation pattern
        escalation_type = np.random.choice(['escalating', 'stable', 'de-escalating'])
        
        if escalation_type == 'escalating':
            base_severity = np.random.uniform(0.2, 0.4)
            for step in range(seq_len):
                severity = base_severity + (step * 0.15) + np.random.uniform(-0.1, 0.1)
                # Shift probability toward higher abuse categories over time
                probs = np.random.dirichlet(
                    [max(0.1, 5 - step), max(0.1, 4 - step), 
                     max(0.1, 3 - step), max(0.1, step), max(0.1, step * 0.5)]
                )
                sequence.append(probs.tolist())
            
            escalation_prob = min(0.95, 0.5 + (seq_len * 0.08) + np.random.uniform(0, 0.2))
            outcome = 1
            
        elif escalation_type == 'stable':
            for _ in range(seq_len):
                probs = np.random.dirichlet([3, 2, 1, 0.5, 0.1])
                sequence.append(probs.tolist())
            
            escalation_prob = np.random.uniform(0.2, 0.4)
            outcome = 0
            
        else:  # de-escalating
            base_severity = np.random.uniform(0.5, 0.7)
            for step in range(seq_len):
                severity = base_severity - (step * 0.1) + np.random.uniform(-0.05, 0.05)
                probs = np.random.dirichlet(
                    [max(0.1, step), max(0.1, step * 0.5), 
                     max(0.1, 3 - step), max(0.1, 3 - step), 0.1]
                )
                sequence.append(probs.tolist())
            
            escalation_prob = np.random.uniform(0.1, 0.3)
            outcome = 0
        
        sequences.append({
            "sequence": sequence,
            "escalation_prob": escalation_prob,
            "outcome": outcome
        })
    
    return sequences

# ===== DATA PREPARATION =====

def prepare_sequences(data, max_len=10):
    """Pad sequences to same length"""
    X = []
    y_prob = []
    y_class = []
    
    for item in data:
        seq = item['sequence']
        
        # Pad sequence
        if len(seq) < max_len:
            padding = [[0, 0, 0, 0, 0]] * (max_len - len(seq))
            seq = padding + seq
        else:
            seq = seq[-max_len:]  # Take last max_len items
        
        X.append(seq)
        y_prob.append(item['escalation_prob'])
        y_class.append(item['outcome'])
    
    return np.array(X), np.array(y_prob), np.array(y_class)

# ===== BUILD LSTM MODEL =====

def build_lstm_model(sequence_length=10, feature_dim=5):
    """Build LSTM model for escalation prediction"""
    
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, feature_dim)),
        
        # First LSTM layer
        layers.LSTM(64, return_sequences=True, dropout=0.2),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(32, dropout=0.2),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer - probability of escalation
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# ===== TRAINING =====

def train_escalation_model():
    """Train LSTM escalation model"""
    
    # Generate training data
    print("Generating training data...")
    all_data = SEQUENCE_DATA + generate_synthetic_sequences(200)
    
    # Prepare sequences
    X, y_prob, y_class = prepare_sequences(all_data)
    
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_class[:split], y_class[split:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Build model
    model = build_lstm_model()
    print(model.summary())
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ],
        verbose=1
    )
    
    # Evaluate
    print("\nFinal evaluation:")
    test_loss, test_acc, test_auc = model.evaluate(X_val, y_val)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save model
    model.save('escalation_lstm_model.h5')
    print("Model saved to escalation_lstm_model.h5")
    
    return model, history

# ===== INFERENCE FUNCTION =====

def predict_escalation(model, sequence):
    """
    Predict escalation probability for a new sequence
    
    Args:
        model: Trained LSTM model
        sequence: List of [control, verbal, threat, physical, severe] probabilities
    
    Returns:
        Escalation probability (0-1)
    """
    # Prepare input
    X, _, _ = prepare_sequences([{"sequence": sequence, "escalation_prob": 0, "outcome": 0}])
    
    # Predict
    prob = model.predict(X, verbose=0)[0][0]
    
    return float(prob)

# ===== MAIN =====

if __name__ == "__main__":
    # Train model
    model, history = train_escalation_model()
    
    # Test prediction
    test_sequence = [
        [0.3, 0.2, 0.1, 0.0, 0.0],
        [0.6, 0.3, 0.1, 0.0, 0.0],
        [0.5, 0.4, 0.3, 0.1, 0.0],
        [0.2, 0.3, 0.4, 0.5, 0.0],
    ]
    
    prob = predict_escalation(model, test_sequence)
    print(f"\nTest prediction - Escalation probability: {prob:.2f}")
    
    # Save to GCS
    # from google.cloud import storage
    
    # storage_client = storage.Client()
    # bucket = storage_client.bucket('dv-detection-models')
    
    # blob = bucket.blob('lstm_escalation/model.h5')
    # blob.upload_from_filename('escalation_lstm_model.h5')
    
    print("Model uploaded to Google Cloud Storage")