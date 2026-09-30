# Best traits — logistic_regression, agent-2, generation 1

### Original code (CO)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 60.0000 s |
| Inference time | TI | 0.0000 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 1 |
| Logical errors | EL | 0 |
| Loss | L | 1.0000 |
| Model accuracy | Ma | 0.0000 |
| Memory usage | Mu | 97.98 MB |
| Computational cost | K | 3.4844 s |
| Time | T | 60.0000 s |
| Model performance | Mp | 0.0000 |
| Novelty | Nθ | 1.0000 |
| Accuracy term | Aθ | 0.5000 |
| Resource | R | 341.3871 |
| Task performance (novelty-free) | P* | 0 |
| **Performance** | **P** | **0** |
| **Trait** | **Tθ** | **0** |



### Top sample 1 (id: gen1_island2_sample1, generation 1)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0209 s |
| Inference time | TI | 0.0007 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.1740 |
| Model accuracy | Ma | 0.9333 |
| Memory usage | Mu | 141.72 MB |
| Computational cost | K | 4.7969 s |
| Time | T | 0.0216 s |
| Model performance | Mp | 5.3648 |
| Novelty | Nθ | 0.1250 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 679.8071 |
| Task performance (novelty-free) | P* | 0.0634793 |
| **Performance** | **P** | **0.00793205** |
| **Trait** | **Tθ** | **0.00793205** |


```python
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def run(data_path: str) -> dict:
    # TRAIT: Memory usage, Resources
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # TRAIT: model accuracy, training time, Novelty
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Introduced stratification
    )

    # TRAIT: model accuracy, Computational cost
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        # TRAIT: model accuracy, training time, computational cost
        max_iter=100,  # Increased max iterations
        C=1.0,  # Increased C value
        solver='lbfgs',
        random_state=42,
    )

    # TRAIT: training time
    _t0 = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - _t0

    # TRAIT: inference time
    _t1 = time.time()
    predictions = model.predict(X_test_scaled)
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss
    try:
        probabilities = model.predict_proba(X_test_scaled)
        loss = float(log_loss(y_test, probabilities, labels=sorted(set(y))))
    except Exception:
        loss = 1.0 - accuracy

    return {
        "accuracy": accuracy,
        "loss": loss,
        "training_time": training_time,
        "inference_time": inference_time,
    }
```

### Top sample 2 (id: gen1_island1_sample3, generation 1)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0187 s |
| Inference time | TI | 0.0010 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.3800 |
| Model accuracy | Ma | 0.8667 |
| Memory usage | Mu | 141.26 MB |
| Computational cost | K | 7.3594 s |
| Time | T | 0.0196 s |
| Model performance | Mp | 2.2808 |
| Novelty | Nθ | 0.1861 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 1039.5980 |
| Task performance (novelty-free) | P* | 0.0424551 |
| **Performance** | **P** | **0.00789936** |
| **Trait** | **Tθ** | **0.00789936** |


```python
def run(data_path: str) -> dict:
    import time
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler  # TRAIT: memory usage, novelty

    df = pd.read_csv(data_path)  # TRAIT: resources
    X = df.drop(columns=["target"])  # TRAIT: computational cost
    y = df["target"]

    # TRAIT: model accuracy, training time, randomness, resources.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # TRAIT: model accuracy, novelty
    )

    # TRAIT: model accuracy, computational cost, novelty. Introduces a scaler to improve model accuracy.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # TRAIT: training time, computational cost
    X_test_scaled = scaler.transform(X_test)  # TRAIT: inference time

    # TRAIT: model accuracy, training time, computational cost. Increases max_iter to improve model accuracy.
    model = LogisticRegression(
        max_iter=100,  # TRAIT: computational cost, model accuracy
        C=0.1,  # TRAIT: model accuracy, training time
        solver='lbfgs',  # TRAIT: computational cost
        random_state=42,  # TRAIT: randomness
    )

    # Training time (TT) is measured around fit() only.
    _t0 = time.time()
    model.fit(X_train_scaled, y_train)  # TRAIT: training time
    training_time = time.time() - _t0  # TRAIT: training time

    # Inference time (TI) is measured around predict() only.
    _t1 = time.time()
    predictions = model.predict(X_test_scaled)  # TRAIT: inference time
    inference_time = time.time() - _t1  # TRAIT: inference time

    # TRAIT: model accuracy
    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss (L). log_loss needs calibrated probabilities; models
    # without predict_proba fall back to the error rate.
    try:
        probabilities = model.predict_proba(X_test_scaled)  # TRAIT: computational cost
        loss = float(log_loss(y_test, probabilities, labels=sorted(set(y))))  # TRAIT: loss
    except Exception:
        loss = 1.0 - accuracy  # TRAIT: loss

    return {
        "accuracy": accuracy,  # TRAIT: model accuracy
        "loss": loss,  # TRAIT: loss
        "training_time": training_time,  # TRAIT: training time
        "inference_time": inference_time,  # TRAIT: inference time
    }
```

### Top sample 3 (id: gen1_island2_sample0, generation 1)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0156 s |
| Inference time | TI | 0.0006 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.1740 |
| Model accuracy | Ma | 0.9333 |
| Memory usage | Mu | 141.43 MB |
| Computational cost | K | 4.7656 s |
| Time | T | 0.0162 s |
| Model performance | Mp | 5.3648 |
| Novelty | Nθ | 0.0666 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 674.0009 |
| Task performance (novelty-free) | P* | 0.0854866 |
| **Performance** | **P** | **0.00568999** |
| **Trait** | **Tθ** | **0.00568999** |


```python
def run(data_path: str) -> dict:
    import time
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler  # TRAIT: model accuracy

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # TRAIT: model accuracy, training time. The split itself is
    # evolvable — test_size, random_state and stratification all change
    # how much data the model sees and how stable the score is.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # TRAIT: model accuracy
    )

    # TRAIT: model accuracy
    scaler = StandardScaler()  # Introducing a scaler for feature scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        # TRAIT: model accuracy, training time, computational cost.
        # Improved parameters for better convergence and regularisation.
        max_iter=1000,  # Increased max_iter for better convergence
        C=1.0,  # Improved regularisation parameter
        solver='lbfgs',
        random_state=42,
    )

    # Training time (TT) is measured around fit() only.
    _t0 = time.time()
    model.fit(X_train_scaled, y_train)  # Using scaled data for training
    training_time = time.time() - _t0

    # Inference time (TI) is measured around predict() only.
    _t1 = time.time()
    predictions = model.predict(X_test_scaled)  # Using scaled data for inference
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss (L). log_loss needs calibrated probabilities; models
    # without predict_proba fall back to the error rate.
    try:
        probabilities = model.predict_proba(X_test_scaled)  # Using scaled data for prediction probabilities
        loss = float(log_loss(y_test, probabilities, labels=sorted(set(y))))
    except Exception:
        loss = 1.0 - accuracy

    return {
        "accuracy": accuracy,
        "loss": loss,
        "training_time": training_time,
        "inference_time": inference_time,
    }
```
