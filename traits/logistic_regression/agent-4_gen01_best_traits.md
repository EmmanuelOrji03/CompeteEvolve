# Best traits — logistic_regression, agent-4, generation 1

### Original code (CO)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0998 s |
| Inference time | TI | 0.0022 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.8289 |
| Model accuracy | Ma | 0.7000 |
| Memory usage | Mu | 141.73 MB |
| Computational cost | K | 4.8125 s |
| Time | T | 0.1020 s |
| Model performance | Mp | 0.8445 |
| Novelty | Nθ | 1.0000 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 682.0779 |
| Task performance (novelty-free) | P* | 0.0100646 |
| **Performance** | **P** | **0.0100646** |
| **Trait** | **Tθ** | **0.0100646** |



### Top sample 1 (id: gen1_island2_sample1, generation 1)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0178 s |
| Inference time | TI | 0.0005 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.1740 |
| Model accuracy | Ma | 0.9333 |
| Memory usage | Mu | 141.47 MB |
| Computational cost | K | 4.7969 s |
| Time | T | 0.0183 s |
| Model performance | Mp | 5.3648 |
| Novelty | Nθ | 0.0985 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 678.6079 |
| Task performance (novelty-free) | P* | 0.0750048 |
| **Performance** | **P** | **0.00738863** |
| **Trait** | **Tθ** | **0.00738863** |


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

    # TRAIT: model accuracy. Feature scaling is applied to improve model performance.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # TRAIT: model accuracy
    X_test_scaled = scaler.transform(X_test)  # TRAIT: model accuracy

    model = LogisticRegression(
        # TRAIT: model accuracy, training time, computational cost.
        # Increased max_iter to improve convergence and reduced regularization to reduce underfitting.
        max_iter=1000,  # TRAIT: model accuracy, training time, computational cost
        C=1.0,  # TRAIT: model accuracy, training time, computational cost
        solver='lbfgs',
        random_state=42,
        n_jobs=-1  # TRAIT: training time, computational cost
    )

    # Training time (TT) is measured around fit() only.
    _t0 = time.time()
    model.fit(X_train_scaled, y_train)  # TRAIT: model accuracy
    training_time = time.time() - _t0

    # Inference time (TI) is measured around predict() only.
    _t1 = time.time()
    predictions = model.predict(X_test_scaled)  # TRAIT: model accuracy
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss (L). log_loss needs calibrated probabilities; models
    # without predict_proba fall back to the error rate.
    try:
        probabilities = model.predict_proba(X_test_scaled)  # TRAIT: model accuracy
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

### Top sample 2 (id: gen1_island2_sample2, generation 1)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0160 s |
| Inference time | TI | 0.0006 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.3800 |
| Model accuracy | Ma | 0.8667 |
| Memory usage | Mu | 141.25 MB |
| Computational cost | K | 4.5469 s |
| Time | T | 0.0166 s |
| Model performance | Mp | 2.2808 |
| Novelty | Nθ | 0.0874 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 642.2283 |
| Task performance (novelty-free) | P* | 0.0815291 |
| **Performance** | **P** | **0.00712377** |
| **Trait** | **Tθ** | **0.00712377** |


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

    # TRAIT: model accuracy. Applying feature scaling to improve model performance.
    scaler = StandardScaler()  # TRAIT: model accuracy
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        # TRAIT: model accuracy, training time, computational cost.
        # Increased max_iter to improve convergence and reduced C to reduce over-regularization.
        max_iter=100,  # TRAIT: model accuracy, training time
        C=0.1,  # TRAIT: model accuracy
        solver='lbfgs',
        random_state=42,
    )

    # Training time (TT) is measured around fit() only.
    _t0 = time.time()
    model.fit(X_train_scaled, y_train)  # TRAIT: model accuracy
    training_time = time.time() - _t0

    # Inference time (TI) is measured around predict() only.
    _t1 = time.time()
    predictions = model.predict(X_test_scaled)  # TRAIT: model accuracy
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss (L). log_loss needs calibrated probabilities; models
    # without predict_proba fall back to the error rate.
    try:
        probabilities = model.predict_proba(X_test_scaled)  # TRAIT: model accuracy
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

### Top sample 3 (id: gen1_island2_sample4, generation 1)

| Trait | Symbol | Value |
| --- | --- | --- |
| Training time | TT | 0.0171 s |
| Inference time | TI | 0.0010 s |
| Syntax errors | SE | 0 |
| Runtime errors | RE | 0 |
| Logical errors | EL | 0 |
| Loss | L | 0.1740 |
| Model accuracy | Ma | 0.9333 |
| Memory usage | Mu | 141.80 MB |
| Computational cost | K | 5.7500 s |
| Time | T | 0.0181 s |
| Model performance | Mp | 5.3648 |
| Novelty | Nθ | 0.0982 |
| Accuracy term | Aθ | 1.0000 |
| Resource | R | 815.3770 |
| Task performance (novelty-free) | P* | 0.0633897 |
| **Performance** | **P** | **0.00622605** |
| **Trait** | **Tθ** | **0.00622605** |


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

    # TRAIT: model accuracy. Feature scaling is applied.
    scaler = StandardScaler()  # TRAIT: model accuracy
    X_train = scaler.fit_transform(X_train)  # TRAIT: model accuracy
    X_test = scaler.transform(X_test)  # TRAIT: model accuracy

    model = LogisticRegression(
        # TRAIT: model accuracy, training time, computational cost.
        # Improved tuning: increased max_iter and adjusted C for better convergence.
        max_iter=1000,  # TRAIT: model accuracy, training time
        C=1.0,  # TRAIT: model accuracy
        solver='lbfgs',  # TRAIT: computational cost
        random_state=42,
        penalty='l2',  # TRAIT: model accuracy
        tol=1e-4,  # TRAIT: model accuracy
    )

    # Training time (TT) is measured around fit() only.
    _t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - _t0

    # Inference time (TI) is measured around predict() only.
    _t1 = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - _t1

    accuracy = float(accuracy_score(y_test, predictions))

    # TRAIT: loss (L). log_loss needs calibrated probabilities; models
    # without predict_proba fall back to the error rate.
    try:
        probabilities = model.predict_proba(X_test)
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
