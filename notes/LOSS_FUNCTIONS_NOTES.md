# LOSS FUNCTIONS - QUICK REVISION NOTES

## 1. WHAT IS A LOSS FUNCTION?

**Definition**: Measures how wrong your model's predictions are. Lower loss = better model.

**Formula**: Loss = f(y_true, y_pred)

---

## 2. MAIN LOSS FUNCTIONS

### A. Mean Squared Error (MSE) - REGRESSION

```
MSE = (1/n) × Σ(y_true - y_pred)²
```

**When**: Regression, penalizes large errors heavily
**Example**: House price prediction

### B. Mean Absolute Error (MAE) - REGRESSION

```
MAE = (1/n) × Σ|y_true - y_pred|
```

**When**: Regression, less sensitive to outliers
**Example**: Temperature prediction

### C. Binary Cross-Entropy (BCE) - CLASSIFICATION

```
BCE = -[y×log(ŷ) + (1-y)×log(1-ŷ)]
```

**When**: Binary classification (0 or 1)
**Example**: Spam detection, disease diagnosis

### D. Categorical Cross-Entropy (CCE) - CLASSIFICATION

```
CCE = -Σ(y_true × log(y_pred))
```

**When**: Multi-class classification
**Example**: Digit recognition (0-9), image classification

---

## 3. QUICK COMPARISON

| Loss | Task | Sensitive to Outliers | Formula Complexity |
|------|------|----------------------|-------------------|
| MSE | Regression | High | Simple |
| MAE | Regression | Low | Simple |
| BCE | Binary | Medium | Medium |
| CCE | Multi-class | Medium | Medium |

---

## 4. KEY PYTORCH CODE

### MSE Loss
```python
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)
```

### BCE Loss
```python
loss_fn = nn.BCELoss()
loss = loss_fn(y_pred, y_true)
```

### Cross-Entropy Loss (includes softmax)
```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y_true)
```

### Manual BCE (from your code)
```python
epsilon = 1e-7
y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)
loss = -(y * torch.log(y_pred) + (1-y) * torch.log(1-y_pred)).mean()
```

---

## 5. BCE EXPLAINED (YOUR NEURAL NETWORK)

```python
def loos_fucntion(self, y_pred, y):
    epsilon = 1e-7                              # Prevents log(0) = -∞
    y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)  # Keeps predictions safe
    loss = -(y * torch.log(y_pred) + (1-y) * torch.log(1-y_pred)).mean()
    return loss
```

**Why epsilon?** log(0) crashes, log(1e-7) = -16 (safe)
**Why clamp?** Ensures y_pred never exactly 0 or 1
**Why negative sign?** log(0.1) = -2.3, we want positive loss
**Why .mean()?** Average loss over all samples

**Example:**
- If y=1, y_pred=0.9: loss = -log(0.9) = 0.105 (small, good!)
- If y=1, y_pred=0.1: loss = -log(0.1) = 2.303 (large, bad!)

---

## 6. CHOOSING THE RIGHT LOSS

| Problem Type | Output | Activation | Loss Function |
|--------------|--------|------------|---------------|
| Binary Classification | 0 or 1 | Sigmoid | BCE |
| Multi-class | Class 0-9 | Softmax | CCE |
| Regression | Continuous | None/ReLU | MSE or MAE |

---

## 7. GRADIENT DESCENT WITH LOSS

```python
for epoch in range(epochs):
    y_pred = model.forward(X)           # Forward pass
    loss = model.loss_function(y_pred, y)  # Calculate loss
    loss.backward()                     # Compute gradients

    # Update weights
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad

    # Reset gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()
```

**Goal**: Minimize loss by adjusting weights

---

## 8. KEY TAKEAWAYS

✅ **Regression** → MSE or MAE
✅ **Binary classification** → BCE
✅ **Multi-class** → CCE
✅ **Lower loss** = better model
✅ **Always use epsilon** to prevent log(0)
✅ **Clamp predictions** to safe range
✅ **Monitor loss** during training (should decrease)

---

**END OF NOTES**
