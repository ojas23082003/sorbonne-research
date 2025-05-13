
# 📊 Sorbonne Research Project

Welcome to the project implementation repository!  
This project contains Python implementations of time series forecasting models and benchmarking pipelines developed for evaluating models.

---

## 📥 Installation & Setup

Follow the steps below to get started:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ojas23082003/sorbonne-research.git
cd sorbonne-research
```

### 2️⃣ Create a Python Virtual Environment

```bash
py -m venv env
```

### 3️⃣ Activate the Virtual Environment

- **On Windows:**

    ```bash
    env\Scripts\activate
    ```

- *(On macOS/Linux, use:)*

    ```bash
    source env/bin/activate
    ```

### 4️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Experiments

To replicate the results:

- Run any Python file ending with `_training_loop.py`
- This will train the models and generate a `.csv` file containing sMAPE evaluation scores


## 📊 Output

After execution, a CSV file will be generated summarizing the **sMAPE (Symmetric Mean Absolute Percentage Error)** values for your experiments.  

---