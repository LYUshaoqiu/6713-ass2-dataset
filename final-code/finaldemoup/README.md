# Final Demo

This folder contains the minimal code package for the final course survey demo.

## Run

```powershell
python app.py
```

Then open:

- `http://127.0.0.1:8011/`

## Required Python Packages

Install these if needed:

```powershell
python -m pip install torch transformers safetensors
```

## Folder Structure

- `app.py`: backend and embedded survey page
- `model/roberta_HateXPlain/`: base RoBERTa encoder files
- `model/classifier_head.pt`: joint text-plus-scores classifier head
- `submissions/`: created automatically after form submission
