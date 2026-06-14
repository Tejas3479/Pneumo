# Model Card — PneumoDetect AI

This Model Card details the operational parameters, intended clinical use cases, demographic fairness metrics, and training specifications of the PneumoDetect AI system.

## Model Details
- **Developer:** PneumoDetect AI Team
- **Model Date:** 2026-06-14
- **Model Type:** Vision Transformer (ViT-B/16) with LoRA Adapters / ResNet-50 Ensemble
- **Version:** 1.0.0
- **License:** Research-only medical evaluation license.
- **Reference:** Google Model Card Toolkit standard schema.

## Intended Use
- **Primary Intended Uses:** Assist clinical radiologists in rapid secondary screening of chest radiographs for signs of Pneumothorax (collapsed lung).
- **Primary Intended Users:** Licensed radiologists, emergency clinicians, and pulmonology specialists.
- **Out-of-Scope Use Cases:** Diagnosis of non-radiological images, multi-abnormality chest pathology screening (without secondary verification), or autonomous diagnostic decision-making without clinician supervision.

## Factors
- **Demographic Factors:** Biological Sex (Male/Female), Patient Age.
- **Clinical/Imaging Factors:** Modality (Digital Radiography DX / Computed Radiography CR), scanner resolution, chest orientation (PA or AP).

## Metrics
- **Performance Evaluation Metric:** Area Under the Receiver Operating Characteristic curve (AUROC), Validation Binary Cross-Entropy Loss.
- **Fairness Metrics:** Demographic Parity Difference (DPD), Equal Opportunity Difference (EOD) evaluated across Patient Sex.

## Training Data
- **Dataset Source:** Synthetic clinical-grade Chest X-Ray database.
- **Training Set Size:** 80 images.
- **Target Label Balance:** Approximately balanced positive (pneumothorax) and negative cases.

## Evaluation Data
- **Evaluation Set Size:** 20 images.
- **Validation Split Strategy:** Reproducible index splits stored in validation indices cache.

## Quantitative Analyses
| Metric | Value | Target Threshold | Status |
|---|---|---|---|
| Validation AUROC | 0.5781 | >= 0.8500 | EVAL |
| Validation Loss | 0.6917 | < 0.5000 | EVAL |
| Demographic Parity Difference | N/A | < 0.1000 | WARNING |
| Equal Opportunity Difference | N/A | < 0.1000 | WARNING |

## Ethical Considerations & Limitations
- The model is not approved for autonomous diagnosis. It must be utilized strictly as a diagnostic aid.
- High demographic differences indicate potential biases in clinical subset distributions; the model should be retrained using debiased adversarial heads or updated training weights if parity metrics exceed 0.1.

---
Generated automatically by PneumoDetect Regulatory Pipeline on 2026-06-14.
