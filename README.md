# 🧅 Onion Grade AI

## AI-Powered Quality Assessment & Digital Grading of Onions

Onion Grade AI is a computer-vision based system designed to assist in the quality assessment and grading of onions.

The system takes a batch image of onions, detects individual onions, analyses their visible quality condition, calculates batch-level quality statistics and generates a digital assessment report.

---

## 🎯 Problem

Manual onion quality assessment can be:

- Subjective
- Time-consuming
- Inconsistent across procurement centres
- Difficult to document and verify

These inconsistencies can lead to disputes between farmers, buyers and procurement authorities.

---

## 💡 Proposed Solution

Onion Grade AI provides an AI-assisted workflow:

```text
Batch Image
     ↓
YOLO26 Detection
     ↓
Individual Onion Detection
     ↓
Image Preprocessing
     ↓
Quality Classification
     ↓
Batch Assessment
     ↓
Digital PDF Report