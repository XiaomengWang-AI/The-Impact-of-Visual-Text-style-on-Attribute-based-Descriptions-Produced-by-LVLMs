# The Impact of Visual Text Style on Attribute-Based Descriptions Produced by LVLMs

This repository provides the official implementation of the paper:
“The Impact of Visual Text Style on Attribute-Based Descriptions Produced by LVLMs”,  
to appear in Proceedings of ACM International Conference on Multimedia Retrieval (ICMR), 2026.

## Overview

This work investigates the influence of visual text style on attribute-based descriptions produced by Large Vision-Language Models (LVLMs).  

We reveal that even when the concept in the visual text is correctly identified, text style influences the
model’s attribute-based descriptions of the concept.

Our findings demonstrate non-trivial **style leakage** from text style into semantic inference and motivate style-aware evaluation and mitigation for LVLM-based multimedia systems.

## Dependencies

All required dependencies are listed in `requirements.txt`.  
We recommend creating a virtual environment and installing dependencies via:

```bash
pip install -r requirements.txt

## Usage

The experimental pipeline consists of the following stages:

### 1. Visual Text Generation
Generate visual text images with different styles:
```bash
python visual_text_generator.py

### 2. Visual Text Recognition
Run LVLM-based recognition:
```bash
python iden_visual_text_qwen.py
python iden_visual_text_gpt.py

### 3. Attribute Output Generation
Collect attribute-based descriptions from LVLMs:
```bash
python attr_eval_qwen_prompt5_run5.py
python attr_eval_gpt_prompt5_run5.py

Clean attribute outputs from the Qwen2.5-VL-3B-Instruct model:
```bash
python llama_clean.py

### 4. Style Distribution Analysis
Compute Total Variation (TV) between style distributions:
```bash
python TV_by_style.py

Examine the differences in the top-3 most-frequent attributes produced in the two styles:
```bash
python style_top3_words_by_breed.py



## Acknowledgements

We acknowledge the use of publicly available font resources:

- https://fontzone.net/  
- https://font.download/  
- https://www.wfonts.com/  
- https://grammarhow.com/best-cursive-fonts-in-microsoft-word/

We also thank the reviewers of ICMR 2026 for their valuable feedback.



