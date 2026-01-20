# Decode Like a Clinician 🩺📈

**Temporal Verbalization for Structured Clinical Reasoning with LLMs**

This repository contains the **official implementation** of the AACL 2025 paper:

> **Decode Like a Clinician: Enhancing LLM Fine-Tuning with Temporal Structured Data Representation**
> Daniel Fadlon, David Dov, Aviya Bennett, Daphna Heller-Miron, Gad Levy,
> Kfir Bar, Ahuva Weiss-Meilik
> *IJCNLP-AACL 2025 (Long Paper)*

- 📄 **Paper:** [https://aclanthology.org/2025.ijcnlp-long.103/](https://aclanthology.org/2025.ijcnlp-long.103/)
- 💻 **Code:** [https://github.com/DanielFadlon/decode-like-a-clinician](https://github.com/DanielFadlon/decode-like-a-clinician)


## 📚 Citation

If you use this code or build upon this work, **please cite**:

```bibtex
@inproceedings{fadlon-etal-2025-decode,
    title = "Decode Like a Clinician: Enhancing {LLM} Fine-Tuning with Temporal Structured Data Representation",
    author = "Fadlon, Daniel  and
      Dov, David  and
      Bennett, Aviya  and
      Heller-Miron, Daphna  and
      Levy, Gad  and
      Bar, Kfir  and
      Weiss-Meilik, Ahuva",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.103/",
    pages = "1906--1922",
    ISBN = "979-8-89176-298-5"
}
```

---

## 🔍 What this work is *actually* about

Large Language Models are increasingly applied to **structured prediction tasks** (e.g., clinical deterioration, mortality, adverse outcomes).
However, **how structured temporal data is *represented*** is often treated as an implementation detail rather than a core modeling decision.

This paper shows that:

> **The way temporal EMRs are encoded into prompts can substantially affect the effectiveness of LLM fine-tuning for clinical prediction, even without architectural modifications.**

We introduce a clinician-inspired decoding paradigm that verbalizes structured EMRs as temporally grounded event narratives, enabling LLMs to model longitudinal patient trajectories. Using this framework, we:
- Achieve strong performance in both in-hospital and cross-hospital evaluation settings across real-world and open-source datasets; and
- Conduct a systematic analysis of how different temporal event verbalization strategies influence predictive performance, revealing both the strengths and limitations of LLMs in modeling temporal clinical structure.

## 📂 Repository Structure (what matters, not boilerplate)

```
decode-like-a-clinician/
├── src/
│   ├── verbalizer/              # Temporal verbalization logic
│   │   ├── indicator_formatter.py
│   │   ├── event_narrative.py
│   │   └── time_encoding.py
│   ├── modeling/                # LLM fine-tuning wrappers
│   └── evaluation/              # Outcome prediction & robustness analysis
│
├── mimic_iv_data_pipeline/      # End-to-end MIMIC-IV preprocessing → verbalized input
│
├── configurations/              # Exact configs used in the paper
│
├── scripts/                     # Training / evaluation entry points
│
└── tests/                       # Verbalizer correctness & consistency tests
```

---

## 🧪 Experimental Setting (as in the paper)

* **Tasks:** Clinical outcome prediction from longitudinal EHR
* **Datasets:**
  * Two real-world hospital datasets
  * MIMIC-IV for external validation - see the 
* **Models:** Decoder-only LLMs (fine-tuned, no architecture changes)
* **Evaluation:** AUC, robustness to representation changes, cross-hospital transfer

## 🧬 MIMIC-IV Setup

This repository includes an end-to-end pipeline for preprocessing MIMIC-IV data and converting it into the temporal verbalized format used in our experiments.
Due to licensing restrictions, MIMIC-IV data is not included in this repository. Users must obtain access independently and follow the official PhysioNet requirements.

🔗 **Upstream Pipeline**: The MIMIC-IV preprocessing pipeline in this repository is **adapted and extended** from the following upstream project: 
<https://github.com/healthylaife/MIMIC-IV-Data-Pipeline>

👉 For detailed instructions on:
- accessing MIMIC-IV,
- preprocessing raw tables,
- generating temporally verbalized EMR inputs,
- and reproducing the experiments reported in the paper,

please refer to the dedicated README in the pipeline directory:

📘 MIMIC-IV Pipeline Documentation [mimic_iv_data_pipeline/README.md](mimic_iv_data_pipeline/README.md)

---

## 🚀 Running the code (minimal, reproducible)

### Environment

```bash
pip install -r requirements.txt
```

### Verbalize structured EHR data

```bash
python src/verbalizer/run_verbalizer.py \
  --input path/to/structured_data \
  --output path/to/verbalized_data \
  --config configurations/verbalization.json
```

### Train a model

```bash
python scripts/train.py \
  --config configurations/train_config.json
```

### Evaluate

```bash
python scripts/evaluate.py \
  --model_dir outputs/experiment_X
```

The configs correspond **exactly** to those reported in the paper.

---

## 📜 License

This repository is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

### Third-Party Components

Parts of the MIMIC-IV preprocessing pipeline are adapted from the following upstream repository:
- <https://github.com/healthylaife/MIMIC-IV-Data-Pipeline>

The upstream code is used in accordance with its original license.  
Modifications include temporal aggregation and clinician-inspired verbalization to support LLM fine-tuning, as described in the accompanying paper.

### Dataset Licensing

This repository does **not** include MIMIC-IV data.  
Use of MIMIC-IV requires credentialed access and compliance with the **PhysioNet Data Use Agreement (DUA)**.

