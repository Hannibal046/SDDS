# Dialogue Summarization with Static-Dynamic Structure Fusion Graph

This repository contains the source code for this paper [Dialogue Summarization with Static-Dynamic Structure Fusion Graph](https://aclanthology.org/2023.acl-long.775/).

Dialogue summarization, one of the most challenging and intriguing text summarization tasks, has attracted increasing attention in recent years.

Since dialogue possesses dynamic interaction nature and presumably inconsistent information flow scattered across multiple utterances by different interlocutors, many researchers address this task by modeling dialogue with pre-computed static graph structure using external linguistic toolkits. 

However, such methods heavily depend on the reliability of external tools and the static graph construction is disjoint with the graph representation learning phase, which could not make the graph dynamically adapt to the downstream summarization task. 

In this paper, we propose a Static-Dynamic graph-based Dialogue Summarization model (SDDS), which fuses prior knowledge from human expertise and implicit knowledge from a PLM, and adaptively adjusts the graph weight, and learns the graph structure in an end-to-end learning fashion from the supervision of summarization task. 

<div align=center>
<img src=model.svg width=60% height=60% />
</div>

---

## Setup
Our code is mainly based on 🤗 [Transformers](https://github.com/huggingface/transformers). 

```bash
## firstly install torch corresponding to the CUDA
pip install transformers==4.8.2 \
            py-rouge nltk numpy datasets stanza dgl
```
---
## Data
For dataset we use, we prepare the `SamSum` dataset in the data folder along with its annotation created by external linguistic tools. 

For MediaSum dataset, we refer to [here](https://github.com/zcgzcgzcg1/MediaSum) and for DialogSum [here](https://github.com/cylnlp/dialogsum).

For linguistic tools for annotation:

- discouse_parsing: https://github.com/shizhouxing/DialogueDiscourseParsing
- keyword extraction: we use [Stanza](https://github.com/stanfordnlp/stanza) for pos tagging and select the NOUN and PROPN.


---
## Training
The dafault config is at `config/graphbart_config.json`
```bash
cd src && CUDA_VISIBLE_DEVICES=x python run_summarization.py
```
