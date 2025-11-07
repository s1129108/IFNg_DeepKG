# IFNg_DeepKG: A Novel Model for Identifying Interferon-Gamma Inducing Epitopes Using Knowledge Graph RAG in Biomedical Applications
Van The Le, Juan Peter Timothy Yuune, Yu-Yen Ou
|[ 🎇&nbsp;Abstract](#abstract) |[📃&nbsp;Dataset](#Dataset) | [ 🚀&nbsp;Quick Prediction ](#colab) |[ 💾&nbsp;Requirements](#requirement)|
|-------------------------------|-----------------------------|------------------------------------- |--------------------------------------|

## 🎇Abstract <a name="abstract"></a>
The accurate and efficient computational identification of interferon-gamma inducing epitopes (IFNgIE) is a critical bottleneck in the design of next-generation vaccines and immunotherapies. Existing computational models, while adept at learning sequence-based patterns, frequently fail to incorporate the rich biological context that governs an epitope's immunogenicity, such as its protein of origin, host, and disease association. To address this limitation, we propose IFNg_DeepKG, a new deep learning framework that synergistically integrates a pre-trained protein language model (ESM2), a custom knowledge graph (KG) using a Retrieval-Augmented Generation (RAG) approach, and a multi-scale convolutional neural network (MSCNN). The model’s central innovation lies in its use of the RAG-KG to enrich sequence embeddings with external, biologically-informed context, thereby significantly enhancing predictive performance. IFNg_DeepKG demonstrates superior performance on independent test datasets, achieving an AUC of 0.99 on the Human H_IFNgInd1 dataset and 0.95 on the Mouse M_IFNgInd1 dataset, a substantial increase over baseline models. The framework successfully identifies and classifies clinically relevant epitopes, including those associated with COVID-19 and Alzheimer's disease. By bridging the gap between sequence-based features and biological contexts, IFNg_DeepKG represents a significant advancement in computational immunology, offering a scalable and powerful platform for rational epitope discovery and precision medicine.
<br>

![workflow](./FIGURE/workflow.png)

## 📃Dataset <a name="Dataset"></a>

| Host            | Set |    Inducing |  Non-inducing    |
|--------------------|------------------|--------------------------|--------------------------|
| Human      |   H_IFNgTrain             |          20394            |20394                     |
|       |   H_IFNgInd1             |          5098            |5098                     |
|       |   H_IFNgInd2             |          5098            |33946                     |
| Mouse       |    M_IFNgTrain         |                    6387    |     6387                 |
|        |    M_IFNgInd1         |                    1596    |     1596                 |
|        |    M_IFNgInd2         |                    1596    |     8139                 |
| RAG database   |          IFNgKG  |                     169   |                 554     |

## 🚀Prediction <a name="colab"></a>

### Step 1: Environment Setup
Use pip install -r requirements.txt for environment dependencies

### Step 2: Submit your fasta file

Upload your own FASTA file
The format of the FASTA file will be as follows:
```bash
>P_te_Seq_1
FVFPTKDV
>P_te_Seq_2
LPRQRAYL
>P_te_Seq_3
QTRQKFHL
>P_te_Seq_4
KRRYKQLL
>P_te_Seq_5
TLTHPVTK
>P_te_Seq_6
VPFYGKAI
```
Alternatively, you may use our testing dataset in TEST_SAMPLE

### Step 3: Generate ESM2 embedding
Use exfold.py for embedding generation. You can download the pretrained model esm2_t33_650M_UR50D from EMS2's offical github https://github.com/facebookresearch/esm

### Step 4: Get the RAG-KG ESM2 embedding
Use get_RAGKGEmb.py tool for merging vanilla ESM2 embeddings with RAG-KG database (we provided fasta files for RAG)

### Step 5: Get the dataset for MSCNN
Use get_dataset.py to generate input feature for MSCNN

### Step 6: Prediction
Use MSCNN.py for prediction


## 📚&nbsp;License <a name="License"></a>
Licensed under the Academic Free License version 3.0
