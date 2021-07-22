# Team Enigma at ArgMining 2021 Shared Task: Leveraging Pretrained Language Models for Key Point Matching

This is our attempt of the shared task on **Quantitative Summarization – Key Point Analysis Shared Task** at the [8th Workshop on Argument Mining](https://2021.argmining.org/shared_task_ibm.html#ibm), part of EMNLP 2021.  

## Authors - Manav Nitin Kapadnis, Sohan Patnaik, Siba Smarak Panigrahi, Varun Madhavan

Key Point Analysis (KPA) is a new NLP task, with strong relations to Computational Argumentation, Opinion Analysis, and Summarization ([Bar-Haim et al., ACL-2020](https://www.aclweb.org/anthology/2020.acl-main.371.pdf); [Bar-Haim et al., EMNLP-2020](https://arxiv.org/pdf/2010.05369.pdf)). 
Given an input corpus, consisting of a collection of relatively short, opinionated texts focused on a topic of interest, the goal of KPA is to produce a succinct list of the most prominent key-points in the input corpus, along with their relative prevalence. Thus, the output of KPA is a bullet-like summary, with an important quantitative angle and an associated well-defined evaluation framework. Successful solutions to KPA can be used to gain better insights from public opinions as expressed in social media, surveys, and so forth, giving rise to a new form of a communication channel between decision makers and people that might be impacted by the decision. 

Official repository of the task can be found [here](https://github.com/IBM/KPA_2021_shared_task).

### Sections
1. [System description paper](#system-description-paper)
2. [Results](#results)
3. [Task Details](#task-details)
4. [Acknowledgements](#acknowledgements)

## System Description Paper  
Our paper can be found [here](https://drive.google.com/file/d/1Dw-xHANOpHaNHW7v-DlZVQes3Uv3Saot/view?usp=sharing).  
Our presentation for the conference can be found [here]().

## Results  
Results of different models on the test dataset can be found here:
The results have been in terms of the **mAP Strict** (mean average precision) and **mAP Relaxed** (mean average precision) scores.

The table below represents results of models on vanilla text, for additional information refer to our [paper](insert link)

| Model         | map Strict (mean) | map Relaxed (mean) |
|---------------|-------------------|--------------------|
| BERT-base     | 0.804             | 0.910              |
| Roberta-base  | 0.826             | 0.930              |
| BART-base     | 0.824             | 0.908              |
| DeBerta-base  | 0.894             | 0.973              |
| BERT-large    | 0.821             | 0.924              |
| Roberta-large | 0.892             | 0.970              |
| BART-large    | 0.909             | 0.982              |
| DeBerta-large | 0.889             | 0.979              |

Our Final Leaderboard Test mAP Strict: **0.872** ; mAP Relaxed: **0.966**  
Post Evaluation Leaderboard Test mAP Strict: **0.921** ; mAP Relaxed: **0.982**

### File descriptions:  


### How to Run:

We have combined the three files of each of the train and dev sets into single train.csv and val.csv files that are too large to upload on github, so I have added them on drive and shared the link over here.

Combined Dataset with Features - https://tinyurl.com/CombinedDatasetWithFeatures


## Task Details

8th ArgMining Workshop Quantitative Summarization – Key Point Analysis Shared Task
=========================================================================


<details><summary><b>Overview</b></summary>
<p>
Key Point Analysis (KPA) is a new NLP task, with strong relations to Computational Argumentation, Opinion Analysis, and Summarization (Bar-Haim et al., ACL-2020; Bar-Haim et al., EMNLP-2020.). Given an input corpus, consisting of a collection of relatively short, opinionated texts focused on a topic of interest, the goal of KPA is to produce a succinct list of the most prominent key-points in the input corpus, along with their relative prevalence. Thus, the output of KPA is a bullet-like summary, with an important quantitative angle and an associated well-defined evaluation framework. Successful solutions to KPA can be used to gain better insights from public opinions as expressed in social media, surveys, and so forth, giving rise to a new form of a communication channel between decision makers and people that might be impacted by the decision.
  
</p>
</details>

<details><summary><b>Important Dates</b></summary>
<p>

* 2021-04-22: Training data release, Development phase leaderboard available 
* 2021-04-06: Test data release; Evaluation start
* 2021-09-21: Evaluation end
* 2021-10-02: System description paper deadline
* 2021-10-18: Deadline for reviews of system description papers
* 2021-10-25: Author notifications
* 2021-11-01: Camera-ready description paper deadline
* 2021-12-13: [TextGraphs-14 workshop](https://sites.google.com/view/textgraphs2020)

Dates are specified in the ISO 8601 format.
</p>
</details>
