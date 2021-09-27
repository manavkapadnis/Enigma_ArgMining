# Team Enigma at ArgMining 2021 Shared Task: Leveraging Pretrained Language Models for Key Point Matching

This is our attempt of the shared task on **Quantitative Summarization – Key Point Analysis Shared Task** at the [8th Workshop on Argument Mining](https://2021.argmining.org/shared_task_ibm.html#ibm), part of EMNLP 2021.  

## Authors - Manav Nitin Kapadnis, Sohan Patnaik, Siba Smarak Panigrahi, Varun Madhavan, Abhilash Nandy

Key Point Analysis (KPA) is a new NLP task, with strong relations to Computational Argumentation, Opinion Analysis, and Summarization ([Bar-Haim et al., ACL-2020](https://www.aclweb.org/anthology/2020.acl-main.371.pdf); [Bar-Haim et al., EMNLP-2020](https://arxiv.org/pdf/2010.05369.pdf)). 
Given an input corpus, consisting of a collection of relatively short, opinionated texts focused on a topic of interest, the goal of KPA is to produce a succinct list of the most prominent key-points in the input corpus, along with their relative prevalence. Thus, the output of KPA is a bullet-like summary, with an important quantitative angle and an associated well-defined evaluation framework. Successful solutions to KPA can be used to gain better insights from public opinions as expressed in social media, surveys, and so forth, giving rise to a new form of a communication channel between decision makers and people that might be impacted by the decision. 

Official repository of the task can be found [here](https://github.com/IBM/KPA_2021_shared_task).

### Sections
1. [System Description Paper](#system-description-paper)
2. [Transformer Architecture](#architecture)
3. [Results](#results)
4. [Task Details](#task-details)
5. [Acknowledgements](#acknowledgements)

## System Description Paper
Our paper can be found [here](https://github.com/manavkapadnis/Enigma_ArgMining/blob/main/Paper.pdf).  
<!--Our presentation for the conference can be found [here]()-->

## Transformer Architecture

The Transformer Architecture used by us is shown in the figure. We used the pre-trained models realeased by [HuggingFace](https://huggingface.co/transformers/pretrained_models.html).

![Transformer Architecture](https://github.com/manavkapadnis/Enigma_ArgMining/blob/main/model_architecture.png)

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
1. Ablation Study - This folder contains the python scripts for the ablations studies carried out by us i.e., boosting, addition of definition of nouns to the input, concatenation of argument and key point, average of hidden states. 
2. Appendix C - This folder contains the python notebooks for results described in section C of Appendix. This includes two files, one which implements the model with POS and Dependency features of both the main dataset and the additional dataset (STS and IBM-Rank-30k) and the other implements tf-idf features for both the datasets.
3. Combined Dataset Files Creation - This folder contains python notebooks which are used to create the train and the test dataset.
4. Track 1 - Key Point Matching - This folder contains the python notebooks which implements our baseline model without any additional features, model with additional features (POS and Dependency), model with tf-idf features and model with training of additional data (STS and IBM-Rank-30k).
5. train_dev_test_dataset - This folder contains the original dataset as provided by the organizers.

### How to Run: 
We have combined the three files of each of the train and dev sets into single train.csv and val.csv files that are too large to upload on github, so we have added them on drive and shared the link over here.

Combined Dataset with Features - https://tinyurl.com/CombinedDatasetWithFeatures

First run the notebooks in the 'Combined Dataset Files Creation' which will create the required train and test data. Then, in order to reproduce our results, run the respective codes from the different folders.

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

* 2021-04-22: Training data release; Development phase leaderboard available 
* 2021-06-24: Test data release; Evaluation start
* 2021-06-30: Evaluation end; submission closed
* 2021-07-08: Results Announce
* 2021-08-20: Paper submission due
* 2021-09-15: Notification to authors
* 2021-09-23: Camera-ready version due
* 2021-11-10: ArgMining 2021 workshop (EMNLP)

Dates are specified in the ISO 8601 format.
</p>
</details>

<details><summary><b>Data</b></summary>
<p>
  
### Training Data
**ArgKP** dataset ([Bar-Haim et al., ACL-2020](https://www.aclweb.org/anthology/2020.acl-main.371.pdf)), divided to train/dev sets. 
This dataset contains ~24K argument/key-point pairs, for 28 controversial topics. 
Each of the pairs is labeled as matching/non-matching, as well as assigned a stance towards the topic. 
Given a set of key points for a topic, an argument could be matched to one or more key points, or to none of them. 
The arguments in this dataset are a subset of the **IBM-ArgQ-Rank-30kArgs** dataset ([Gretz et al., 2020](https://arxiv.org/abs/1911.11408)), available [here](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml).

For track 2, participants are also encouraged to utilize the remainder of **IBM-ArgQ-Rank-30kArgs** dataset. 
This dataset contains ~30K crowd-sourced arguments on 71 controversial topics, collected with strict length limitations and accompanied by extensive quality control measures. 
Note that this dataset contains quality score per argument, which will not be available in the test data, but may be utilized for training and analysis.
Participants may use existing services and additional datasets, as long as they are publicly available to the community. 
Participants may not use labeled data unavailable to the community.

### Test Data
A test dataset of three debatable topics will be collected according to guidelines in [Gretz et al., 2020](https://arxiv.org/abs/1911.11408) for the **IBM- ArgQ-Rank-30kArgs** dataset. 
Stance will be provided for each argument, but quality score will not be available in the test setting, even though it is available in the train data and may be utilized for training and analysis.

<ins>Track 1</ins> - In addition to the arguments and topics, the input will contain key points that are expected a-priori to be found in above arguments regarding each topic and stance. 
These key points are compiled by an expert debater, similarly to the key points created in [Bar-Haim et al., EMNLP-2020](https://arxiv.org/pdf/2010.05369.pdf) **ArgKP** dataset. 



**We have combined the three files of each of the train and dev sets into single train.csv and val.csv files that are too large to upload on github, so I have added them on drive and shared the link over here.

Combined Dataset with Features - https://tinyurl.com/CombinedDatasetWithFeatures** 
  
  </p>
</details>

<details><summary><b>Task</b></summary>
<p> 
  
### Track 1 – Key-Point Matching
Given a debatable topic, a set of key points per stance, and a set of crowd arguments supporting or contesting the topic, report for each argument its match score for each of the key points under the same stance towards the topic.

### Track 2 - Key Points Generation and Matching
Given a debatable topic and a set of crowd arguments supporting or contesting the topic, generate a set of key points for each stance of the topic and report for each given argument its match score for each of the key points under the same topic and in the same stance.

### Key points analysis example

Following is an example of key point analysis, as obtained by human labeling on key points provided by an expert, on the topic "Homeschooling should be banned", on the pro stance arguments (taken from Arg-KP dataset):

| Key point  | Matched arguments count |
| ------------- | ------------- |
| Mainstream schools are essential to develop social skills.   | 61 |
| Parents are not qualified as teachers.   | 20 |
| Homeschools cannot be regulated/standardized. | 15 |
| Mainstream schools are of higher educational quality. | 9 |

A few examples of concrete key point to argument matches:

<table>
    <thead>
        <tr>
            <th>Argument</th>
            <th>Matching key point</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>children can not learn to interact with their peers when taught at home</td>
            <td rowspan=3>Mainstream schools are essential to develop social skills</td>
        </tr>
        <tr>
            <td>homeschooling a child denies them valuable lifeskills, particularly interaction with their own age group and all experiences stemming from this.</td>
        </tr>
        <tr>
            <td>to homeschool is in one way giving a child an immersive educational experience, but not giving them the social skills and cooperative skills they need throughout life, so should be banned.</td>
        </tr>
        <tr>
            <td>parents are usually not qualified to provide a suitable curriculum for their children. additionally, children are not exposed to the real world.</td>
            <td>Parents are not qualified as teachers</td>
        </tr>
        <tr>
            <td>it is impossible to ensure that homeschooled children are being taught properly</td>
            <td>Homeschools cannot be regulated/standardized.</td>
        </tr>
    </tbody>
</table>
  
### Track 1 - Key-Point Matching
<ins>Input</ins> : 

Arguments and expert key points for topic and stance in the test dataset.
The input consist of three files:
- arguments.csv - This file lists all the arguments for each topic, along with the stance of each argument towards the topic.
- key_points.csv - This file lists all the key points for each topic, along with the stance of each key point towards the topic.
- labels.csv - This file contains the labeled pairs of argument id and key point id. Note that not all the possible pairs are labeled.

The dataset, split to train and dev, can be found in the folder `kpm_data`

<ins>Output</ins> : 

For each argument, its match score for each of the key points under the same topic and in the same stance towards the topic.

The output file should have the following json format: 

> {"arg_15_0": {"kp_15_0": 0.8282181024551392, "kp_15_2": 0.9438725709915161}, "arg_15_1": {"kp_15_0": 0.9994438290596008, "kp_15_2":0}}

Here for instance, arg_15_0 is matched with two key points. The score for the match with kp_15_2 is 0.9438725709915161.

The submitted zip file should contain a single file named *predictions.p*.

<ins>Evaluation</ins> : 

Test dataset will be pre-labeled according to the guidelines in [Bar-Haim et al., ACL-2020](https://www.aclweb.org/anthology/2020.acl-main.371.pdf), for pairs of argument/key-point as matching/non-matching. 
In the labeling task, each argument is presented in the context of its debatable topic, and the list of key points follows. 
Annotators are guided to mark all of the key points this argument can be associated with, and if none are relevant, select the 'None' option. 

Two scores will be calculated for track 1 - *relaxed* and *strict* mean Average Precision, as follows:
1.	For evaluation purposes, each argument will be paired with the highest scoring key point assigned to it (randomly chosen in case of a tie).
2.	50% of above-described pairs, with lowest matching score, will be removed from the evaluation process. This is since we expect any set of arguments to contain some number of unique claims which do not match any of the key points offered. Based upon what we see in the public dataset, where the fraction arguments not matching any of the given key points is 0.35, yet ranging widely, we choose to evaluate only on top 50% of the pairs for each motion and stance.
3.	Precision for remaining pairs will be calculated based on labeled data. Note that Some of the pairs created this way might form an ambiguous labeling pair, as detailed in [Bar-Haim et al., ACL-2020](https://www.aclweb.org/anthology/2020.acl-main.371.pdf): pairs of argument and key point with undecided labeling (more than 15% of the annotators, yet less than 60% of them marked the pair as a match). Such pairs are excluded from the labeled data. In the strict evaluation score, these pairs will be considered as no match in ground truth, and in the relaxed evaluation score they will be considered as match. 
4.	The final score of a system would be the average rank of the strict and relaxed scores. Each such score is obtained by calculating macro-average of the 6 mean Average Precision values for this system on each topic and stance combination

The evaluation script is: `track_1_kp_matching.py`. To run it, execute:
> `python track_1_kp_matching.py kpm_data_dir predictions_file`

When *kpm_data_dir* stands for the input folder, and *predictions_file* stands for the predictions json file.
  
**This evaluation script is embedded in our code itself so the track_1_kp_matching.py need not be used separately**
</p>
</details>

<details><summary><b>Competition Details</b></summary>
<p>
  
Submission
--------------------

Please submit your solutions via CodaLab: <https://competitions.codalab.org/competitions/31166#participate>

Contacts
--------------------
  
Contact the organizers at `KPA_2021_shared_task_organizers@googlegroups.com'

Terms and Conditions
--------------------
 
By submitting results to this competition, you consent to the public release of your scores at the ArgMining workshop and in the associated proceedings, at the task organizers' discretion. Scores may include but are not limited to, automatic and manual quantitative judgments, qualitative judgments, and such other metrics as the task organizers see fit. You accept that the ultimate decision of metric choice and score value is that of the task organizers. You further agree that the task organizers are under no obligation to release scores and that scores may be withheld if it is the task organizers' judgment that the submission was incomplete, erroneous, deceptive, or violated the letter or spirit of the competition's rules. Inclusion of a submission's scores is not an endorsement of a team or individual's submission, system, or science. You further agree that your system may be named according to the team name provided at the time of submission, or to a suitable shorthand as determined by the task organizers. Wherever appropriate, academic citation for the sending group would be added (e.g. in a paper summarizing the task).

Competitions should comply with any general rules of EMNLP. The organizers are free to penalize or disqualify for any violation of the above rules or for misuse, unethical behaviour or other behaviours they agree are not accepted in a scientific competition in general and in the specific one at hand.
  

  
  


