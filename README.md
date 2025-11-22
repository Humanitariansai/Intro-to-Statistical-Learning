
# **Humanitarians AI — Intro to Statistical Learning**

### *A Work-in-Progress YouTube Course Built by Volunteers Learning-by-Teaching*

**A Program of Humanitarians AI, a 501(c)(3) nonprofit dedicated to ethical AI for social good.**
[https://www.humanitarians.ai/](https://www.humanitarians.ai/)

---

# **Course Overview**

**Intro to Statistical Learning** is a volunteer-built, open-access Humanitarians AI YouTube course where contributors learn foundational statistics and machine learning by teaching it.

Unlike traditional intros to ML, this course emphasizes **computational biology, health, and life sciences**, making it ideal for learners who want to apply statistical learning to real biological and medical questions.

This course evolves continuously through the learning-by-teaching model:

* Volunteers refine the outline
* Labs and notebooks are built first
* The Humanitarians Courses GPT helps generate slides, explanations, visuals, and demos
* YouTube lectures are recorded after the labs stabilize
* New fellows continue adding case studies & updated biological datasets

The course is designed for beginners, recent graduates, and early-career technologists who want to build competence in **statistics, machine learning, data analysis, and computational thinking** with an AI-for-Good orientation.

---

# **Course Structure & Pedagogy**

### The Humanitarians AI “Learn → Build → Teach” model:

1. **Start with an Outline**
   This document is the seed for the evolving course.

2. **Build Hands-On Labs First**
   Labs will cover:

   * Exploratory data analysis
   * Probability and statistics
   * Supervised & unsupervised learning
   * Genomics & health datasets
   * ML pipelines
   * Validation & evaluation

3. **Use Humanitarians Courses GPT to Generate Teaching Materials**
   Volunteers produce:

   * Slides
   * Lecture scripts
   * Python notebooks
   * Assignments
   * Demos
   * Mini-projects

4. **Record YouTube Lectures**
   Short teaching segments:

   * Concept explanations
   * Live coding demos
   * Interpretability walk-throughs
   * Research-oriented biology case studies

5. **Continuous Updating**
   New contributors add:

   * Additional examples
   * Updated datasets
   * New visualizations
   * Health & genomics use cases

---

# **Part I — Foundations of Statistical Learning**

---

## **Module 1 — Introduction + Python + Exploratory Data Analysis**

Topics:

* What is statistical learning?
* Applications in biology & medicine
* Python basics for ML
* Numpy, Pandas, Matplotlib/Seaborn
* EDA fundamentals:

  * summaries
  * distributions
  * correlations
  * biological datasets (gene expression, clinical data)

Labs:

* Load & explore a small biological dataset
* Visualize distributions, missingness, and correlations

---

## **Module 2 — Probability Foundations**

Topics:

* Counting methods
* Random variables, distributions
* PDFs, CDFs, quantiles
* Mean, variance, covariance
* Conditional probability & Bayes’ theorem
* Base rate fallacy
* Joint distributions
* Correlation vs independence
* Bias–variance intuition
* Central Limit Theorem

Labs:

* Probability simulations in Python
* Biological probability examples (e.g., mutation frequencies)

---

## **Module 3 — Core Statistics for Learning**

Topics:

* Bayesian inference (known & unknown priors)
* Conjugate priors
* Frequentist inference
* Confidence intervals
* Significance tests
* Hypothesis testing do/don’t
* Bootstrapping & permutation tests
* Biological example: estimating gene expression differences

Labs:

* Bootstrap confidence intervals
* Permutation test for biological group differences

---

# **Part II — Unsupervised and Supervised Learning**

---

## **Module 4 — Unsupervised Learning & Model Evaluation**

Topics:

* Clustering: k-means, hierarchical, density-based
* Dimensionality reduction: PCA, t-SNE, UMAP
* Validation & cross-validation
* ROC, AUC, confusion matrices
* Choosing the right model
* Biological examples:

  * clustering patient cohorts
  * dimensionality reduction for gene expression

Labs:

* Cluster biological data
* PCA/UMAP visualizations

---

## **Module 5 — Linear Models & Regularization**

Topics:

* Simple & multiple linear regression
* Logistic regression
* Model inference
* Interpretation of coefficients
* Lasso, Ridge, ElasticNet
* Biological examples:

  * predicting disease outcomes
  * modeling protein abundance

Labs:

* Fit & interpret regression models
* Explore regularization effects on noisy biological data

---

## **Module 6 — Classic Supervised Learning Algorithms**

Topics:

* k-NN
* Decision trees
* Random forests
* Support Vector Machines
* Model comparison & selection
* Visualization of decision boundaries
* Biology examples:

  * phenotype prediction
  * medical classification

Labs:

* Train & compare SVM, kNN, trees on medical datasets

---

## **Module 7 — Text Mining & NLP (Applied to Biology)**

Topics:

* Tokenization, vectorization
* TF-IDF
* Word embeddings
* Biomedical text mining
* PubMed abstracts
* Clinical notes preprocessing

Labs:

* Build a small text classifier using biomedical abstracts

---

## **Module 8 — Project Check-In & Research Methods**

Topics:

* How to run a small statistical learning research project
* Reproducible ML pipelines
* Data provenance & documentation
* Reporting early results

Labs:

* Draft a research project plan
* Create a reproducible notebook pipeline

---

## **Module 9 — Time Series & Biological Signals**

Topics:

* Time series decomposition
* ARIMA
* Trend & seasonality
* Biological examples:

  * heart rate
  * sleep cycles
  * gene time-series

Labs:

* Fit & evaluate ARIMA models
* Analyze a physiological time series dataset

---

## **Module 10 — Recommender Systems & Web Data**

Topics:

* Collaborative filtering
* Content-based recommendation
* API-based data collection
* Applications:

  * drug repurposing recommendation
  * personalized health

Labs:

* Build a simple recommender with medical or genomics data

---

## **Module 11 — Probabilistic Models & Networks**

Topics:

* Naïve Bayes
* Bayesian networks
* Social network analysis (SNA)
* Network motifs in biology
* Gene interaction networks

Labs:

* Build Naïve Bayes classifier
* Visualize a biological network

---

## **Module 12 — Neural Networks & Deep Learning**

Topics:

* Perceptrons
* MLPs
* Activation functions
* Loss and optimization
* Intro to deep learning for biology:

  * images
  * sequences
  * expression data

Labs:

* Train a simple neural network in PyTorch or TensorFlow

---

## **Module 13 — Course Review + Final Projects**

Topics:

* Review of statistical learning
* Integrating concepts
* Presenting ML work
* Ethical considerations in biomedical ML

Labs:

* Capstone: Build & present a full ML project

---

# **Part III — Social Impact Labs (AI for Good)**

Every Humanitarians AI course includes modules aligned with the mission.

Volunteers can apply statistical learning to:

* Food insecurity
* Public health
* Resource allocation
* Medical diagnostics
* Environmental health
* Accessible biological education

Labs:

* Build a model that directly supports a humanitarian or nonprofit use-case

---

# **Course Objectives**

### Understanding

* Explain foundational statistical learning concepts
* Understand probability, inference, and evaluation

### Technical Skills

* Use Python for ML
* Build supervised & unsupervised models
* Analyze biological datasets
* Validate & interpret models

### Teaching Skills

* Create demos, walkthroughs, explanations
* Record short videos
* Use Humanitarians Courses GPT for materials

### Social Impact

* Apply statistical learning to real humanitarian challenges

---

# **Course Materials**

### Primary

* GitHub repository
* Jupyter notebooks
* Humanitarians Courses GPT

### Additional

* Open datasets (biology, health, medical)
* Papers and blog posts
* YouTube lectures (uploaded as produced)

---

# **Repository Layout**

```
Outline/
Labs/
Lectures/
Assignments/
Syllabus/
Module_01_Intro/
Module_02_Probability/
Module_03_Statistics/
Module_04_Unsupervised/
Module_05_LinearModels/
Module_06_Supervised/
Module_07_NLP/
Module_08_Research/
Module_09_TimeSeries/
Module_10_Recommenders/
Module_11_Networks/
Module_12_DeepLearning/
Module_13_Review/
Datasets/
Cheatsheets/
Reading/
Notebooks/
MD/
GEM/
GPT_Assistant/
```
