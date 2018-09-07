---
title: summary of Structuring Machine Learning Projects
date: 2018-04-03
copyright: true
categories: english
tags: [Structuring Machine Learning Projects, deep learning]
mathjax: true
mathjax2: true
toc: true
top: true
---

## Note
This is my personal summary after studying the course, [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects) and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

### Content Table of my personal notes

$1_{st}$ week: [01_ml-strategy-1](/2018-04-01/01_ml-strategy-1/)
- [01_introduction-to-ml-strategy](/2018-04-01/01_ml-strategy-1/#01_introduction-to-ml-strategy)
  - [01_why-ml-strategy](/2018-04-01/01_ml-strategy-1/#01_why-ml-strategy)
  - [02_orthogonalization](/2018-04-01/01_ml-strategy-1/#02_orthogonalization)
    - [Orthogonalization](/2018-04-01/01_ml-strategy-1/#Orthogonalization)
- [02_setting-up-your-goal](/2018-04-01/01_ml-strategy-1/#02_setting-up-your-goal)
  - [01_single-number-evaluation-metric](/2018-04-01/01_ml-strategy-1/#01_single-number-evaluation-metric)
  - [02_satisficing-and-optimizing-metric](/2018-04-01/01_ml-strategy-1/#02_satisficing-and-optimizing-metric)
    - [summary](/2018-04-01/01_ml-strategy-1/#summary)
  - [03_train-dev-test-distributions](/2018-04-01/01_ml-strategy-1/#03_train-dev-test-distributions)
    - [summary](/2018-04-01/01_ml-strategy-1/#summary)
  - [04_size-of-the-dev-and-test-sets](/2018-04-01/01_ml-strategy-1/#04_size-of-the-dev-and-test-sets)
    - [Summary](/2018-04-01/01_ml-strategy-1/#Summary)
  - [05_when-to-change-dev-test-sets-and-metrics](/2018-04-01/01_ml-strategy-1/#05_when-to-change-dev-test-sets-and-metrics)
    - [summary](/2018-04-01/01_ml-strategy-1/#summary)
- [03_comparing-to-human-level-performance](/2018-04-01/01_ml-strategy-1/#03_comparing-to-human-level-performance)
  - [01_why-human-level-performance](/2018-04-01/01_ml-strategy-1/#01_why-human-level-performance)
    - [summary](/2018-04-01/01_ml-strategy-1/#summary)
  - [02_avoidable-bias](/2018-04-01/01_ml-strategy-1/#02_avoidable-bias)
    - [summary](/2018-04-01/01_ml-strategy-1/#summary)
  - [03_understanding-human-level-performance](/2018-04-01/01_ml-strategy-1/#03_understanding-human-level-performance)
    - [Summary](/2018-04-01/01_ml-strategy-1/#Summary)
- [04_surpassing-human-level-performance](/2018-04-01/01_ml-strategy-1/#04_surpassing-human-level-performance)
  - [summary](/2018-04-01/01_ml-strategy-1/#summary)
- [05_improving-your-model-performance](/2018-04-01/01_ml-strategy-1/#05_improving-your-model-performance)


$2_{nd}$ week: [02_ml-strategy-2](/2018-04-02/02_ml-strategy-2/)
- [01_error-analysis](/2018-04-02/02_ml-strategy-2/#01_error-analysis)
  - [01_carrying-out-error-analysis](/2018-04-02/02_ml-strategy-2/#01_carrying-out-error-analysis)
  - [02_cleaning-up-incorrectly-labeled-data](/2018-04-02/02_ml-strategy-2/#02_cleaning-up-incorrectly-labeled-data)
  - [03_build-your-first-system-quickly-then-iterate](/2018-04-02/02_ml-strategy-2/#03_build-your-first-system-quickly-then-iterate)
- [02_mismatched-training-and-dev-test-set](/2018-04-02/02_ml-strategy-2/#02_mismatched-training-and-dev-test-set)
  - [01_training-and-testing-on-different-distributions](ns)
  - [02_bias-and-variance-with-mismatched-data-distributions](ns)
  - [03_addressing-data-mismatch](/2018-04-02/02_ml-strategy-2/#03_addressing-data-mismatch)
- [03_learning-from-multiple-tasks](/2018-04-02/02_ml-strategy-2/#03_learning-from-multiple-tasks)
  - [01_transfer-learning](/2018-04-02/02_ml-strategy-2/#01_transfer-learning)
  - [02_multi-task-learning](/2018-04-02/02_ml-strategy-2/#02_multi-task-learning)
- [04_end-to-end-deep-learning](/2018-04-02/02_ml-strategy-2/#04_end-to-end-deep-learning)
  - [01_what-is-end-to-end-deep-learning](/2018-04-02/02_ml-strategy-2/#01_what-is-end-to-end-deep-learning)
  - [02_whether-to-use-end-to-end-deep-learning](/2018-04-02/02_ml-strategy-2/#02_whether-to-use-end-to-end-deep-learning)
