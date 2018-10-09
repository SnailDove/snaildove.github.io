---
title: summary of nlp sequence models
mathjax: true
mathjax2: true
categories: english
tags: [nlp-sequence-models, deep learning]
date: 2018-06-06
commets: true
copyright: true
toc: true
top: true
---

## Note

This is my personal summary after studying the course, [nlp sequence models](https://www.coursera.org/learn/nlp-sequence-models/), which belongs to Deep Learning Specialization. and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

## My personal note

$1_{st}$ week : [Building a Recurrent Neural Network Step by Step](/2018/06/01/01_recurrent-neural-networks/)
- [01_why-sequence-models](/2018/06/01/01_recurrent-neural-networks/###01_why-sequence-models)
- [02_notation](/2018/06/01/01_recurrent-neural-networks/###02_notation)
- [03_recurrent-neural-network-model](/2018/06/01/01_recurrent-neural-networks/###03_recurrent-neural-network-model)
- [04_backpropagation-through-time](/2018/06/01/01_recurrent-neural-networks/###04_backpropagation-through-time)
- [05_different-types-of-rnns](/2018/06/01/01_recurrent-neural-networks/###05_different-types-of-rnns)
- [06_language-model-and-sequence-generation](/2018/06/01/01_recurrent-neural-networks/###06_language-model-and-sequence-generation)
- [07_sampling-novel-sequences](/2018/06/01/01_recurrent-neural-networks/###07_sampling-novel-sequences)
- [08_vanishing-gradients-with-rnns](/2018/06/01/01_recurrent-neural-networks/###08_vanishing-gradients-with-rnns)
- [09_gated-recurrent-unit-gru](/2018/06/01/01_recurrent-neural-networks/###09_gated-recurrent-unit-gru)
- [10_long-short-term-memory-lstm](/2018/06/01/01_recurrent-neural-networks/###10_long-short-term-memory-lstm)
- [11_bidirectional-rnn](/2018/06/01/01_recurrent-neural-networks/###11_bidirectional-rnn)
- [12_deep-rnns](/2018/06/01/01_recurrent-neural-networks/###12_deep-rnns)

$2_{nd}$ week : [natural language processing word embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/)
- [01_introduction-to-word-embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/##01_introduction-to-word-embeddings)
  - [01_word-representation](/2018/06/02/02_natural-language-processing-word-embeddings/###01_word-representation)
  - [02_using-word-embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/###02_using-word-embeddings)
  - [03_properties-of-word-embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/###03_properties-of-word-embeddings)
  - [04_embedding-matrix](/2018/06/02/02_natural-language-processing-word-embeddings/###04_embedding-matrix)
- [02_learning-word-embeddings-word2vec-glove](/2018/06/02/02_natural-language-processing-word-embeddings/##02_learning-word-embeddings-word2vec-glove)
  - [01_learning-word-embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/###01_learning-word-embeddings)
  - [02_word2vec](/2018/06/02/02_natural-language-processing-word-embeddings/###02_word2vec)
  - [03_negative-sampling](/2018/06/02/02_natural-language-processing-word-embeddings/###03_negative-sampling)
  - [04_glove-word-vectors](/2018/06/02/02_natural-language-processing-word-embeddings/###04_glove-word-vectors)
- [03_applications-using-word-embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/##03_applications-using-word-embeddings)
  - [01_sentiment-classification](/2018/06/02/02_natural-language-processing-word-embeddings/###01_sentiment-classification)
  - [02_debiasing-word-embeddings](/2018/06/02/02_natural-language-processing-word-embeddings/###02_debiasing-word-embeddings)

$3_{rd}$ week : [sequence models attention mechanism](/2018/06/03/03_sequence-models-attention-mechanism/)
- [01_various-sequence-to-sequence-architectures](/2018/06/03/03_sequence-models-attention-mechanism/##01_various-sequence-to-sequence-architectures)
  - [01_basic-models](/2018/06/03/03_sequence-models-attention-mechanism/###01_basic-models)
  - [02_picking-the-most-likely-sentence](/2018/06/03/03_sequence-models-attention-mechanism/###02_picking-the-most-likely-sentence)
  - [03_beam-search](/2018/06/03/03_sequence-models-attention-mechanism/###03_beam-search)
  - [04_refinements-to-beam-search](/2018/06/03/03_sequence-models-attention-mechanism/###04_refinements-to-beam-search)
  - [05_error-analysis-in-beam-search](/2018/06/03/03_sequence-models-attention-mechanism/###05_error-analysis-in-beam-search)
  - [06_bleu-score-optional](/2018/06/03/03_sequence-models-attention-mechanism/###06_bleu-score-optional)
  - [07_attention-model-intuition](/2018/06/03/03_sequence-models-attention-mechanism/###07_attention-model-intuition)
  - [08_attention-model](/2018/06/03/03_sequence-models-attention-mechanism/###08_attention-model)
- [02_speech-recognition-audio-data](/2018/06/03/03_sequence-models-attention-mechanism/##02_speech-recognition-audio-data)
  - [01_speech-recognition](/2018/06/03/03_sequence-models-attention-mechanism/###01_speech-recognition)
  - [02_trigger-word-detection](/2018/06/03/03_sequence-models-attention-mechanism/###02_trigger-word-detection)

[conclusion of Deep Learning Specialization and thank-you](/2018/06/03/03_sequence-models-attention-mechanism/###conclusion-and-thank-you)

## My personal programming assignments
$1_{st}$ week:
1. [Building a Recurrent Neural Network Step by Step](/2018/06/02/Building+a+Recurrent+Neural+Network+-+Step+by+Step+-+v3)
2. [Dinosaurus Island Character level language model final](/2018/06/02/Dinosaurus+Island+--+Character+level+language+model+final+-+v3/)
3. [Improvise a Jazz Solo with an LSTM Network](/2018/06/02/Improvise+a+Jazz+Solo+with+an+LSTM+Network+-+v3/)

$2_{nd}$ week:
1. [Word Vector Representation](/2018/06/03/Operations+on+word+vectors+-+v2/)
2. [Emojify](/2018/06/03/Emojify+-+v2/)


$3_{rd}$ week:
1. [machine translation](/2018/06/05/Neural+machine+translation+with+attention+-+v4/)
2. [Trigger word](/2018/06/06/Trigger%20word%20detection%20-%20v1/)
