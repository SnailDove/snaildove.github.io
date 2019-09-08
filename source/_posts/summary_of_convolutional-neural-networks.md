---
title: summary of convolutional neural networks
date: 2018-05-04
copyright: true
categories: English
tags: [convolutional-neural-networks, deep learning]
mathjax: true
mathjax2: true
toc: true
top: true
---

## Note
This is my personal summary after studying the course, [convolutional neural networks](https://www.coursera.org/learn/convolutional-neural-networks), which belongs to Deep Learning Specialization. and the copyright belongs to [deeplearning.ai](https://www.deeplearning.ai/).

## My personal notes

${1_{st}}$ week: [01_foundations-of-convolutional-neural-networks](/2018/05/01/01_foundations-of-convolutional-neural-networks)
- [01_computer-vision](/2018/05/01/01_foundations-of-convolutional-neural-networks/##01_computer-vision)
- [02_edge-detection-example](/2018/05/01/01_foundations-of-convolutional-neural-networks/##02_edge-detection-example)
- [03_more-edge-detection](/2018/05/01/01_foundations-of-convolutional-neural-networks/##03_more-edge-detection)
- [04_padding](/2018/05/01/01_foundations-of-convolutional-neural-networks/##04_padding)
- [05_strided-convolutions](/2018/05/01/01_foundations-of-convolutional-neural-networks/##05_strided-convolutions)
- [06_convolutions-over-volume](/2018/05/01/01_foundations-of-convolutional-neural-networks/##06_convolutions-over-volume)
- [07_one-layer-of-a-convolutional-network](/2018/05/01/01_foundations-of-convolutional-neural-networks/##07_one-layer-of-a-convolutional-network)
- [08_simple-convolutional-network-example](/2018/05/01/01_foundations-of-convolutional-neural-networks/##08_simple-convolutional-network-example)
- [09_pooling-layers](/2018/05/01/01_foundations-of-convolutional-neural-networks/##09_pooling-layers)
- [10_cnn-example](/2018/05/01/01_foundations-of-convolutional-neural-networks/##10_cnn-example)
- [11_why-convolutions](/2018/05/01/01_foundations-of-convolutional-neural-networks/##11_why-convolutions)

$2_{nd}$ week: [02_deep-convolutional-models-case-studies](/2018/05/01/02_deep-convolutional-models-case-studies)
- [01_case-studies](/2018/05/01/02_deep-convolutional-models-case-studies/##01_case-studies)
    - [01_why-look-at-case-studies](/2018/05/01/02_deep-convolutional-models-case-studies/###01_why-look-at-case-studies)
    - [02_classic-networks](/2018/05/01/02_deep-convolutional-models-case-studies/###02_classic-networks)
    - [03_resnets](/2018/05/01/02_deep-convolutional-models-case-studies/###03_resnets)
    - [04_why-resnets-work](/2018/05/01/02_deep-convolutional-models-case-studies/###04_why-resnets-work)
    - [05_networks-in-networks-and-1x1-convolutions](/2018/05/01/02_deep-convolutional-models-case-studies/###05_networks-in-networks-and-1x1-convolutions)
    - [06_inception-network-motivation](/2018/05/01/02_deep-convolutional-models-case-studies/###06_inception-network-motivation)
    - [07_inception-network](/2018/05/01/02_deep-convolutional-models-case-studies/###07_inception-network)
- [02_practical-advices-for-using-convnets](/2018/05/01/02_deep-convolutional-models-case-studies/##02_practical-advices-for-using-convnets)
    - [01_using-open-source-implementation](/2018/05/01/02_deep-convolutional-models-case-studies/###01_using-open-source-implementation)
    - [02_transfer-learning](/2018/05/01/02_deep-convolutional-models-case-studies/###02_transfer-learning)
    - [03_data-augmentation](/2018/05/01/02_deep-convolutional-models-case-studies/###03_data-augmentation)
    - [04_state-of-computer-vision](/2018/05/01/02_deep-convolutional-models-case-studies/###04_state-of-computer-vision)

$3_{rd}$ week : [03_object-detection](/2018/05/03/03_object-detection/)
- [01_object-localization](/2018/05/03/03_object-detection/##01_object-localization)
- [02_landmark-detection](/2018/05/03/03_object-detection/##02_landmark-detection)
- [03_object-detection](/2018/05/03/03_object-detection/##03_object-detection)
- [04_convolutional-implementation-of-sliding-windows](/2018/05/03/03_object-detection/##04_convolutional-implementation-of-sliding-windows)
- [05_bounding-box-predictions](/2018/05/03/03_object-detection/##05_bounding-box-predictions)
- [06_intersection-over-union](/2018/05/03/03_object-detection/##06_intersection-over-union)
- [07_non-max-suppression](/2018/05/03/03_object-detection/##07_non-max-suppression)
- [08_anchor-boxes](/2018/05/03/03_object-detection/##08_anchor-boxes)
- [09_yolo-algorithm](/2018/05/03/03_object-detection/##09_yolo-algorithm)
- [10_optional-region-proposals](/2018/05/03/03_object-detection/##10_optional-region-proposals)

$4_{th}$ week : [04_special-applications-face-recognition-neural-style-transfer](/2018/05/04/04_special-applications-face-recognition-neural-style-transfer/) 
- [01_face-recognition](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/##01_face-recognition)
  - [01_what-is-face-recognition](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###01_what-is-face-recognition)
  - [02_one-shot-learning](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###02_one-shot-learning)
  - [03_siamese-network](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###03_siamese-network)
  - [04_triplet-loss](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###04_triplet-loss)
  - [05_face-verification-and-binary-classification](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###05_face-verification-and-binary-classification)
- [02_neural-style-transfer](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/##02_neural-style-transfer)
  - [01_what-is-neural-style-transfer](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###01_what-is-neural-style-transfer)
  - [02_what-are-deep-convnets-learning](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###02_what-are-deep-convnets-learning)
  - [03_cost-function](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###03_cost-function)
  - [04_content-cost-function](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###04_content-cost-function)
  - [05_style-cost-function](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###05_style-cost-function)
  - [06_1d-and-3d-generalizations](/2018/05/03/04_special-applications-face-recognition-neural-style-transfer/###06_1d-and-3d-generalizations)

## My personal programming assignments

$1_{st}$ week : [Convolution model Step by Step](/2018/05/01/Convolution+model+-+Step+by+Step+-+v2/)
$2_{nd}$ week : [Keras Tutorial Happy House](/2018/05/02/Keras+-+Tutorial+-+Happy+House+v2/), [Residual Networks](/2018/05/02/Residual+Networks+-+v2/)
$3_{rd}$ week : [Autonomous driving - Car detection](/2018/05/03/Autonomous+driving+application+-+Car+detection+-+v3/)
$4_{th}$ week : [Deep Learning & Art Neural Style Transfer](/2018/05/04/Art+Generation+with+Neural+Style+Transfer+-+v3/), [Face Recognition for the Happy House](/2018/05/04/Face+Recognition+for+the+Happy+House+-+v3/)
