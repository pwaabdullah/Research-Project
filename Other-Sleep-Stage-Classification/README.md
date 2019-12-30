# Deep Learning in Classifying Sleep Stages
This work presents a deep feed-forward neural network classifier to automatically classify the stages of sleep using raw data taken from a single electropalatogram channel (Fpz-Cz). No features are extracted at all from the data, and the network can classify the five sleep stages: waking, Nl, N2, N3, N4, and rapid eye movement. The network has three layers, takes as an input a l-s epochs to be classified, and requires no signal pre-processing nor feature extraction. We trained and evaluated our system using DeepLearning4J, the free Java framework for test data taken from PhysioNet's Polysomnography Sleep database. An accuracy of 0.99 within a constrained environment has been reached.

* Language/Tools: Java, deepLearning4J
* Published Article: https://ieeexplore.ieee.org/document/8846973

M. H. Al-Meer and Al Mamun, Abdullah "Deep Learning in Classifying Sleep Stages," 2018 Thirteenth International Conference on Digital Information Management (ICDIM), Berlin, Germany, 2018, pp. 12-17.
