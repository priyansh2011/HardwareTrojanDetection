# HardwareTrojanDetection
This repository contains the code for detection of hardware trojan using GNN and Machine Learning and Deep Learning.

Methodolgy used:

Verilog Code⇒ Data Flow Graph(DFG) and Abstract Systax Tree(AST) => GNN
=> Graph Embeddings => Hardware Trojan detection

We reffered to the following repository for graph embeddings generation: https://github.com/AICPS/hw2vec

We have provided the generated embeddings in the AST3 and DFG3 folders along with GNN model weights for both and have also provided the cached graphs files.

One can directly use the embeddings for their use.

We have used TrustHub dataset for this project.

If one wishes to work in this project on their local environment. You would need to clone the repository https://github.com/AICPS/hw2vec and then use it. There would be many
errors due to environment mismatches. To use this code we had to patch the codes of pytorch and pytoch-gemometric. The files are in the repository. Add the linear.py file
in the path /usr/local/lib/python3.7/dist-packages/torch-geometric/nn/Dense/linear.py and module.py file in the path /usr/local/lib/python3.7/dist-packages/torch/nn/Module/module.py.

The file use_case_2_modified2.py should be used if one whished to train GNN first on AST and then the same model on DFG using the same weights.
