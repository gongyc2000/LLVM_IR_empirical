# An Empirical Study on Divergence of Differently-Sourced LLVM IRs
In solving binary code similarity detection, many approaches choose to operate on certain unified intermediate representations (IRs), such as Low Level Virtual Machine (LLVM) IR, to overcome the cross-architecture analysis challenge induced by the significant morphological and syntactic gaps across the diverse instruction set architectures (ISAs). However, the LLVM IRs of the same program can be affected by diverse factors, such as the acquisition source, i.e., compiled from source code or disassembled and lifted from binary code. While the impact of compilation settings on binary code has been explored, the specific differences between LLVM IRs from varied sources remain underexamined. To this end, we pioneer an in-depth empirical study to assess the discrepancies in LLVM IRs derived from different sources. Correspondingly, an extensive dataset containing nearly 98 million LLVM IR instructions distributed in 808,431 functions is curated with respect to these potential IR-influential factors. On this basis, three types of code metrics detailing the syntactic, structural, and semantic aspects of the IR samples are devised and leveraged to assess the divergence of the IRs across different origins. The findings offer insights into how and to what extent the various factors affect the IRs, providing valuable guidance for assembling a training corpus aimed at developing robust LLVM IR-oriented pre-training models, as well as facilitating relevant program analysis studies that operate on the LLVM IRs.
# Dataset
The dataset can be downloaded at: https://drive.google.com/file/d/12RBiUuW1u3JTmOPPy3pdUy__UtN3uyvl/view?usp=drive_link
# Source
## Step1:Code normalization
Normalize LLVM IR from source code using (src)reg.py  
```python  
python ./dataset_constr/(src)reg.py
```
Normalize LLVM IR from source code using (bin)reg1.py  
```python  
python ./dataset_constr/(bin1)reg1.py
```
## Step2:Model pre-training  
```python  
python ./pretraining_model/run_pretraining.py
```
## Step3:Similarity detection  
```python  
python ./pretraining_model/run_test.py
```

