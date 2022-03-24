# SiameseXML

Code for _SiameseXML: Siamese networks meet extreme classifiers with 100M labels_

---

## Best Practices for features creation

---

* Adding sub-words on top of unigrams to the vocabulary can help in training more accurate embeddings and classifiers.

---

## Setting up

---

### Expected directory structure

```txt
+-- <work_dir>
|  +-- programs
|  |  +-- siamesexml
|  |    +-- siamesexml
|  +-- data
|    +-- <dataset>
|  +-- models
|  +-- results

```

### Download data for SiameseXML

```txt
* Download the (zipped file) BoW features from XML repository.  
* Extract the zipped file into data directory. 
* Yf.txt file contains label features; Either change the file name of make a soft-link to lbl_X_Xf.txt
* The following files should be available in <work_dir>/data/<dataset> for new datasets (ignore the next step)
    - trn_X_Xf.txt
    - trn_X_Y.txt
    - tst_X_Xf.txt
    - lbl_X_Xf.txt
    - tst_X_Y.txt
    - fasttextB_embeddings_300d.npy or fasttextB_embeddings_512d.npy
* The following files should be available in <work_dir>/data/<dataset> if the dataset is in old format (please refer to next step to convert the data to new format)
    - train.txt
    - test.txt
    - fasttextB_embeddings_300d.npy or fasttextB_embeddings_512d.npy 
```

### Convert to new data format

```perl
# A perl script is provided (in siamesexml/tools) to convert the data into new format
# Either set the $data_dir variable to the data directory of a particular dataset or replace it with the path
perl convert_format.pl $data_dir/train.txt $data_dir/trn_X_Xf.txt $data_dir/trn_X_Y.txt
perl convert_format.pl $data_dir/test.txt $data_dir/tst_X_Xf.txt $data_dir/tst_X_Y.txt
```

## Example use cases

---

### A single learner

The given code can be utilized as follows. A json file is used to specify architecture and other arguments. Please refer to the full documentation below for more details.

```bash
./run_main.sh 0 SiameseXML LF-AmazonTitles-131K 0 108
```

## Full Documentation

```txt
./run_main.sh <gpu_id> <type> <dataset> <version> <seed>

* gpu_id: Run the program on this GPU.

* type
  SiameseXML uses DeepXML[2] framework for training. The classifier is trained in M-IV.
  - SiameseXML: The intermediate representation is not fine-tuned while training the classifier (more scalable; suitable for large datasets).
  - SiameseXML++: The intermediate representation is fine-tuned while training the classifier (leads to better accuracy on some datasets).

* dataset
  - Name of the dataset.
  - SiameseXML expects the following files in <work_dir>/data/<dataset>
    - trn_X_Xf.txt
    - trn_X_Y.txt
    - tst_X_Xf.txt
    - lbl_X_Xf.txt
    - tst_X_Y.txt
    - fasttextB_embeddings_300d.npy or fasttextB_embeddings_512d.npy
  - You can set the 'embedding_dims' in config file to switch between 300d and 512d embeddings.

* version
  - different runs could be managed by version and seed.
  - models and results are stored with this argument.

* seed
  - seed value as used by numpy and PyTorch.
```

### Notes

```txt
* Other file formats such as npy, npz, pickle are also supported.
* Initializing with token embeddings (computed from FastText) leads to noticible accuracy gains. Please ensure that the token embedding file is available in data directory, if 'init=token_embeddings', otherwise it'll throw an error.
* Config files are made available in siamesexml/configs/<framework>/<method> for datasets in XC repository. You can use them when trying out the given code on new datasets.
* We conducted our experiments on a 24-core Intel Xeon 2.6 GHz machine with 440GB RAM with a single Nvidia P40 GPU. 128GB memory should suffice for most datasets.
* The code make use of CPU (mainly for hnswlib) as well as GPU. 
```

## Cite as

```bib
@InProceedings{Dahiya21b,
    author = "Dahiya, K. and Agarwal, A. and Saini, D. and Gururaj, K. and Jiao, J. and Singh, A. and Agarwal, S. and Kar, P. and Varma, M",
    title = "SiameseXML: Siamese Networks meet Extreme Classifiers with 100M Labels",
    booktitle = "Proceedings of the International Conference on Machine Learning",
    month = "July",
    year = "2021"
}
```
## YOU MAY ALSO LIKE
- [DeepXML: A Deep Extreme Multi-Label Learning Framework Applied to Short Text Documents](https://github.com/Extreme-classification/deepxml)
- [DECAF: Deep Extreme Classification with Label Features](https://github.com/Extreme-classification/DECAF)
- [ECLARE: Extreme Classification with Label Graph Correlations](https://github.com/Extreme-classification/ECLARE)
- [GalaXC: Graph Neural Networks with Labelwise Attention for Extreme Classification](https://github.com/Extreme-classification/GalaXC)
## References

---
[1] K. Dahiya, A. Agarwal, D. Saini, K. Gururaj, J. Jiao, A. Singh, S. Agarwal, P. Kar and M. Varma. SiameseXML: Siamese networks meet extreme classifiers with 100M labels. In ICML, July 2021

[2] K. Dahiya, D. Saini, A. Mittal, A. Shaw, K. Dave, A. Soni, H. Jain, S. Agarwal, and M. Varma. Deepxml:  A deep extreme multi-label learning framework applied to short text documents. In WSDM, 2021.

[3] pyxclib: <https://github.com/kunaldahiya/pyxclib>
