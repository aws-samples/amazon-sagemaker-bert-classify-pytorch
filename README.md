# Sagemaker BERT text classification using PyTorch
 
 This sample show you how to train and deploy BERT on Sagemaker. This uses a text classifiation example using the [Dbpedia ontology dataset](https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2)
 
 To get started, use the notebook [BertTextClassification.ipynb](BertTextClassification.ipynb)
 
 ## Dataset
 We use the Dbpedia ontology dataset, for more details, see https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2
 
 ### Customise for your dataset
 In order to customise this sample, for your own dataset, perform the following steps
 
 1. Create the dataset class, similar to [dbpedia_dataset.py](dbpedia_dataset.py).
 2. Create the label mapper class which maps string labels to zero indexed integer labels, similar to [dbpedia_dataset_label_mapper.py](dbpedia_dataset_label_mapper.py).
 3. Replace the use of classes `DbpediaDataset` and `DbpediaLabelMapper` in [builder.py](builder.py) with your own custom dataset and label mapper class
 
 ## Security
 
 See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
 
 ## License
 
 This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.                 
