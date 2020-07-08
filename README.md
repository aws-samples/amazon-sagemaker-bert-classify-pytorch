# Amazon Sagemaker BERT text classification using PyTorch
 
 This sample show you how to 
 
 - Train [BERT](https://www.aclweb.org/anthology/N19-1423/), using [huggingface](https://huggingface.co/transformers/pretrained_models.html) on [Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html) using **[SPOT instances](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)**
 - This samples also implements **[Sagemaker checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)** , so when a spot instance terimnates, you can resume training from the checkpoint
 - **[Deploy](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)** the BERT for inference. 
 
 This uses a text classifiation example using the [Dbpedia ontology dataset](https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2)
 
 To get started, use the notebook [BertTextClassification.ipynb](BertTextClassification.ipynb)
 
 ## Dataset
 We use the Dbpedia ontology dataset, for more details, see https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2
 
 ### Customise for your dataset
 In order to customise this sample, for your own dataset, perform the following steps
 
 1. Create the dataset class, similar to [dbpedia_dataset.py](src/dbpedia_dataset.py).
 2. Create the label mapper class which maps string labels to zero indexed integer labels, similar to [dbpedia_dataset_label_mapper.py](src/dbpedia_dataset_label_mapper.py).
 3. Replace the use of classes `DbpediaDataset` and `DbpediaLabelMapper` in [builder.py](src/builder.py) with your own custom dataset and label mapper class
 
 ## Security
 
 See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
 
 ## License
 
 This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.                 
