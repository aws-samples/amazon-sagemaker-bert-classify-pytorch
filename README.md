# Amazon Sagemaker BERT text classification using PyTorch
 
 This sample show you how to 
 
 - Train [BERT](https://www.aclweb.org/anthology/N19-1423/), using [huggingface](https://huggingface.co/transformers/pretrained_models.html) on [Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html) using [Spot instances](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html). Spot instances allow you to lower training costs.
 - Use **multi-gpu training** where the instance has multiple gpus
 - Implement [Sagemaker checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html) , so when a spot instance terminates, you can resume training from the checkpoint
 - Use [gradient accumulation](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) to train with normal batch sizes even with longer sequences
 - [Deploy](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html) the BERT model for inference. 

  
 To get started, use the notebook [BertTextClassification.ipynb](BertTextClassification.ipynb)
 
 ## Dataset
 We use the [Dbpedia ontology dataset](https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2), for more details, see https://wiki.dbpedia.org/services-resources/dbpedia-data-set-2014#2
 
 ### Customise for your dataset
 In order to customise this sample, for your own dataset, perform the following steps
 
 1. Create a dataset class, that implements the [PyTorch Dataset abstract class](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), see [dbpedia_dataset.py](src/dbpedia_dataset.py) as an example implementation.
 2. Create a label mapper class, that implements abstract class [LabelMapperBase](src/label_mapper_base.py), to maps string labels to zero indexed integer labels. See an example implementation [dbpedia_dataset_label_mapper.py](src/dbpedia_dataset_label_mapper.py).
 3. Replace the use of classes `DbpediaDataset` and `DbpediaLabelMapper` in [builder.py](src/builder.py) with your own custom dataset and label mapper class
 
 ## Security
 
 See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
 
 ## License
 
 This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.                 
