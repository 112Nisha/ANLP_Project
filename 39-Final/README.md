# ANLP_Project

## Dataset

- Full model dataset - Writing Prompts:
  
[Link to Dataset](https://www.kaggle.com/datasets/ratthachat/writing-prompts)

- Golden BERT dataset - Book8:
    - [Test](wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_test.tsv)
    - [Train](wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_train.tsv)
    - [Validate](wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_valid.tsv)


## Instructions for execution

### NOTE: We have implemented the transformers from scratch as well as by using the gpt2 config.

- Clone the final_submission branch of the gitrepo.

- In order to train/test the transformers from scratch, replace the source and target datasets in the FirstTransformerScratch.py and SecondTransformerScratch.py files to train/test the first transformer (from scratch) and the second transformer (from scratch) respectively. There is a comment above the line where the changes must be made in the files. 

- In order to train/test the transformers, replace the source and target datasets in the FirstTransformer.py and SecondTransformer.py files to train/test the first transformer and the second transformer respectively. There is a comment above the line where the changes must be made in the files. 

- In order to evaluate the entire pipeline, where the output of the first transformer is fed as input to the second transformer, change the source and target datasets in the CombinedEvaluate.py file. Load the saved pre-trained models for golden BERT, the first transformer and the second transformer into the file. There are comments above the lines where the changes must be made in the file.

Run the following command to run the files:

```
python3 filename.py
```

## Assumptions

- We have reduced the size of our maximum story length and the size of our dataset due to computational restrictions.

- We do not have a separate output file, our outputs have been added at the bottom of our report.



