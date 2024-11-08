# ANLP_Project

## Dataset

- Full model dataset:
  
[Link to Dataset](https://www.kaggle.com/datasets/ratthachat/writing-prompts)

- Golden BERT dataset - Book8:
    - [Test](wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_test.tsv)
    - [Train](wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_train.tsv)
    - [Validate](wget https://dissent.s3-us-west-2.amazonaws.com/data/discourse_EN_EIGHT_and_but_because_if_when_before_so_though_2017dec18_valid.tsv)

## Directory structure

## Instructions for execution


## To change:
- The stubbing (input into the 2nd transformer)
- Use the [WP] etc things in the first part of the source for each dataset


## Doubts:
- 1st transformer - joining the prompt and outline (or smth like this)
- Loss with the words or probability distro?
- What do we do for 1 sentence (no punctuation)? Do we return 0 bert loss?
