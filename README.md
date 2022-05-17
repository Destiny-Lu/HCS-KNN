## Enhance Representational Differentiation Step By Step: A Two-Stage Encoder-Decoder Network for Implicit Discourse Relation Classification

### Data

We use PDTB 2.0 to evaluate our models. Due to the LDC policy, we cannot release the PDTB data. If you have bought data from LDC, please put the pdtb2.csv file in ./data/pdtb2.csv.

### Package Dependencies

- numpy==1.18.1
- pandas==1.3.4
- scikit-learn==1.0.1
- scipy==1.7.2
- torch==1.10.0
- tqdm==4.62.3
- transformers==4.12.5
- wandb==0.12.10

### Preprocessing

You can complete data preprocessing by simply running the process.py script

### Training

You can train models in different settings, including 4-way or the binary classification for each relation.

#### stage-one

The first stage runs the following code. You can also directly use the first stage checkpoint provided by the folder Full and put it into the model folder.

```
python main.py \
    --project trans_Full \
    --batch_size 32 \
    --epochs 20 \
    --learning_rate 2e-5 \
    --name baseline \
    --cuda_no 0 \
    --do_full_train
```

#### stage-two

For the 4-way classification, please set the **task** parameter to **pdtb2_4** and for the binary classification, please et the **task** parameter to **Temporal**、**Comparison**、**Contingency** and **Expansion** respectively. The **project** and **name** is the settings for [wandb](wandb.ai), please read its official documentation for more details.

```
python3 main.py \
    --project \
    --task \
    --name \
    --cuda_no 0 \
    --epochs 20 \
    --learning_rate 1e-5 \
    --batch_size 32 \
    --logging_steps 100 \
    --seed \
    --do_trans_train
```

