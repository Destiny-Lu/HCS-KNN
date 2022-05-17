# %%
import torch
import torch.nn as nn
import logging
import wandb
import numpy as np
import os
import random

from transformers import (
    get_linear_schedule_with_warmup, 
    AdamW,
    set_seed
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (
    RandomSampler, 
    SequentialSampler, 
    DataLoader
)
from utils.Loss import lifted_loss
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from Models.models import Model
from Models.RoBERTaBaseLineModel import RobertaPDTBModel
from Config import get_Config
from utils.KNN import  KNeighborsClassifier
from sklearn.metrics import f1_score


# %%
logger = logging.getLogger(__name__)

# %%
def train(args, model, train_dataset):

    model.train()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay' : args.weight_decay
        },
        {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay' : 0.0
        }
    ]

    total_steps = len(train_dataset) * args.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  batch size = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", total_steps)

    global_steps = 0
    train_loss, logging_loss = 0.0, 0.0 
    model.zero_grad()
    
    cse = CrossEntropyLoss()
    for epoch, _ in tqdm(enumerate(range(args.epochs)), desc='Epochs', total=args.epochs):#, disable=args.local_rank not in [-1, 0]):
        for step, batch in tqdm(enumerate(train_dataloader), desc='Iteration', total=len(train_dataset) // args.batch_size):#, disable=args.local_rank not in [-1, 0]):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs ={
                'input_ids':batch[0], 
                'attention_mask':batch[1], 
                'src_mask':batch[2], 
                'trg_mask':batch[3]
            }

            output = model(**inputs)

            lifted = lifted_loss(output[0], batch[4])
            generation_label = torch.cat((batch[0][:,:64], batch[0][:,128:192]), dim=1).reshape(-1,1).squeeze(-1)
            cseloss = 0.0*cse(output[1].reshape(-1, 50265), generation_label).mean()

            loss = lifted #+ cseloss 

            train_loss += loss.item() 

            optimizer.zero_grad()



            loss.backward()


            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_steps += 1

            if global_steps % 10 == 0:
                wandb.log(
                    {
                        'loss':train_loss / 10
                    },
                    step=global_steps,
                )
                train_loss = 0
    
    model.encoder.save_pretrained(args.save_path + "Full")

def trans_train(args, model, clf, train_dataset, dev_dataset, test_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay' : args.weight_decay
        },
        {'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay' : 0.0
        }
    ]

    total_steps = len(train_dataloader) * args.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Total optimization steps = %d", total_steps)
    
    global_steps = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    best_dev_metric = 0.0
    best_test_metric = 0.0
    best_test_acc = 0.0
    best_test_f1 = 0.0

    # patience = args.aptience

    for epoch, _ in tqdm(enumerate(range(args.epochs)), desc='Epochs', total=args.epochs):
        for batch in tqdm(train_dataloader, desc='Iteration', total=len(train_dataset) // args.batch_size):
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "args" : args, 
                'input_ids':batch[0], 
                'attention_mask':batch[1], 
                'labels':batch[2]
            }

            loss, logits = model(**inputs)

            train_loss += loss.item() 

            optimizer.zero_grad()


            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_steps += 1

            if global_steps % 10 == 0:

                #记录loss
                wandb.log(
                    {
                        'loss':(train_loss - logging_loss) / 10
                    },
                    step=global_steps,
                )
                logging_loss = train_loss

            if global_steps % args.logging_steps == 0:
                results = evaluate(args, model, dev_dataset, clf=clf)
                results_test = evaluate(args, model, test_dataset, test=True, clf=clf)
                # fitlog.add_metric({'eval_acc': results['acc'], 'eval_f1': results['f1']}, step=global_step)
                # fitlog.add_metric({'test_acc': results_test['acc'], 'test_f1': results_test['f1']}, step=global_step)
                wandb.log(
                    {
                        'eval_acc': results['acc'], 
                        'eval_f1': results['f1'], 
                        'test_acc': results_test['acc'], 
                        'test_f1': results_test['f1']
                    },
                    step=global_steps,
                )
                    # if (best_dev_metric < results['f1'] and best_test_metric < results_test['f1']) or (best_test_metric < results_test['f1']):
                if best_test_acc < results_test['acc']:
                # if best_test_metric < results_test['f1']:
                    best_test_acc = results_test['acc']
                    # patience = args.patience              # 每次迭代都是回复patience次
                    wandb.run.summary["best_acc"] = best_test_acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Save to main
                    model_to_save.save_pretrained(args.output_dir + args.task + "_" + str(args.seed) + 'best_acc')
                    logger.info("Saving model checkpoint to %s", args.output_dir)

                if best_dev_metric < results['f1']:
                    best_dev_metric = results['f1']
                    best_test_metric = results_test['f1']
                    # patience = args.patience              # 每次迭代都是回复patience次
                    wandb.run.summary["best_f1"] = best_dev_metric
                    wandb.run.summary["best_test_f1"] = best_test_metric
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Save to main
                    model_to_save.save_pretrained(args.output_dir + args.task + "_" + str(args.seed) + 'best_dev')
                    logger.info("Saving model checkpoint to %s", args.output_dir)


                if best_test_f1 < results_test['f1']:
                # if best_test_metric < results_test['f1']:
                    best_test_f1 = results_test['f1']
                    # patience = args.patience              # 每次迭代都是回复patience次
                    wandb.run.summary["best_test_f1"] = best_test_f1
                    model_to_save = model.module if hasattr(model, 'module') else model
                    # Save to main
                    model_to_save.save_pretrained(args.output_dir + args.task + "_" + str(args.seed) + 'best_f1')
                    logger.info("Saving model checkpoint to %s", args.output_dir)

                model.train()


def evaluate(args, model, dataset, test=False, clf=None, k=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    results = {}
    

    # logger.info(eval_task_names)      # ('pdtb2_level1',)
    # logger.info(eval_outputs_dirs)  # ('./model/',)


    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    
    preds = None
    out_label_ids = None
    alt_label_ids = None
    batch_alt_label_ids = None
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            if k == None:
                logits = clf.predict(logits)
            else:
                logits = clf.predict(logits, k=k)
                
            

            if preds is None:
                preds = logits
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                if batch_alt_label_ids is not None:
                    alt_label_ids = batch_alt_label_ids.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                if batch_alt_label_ids is not None:
                    alt_label_ids = np.append(alt_label_ids, batch_alt_label_ids.detach().cpu().numpy(), axis=0)

        # 把preds 都转成0,1
        
    result = comput_metrics(preds, out_label_ids, args.task)
    results.update(result)

    return results

# 评价指标
def binary_accuracy(preds, labels):
    acc = (preds == labels).astype(int)
    # acc = (preds == labels).astype(int) + ((alt_labels != labels) & (preds == alt_labels)).astype(int)
    return {'acc': acc.mean()}

def binary_acc_and_f1(preds, labels, task_name=None):                 # 二分类看F1, 多分类看macro—f1
    acc = binary_accuracy(preds, labels)['acc']
    f1 = 0
    if task_name == 'pdtb2_4':
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    else:
        f1 = f1_score(y_true=labels, y_pred=preds)

    logger.info('labels:' + str(labels) + str(len(labels)))
    logger.info('preds: ' + str(preds) + str(len(preds)))
    logger.info("f1:" + str(f1))
    logger.info("acc:" + str(acc))
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2 
    }

def comput_metrics(preds, labels, task_name):
    assert len(preds) == len(labels)
    return binary_acc_and_f1(preds, labels, task_name)

def create_cls_dataset(args, model, dataset, task, nums):
    label_map = {
        "Expansion":0, 
        "Contingency":1, 
        "Comparison":2, 
        "Temporal":3
    }
    relset_map = {
        0:[],
        1:[], 
        2:[],
        3:[],  
    }

    relcenter_map = [np.array([0.0 for _ in range(768)]) for _ in range(4)]

    for data in tqdm(dataset, desc='searching cls data...', total=len(dataset)):
        model_output = model(input_ids=data[0].unsqueeze(0).to(args.device), attention_mask=data[1].unsqueeze(0).to(args.device), state='eval').squeeze(0).cpu().detach()
        model_output = torch.nn.functional.normalize(model_output, dim=0).numpy()
        key = int(data[2])
        relset_map[key].append(model_output)
        relcenter_map[key] += model_output

    for i in range(4):
        relcenter_map[i] /= len(relset_map[i])

    data = []
    Y = []
    relcet = None
    
    for i in range(4):
        relcet = torch.tensor(relset_map[i]).float()
        relcenter = torch.tensor(relcenter_map[i]).unsqueeze(-1).float()
        scores = torch.mm(relcet, relcenter)
        relcet = torch.cat((relcet, scores), dim=1)
        relcet = sorted(relcet.tolist(), key=lambda x:x[768], reverse=True)
        if data == None:
            data = torch.tensor(relcet)[:nums, :768].tolist()
            if task == 'pdtb2_4':
                Y = nums * [i]
            elif i == label_map[task]:
                Y = nums * [1]
            else:
                Y = nums * [0]
        else:
            data = data + torch.tensor(relcet)[:nums, :768].tolist()
            if task == 'pdtb2_4':
                Y = Y + nums * [i]
            elif i == label_map[task]:
                Y = Y + nums * [1] 
            else:
                Y = Y + nums * [0]

    return np.array(data), np.array(Y)


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ[ 'PYTHONHASHSEED ' ] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
if __name__ == "__main__":

    args = get_Config()

    if args.do_full_train:
        import os 
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_no)
        train_dataset = torch.load(args.data_dir + "Full_pdtb2_4_train")
        device = torch.device('cuda:' + str(args.cuda_no) if torch.cuda.is_available() else "cpu")
        args.device = device

        model = Model(args)
        model.to(args.device)
        wandb.init(
            project=f"{args.project}", 
            config=args, 
            entity=args.wandb_entity, 
            name=args.name, 
        )

        train(args, model, train_dataset)

    if args.do_trans_train:
        args.device = torch.device('cuda:' + str(args.cuda_no) if torch.cuda.is_available() else "cpu")
        set_seed(args.seed)
        # model = Model(args).encoder.from_pretrained("./model/Full")
        model = RobertaPDTBModel.from_pretrained("./model/Full")
        # model.from_pretrained("./model/Full")
        model.to(args.device)
        model.eval()
        #选择置信度高的样本作为分类器样本
        if os.path.exists("./data/" + args.task + "/clf_dataset"):
            cls_dataset = torch.load("./data/" + args.task + "/clf_dataset")
        else:
            pdtb24_data = torch.load('./data/pdtb2_4/Trans_train')
            cls_dataset = create_cls_dataset(args, model, pdtb24_data, args.task, nums=args.clf_examples_nums)
            torch.save(cls_dataset, "./data/" + args.task + "/clf_dataset")
        # model.from_pretrained("./model/Full")
        clf = KNeighborsClassifier(n_neighbors=10, weights="distance")
        clf.fit(cls_dataset[0], cls_dataset[1]) 

        train_dataset = torch.load('{}{}/Trans_train'.format(args.data_dir, args.task))
        dev_dataset = torch.load('{}{}/Trans_dev'.format(args.data_dir, args.task))
        test_dataset = torch.load('{}{}/Trans_test'.format(args.data_dir, args.task))

        wandb.init(
            project=f"{args.project}", 
            config=args, 
            entity=args.wandb_entity, 
            name=args.name, 
        )

        model.train()

        trans_train(args, model, clf, train_dataset, dev_dataset, test_dataset)

    

# # %%
# torch.__version__

# %%



