import argparse

def get_Config():
    parser = argparse.ArgumentParser()

    ## Wandb settings, please read its official documentation for more details. 
    parser.add_argument('--project', default='', type=str, required=False)
    parser.add_argument('--wandb_entity', default='', type=str, required=False)
    parser.add_argument('--name', default='', type=str, required=False)

    parser.add_argument('--task', default='pdtb2_4', type=str, required=False)
    parser.add_argument('--save_path', default='./model/', type=str, required=False)
    parser.add_argument('--seed', default=42, type=int, required=False)
    parser.add_argument('--batch_size', default=8, type=int, required=False)
    parser.add_argument('--epochs', default=10, type=int, required=False)
    parser.add_argument('--learning_rate', default=1e-4, type=float, required=False)
    parser.add_argument('--weight_decay', default=1e-2, type=float, required=False)
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--cuda_no", type=int, default=3)
    parser.add_argument("--do_full_train", action='store_true')
    parser.add_argument("--do_trans_train", action='store_true')
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default='./model/')
    parser.add_argument("--do_select_k", action='store_true')
    parser.add_argument("--metric_type", type=str, default='acc')
    parser.add_argument("--clf_examples_nums", type=int, default=500)

    args = parser.parse_args()

    return args