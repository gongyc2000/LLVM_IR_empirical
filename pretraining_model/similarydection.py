import torch
from torch.utils.data import DataLoader, Dataset

from models import Similary
import torch.nn.functional as F
import json
from tqdm import tqdm
import logging
import argparse
from transformers import (RobertaTokenizer, RobertaConfig, RobertaModel, AdamW,
                          get_linear_schedule_with_warmup)
import logging
from utils import TextDataset, set_seed, draw_plot, get_date, makedir
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.metrics import confusion_matrix
import os
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def load_data_from_json(file_path):
    with open(file_path,'r') as file:
        data = json.load(file)
    return [(item['code_before'],item['code_after'],item['label'])for item in data]

def convert_to_encoding(encoded_code, tokenizer, max_token_length):
    code = encoded_code

    # 将文本数据编码为模型可以接受的格式
    func_after_encoding = tokenizer(code, return_tensors="pt", padding="max_length", max_length=max_token_length,
                                    truncation=True)
    return func_after_encoding

def train(model, train_data_loader, tokenizer,device, args):
    ymd, hms = get_date()
    dataset_name = args.train_json_path
    checkpoint_path = os.path.join(args.output_dir, 'best-acc-checkpoint', ymd + '-' + hms)
    tensorboard_path = os.path.join(args.output_dir, 'tbd-logs', 'no-pretraining', ymd, hms)
    logger.info("Best accuracy checkpoint path created in: %s", checkpoint_path)
    path_dict = {'best_acc': checkpoint_path}
    makedir(path_dict)
    writer = SummaryWriter(log_dir=str(tensorboard_path))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(args.num_epochs * len(train_data_loader)) * 0.1,
                                                num_training_steps=(args.num_epochs * len(train_data_loader)))

    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=(args.num_epochs * len(train_data_loader)) * 0.1, num_training_steps=(args.num_epochs * len(train_data_loader)))

    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1_score = 0.0

    model.to(device)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_steps = 0
        pbar = tqdm(train_data_loader, mininterval=20)
        for encoded_code1, encoded_code2, labels in pbar:
            optimizer.zero_grad()
            input1_encoding=convert_to_encoding(encoded_code1,tokenizer,args.max_token_length)
            input2_encoding=convert_to_encoding(encoded_code2,tokenizer,args.max_token_length)
            loss , _= model(input1_encoding.to(device), input2_encoding.to(device), labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            train_steps += 1
            pbar.set_description(f"epoch {epoch} loss {round(total_loss / train_steps, 5)}")
        logger.info(f"Epoch {epoch} is finished, the average loss is {total_loss / train_steps}")
        if args.test_json_path != '':
            eval_params = evaluate(model, device, tokenizer,args)
            if eval_params['accuracy'] > best_accuracy:
                # 保存准确率最高的模型
                #torch.save(model.state_dict(), os.path.join(str(checkpoint_path), 'best-acc-model.bin'))    # 暂时关闭保存
                best_accuracy = eval_params['accuracy']
            if eval_params['precision'] > best_precision:
                best_precision = eval_params['precision']
            if eval_params['recall'] > best_recall:
                best_recall = eval_params['recall']
            if eval_params['f1_score'] > best_f1_score:
                best_f1_score = eval_params['f1_score']
            writer.add_scalar(dataset_name + '/Evaluation/Accuracy', eval_params['accuracy'], epoch)
            writer.add_scalar(dataset_name + '/Evaluation/Precision', eval_params['precision'], epoch)
            writer.add_scalar(dataset_name + '/Evaluation/Recall', eval_params['recall'], epoch)
            writer.add_scalar(dataset_name + '/Evaluation/F1 Score', eval_params['f1_score'], epoch)
            writer.add_scalar(dataset_name + '/Evaluation/Avg Loss', eval_params['avg_loss'], epoch)
            writer.add_scalar(dataset_name + '/Train/Avg Loss', total_loss / train_steps, epoch)
    writer.add_hparams(
        {'max_token_length': args.max_token_length, 'lr': args.learning_rate, 'bsize': args.batch_size,
         'epochs': args.num_epochs, 'seed': args.seed},
        {
            dataset_name + '/best_accuracy': best_accuracy,
            dataset_name + '/best_precision': best_precision,
            dataset_name + '/best_recall': best_recall,
            dataset_name + '/best_f1_score': best_f1_score
        })
    writer.close()

def evaluate(model, device,tokenizer, args):
    # model.to(device)
    eval_data_list = load_data_from_json(args.test_json_path)
    eval_dataset = TextDataset(eval_data_list)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_data_loader = DataLoader(eval_dataset, args.batch_size, sampler=eval_sampler)
    model.eval()
    cm_all_predicted = []
    cm_all_labels = []
    eval_loss = 0.0
    eval_steps = 0
    for encoded_code1, encoded_code2, labels in eval_data_loader:
        input1_encoding = convert_to_encoding(encoded_code1, tokenizer, args.max_token_length).to(device)
        input2_encoding = convert_to_encoding(encoded_code2, tokenizer, args.max_token_length).to(device)
        loss , output= model(input1_encoding, input2_encoding, labels.to(device))
        custom_threshold=0.5
        predicted = output>=custom_threshold
        cm_all_predicted.extend(predicted.cpu().numpy())
        cm_all_labels.extend(labels.int().cpu().numpy())
        eval_loss += loss.item()
        eval_steps += 1
    avg_loss = eval_loss / eval_steps
    conf_matrix = confusion_matrix(cm_all_labels, cm_all_predicted)
    tn, fp, fn, tp = conf_matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_pos = tp / (tp + fp)
    precision_neg = tn / (tn + fn)
    recall_pos = tp / (tp + fn)
    recall_neg = tn / (tn + fp)
    f1_score_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
    f1_score_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)
    logger.info(f'Confusion Matrix: {conf_matrix}')
    logger.info(f'Accuracy: {accuracy * 100:.4f}%')
    logger.info(f'Positive Precision: {precision_pos * 100:.4f}%')
    logger.info(f'Negative Precision: {precision_neg * 100:.4f}%')
    logger.info(f'Positive Recall: {recall_pos * 100:.4f}%')
    logger.info(f'Negative Recall: {recall_neg * 100:.4f}%')
    logger.info(f'Positive F1 Score: {f1_score_pos:.4f}')
    logger.info(f'Negative F1 Score: {f1_score_neg:.4f}')
    return {'accuracy': accuracy, 'precision': precision_pos, 'recall': recall_pos, 'f1_score': f1_score_pos,
            'avg_loss': avg_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json_path", default='', type=str, required=True,
                        help="train data file name")
    parser.add_argument("--test_json_path", default='', type=str, required=True,
                        help="train data file name")
    parser.add_argument("--pretraining_model_path", default='', type=str, required=False,
                        help="contrastive learned model")
    parser.add_argument("--learning_rate", default='2e-5', type=float, required=True,
                        help="classifier learned rate")
    parser.add_argument("--max_token_length", default=256, type=int, required=False,
                        help="the max number of token length")
    parser.add_argument("--batch_size", default=4, type=int, required=False,
                        help="the number of batch size")
    parser.add_argument("--num_epochs", default=10, type=int, required=False,
                        help="the number of epochs to train")
    parser.add_argument("--seed", default=42, type=int, required=True,
                        help="the number of seed to use")
    parser.add_argument("--gpu", default=0, type=int, required=False,
                        help="the number of gpu to use")

    args = parser.parse_args()
    args.output_dir = '/home/yons/person/gyc/ContraCVD(1)/ContraCVD/output/similarydection'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("args: %s", args)
    set_seed(args.seed)  # TODO: 加入到args中
    # device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")



    similary_encoder = RobertaModel.from_pretrained(args.pretraining_model_path, config=config)
    similary_model=Similary(similary_encoder)
    train_data_list = load_data_from_json(args.train_json_path)
    train_dataset = TextDataset(train_data_list)
    train_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, args.batch_size, sampler=train_sampler)
    train(similary_model, train_data_loader, tokenizer, device, args)
    # eval_params = evaluate(similary_model, device, tokenizer, args)


if __name__ == '__main__':
    main()
