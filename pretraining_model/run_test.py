import torch
from transformers import (RobertaTokenizer, RobertaConfig, RobertaModel, AdamW,
                          get_linear_schedule_with_warmup)
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse
from utils import TextDataset
from models import Similary
import json
import logging

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


def evaluate(model, device,tokenizer, args):
    eval_data_list = load_data_from_json(args.test_filename)
    eval_dataset = TextDataset(eval_data_list)

    eval_data_loader = DataLoader(eval_dataset, args.batch_size)
    model.eval()
    cm_all_predicted = []
    cm_all_labels = []
    eval_loss = 0.0
    eval_steps = 0
    for encoded_code1, encoded_code2, labels in eval_data_loader:
        input1_encoding = convert_to_encoding(encoded_code1, tokenizer, args.max_token_length)
        input2_encoding = convert_to_encoding(encoded_code2, tokenizer, args.max_token_length)
        loss , output= model(input1_encoding.to(device), input2_encoding.to(device), labels.to(device))
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
    print(accuracy)
    logger.info(f'Positive Precision: {precision_pos * 100:.4f}%')
    logger.info(f'Negative Precision: {precision_neg * 100:.4f}%')
    logger.info(f'Positive Recall: {recall_pos * 100:.4f}%')
    logger.info(f'Negative Recall: {recall_neg * 100:.4f}%')
    logger.info(f'Positive F1 Score: {f1_score_pos:.4f}')
    logger.info(f'Negative F1 Score: {f1_score_neg:.4f}')
    return accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_filename", default='', type=str, required=True,
                        help="test data file name")
    parser.add_argument("--pretraining_model_path", default='', type=str, required=True,
                        help="classifier model")
    parser.add_argument("--max_token_length", default=256, type=int, required=False,
                        help="the max number of token length")
    parser.add_argument("--batch_size", default=4, type=int, required=False,
                        help="the number of batch size")
    parser.add_argument("--gpu", default=0, type=int, required=False,
                        help="the number of gpu to use")

    args = parser.parse_args()

    print(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")

    similary_encoder = RobertaModel.from_pretrained(args.pretraining_model_path, config=config)
    similary_model = Similary(similary_encoder)
    # similary_model.load_state_dict(torch.load(args.pretraining_model_path, map_location=device))
    # model = torch.load(args.classifier_model)
    similary_model.to(device)
    accuracy_new = evaluate(similary_model, device, tokenizer, args)


if __name__ == '__main__':
    main()
