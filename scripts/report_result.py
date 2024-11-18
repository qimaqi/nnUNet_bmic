import os 
import json 
import numpy as np


amos_labels = ["spleen", "right kidney", "left kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior vena cava", "pancreas", "right adrenal gland"," left adrenal gland", "duodenum", "bladder", "uterus"]

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='report result')
    parser.add_argument('--result_dir', type=str, default='result', help='result directory')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    basepath = '/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/nnUNetTrainer__VideoMAELPlans__3d_fullres_video_mae_conv_decoder'
    result_dirs=["fold_0",  "fold_1",  "fold_2"]
    collect_metric = {} 
    for result_dir in result_dirs:
        summary_file = os.path.join(basepath, result_dir, "validation" ,"summary.json")


        with open(summary_file, "r") as f:
            summary = json.load(f)

        metric_mean_cases = summary["mean"]
        # print("metric_mean_cases",len(metric_mean_cases), metric_mean_cases)
        for i, name in enumerate(amos_labels):
            key_i = str(i+1)
            # print(name, metric_mean_cases[key_i]['Dice'])
            collect_metric[name] = collect_metric.get(name, []) + [metric_mean_cases[key_i]['Dice']]

    mean_all = []
    for name in amos_labels:
        print(name, np.mean(collect_metric[name]))
        mean_all.append(np.mean(collect_metric[name]))

    # overall mean dice
    print("overall mean dice", np.mean(mean_all))
    


    # find best 2 cases
    # summary_per_case = summary["metric_per_case"]
    # cases_mean_dice = {}
    # for case_i in summary_per_case:
    #     case_file = os.path.basename(case_i['prediction_file'])
    #     mean_list = []
    #     for i, name in enumerate(amos_labels):
    #         key_i = str(i+1)
    #         dice_i = case_i['metrics'][key_i]['Dice']
    #         if not np.isnan(dice_i):
    #             mean_list.append(dice_i)
    #     cases_mean_dice[case_file] = np.mean(mean_list)

    # cases_mean_dice = sorted(cases_mean_dice.items(), key=lambda x: x[1], reverse=True)
    # print("cases_mean_dice", cases_mean_dice)

    # averga all fold /usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_results/Dataset219_AMOS2022_postChallenge_task2/nnUNetTrainer__VideoMAELPlans__3d_fullres_video_mae_vit_decoder/pretrain_fold_0


 