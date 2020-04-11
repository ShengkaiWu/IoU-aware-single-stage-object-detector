import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from argparse import ArgumentParser
import json

def split_JSON_file(JSON_file):
    """
    split one JSON file to 5 JSON file with equally size
    :param JSON_file: the JSON file to be splited
    :return:
    """
    if type(JSON_file) == str or type(JSON_file) == unicode:
        anns = json.load(open(JSON_file))
    max_num = int(len(anns)/5) + 1
    files_name = ['file_1.json', 'file_2.json', 'file_3.json', 'file_4.json', 'file_5.json']
    json_results_1 = []
    json_results_2 = []
    json_results_3 = []
    json_results_4 = []
    json_results_5 = []

    for ind, ann in enumerate(anns):
        if ind < max_num:
            json_results_1.append(ann)
        elif ind >= max_num and ind <2*max_num:
            json_results_2.append(ann)
        elif ind >= 2*max_num and ind < 3*max_num:
            json_results_3.append(ann)
        elif ind >= 3*max_num and ind < 4*max_num:
            json_results_4.append(ann)
        elif ind >= 4*max_num:
            json_results_5.append(ann)
    mmcv.dump(json_results_1, files_name[0])
    mmcv.dump(json_results_2, files_name[1])
    mmcv.dump(json_results_3, files_name[2])
    mmcv.dump(json_results_4, files_name[3])
    mmcv.dump(json_results_5, files_name[4])




def main():
    parser = ArgumentParser(description='Split one JSON file to multi JSON file')
    parser.add_argument('--JSON_file', help='the JSON file to be splited')


    args = parser.parse_args()
    split_JSON_file(args.JSON_file)


if __name__ == '__main__':
    main()