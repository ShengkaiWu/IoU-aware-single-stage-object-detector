import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from argparse import ArgumentParser

import matplotlib.pyplot as plt

def analyze_results(result_file, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    # if result_types == ['proposal_fast']:
    #     ar = fast_eval_recall(result_file, coco, np.array(max_dets))
    #     for i, num in enumerate(max_dets):
    #         print('AR@{}\t= {:.4f}'.format(num, ar[i]))
    #     return

    assert result_file.endswith('.json')
    coco_dets = coco.loadRes(result_file)

    img_ids = coco.getImgIds()
    for res_type in result_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        # cocoEval.evaluate()
        dt_with_iou = cocoEval.analyze_results()
        result_file = 'results_with_iou.json'
        mmcv.dump(dt_with_iou, result_file)

        ious = []
        scores = []
        print('the number of detections is ', len(dt_with_iou))
        num = np.zeros([20], dtype=np.int)
        sum = np.zeros([20], dtype=np.float)
        for ind, dt in enumerate(dt_with_iou):
            # if dt['iou'] < 0.5:
            #     continue

            if dt['iou'] >=0 and dt['iou'] <0.05:
                num[0] = num[0] +1
                sum[0]=sum[0]+dt['score']
            elif dt['iou'] >=0.05 and dt['iou'] <0.1:
                num[1] = num[1] + 1
                sum[1] = sum[1] + dt['score']
            elif dt['iou'] >=0.1 and dt['iou'] <0.15:
                num[2] = num[2] + 1
                sum[2] = sum[2] + dt['score']
            elif dt['iou'] >= 0.15 and dt['iou'] < 0.2:
                num[3] = num[3] + 1
                sum[3] = sum[3] + dt['score']
            elif dt['iou'] >= 0.2 and dt['iou'] < 0.25:
                num[4] = num[4] + 1
                sum[4] = sum[4] + dt['score']
            elif dt['iou'] >= 0.25 and dt['iou'] < 0.3:
                num[5] = num[5] + 1
                sum[5] = sum[5] + dt['score']
            elif dt['iou'] >= 0.3 and dt['iou'] < 0.35:
                num[6] = num[6] + 1
                sum[6] = sum[6] + dt['score']
            elif dt['iou'] >= 0.35 and dt['iou'] < 0.4:
                num[7] = num[7] + 1
                sum[7] = sum[7] + dt['score']
            elif dt['iou'] >= 0.4 and dt['iou'] < 0.45:
                num[8] = num[8] + 1
                sum[8] = sum[8] + dt['score']
            elif dt['iou'] >= 0.45 and dt['iou'] <= 0.5:
                num[9] = num[9] + 1
                sum[9] = sum[9] + dt['score']
            elif dt['iou'] >=0.5 and dt['iou'] <0.55:
                num[10] = num[10] +1
                sum[10]=sum[10]+dt['score']
            elif dt['iou'] >=0.55 and dt['iou'] <0.6:
                num[11] = num[11] + 1
                sum[11] = sum[11] + dt['score']
            elif dt['iou'] >=0.6 and dt['iou'] <0.65:
                num[12] = num[12] + 1
                sum[12] = sum[12] + dt['score']
            elif dt['iou'] >= 0.65 and dt['iou'] < 0.7:
                num[13] = num[13] + 1
                sum[13] = sum[13] + dt['score']
            elif dt['iou'] >= 0.7 and dt['iou'] < 0.75:
                num[14] = num[14] + 1
                sum[14] = sum[14] + dt['score']
            elif dt['iou'] >= 0.75 and dt['iou'] < 0.8:
                num[15] = num[15] + 1
                sum[15] = sum[15] + dt['score']
            elif dt['iou'] >= 0.8 and dt['iou'] < 0.85:
                num[16] = num[16] + 1
                sum[16] = sum[16] + dt['score']
            elif dt['iou'] >= 0.85 and dt['iou'] < 0.9:
                num[17] = num[17] + 1
                sum[17] = sum[17] + dt['score']
            elif dt['iou'] >= 0.9 and dt['iou'] < 0.95:
                num[18] = num[18] + 1
                sum[18] = sum[18] + dt['score']
            elif dt['iou'] >= 0.95 and dt['iou'] <= 1.0:
                num[19] = num[19] + 1
                sum[19] = sum[19] + dt['score']

            ious.append(dt['iou'])
            scores.append(dt['score'])
            # if ind >200000:
            #     break
        average = sum/num
        plt.scatter(ious, scores, 0.1)
        ious_scores = np.stack((ious, scores), axis=-1)
        np.savetxt('ious_scores.csv', ious_scores, delimiter=',', fmt='%.3e', header='iou, score')
        np.savetxt('average_scores.csv', average,  fmt='%f', header='average_score')
        np.savetxt('num_samples.csv', num, fmt='%d', header='num_samples')

        plt.ylabel('classification score')
        plt.xlabel('IoU with ground truth')
        plt.savefig('score_vs_iou')
        plt.show()


def main():
    parser = ArgumentParser(description='Analysis of Detection Results')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    args = parser.parse_args()
    analyze_results(args.result, args.types, args.ann, args.max_dets)


if __name__ == '__main__':
    main()