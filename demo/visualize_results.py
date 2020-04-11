from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

config_file = 'configs/retinanet_r50_fpn_1x_2gpu.py'
checkpoint_file = '/home/hustget/mmdetection/work_dirs/0102retinanet_r50_fpn_1x_2gpu/epoch_12.pth'
# checkpoint_file = '/home/hustget/mmdetection/work_dirs/0602retinanet_r50_fpn_1x_2gpu/epoch_12.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img = 'demo/test_images/000000000872.jpg'  # or img = mmcv.imread(img), which will only load it once
# img = 'demo/test_images/000000001490.jpg'
# img = 'demo/test_images/000000002261.jpg'
# img = 'demo/test_images/000000002592.jpg'
# img = 'demo/test_images/000000004765.jpg'
# img = 'demo/test_images/000000005037.jpg'
# img = 'demo/test_images/000000005477.jpg'
# img = 'demo/test_images/000000007784.jpg'
img = 'data/coco/val2017/000000007088.jpg'

result = inference_detector(model, img)
# visualize the results in a new window
show_result(img, result, model.CLASSES)
# or save the visualization results to image files
show_result(img, result, model.CLASSES, out_file='demo/test_results/000000007088.png')

# # add by WSK
# for i in range(len(result)):
#     # visualize the results in a new window
#     show_result(img, result[i], model.CLASSES)
#     # or save the visualization results to image files
#     show_result(img, result[i], model.CLASSES, out_file='result.jpg')


# # added by WSK
# # build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file_IoU_balanced, device='cuda:0')
#
# # test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# show_result(img, result, model.CLASSES)
# # or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')


# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     show_result(frame, result, model.CLASSES, wait_time=1)