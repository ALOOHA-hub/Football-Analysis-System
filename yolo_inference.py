from ultralytics import YOLO

from utils.config_loader import cfg

model_path = cfg['settings']['model_path']
input_video_path = cfg['settings']['input_video_path']
conf_thres = cfg['settings']['confidence_threshold']
iou_thres = cfg['settings']['iou_threshold']
output_path = cfg['settings']['output_path']


model = YOLO(model_path)

results = model.predict(input_video_path, save=True, project=output_path, name="inference", exist_ok=True, conf=conf_thres, iou=iou_thres,output=output_path)
print(results[0])
print("===========================")
for box in results[0].boxes:
    print(box)
