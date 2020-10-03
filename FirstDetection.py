from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

print(execution_path)

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# Image is saved only if it crosses the minimum specified threshold.
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "test.jpeg"), output_image_path=os.path.join(execution_path , "imagenew1.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"],  " : ", eachObject["box_points"] )
    print("--------------------------------")