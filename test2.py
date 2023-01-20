import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
ins.segmentImage("files/ex.png", show_bboxes=True, extract_segmented_objects=True,
save_extracted_objects=True, output_image_name="output_image.jpg")

results, output = ins.segmentImage("files/ex.png", show_bboxes=True, output_image_name="result.jpg")

#access the extracted objects from the results
results["extracted_objects"]