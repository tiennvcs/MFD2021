
#run code
python ./src/post_processing/filter_bboxs.py \
    --input_path "/storageStudents/K2018/tiendv/tiennv/mfd_2021/detectron2/TOOL_EVALUATION/Predict_folder/Ts00+01_cfg2submission.csv" \
    --output_path "./TOOL_EVALUATION/filter_bbox_submission.csv" \
    --threshold 0.3