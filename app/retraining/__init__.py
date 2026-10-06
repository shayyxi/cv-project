"""
Retraining loop for the PPE detector, kept apart from the pipeline's
own modules:

    label queue  --(scripts.roboflow_upload)-->  Roboflow  -->  human labelling
    Roboflow version  --(scripts.train_ppe)-->  ultralytics  -->  weights/best.pt

Modules:
    config   ROBOFLOW_* settings from the environment / .env, Roboflow connection
    upload   push new label-queue crops (with COCO pre-labels) to Roboflow, once each
    dataset  generate / download a dataset version in YOLO format, fix data.yaml
    train    fine-tune with ultralytics and install best.pt
"""
