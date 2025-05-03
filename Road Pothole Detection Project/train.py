if __name__=='__main__':
    from ultralytics import YOLO


    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    results = model.train(data="D:/project_kuntal/Project/class_new.yaml", epochs=40)