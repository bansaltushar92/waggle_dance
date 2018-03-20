# Automated Bee Waggle Dance Detection

## Requirements

```
pip install requirements.txt
```

You'll need opencv for this package. The istructions are given at <https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html>

## Multiple Waggle Detection

```
python detect.py -v=video_file -m=model_file -d=destination_video_file
```
-v video file for detection (should be less than 20000 frames)
-m pickle file to load model from. default: model.pickle
-a video file for results


```
python train.py -v=video_directory -m=model_file -a=annotation_directory
```
-v directory with videos to train. Each video should be less than 20000 frames
-m pickle file name for saving the model
-a annotation files. For each video name 'vid.MP4' use name 'vid_annot.csv'