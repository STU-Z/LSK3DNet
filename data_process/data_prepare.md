### prepare memory local map data

**Example folder structure**:
```
./dataset/SemanticKitti/
   └── sequences/
       ├── 00/
       │   ├── velodyne/    # .bin files
       │   ├── labels/      # .label files
       │   ├── calib.txt    # calibration file
       ├── ├── poses.txt    # pose in lidar frame
       ├── ├── times.txt    # time for every lidar frame
       ├── └── local_map/   # memory local map .bin files
       ├── 08/              # validation
       ├── 11/              # testing
       └── ...
```

