#This directory contains the files of syncronization timestamps file for Video and isit ground_truth signal. In our case, the video and ground_truth were not in sync.
- if your data is in sync then comment line
```
self.videcg_sync = VidEcg_sync(self.sync_sig_file)
```
in class thermaldataset() in dataloader.py file
