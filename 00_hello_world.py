import lightning as L
import os

class YourComponent(L.LightningWork):
   def run(self):
      print('RUN ANY PYTHON CODE HERE')
      # print(os.listdir('/foo'))

# basic component
# component = YourComponent()

# run on a cloud machine ("cpu", "gpu", ...)
# compute = L.CloudCompute("gpu")

# stop the machine when idle for 10 seconds
# compute = L.CloudCompute('gpu', idle_timeout=10)

# if the machine hasn't started after 60 seconds, cancel the work
# compute = L.CloudCompute('gpu', wait_timeout=60)

# spot machines can be turned off without notice, use for non-critical, resumable work
# request a spot machine, after 60 seconds of waiting switch to full-price
# compute = L.CloudCompute('gpu', wait_timeout=60, spot=True)

# use 100 GB of space on that machine (max size: 64 TB)
# compute = L.CloudCompute('gpu', disk_size=100)

# mount the files on the s3 bucket under this path
mount = L.storage.Mount(source="s3://lightning-example-public/", mount_path="/foo") # no va
compute = L.CloudCompute(mounts=mount)

component = YourComponent(cloud_compute=compute)
app = L.LightningApp(component)
