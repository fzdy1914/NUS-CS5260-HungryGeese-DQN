import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/ConvD3QNAgent.py', arcname="main.py")
    tar.add('state/ConvD3QN/model_0.pt', arcname="model.pt")
