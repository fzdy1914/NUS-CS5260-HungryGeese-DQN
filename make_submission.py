import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/ConvDQNWithLengthAgentWithSafeGuards.py', arcname="main.py")
    tar.add('state/ConvDQNWithLength/model_1.pt', arcname="model.pt")
