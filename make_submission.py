import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/ConvDQNAgentSilent.py', arcname="main.py")
    tar.add('state/ConvDQN/model_3.pt', arcname="model.pt")
