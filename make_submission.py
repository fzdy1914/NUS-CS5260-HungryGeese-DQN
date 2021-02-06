import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/ConvDQNAgentWithSafeGuards.py', arcname="main.py")
    tar.add('state/model_4.pt', arcname=os.path.basename('state/model.pt'))