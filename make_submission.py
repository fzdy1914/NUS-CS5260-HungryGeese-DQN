import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/ConvD3QNAgentSilent_2.py', arcname="main.py")
    tar.add('state/ConvD3QN_2.pt', arcname="model.pt")
    tar.add('model.py', arcname="model.py")
    tar.add('parameters.py', arcname="parameters.py")
    tar.add('board.py', arcname="board.py")
    tar.add('silent_agent_helper.py', arcname="silent_agent_helper.py")