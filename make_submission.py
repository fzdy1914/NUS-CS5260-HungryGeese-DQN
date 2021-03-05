import tarfile
import os.path

with tarfile.open('submission.tar.gz', "w:gz") as tar:
    tar.add('agent/GeeseNetAgent.py', arcname="main.py")
    tar.add('state/GeeseNet_0.pt', arcname="model.pt")
    tar.add('model.py', arcname="model.py")
    tar.add('dense_model.py', arcname="dense_model.py")
    tar.add('parameters.py', arcname="parameters.py")
    tar.add('board.py', arcname="board.py")
    tar.add('board_stack.py', arcname="board_stack.py")
    tar.add('silent_agent_helper.py', arcname="silent_agent_helper.py")
    tar.add('geese_net.py', arcname="geese_net.py")