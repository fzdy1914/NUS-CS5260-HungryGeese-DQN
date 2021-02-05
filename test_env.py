from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

test_env = make("hungry_geese", debug=True)

# test_env.run(["test_agent.py", "test_agent.py"])
# test_env.step()

print(test_env.configuration)