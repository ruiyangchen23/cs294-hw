import mujoco_py
from os.path import dirname


model = mujoco_py.load_model_from_path('/home/aligula/.mujoco/mjpro150/model/humanoid.xml')
sim = mujoco_py.MjSim(model)

