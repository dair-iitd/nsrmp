#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : objects.py
# Author : Rahul Jain
# Email  : rahuljain13101999@gmail.com
# Date   : 19/06/2021

import math
import random
import numpy as np
from panda.settings import Objects, Color, DiceUrdf, ColorList

class Cube():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
    objct = bullet.loadURDF(Objects.Block.value, position, orn, flags=flags, globalScaling=1)
    bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
    self.type = 'Cube'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion


class Dice():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
    objct = bullet.loadURDF(DiceUrdf[color_idx].value, position, orn, flags=flags, globalScaling=1)
    self.type = 'Dice'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion


class Lego():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    orn = bullet.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
    objct = bullet.loadURDF(Objects.Lego.value, position, orn, flags=flags, globalScaling=2.10) ## 1.25*(31/18.5)
    bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
    self.type = 'Lego'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    self.rotation = orn #rotation is measured in Quaternion

class Tray():
  def __init__(self, bullet, flags, position, color_idx = 1, orn = None) -> None:
    objct = bullet.loadURDF(Objects.Tray.value, position, flags=flags, globalScaling=0.5)
    bullet.changeVisualShape(objct, -1, rgbaColor=ColorList[color_idx])
    self.type = 'Tray'
    self.object_idx = objct
    self.color = (color_idx, Color(color_idx).name)
    self.position = list(position)
    
