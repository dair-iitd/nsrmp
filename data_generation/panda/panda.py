import itertools
import random
import time
import numpy as np
import math
from enum import Enum
from .settings import ColorList, Color

# Panda Arm Position/Orientaion
PANDA_NUM_DOF = 7
END_EFFECTOR_INDEX = 8 
PANDA_POSITION = np.array([0, 0.5, 0.5])

# Inverse Kinematics
lower_limit = [-7]*PANDA_NUM_DOF
upper_limit = [7]*PANDA_NUM_DOF
joint_range = [7]*PANDA_NUM_DOF
rest_position = [0, 0, 0, -2.24, -0.30, 2.66, 2.32, 0, 0]

# Gripping
TABLE_OFFSET = 0.64
MOVE_HEIGHT = TABLE_OFFSET + 0.30
GRASP_WIDTH = 0.01
RELEASE_WIDTH = 0.05
GRASP_HEIGHT = 0.10
ALPHA = 0.99

TIME_STEP = [1.25, 2.0, 1.25, 0.5, 1.25, 2.0, 1.25, 0.5, 1.25]

TrayPositions = [
	[-0.5,  0.15, TABLE_OFFSET],
	[-0.5, -0.15, TABLE_OFFSET],
	[0.5,  0.15, TABLE_OFFSET],
	[0.5, -0.15, TABLE_OFFSET],
]

class PandaState(Enum):
	INIT = 0
	MOVE = 1
	PRE_GRASP = 2
	GRASP = 3
	POST_GRASP = 4
	MOVE_BW = 5
	PRE_RELEASE = 6
	RELEASE = 7
	POST_RELEASE = 8
	IDLE = 9

class PandaPrimitive(object):
	def __init__(self, bullet_client, offset, config):
		self.bullet_client = bullet_client
		self.offset = np.array(offset)
		flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
		num_blocks = config['num_blocks']
		num_trays = config['num_trays']

		# Panda ARM
		orn = self.bullet_client.getQuaternionFromEuler([0, 0, -math.pi/2])
		self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", self.offset + PANDA_POSITION, orn, useFixedBase=True, flags=flags)
		finger_constraint = self.bullet_client.createConstraint(
			self.panda, 9, self.panda, 10,
			jointType=self.bullet_client.JOINT_GEAR,
			jointAxis=[1, 0, 0],
			parentFramePosition=[0, 0, 0],
			childFramePosition=[0, 0, 0])
		self.bullet_client.changeConstraint(finger_constraint, gearRatio=-1, erp=0.1, maxForce=50)

		# Joint Panda Frame
		index = 0
		num_panda_joints = self.bullet_client.getNumJoints(self.panda)
		for joint_idx in range(num_panda_joints):
			self.bullet_client.changeDynamics(self.panda, joint_idx, linearDamping=0, angularDamping=0)
			info = self.bullet_client.getJointInfo(self.panda, joint_idx)
			jointType = info[2]
			if (jointType == self.bullet_client.JOINT_PRISMATIC) or (jointType == self.bullet_client.JOINT_REVOLUTE):
				self.bullet_client.resetJointState(self.panda, joint_idx, rest_position[index])
				index = index+1

		# Static Objects
		self.bullet_client.loadURDF("plane.urdf", offset, flags=flags)
		self.bullet_client.loadURDF("table/table.urdf", offset, flags=flags)

		def get_random_table_position():
			var_xy = 0.5*(np.random.rand(2) + np.array([-0.5, -0.5]))
			return self.offset + np.array([var_xy[0], var_xy[1], TABLE_OFFSET])

		# Previous : Blocks Implementation 
		self.blocks = []
		self.block_color_sequence = np.random.permutation(len(ColorList))[: num_blocks]
		self.big_block_color_sequence = np.random.permutation(len(ColorList))[: num_blocks]
		self.block_colors = []
		
		for idx in range(num_blocks):
			position = get_random_table_position()
			if random.choice([True, False]):
				orn = self.bullet_client.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
				block = self.bullet_client.loadURDF("cube_small.urdf", position, orn, flags=flags)
				self.bullet_client.changeVisualShape(block, -1, rgbaColor=ColorList[self.block_color_sequence[idx]])
				self.blocks.append(block)
				self.block_colors.append((Color(self.block_color_sequence[idx]).name, 'Small'))
			else:
				orn = self.bullet_client.getQuaternionFromEuler([0, 0, np.random.normal(0, math.pi/24, 1)[0]]) if orn is None else orn
				block = self.bullet_client.loadURDF("lego/lego.urdf", position, orn, flags=flags, globalScaling=2.1)
				self.bullet_client.changeVisualShape(block, -1, rgbaColor=ColorList[self.big_block_color_sequence[idx]])
				self.blocks.append(block)
				self.block_colors.append((Color(self.big_block_color_sequence[idx]).name, 'Lego'))

		# Tray
		self.trays = []
		self.tray_position_sequence = list(itertools.permutations(range(len(TrayPositions)), num_trays))
		self.tray_position_sequence = random.choice(self.tray_position_sequence)
		self.tray_color_sequence = random.choices(range(len(ColorList)), k=num_trays)
		tray_positions = [np.array(TrayPositions[i]) for i in self.tray_position_sequence]
		for idx in range(num_trays):
			tray = self.bullet_client.loadURDF("tray/tray.urdf", self.offset + tray_positions[idx], globalScaling=0.5, flags=flags)
			self.bullet_client.changeVisualShape(tray, -1, rgbaColor=ColorList[self.tray_color_sequence[idx]])
			self.trays.append(tray)

		self.PANDA_FINGER_HORIZONTAL = np.array([0, math.pi, math.pi/2])
		self.PANDA_FINGER_VERTICAL = np.array([0, math.pi, 0])

		# Initialize Variables
		self.t = 0
		self.control_dt = 1./240.
		self.state = PandaState.INIT
		self.finger_width = RELEASE_WIDTH
		self.gripper_height = MOVE_HEIGHT
		self.finder_orientation = self.PANDA_FINGER_VERTICAL
		self.block_position = None
		self.target_position = None
		self.pos = None
		self.panda_visible = True

	def hide_panda_body(self):
		for link_idx in range(-1, 11):
			self.bullet_client.changeVisualShape(self.panda, link_idx, rgbaColor=[0, 0, 0, 0])
		self.panda_visible = False
	
	def show_panda_body(self):
		for link_idx in range(-1, 11):
			self.bullet_client.changeVisualShape(self.panda, link_idx, rgbaColor=[1., 1., 1., 1.])
		self.panda_visible = True

	def update_state(self):
		""" 
			The Control Time of 1 Second For Each State of Panda Execution
			INIT -> Initialization State
			MOVE -> Source State, IDLE -> Terminal State
		"""
		if (self.state != PandaState.IDLE):
			self.t += self.control_dt
			if (self.t > TIME_STEP[self.state.value]):
				self.t = 0
				if self.state == PandaState.INIT:
					self.state = PandaState.IDLE
				else:
					self.state = PandaState(self.state.value+1)

	def pre_execute_command(self):
		pass

	def executeCommand(self, block_pos, target_pos):
		""" Execute Command / Initiliaze State, dt """
		if self.state == PandaState.IDLE:
			self.pre_execute_command()
			self.block_position = block_pos
			self.target_position = target_pos
			self.state = PandaState.MOVE
			self.t = 0

	def isExecuting(self):
		""" If the Panda is in the middle of command Execution """
		return self.state != PandaState.IDLE

	def movePanda(self, pos):
		"""
			Given The Position of the Panda (End_Effector_Part)
			Move The Panda/Joints to reach the location 
		"""
		# Move Panda Body
		self.bullet_client.submitProfileTiming("IK")
		orn = self.bullet_client.getQuaternionFromEuler(self.finder_orientation)
		jointPoses = self.bullet_client.calculateInverseKinematics(
			self.panda, END_EFFECTOR_INDEX, pos, orn, lower_limit, upper_limit, joint_range, rest_position)
		self.bullet_client.submitProfileTiming()
		control_mode = self.bullet_client.POSITION_CONTROL
		for idx in range(PANDA_NUM_DOF):
			self.bullet_client.setJointMotorControl2(self.panda, idx, control_mode, jointPoses[idx], force=1200., maxVelocity=1.0)
		# Move Gripper 
		control_mode = self.bullet_client.POSITION_CONTROL
		self.bullet_client.setJointMotorControl2(self.panda, 9, control_mode, self.finger_width, force=100)
		self.bullet_client.setJointMotorControl2(self.panda, 10, control_mode, self.finger_width, force=100)

	def step(self):

		alpha_change_gripper_height = lambda x: ALPHA * self.gripper_height + (1.0-ALPHA)* x
		
		if self.state == PandaState.INIT:
			self.pos = np.zeros(3)
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		if self.state == PandaState.MOVE:
			self.pos = self.block_position.copy()
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif (self.state == PandaState.PRE_GRASP):
			self.pos = self.block_position.copy()
			self.gripper_height = alpha_change_gripper_height(GRASP_HEIGHT+self.pos[2])
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif self.state == PandaState.GRASP:
			self.finger_width = GRASP_WIDTH
			self.movePanda(self.pos)

		elif self.state == PandaState.POST_GRASP:
			self.pos = self.block_position.copy()
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif self.state == PandaState.MOVE_BW:
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos = self.target_position.copy()
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif self.state == PandaState.PRE_RELEASE:
			self.pos = self.target_position.copy()
			self.gripper_height = alpha_change_gripper_height(GRASP_HEIGHT + self.pos[2])
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif self.state == PandaState.RELEASE:
			self.finger_width = RELEASE_WIDTH
			self.movePanda(self.pos)

		elif self.state == PandaState.POST_RELEASE:
			self.pos = self.target_position.copy()
			self.gripper_height = alpha_change_gripper_height(MOVE_HEIGHT)
			self.pos[2] = self.gripper_height
			self.movePanda(self.pos)

		elif self.state == PandaState.IDLE:
			pass
