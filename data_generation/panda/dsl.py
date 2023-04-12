from .panda import PandaPrimitive, PandaState
from scipy.spatial.transform import Rotation
from enum import Enum
import numpy as np

MARGIN = 0.08

class DSL(Enum):
	MOVE_BLOCK = 1
	MOVE_BLOCK_LEFT = 2
	MOVE_BLOCK_RIGHT = 3
	MOVE_BLOCK_TOP = 4
	MOVE_BLOCK_BEFORE = 5
	MOVE_BLOCK_AFTER = 6
	MOVE_BLOCK_TRAY = 7

class PandaDSL(PandaPrimitive):
	""" Define the DSL Code of the Robotic Arm In This Class """
	def __init__(self, bullet_client, offset, config):
		PandaPrimitive.__init__(self, bullet_client, offset, config)

	# =====================================
	# Utility Functions
	# =====================================

	def get_axis_dis(self, sigma_p=0.005, sigma_s=0.005):
		p, s = abs(np.random.normal(0, sigma_p, 1)[0]), np.random.normal(0, sigma_s, 1)[0]
		return 0, 0
		# return (p, s) if p > abs(s) else self.get_axis_dis(sigma_p, sigma_s)

	def get_block_pos_orn(self, block):
		block_id = self.blocks[block]
		block_pos, block_orn = self.bullet_client.getBasePositionAndOrientation(block_id)
		return list(block_pos), list(block_orn)

	def get_block_pos_dim(self, block_target):
		block_target_id = self.blocks[block_target]
		pos, _ = self.bullet_client.getBasePositionAndOrientation(block_target_id)
		dimensions = self.bullet_client.getVisualShapeData(block_target_id)[0][3]
		return list(pos), list(dimensions)
	
	def quaternion_to_Z(self, orn):
		w, x, y, z = orn
		t3 = +2.0*(w*z + x*y)
		t4 = +1.0-2.0*(y*y + z*z)
		Z = np.degrees(np.arctan2(t3, t4))
		return Z

	# =====================================
	# Robot Instructions 
	# =====================================
 
	def MoveBlock(self, block, pos):
		""" Basic instruction to the robot. Moves the block to the given position"""
		block_pos, block_orn = self.get_block_pos_orn(block)
		# rot_euler = Rotation.from_quat(block_orn).as_euler('xyz', degrees=False)
		# self.finder_orientation[2] += rot_euler[2]
		self.executeCommand(block_pos, pos)

	def MoveBlockLeft(self, block, block_target):
		""" Place block to the left of target_block """
		trg, dimensions = self.get_block_pos_dim(block_target)
		variation = self.get_axis_dis()
		trg = [trg[0] - MARGIN, trg[1] + variation[1], trg[2]]
		self.finder_orientation = self.PANDA_FINGER_VERTICAL
		self.MoveBlock(block, trg)

	def MoveBlockRight(self, block, block_target):
		""" Place block to the right of target_block """
		trg, dimensions = self.get_block_pos_dim(block_target)
		trg = [trg[0] + MARGIN, trg[1], trg[2]]
		self.finder_orientation = self.PANDA_FINGER_VERTICAL
		self.MoveBlock(block, trg)

	def MoveBlockTop(self, block, block_target):
		""" Place block on the top of target_block """
		trg, dimensions = self.get_block_pos_dim(block_target)
		variation = self.get_axis_dis()
		# trg = [trg[0]+variation[1], trg[1]+variation[1], trg[2]+dimensions[2]+ MARGIN]
		trg = [trg[0]+variation[1], trg[1]+variation[1], trg[2]+0.06+ MARGIN]
		self.MoveBlock(block, trg)

	def MoveBlockBefore(self, block, block_target):
		""" Place block before/infront of the target_block """
		trg, dimensions = self.get_block_pos_dim(block_target)
		variation = self.get_axis_dis()
		trg = [trg[0] + variation[1], trg[1] - dimensions[1] - variation[0] - MARGIN, trg[2]]
		self.finder_orientation = self.PANDA_FINGER_HORIZONTAL
		self.MoveBlock(block, trg)

	def MoveBlockAfter(self, block, block_target):
		""" Place block behind/after the target_block """
		trg, dimensions = self.get_block_pos_dim(block_target)
		variation = self.get_axis_dis()
		trg = [trg[0] + variation[1], trg[1] + dimensions[1] + variation[0] + MARGIN, trg[2]]
		self.finder_orientation = self.PANDA_FINGER_HORIZONTAL
		self.MoveBlock(block, trg)

	def MoveBlockTray(self, block, tray_target):
		""" Places block on the center of the tray"""
		tray_target_id = self.trays[tray_target]
		trg, _ = self.bullet_client.getBasePositionAndOrientation(tray_target_id)
		variation = self.get_axis_dis()
		trg = [trg[0]+variation[1], trg[1] + variation[1], trg[2] + 0.02]
		self.MoveBlock(block, trg)

	# =====================================
	# Execute DSL
	# =====================================

	def ExecuteDSL(self, command):
		""" Reconfigures the Panda With the DSL Command
				The State of the Panda Idle -> Busy and Executing """
		if self.isExecuting(): return False

		print(f"Command Executing - {command}")
		dsl, arg1, arg2 = command
		if dsl == DSL.MOVE_BLOCK: 				self.MoveBlock(arg1, arg2)
		if dsl == DSL.MOVE_BLOCK_AFTER: 	self.MoveBlockAfter(arg1, arg2)
		if dsl == DSL.MOVE_BLOCK_BEFORE:	self.MoveBlockBefore(arg1, arg2)
		if dsl == DSL.MOVE_BLOCK_LEFT: 		self.MoveBlockLeft(arg1, arg2)
		if dsl == DSL.MOVE_BLOCK_RIGHT: 	self.MoveBlockRight(arg1, arg2)
		if dsl == DSL.MOVE_BLOCK_TOP: 		self.MoveBlockTop(arg1, arg2)
		if dsl == DSL.MOVE_BLOCK_TRAY:		self.MoveBlockTray(arg1, arg2)
		return True
