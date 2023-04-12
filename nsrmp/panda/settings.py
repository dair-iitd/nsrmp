from enum import Enum

ColorList = [
	(0, 0, 1, 1), #BLUE
	(0, 1, 0, 1), #GREEN
	(1, 0, 0, 1), #RED
	(0, 1, 1, 1), #CYAN
	(1, 1, 0, 1), #YELLOW
	(1, 0, 1, 1), #MAGENTA
	(1, 1, 1, 1), #WHITE
]

class Color(Enum):
	BLUE = 0
	GREEN = 1
	RED = 2
	CYAN = 3
	YELLOW = 4
	MAGENTA = 5
	WHITE = 6

class Objects(Enum):
	Block = "cube_small.urdf"
	Tray = "tray/tray.urdf"
	Lego = "lego/lego.urdf"
	BlueDice = "./panda/urdf/dice/blue/Dice.urdf"
	GreenDice = "./panda/urdf/dice/green/Dice.urdf"
	RedDice = "./panda/urdf/dice/red/Dice.urdf"
	CyanDice = "./panda/urdf/dice/cyan/Dice.urdf"
	YellowDice = "./panda/urdf/dice/yellow/Dice.urdf"
	MagentaDice = "./panda/urdf/dice/magenta/Dice.urdf"
	WhiteDice = "./panda/urdf/dice/white/Dice.urdf"

DiceUrdf =  [
	Objects.BlueDice,
	Objects.GreenDice,
	Objects.RedDice,
	Objects.CyanDice,
	Objects.YellowDice,
	Objects.MagentaDice,
	Objects.WhiteDice,
]

WORK_AREA_SIDE = 0.6
PANDA_POSITION = [0, 0.5, 0.5]
TABLE_OFFSET = 0.67
MARGIN = 0.03
TrayPositions = [
	[-0.5,  0.15, TABLE_OFFSET], # LEFT
	[0.5,  0.15, TABLE_OFFSET],  # RIGHT
]

object_dimensions = {
	'Small': 0.0500,
	'Dice': 0.0500,
	'Lego': 0.0625,
}

camera_settings = {
	'small_table_view' : {
		'cameraDistance': 12.4,
		'cameraYaw': 0,
		'cameraPitch': -47,
		'cameraTargetPosition':[0, 8.0, -8.0],
	},
	'small_table_view_2' : {
		'cameraDistance': 12.25,
		'cameraYaw': 0,
		'cameraPitch': -47,
		'cameraTargetPosition':[0, 8.0, -8.0],
	},
	'small_table_view_3' : {
		'cameraDistance': 0.50,
		'cameraYaw': 0,
		'cameraPitch': -47,
		'cameraTargetPosition':[0, 0, 0.64],
	},
	'big_table_view' : {
		'cameraDistance': 12.9,
		'cameraYaw': 0,
		'cameraPitch': -60,
		'cameraTargetPosition': [0, 6.0, -10.0],
	}
}

directions = {
    'above': [0., 0., 1.],
    'behind': [0., 1., 0.],
    'below': [0., 0., -1.],
    'front': [0., -1., 0.],
    'left': [-1., 0., 0.],
    'right': [1., 0., 0.] 
}