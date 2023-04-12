from enum import Enum

#4-vec specification of the colour (R,G,B,A): R = Red, G = Green, B = Blue, A = Alpha (0 = invisible, 1 = visible)
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
	Blue = 0
	Green = 1
	Red = 2
	Cyan = 3
	Yellow = 4
	Magenta = 5
	White = 6

class Objects(Enum):
	Block = "cube_small.urdf"
	Tray = "tray/tray.urdf"
	Lego = "lego/lego.urdf"
	BlueDice = "./urdf/dice/blue/Dice.urdf"
	GreenDice = "./urdf/dice/green/Dice.urdf"
	RedDice = "./urdf/dice/red/Dice.urdf"
	CyanDice = "./urdf/dice/cyan/Dice.urdf"
	YellowDice = "./urdf/dice/yellow/Dice.urdf"
	MagentaDice = "./urdf/dice/magenta/Dice.urdf"
	WhiteDice = "./urdf/dice/white/Dice.urdf"

DiceUrdf =  [
	Objects.BlueDice,
	Objects.GreenDice,
	Objects.RedDice,
	Objects.CyanDice,
	Objects.YellowDice,
	Objects.MagentaDice,
	Objects.WhiteDice,
]
#The amount of area that we use to place the objects
WORK_AREA_SIDE = 0.6

PANDA_POSITION = [0, 0.5, 0.5]
# The height of the table in the urdf file is 0.6 from (0,0). Hence, the offset should be more than 0.6 otherwise the objects will not be visible
TABLE_OFFSET = 0.67

MARGIN = 0.03
TrayPositions = [
	[-0.5,  0.15, TABLE_OFFSET], # LEFT
	[0.5,  0.15, TABLE_OFFSET],  # RIGHT
]

# the size of the objects. Currently the size of cube, dice and lego objects loaded from urdf files are <0.05,0.05,0.05>. 
# If you want to increase the size of the object, please change the values here. The default will be 0.05
object_dimensions = {
	'Cube': 0.0500,
	'Dice': 0.0500,
	'Lego': 0.0625,
}

directions = {
    'above': [0., 0., 1.],
    'behind': [0., 1., 0.],
    'below': [0., 0., -1.],
    'front': [0., -1., 0.],
    'left': [-1., 0., 0.],
    'right': [1., 0., 0.] 
}

camera_settings = {
	'small_table_view' : {
		'cameraDistance': 12.4,
		'cameraYaw': 0,
		'cameraPitch': -47,
		'cameraTargetPosition':[0, 8.0, -8.0],
	},
	'small_table_view_2' : {
		'cameraDistance': 12.3,
		'cameraYaw': 0,
		'cameraPitch': -47,
		'cameraTargetPosition':[0, 8.0, -8.0],
	},
	'small_table_view_3' : {
		'cameraDistance': 1.20,
		'cameraYaw': 0,
		'cameraPitch': -47,
		'cameraTargetPosition':[0, 0, 0.64],
	},
	'big_table_view' : {
		'cameraDistance': 12.9,
		'cameraYaw': 0,
		'cameraPitch': -60,
		'cameraTargetPosition': [0, 6.0, -10.0],
	},
	'big_table_view_2' : {
		'cameraDistance': 12.3,
		'cameraYaw': 0,
		'cameraPitch': -46,
		'cameraTargetPosition':[0, 8.0, -7.5],
	},
	'big_table_view_3' : {
		'cameraDistance': 13.0,
		'cameraYaw': 0,
		'cameraPitch': -49,
		'cameraTargetPosition':[0, 8.0, -8.2],
	},
}
