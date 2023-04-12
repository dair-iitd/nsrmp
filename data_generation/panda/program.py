from .dsl import PandaDSL, DSL

class PandaProgram(PandaDSL):
	def __init__(self, bullet_client, offset, panda_config):
		PandaDSL.__init__(self, bullet_client, offset, panda_config)
		self.index = 0
		self.program = []

	def isProgramExecuting(self):
		return len(self.program) > 0

	def setProgram(self, program):
		if self.isProgramExecuting() or self.isExecuting(): return False
		print("Seting up new program ...")
		self.program = program
		return True

	def update_state(self):
		super().update_state()
		if(self.isProgramExecuting()):
			success = self.ExecuteDSL(self.program[0])
			if (success): self.program.pop(0)

class PandaProgramManager(PandaProgram):
	""" Maintain a buffer for program """
	def __init__(self, bullet_client, offset, panda_config):
		PandaProgram.__init__(self, bullet_client, offset, panda_config)
		self.program_queue = []

	def pop_queue(self): self.program_queue.pop(0)
	def is_empty_queue(self): return len(self.program_queue) == 0
	def next_program(self): return self.program_queue[0]
	def queue_program(self, p): self.program_queue.append(p.copy())

	def schedule_next_program(self):
		if self.is_empty_queue(): return
		next_program = self.next_program()
		success = self.setProgram(next_program)
		if success: self.pop_queue()	

	def update_state(self):
		super().update_state()
		self.schedule_next_program()

	def is_simulation_complete(self):
		return self.is_empty_queue() and (not self.isProgramExecuting()) and (not self.isExecuting())