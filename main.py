# Demonstrates a method of input for human players

from klask_simulator import KlaskSimulator

import pygame

class KeyboardController():
    __position_x = 0
    __position_y = 0

    def __init__(self, force):
        self.force = force

    def getAction(self):
        return (self.__position_x * self.force, self.__position_y * self.force)

    def keyUp_pressed(self):
        self.__position_y += 1

    def keyUp_released(self):
        self.__position_y -= 1

    def keyDown_pressed(self):
        self.__position_y -= 1

    def keyDown_released(self):
        self.__position_y += 1

    def keyLeft_pressed(self):
        self.__position_x -= 1

    def keyLeft_released(self):
        self.__position_x += 1

    def keyRight_pressed(self):
        self.__position_x += 1

    def keyRight_released(self):
        self.__position_x -= 1

# Initialize the simulator
sim = KlaskSimulator(render_mode="human")

sim.reset()

# Initialize the controllers
force = 0.005
p1 = KeyboardController(force)
p2 = KeyboardController(force)

running = True

while running:
    # Check the event queue (only accessable if render_mode="human", is optional)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            # The user closed the window or pressed escape
            running = False

        # Handle inputs for the keyboard controller
        if event.type == pygame.KEYDOWN:
            # P1
            if event.key == pygame.K_a:
                p1.keyLeft_pressed()
            if event.key == pygame.K_d:
                p1.keyRight_pressed()
            if event.key == pygame.K_w:
                p1.keyUp_pressed()
            if event.key == pygame.K_s:
                p1.keyDown_pressed()

            # P2
            if event.key == pygame.K_LEFT:
                p2.keyLeft_pressed()
            if event.key == pygame.K_RIGHT:
                p2.keyRight_pressed()
            if event.key == pygame.K_UP:
                p2.keyUp_pressed()
            if event.key == pygame.K_DOWN:
                p2.keyDown_pressed()
        if event.type == pygame.KEYUP:
            # P1
            if event.key == pygame.K_a:
                p1.keyLeft_released()
            if event.key == pygame.K_d:
                p1.keyRight_released()
            if event.key == pygame.K_w:
                p1.keyUp_released()
            if event.key == pygame.K_s:
                p1.keyDown_released()

            # P2
            if event.key == pygame.K_LEFT:
                p2.keyLeft_released()
            if event.key == pygame.K_RIGHT:
                p2.keyRight_released()
            if event.key == pygame.K_UP:
                p2.keyUp_released()
            if event.key == pygame.K_DOWN:
                p2.keyDown_released()

    p1_action = p1.getAction()
    p2_action = p2.getAction()

    sim.step((p1_action[0] * 1000, p1_action[1] * 1000), sim.PlayerPuck.P1)
    sim.step((p2_action[0] * 1000, p2_action[1] * 1000), sim.PlayerPuck.P2)


sim.close()
