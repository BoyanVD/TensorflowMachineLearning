import pygame
import os
import random

PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
GAP = 250
VELOCITY = 5 # In out implementation the pipes are moving towards the bird

class Pipe:

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMAGE, False, True)
        self.PIPE_BOTTOM = PIPE_IMAGE

        self.passed = False
        self.set_height()


    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + GAP


    def move(self):
        self.x -= VELOCITY


    def draw(self, window):
        window.blit(self.PIPE_TOP, (self.x, self.top))
        window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))


    def get_top_pipe_mask(self):
        return pygame.mask.from_surface(self.PIPE_TOP)


    def get_bottom_pipe_mask(self):
        return pygame.mask.from_surface(self.PIPE_BOTTOM)


    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = self.get_top_pipe_mask()
        bottom_mask = self.get_bottom_pipe_mask()

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        bottom_collision_point = bird_mask.overlap(bottom_mask, bottom_offset)
        top_collision_point = bird_mask.overlap(top_mask, top_offset)

        return (bottom_collision_point or top_collision_point)
