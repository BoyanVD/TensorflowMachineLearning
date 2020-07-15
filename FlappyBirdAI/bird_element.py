import pygame
import os

class Bird:
    IMAGES = [
        pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))),
        pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))),
        pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))
    ]
    MAX_ROTATION = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x # the x coordinate of bird
        self.y = y # the y coordinate of bird
        self.tilt = 0 #
        self.tick_count = 0 # tracks when we last jumped
        self.height = self.y
        self.image_count = 0
        self.image = self.IMAGES[0]
        self.velocity = 0


    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y


    def move(self):
        self.tick_count += 1
        # Calculating the arc of the bird's jump
        displacement = self.velocity * self.tick_count + 1.5 * self.tick_count**2

        if displacement >= 16:
            displacement = 16
        elif displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        # Making the tilting
        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VELOCITY


    def draw(self, window):
        self.image_count += 1

        # Makes the flapping bird animation simulation (makes the wings of the bird flapping up and down)
        if self.image_count < self.ANIMATION_TIME:
            self.image = self.IMAGES[0]
        elif self.image_count < self.ANIMATION_TIME*2:
            self.image = self.IMAGES[1]
        elif self.image_count < self.ANIMATION_TIME*3:
            self.image = self.IMAGES[2]
        elif self.image_count < self.ANIMATION_TIME*4:
            self.image = self.IMAGES[1]
        elif self.image_count == self.ANIMATION_TIME*4 + 1:
            self.image = self.IMAGES[0]
            self.image_count = 0

        if self.tilt <= -80:
            self.image = self.IMAGES[1]
            self.image_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)

        window.blit(rotated_image, new_rect.topleft)


    def get_mask(self):
        return pygame.mask.from_surface(self.image)
