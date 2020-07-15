import pygame
import neat
import time
import os
import random

pygame.font.init()

# Loading images and defying window dimensionality
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

BIRD_IMAGES = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))
]

PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STATISTICS_FONT = pygame.font.SysFont("comicsans", 50)

class Bird:
    IMAGES = BIRD_IMAGES
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



class Pipe:
    GAP = 250
    VELOCITY = 5 # In out implementation the pipes are moving towards the bird

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
        self.bottom = self.height + self.GAP


    def move(self):
        self.x -= self.VELOCITY


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


class Base:
    VELOCITY = 5
    WIDTH = BASE_IMAGE.get_width()
    IMAGE = BASE_IMAGE

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH


    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH


    def draw(self, window):
        window.blit(self.IMAGE, (self.x1, self.y))
        window.blit(self.IMAGE, (self.x2, self.y))



def draw_window(window, birds, pipes, base, score):
    window.blit(BACKGROUND_IMAGE, (0, 0))

    for pipe in pipes:
        pipe.draw(window)

    base.draw(window)

    for bird in birds:
        bird.draw(window)

    text = STATISTICS_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    window.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))

    pygame.display.update()


def fitness(genomes, config):
    neural_networks = []
    genomes_birds = []
    birds = []

    for _, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        neural_networks.append(network)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        genomes_birds.append(genome)


    base = Base(730)
    pipes = [Pipe(700)]

    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    run = True
    score = 0

    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # There is possibility to have two pipes on screen
        pipe_index = 0
        if len(birds) > 0 and len(pipes) > 1 and birds[0].x > pipes[0].PIPE_TOP.get_width():
            pipe_index = 1
        elif len(birds) == 0:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            genomes_birds[x].fitness += 0.1

            output = neural_networks[x].activate((bird.y, abs(bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))
            if output[0] > 0.5:
                bird.jump()

        pipes_to_remove = []
        add_pipe = False

        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    genomes_birds[x].fitness -= 1
                    birds.pop(x)
                    neural_networks.pop(x)
                    genomes_birds.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes_to_remove.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for genome in genomes_birds:
                genome.fitness += 5 # Adding bonus if the bird passed a pipe

            pipes.append(Pipe(700))

        for pipe in pipes_to_remove:
            pipes.remove(pipe)
            pipes_to_remove.remove(pipe)

        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                neural_networks.pop(x)
                genomes_birds.pop(x)

        if score >= 50:
            break

        base.move()
        draw_window(window, birds, pipes, base, score)


def run(configuration_path):
    configuration = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configuration_path)

    population = neat.Population(configuration)
    population.add_reporter(neat.StdOutReporter(True))
    statistics = neat.StatisticsReporter()
    population.add_reporter(statistics)

    # Best object, think of saving it
    winner = population.run(fitness, 50)

if __name__ == "__main__":
    local_directory = os.path.dirname(__file__)
    configuration_path = os.path.join(local_directory, "config-feedforward.txt")
    run(configuration_path)

    main()
