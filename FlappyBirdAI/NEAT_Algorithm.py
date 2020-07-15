import neat
import pygame

from bird_element import Bird
from pipe_element import Pipe
from window_element import draw_window, WINDOW_WIDTH, WINDOW_HEIGHT
from base_element import Base


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
