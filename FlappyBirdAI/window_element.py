import pygame
import os

pygame.font.init()

# Loading images and defying window dimensionality
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bg.png")))

STATISTICS_FONT = pygame.font.SysFont("comicsans", 50)



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
