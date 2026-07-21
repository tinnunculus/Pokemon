import pygame
import os
os.environ["SDL_VIDEODRIVER"] = "x11"

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("SDL Test")

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

pygame.quit()