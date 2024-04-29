import pygame
pygame.init()

sound_file = "./audio/explosion.mp3" 
pygame.mixer.music.load(sound_file)

pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

pygame.quit()