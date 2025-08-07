import pygame, sys, math, random
from pygame.math import Vector2
from pygame.locals import *
import cv2
import mediapipe as mp
import threading
import numpy as np
from colors import Color 

pygame.init()
clock = pygame.time.Clock()
pygame.display.set_caption("Particles Simulation")
WIDTH = 1200
HEIGHT = 800
screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
scale = 0.1
particles_surface =  pygame.Surface((WIDTH/scale, HEIGHT/scale), pygame.SRCALPHA)
color = Color()

# Choose Yours ! ------------------------------------- #
color = color.TURQUOISE # Go check in colors file!
show_camera = True # True or False
pake_tangan = True # True or False
show_pointer = True # True or False
# ---------------------------------------------------- #

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

class Particles:
    def __init__(self, num_particles=20000): 
        np.random.seed(42)
        self.num_particles = num_particles
        self.particles_pos, self.particles_v = self._makeParticles()
        # particles' parameter
        self.pBest = self.particles_pos.copy()
        self.gBest = np.zeros((1, 2), dtype=np.int16)
        self.c1 = np.abs(np.random.normal(0, 0.001, 1)).astype(np.float32)
        self.c2 = np.abs(np.random.normal(0, 0.001, 1)).astype(np.float32)
        self.r1 = np.random.random((self.num_particles, 1)).astype(np.float32)
        self.r2 = np.random.random((self.num_particles, 1)).astype(np.float32)

    def _makeParticles(self):
        rects_pos = np.random.randint(0, WIDTH, (self.num_particles, 2))
        rects_v = np.random.randn(self.num_particles, 2)
        return rects_pos.astype(np.int16), rects_v.astype(np.float32)
        
    def drawParticles(self, colors):
        pixels = pygame.surfarray.pixels3d(screen)  
    
        colors_array = np.array(colors, dtype=np.uint8) 
    
        particle_x = self.particles_pos[:, 0].astype(np.int32)
        particle_y = self.particles_pos[:, 1].astype(np.int32)
    
        valid_x = (particle_x >= 0) & (particle_x < WIDTH)
        valid_y = (particle_y >= 0) & (particle_y < HEIGHT)
        valid = valid_x & valid_y     
        num_valid = np.count_nonzero(valid)
        random_colors = colors_array[np.random.randint(0, len(colors), size=num_valid)]

        pixels[particle_x[valid], particle_y[valid]] = random_colors

        del pixels

    def _fitness(self, particles_pos, target_pos):
        distance = np.array(target_pos) - particles_pos
        return np.sqrt(np.sum(np.square(distance), axis=1)).astype(np.float32)

    
    def _getNormalizedDirection(self, target_pos):
        distance = self._fitness(self.particles_pos, target_pos)

        distance = np.where(distance==0, 1, distance)        
        normal = self.particles_pos - target_pos
        return np.array([normal[:,0] / distance, normal[:,1] / distance]).astype(np.float32).T

    def _adjustVel(self, target_pos, speed_multiplier=10.0, force=5.0):
        dist = self._fitness(self.particles_pos, target_pos)
        dist = np.where(dist<force, force, dist)
        normal = self._getNormalizedDirection(target_pos)
        self.particles_v -= (normal * speed_multiplier / dist[:, np.newaxis]).astype(np.float32)

    def _velFriction(self, friction=0.99):
        self.particles_v *= friction 

    def _updatePersonalBest(self, target_pos):
        current_fitness = self._fitness(self.particles_pos, target_pos)
        best_fitness = self._fitness(self.pBest, target_pos)
        bestFitness = np.minimum(current_fitness, best_fitness)
        mask = (current_fitness <= bestFitness) 
        self.pBest[mask] = self.particles_pos[mask] 

    def _updateGlobalBest(self, target_pos):
        current_best_idx = np.argmin(self._fitness(self.particles_pos, target_pos))
        current_best_fitness = self._fitness(self.particles_pos[current_best_idx, np.newaxis], target_pos)
        gBest_fitness = self._fitness(self.gBest, target_pos)
        self.gBest = np.where(gBest_fitness < current_best_fitness, gBest_fitness, self.particles_pos[current_best_idx, np.newaxis]).astype(np.int16)

    def update(self, target_pos):
        self._updatePersonalBest(target_pos)
        self._updateGlobalBest(target_pos)

        personal_based = self.c1 * self.r1 * (self.pBest - self.particles_pos).astype(np.float32)
        global_based =  self.c2 * self.r2 * (self.gBest - self.particles_pos).astype(np.float32)
        self.particles_v += personal_based + global_based

        self._adjustVel(target_pos, speed_multiplier=20.0, force=10.0)
        self._velFriction()

        self.particles_pos = self.particles_pos + self.particles_v

        # keep the particle on the screen
        # self.particles_pos %= (WIDTH, HEIGHT)

    def spreadParticles(self, force=10):
        direction = self.particles_pos - self.gBest
        distance = np.linalg.norm(direction, axis=1, keepdims=True)

        distance = np.where(distance == 0, 1, distance)

        normalized_direction = direction / distance
        self.particles_v += normalized_direction * force

        self.particles_pos += self.particles_v

        # Keep particles within screen bounds
        # self.particles_pos %= (WIDTH, HEIGHT)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def hand_detection():
    global telunjuk_x, telunjuk_y, hand_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                telunjuk_x = int(hand_landmarks.landmark[8].x * WIDTH)
                telunjuk_y = int(hand_landmarks.landmark[8].y * HEIGHT)
                if show_pointer == True:
                    pygame.draw.circle(screen, (0, 255, 0), (telunjuk_x, telunjuk_y), 2)
                hand_detected = True

        if show_camera == True:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # q for QUITTT
                # break
                pygame.quit()
                sys.exit()

if pake_tangan == True:
    telunjuk_x, telunjuk_y = WIDTH // 2, HEIGHT // 2
    hand_detected = False
    thread = threading.Thread(target=hand_detection)
    thread.daemon = True
    thread.start()

def main():
    particles = Particles(20000)
    target_pos = (0, 0)
    while True:
        screen.fill((0, 0, 0))

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_Q:
                cap.release()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                particles.spreadParticles()

        if pake_tangan == True:
            target_pos = (telunjuk_x, telunjuk_y)
        else:
            target_pos = pygame.mouse.get_pos()

        particles.update(target_pos)
        particles.drawParticles([color])
        pygame.display.flip()

        # Display FPS
        font = pygame.font.SysFont(None, 35)
        fps = font.render(f"FPS: {int(clock.get_fps())}", True, pygame.Color('white'))
        screen.blit(fps, (10, 10))

        pygame.display.update()
        clock.tick(60)

if __name__ == '__main__':
    main()
