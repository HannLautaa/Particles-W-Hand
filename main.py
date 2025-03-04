import pygame, sys, math, random
from pygame.math import Vector2
from pygame.locals import *
import cv2
import mediapipe as mp
import threading
import numpy as np
from colors import Color 

# Setup pygame/windos -------------------------------- #
pygame.init()
clock = pygame.time.Clock()
pygame.display.set_caption("Particles Simulation")
WIDTH = 1200
HEIGHT = 800
screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
scale = 0.1
particles_surface =  pygame.Surface((WIDTH/scale, HEIGHT/scale), pygame.SRCALPHA)
color = Color()
# ---------------------------------------------------- #

# Choose Yours ! ------------------------------------- #
color = color.TURQUOISE # Go check in colors file!
show_camera = True # True or False
pake_tangan = True # True or False
# ---------------------------------------------------- #

# mp ------------------------------------------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
# ---------------------------------------------------- #

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
        pixels = pygame.surfarray.pixels3d(screen)  # Get pixel array
    
        # Convert colors to a NumPy array
        colors_array = np.array(colors, dtype=np.uint8)  # Shape: (num_colors, 3)
    
        # Convert particle positions to integers
        particle_x = self.particles_pos[:, 0].astype(np.int32)
        particle_y = self.particles_pos[:, 1].astype(np.int32)
    
        # Ensure positions are within screen bounds
        valid_x = (particle_x >= 0) & (particle_x < WIDTH)
        valid_y = (particle_y >= 0) & (particle_y < HEIGHT)
        valid = valid_x & valid_y  # Combine conditions
    
        # Randomly assign colors to valid particles
        num_valid = np.count_nonzero(valid)
        random_colors = colors_array[np.random.randint(0, len(colors), size=num_valid)]

        # Apply colors to valid positions
        pixels[particle_x[valid], particle_y[valid]] = random_colors

        del pixels
   
    # def drawParticles(self, color):
    #     for pos in self.particles_pos:
    #         pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), 1)  # Draw small circles directly

    def _fitness(self, particles_pos, target_pos):
        distance = np.array(target_pos) - particles_pos
        return np.sqrt(np.sum(np.square(distance), axis=1)).astype(np.float32)

    
    def _getNormalizedDirection(self, target_pos):
        # find the distance
        distance = self._fitness(self.particles_pos, target_pos)

        distance = np.where(distance==0, 1, distance) # avoids division by 0
        normal = self.particles_pos - target_pos
        return np.array([normal[:,0] / distance, normal[:,1] / distance]).astype(np.float32).T # transpose it to be (x, y) arrays

    def _adjustVel(self, target_pos, speed_multiplier=10.0, force=5.0):
        dist = self._fitness(self.particles_pos, target_pos)
        dist = np.where(dist<force, force, dist) # prevent extreme forces when the particle is very close to the target.
        normal = self._getNormalizedDirection(target_pos)
        self.particles_v -= (normal * speed_multiplier / dist[:, np.newaxis]).astype(np.float32)

    def _velFriction(self, friction=0.99):
        self.particles_v *= friction # prevent very high velocity from multiple iteration by always only take friction*100% of it

    def _updatePersonalBest(self, target_pos):
        current_fitness = self._fitness(self.particles_pos, target_pos)
        best_fitness = self._fitness(self.pBest, target_pos)
        bestFitness = np.minimum(current_fitness, best_fitness)
        mask = (current_fitness <= bestFitness) # mask it to find which personal best should be changed
        self.pBest[mask] = self.particles_pos[mask] # update the pBest

    def _updateGlobalBest(self, target_pos):
        current_best_idx = np.argmin(self._fitness(self.particles_pos, target_pos))
        current_best_fitness = self._fitness(self.particles_pos[current_best_idx, np.newaxis], target_pos)
        gBest_fitness = self._fitness(self.gBest, target_pos)
        # update gBest with the lowest fitness
        self.gBest = np.where(gBest_fitness < current_best_fitness, gBest_fitness, self.particles_pos[current_best_idx, np.newaxis]).astype(np.int16)

    def update(self, target_pos):
        # update personal best and global best
        self._updatePersonalBest(target_pos)
        self._updateGlobalBest(target_pos)

        # update velocity
        personal_based = self.c1 * self.r1 * (self.pBest - self.particles_pos).astype(np.float32)
        global_based =  self.c2 * self.r2 * (self.gBest - self.particles_pos).astype(np.float32)
        self.particles_v += personal_based + global_based

        # adjust the velocity to move smoothly 
        self._adjustVel(target_pos, speed_multiplier=20.0, force=10.0)
        self._velFriction()

        # update position 
        self.particles_pos = self.particles_pos + self.particles_v

        # keep the particle on the screen
        # self.particles_pos %= (WIDTH, HEIGHT)

    def spreadParticles(self, force=10):
        # Compute direction away from gBest
        direction = self.particles_pos - self.gBest
        distance = np.linalg.norm(direction, axis=1, keepdims=True)

        # Avoid division by zero
        distance = np.where(distance == 0, 1, distance)

        # Normalize direction and apply force
        normalized_direction = direction / distance
        self.particles_v += normalized_direction * force

        # Update positions based on new velocity
        self.particles_pos += self.particles_v

        # Keep particles within screen bounds
        # self.particles_pos %= (WIDTH, HEIGHT)

# Call OpenCV ---------------------------------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# ---------------------------------------------------- #

# Hand detection function ---------------------------- #
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
                pygame.draw.circle(screen, (0, 255, 0), (telunjuk_x, telunjuk_y), 2)
                hand_detected = True

        if show_camera == True:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # q for QUITTT
                # break
                pygame.quit()
                sys.exit()
# --------------------------------------------------- #

# Start hand detection in a separate thread --------- #
if pake_tangan == True:
    telunjuk_x, telunjuk_y = WIDTH // 2, HEIGHT // 2
    hand_detected = False
    thread = threading.Thread(target=hand_detection)
    thread.daemon = True
    thread.start()
# --------------------------------------------------- #

# Loop ------------------------------------------------------- #
def main():
    particles = Particles(20000)
    target_pos = (0, 0)
    while True:
        screen.fill((0, 0, 0))

        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_Q:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == pygame.K_SPACE:
                particles.spreadParticles()

        # Generate
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

        # Update display
        pygame.display.update()
        clock.tick(60)
# ------------------------------------------------------------- #

if __name__ == '__main__':
    main()
