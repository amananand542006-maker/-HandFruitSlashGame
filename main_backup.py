import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass

import cv2
import pygame
import mediapipe as mp

# -----------------------------
# Configuration
# -----------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 60
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join("models", "hand_landmarker.task")

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Fruit Slash")
clock = pygame.time.Clock()

FONT_BIG = pygame.font.SysFont("Arial", 56, bold=True)
FONT = pygame.font.SysFont("Arial", 30, bold=True)
FONT_SMALL = pygame.font.SysFont("Arial", 22)

# Fruit definitions: name, radius, points, drawing color, bonus
FRUITS = [
    ("KIWI", 30, 10, (130, 190, 55), 0),
    ("STRAWBERRY", 27, 12, (220, 55, 70), 0),
    ("CHERRY", 24, 15, (190, 35, 55), 0),
    ("ORANGE", 38, 8, (245, 145, 35), 0),
    ("APPLE", 42, 7, (220, 50, 45), 0),
    ("WATERMELON", 62, 5, (45, 175, 80), 0),
    ("PINEAPPLE", 45, 20, (245, 190, 45), 0),
    ("STAR", 28, 25, (255, 220, 55), 5),
]

@dataclass
class FallingObject:
    kind: str
    name: str
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    points: int
    color: tuple
    bonus: int = 0
    alive: bool = True
    rotation: float = 0

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 720 * dt
        self.rotation += self.vx * dt * 0.03

    def draw(self, surface):
        x, y, r = int(self.x), int(self.y), self.radius
        if self.kind == "bomb":
            pygame.draw.circle(surface, (25, 25, 30), (x, y), r)
            pygame.draw.circle(surface, (220, 220, 220), (x, y), r, 3)
            pygame.draw.line(surface, (255, 160, 30), (x + r//2, y-r//2),
                             (x+r, y-r-15), 5)
            pygame.draw.circle(surface, (255, 120, 20), (x+r+2, y-r-17), 7)
            label = FONT_SMALL.render("BOMB", True, (255, 80, 80))
            surface.blit(label, label.get_rect(center=(x, y)))
            return

        pygame.draw.circle(surface, self.color, (x, y), r)
        pygame.draw.circle(surface, (255, 255, 255), (x-r//3, y-r//3), max(3, r//8))

        # Simple fruit-specific visual details
        if self.name == "KIWI":
            pygame.draw.circle(surface, (220, 245, 145), (x, y), int(r*.62))
            pygame.draw.circle(surface, (70, 120, 45), (x, y), 5)
        elif self.name == "WATERMELON":
            pygame.draw.arc(surface, (20, 100, 50), (x-r, y-r, 2*r, 2*r), 0.2, 2.8, 5)
        elif self.name == "PINEAPPLE":
            for a in (-0.5, 0, 0.5):
                pygame.draw.line(surface, (40, 150, 60), (x, y-r),
                                 (x+int(a*r), y-r-18), 5)
        elif self.name == "STAR":
            pts = []
            for i in range(10):
                a = -math.pi/2 + i*math.pi/5
                rr = r if i % 2 == 0 else r*.45
                pts.append((x+math.cos(a)*rr, y+math.sin(a)*rr))
            pygame.draw.polygon(surface, (255, 235, 70), pts)

        label = FONT_SMALL.render(str(self.points), True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=(x, y)))

def ensure_model():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Hand model downloaded.")

# -----------------------------
# Hand tracking
# -----------------------------
ensure_model()

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check macOS Camera permission for VS Code/Python.")

# -----------------------------
# Game state
# -----------------------------
objects = []
slash_points = []
particles = []
score = 0
combo = 0
best_combo = 0
game_over = False
spawn_timer = 0.0
bomb_timer = 0.0
start_time = time.time()
last_finger = None
slash_cooldown = 0.0

def reset_game():
    global objects, slash_points, particles, score, combo, best_combo
    global game_over, spawn_timer, bomb_timer, start_time, last_finger, slash_cooldown
    objects = []
    slash_points = []
    particles = []
    score = 0
    combo = 0
    best_combo = 0
    game_over = False
    spawn_timer = 0
    bomb_timer = 0
    start_time = time.time()
    last_finger = None
    slash_cooldown = 0

def spawn_fruit(level):
    name, radius, points, color, bonus = random.choice(FRUITS)
    x = random.randint(radius + 30, WIDTH - radius - 30)
    y = HEIGHT + radius
    vx = random.uniform(-180, 180)
    vy = random.uniform(-900, -680) - level * 15
    objects.append(FallingObject("fruit", name, x, y, vx, vy, radius, points, color, bonus))

def spawn_bomb(level):
    radius = random.randint(28, 43)
    x = random.randint(radius + 30, WIDTH - radius - 30)
    y = HEIGHT + radius
    vx = random.uniform(-160, 160)
    vy = random.uniform(-850, -650) - level * 12
    objects.append(FallingObject("bomb", "BOMB", x, y, vx, vy, radius, 0, (25,25,30)))

def distance_point_segment(px, py, ax, ay, bx, by):
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    denom = abx*abx + aby*aby
    if denom == 0:
        return math.hypot(px-ax, py-ay)
    t = max(0, min(1, (apx*abx + apy*aby) / denom))
    cx, cy = ax+t*abx, ay+t*aby
    return math.hypot(px-cx, py-cy)

def add_particles(x, y, color, count=18):
    for _ in range(count):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(80, 320)
        particles.append({
            "x": x, "y": y,
            "vx": math.cos(angle)*speed,
            "vy": math.sin(angle)*speed,
            "life": random.uniform(.25, .55),
            "color": color
        })

def slice_object(obj):
    global score, combo, best_combo, game_over
    if obj.kind == "bomb":
        game_over = True
        return

    combo += 1
    best_combo = max(best_combo, combo)
    multiplier = 2 if combo >= 5 else 1
    gained = obj.points * multiplier + obj.bonus
    score += gained
    add_particles(obj.x, obj.y, obj.color, 22)
    obj.alive = False

def draw_background(surface, level):
    surface.fill((12, 18, 35))
    # Moving-style decorative grid
    for x in range(0, WIDTH, 60):
        pygame.draw.line(surface, (20, 30, 55), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 60):
        pygame.draw.line(surface, (20, 30, 55), (0, y), (WIDTH, y))
    title = FONT.render("HAND FRUIT SLASH", True, (240, 245, 255))
    surface.blit(title, (25, 18))
    level_text = FONT_SMALL.render(f"LEVEL {level}", True, (180, 220, 255))
    surface.blit(level_text, (WIDTH-130, 28))

def draw_ui(surface):
    score_text = FONT.render(f"Score: {score}", True, (255, 255, 255))
    combo_text = FONT.render(f"Combo: {combo}", True, (255, 210, 80))
    surface.blit(score_text, (25, 70))
    surface.blit(combo_text, (25, 108))

    hint = FONT_SMALL.render("Move your INDEX FINGER quickly through fruits. Avoid bombs!", True, (190, 200, 220))
    surface.blit(hint, (25, HEIGHT-35))

def main():
    global spawn_timer, bomb_timer, last_finger, slash_cooldown, combo, game_over

    reset_game()

    with HandLandmarker.create_from_options(hand_options) as landmarker:
        running = True

        while running:
            dt = clock.tick(FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r and game_over:
                        reset_game()

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            finger = None
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                # Index fingertip landmark = 8
                lm = hand[8]
                finger = (int(lm.x * WIDTH), int(lm.y * HEIGHT))

            # Current difficulty level
            elapsed = time.time() - start_time
            level = min(10, 1 + int(elapsed // 20))

            if not game_over:
                spawn_timer -= dt
                bomb_timer -= dt
                slash_cooldown = max(0, slash_cooldown - dt)

                fruit_interval = max(0.32, 0.85 - level * 0.05)
                if spawn_timer <= 0:
                    spawn_fruit(level)
                    spawn_timer = fruit_interval

                bomb_interval = max(0.75, 2.4 - level * 0.12)
                if bomb_timer <= 0:
                    spawn_bomb(level)
                    bomb_timer = bomb_interval

                # Build slash trail
                if finger is not None:
                    if last_finger is not None:
                        fx, fy = finger
                        lx, ly = last_finger
                        movement = math.hypot(fx-lx, fy-ly)

                        # A fast enough movement counts as a slash.
                        if movement > 18:
                            slash_points.append((fx, fy, 0.16))

                            # Collision against the movement segment
                            for obj in objects:
                                if not obj.alive:
                                    continue
                                d = distance_point_segment(obj.x, obj.y, lx, ly, fx, fy)
                                if d < obj.radius + 10:
                                    slice_object(obj)
                                    if obj.kind == "bomb":
                                        break

                    last_finger = finger
                else:
                    last_finger = None

                # If player is not slashing, combo gradually resets.
                if finger is None:
                    combo = max(0, combo - 1 if combo > 0 else 0)

                for obj in objects:
                    obj.update(dt)
                    if obj.y > HEIGHT + 150:
                        obj.alive = False
                        if obj.kind == "fruit":
                            combo = 0

                objects[:] = [o for o in objects if o.alive]

            # Update slash trail
            slash_points = [(x, y, life-dt) for x, y, life in slash_points if life-dt > 0]

            # Update particles
            for p in particles:
                p["x"] += p["vx"] * dt
                p["y"] += p["vy"] * dt
                p["vy"] += 650 * dt
                p["life"] -= dt
            particles[:] = [p for p in particles if p["life"] > 0]

            # Draw
            draw_background(screen, level)
            for obj in objects:
                obj.draw(screen)

            for p in particles:
                pygame.draw.circle(screen, p["color"], (int(p["x"]), int(p["y"])), 4)

            # Draw slash trail
            if len(slash_points) >= 2:
                for i in range(1, len(slash_points)):
                    x1, y1, life1 = slash_points[i-1]
                    x2, y2, life2 = slash_points[i]
                    width = max(2, int(12 * min(life1, life2) / .16))
                    pygame.draw.line(screen, (220, 245, 255), (x1, y1), (x2, y2), width)

            # Finger cursor
            if finger is not None:
                pygame.draw.circle(screen, (255, 255, 255), finger, 13, 3)
                pygame.draw.circle(screen, (120, 220, 255), finger, 5)

            draw_ui(screen)

            if game_over:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 175))
                screen.blit(overlay, (0, 0))
                text = FONT_BIG.render("GAME OVER", True, (255, 80, 80))
                screen.blit(text, text.get_rect(center=(WIDTH//2, HEIGHT//2-60)))
                sc = FONT.render(f"Final Score: {score}", True, (255,255,255))
                screen.blit(sc, sc.get_rect(center=(WIDTH//2, HEIGHT//2+5)))
                bc = FONT_SMALL.render(f"Best Combo: {best_combo}", True, (255,220,90))
                screen.blit(bc, bc.get_rect(center=(WIDTH//2, HEIGHT//2+45)))
                restart = FONT.render("Press R to restart  |  ESC to quit", True, (210,220,235))
                screen.blit(restart, restart.get_rect(center=(WIDTH//2, HEIGHT//2+100)))

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
