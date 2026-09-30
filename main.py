import math
import os
import random
import time
import urllib.error
import urllib.request
import textwrap
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
import mediapipe as mp

# -----------------------------
# Configuration
# -----------------------------
WIDTH, HEIGHT = 1280, 720
FPS = 60
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "hand_landmarker.task")

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Fruit Slash")
clock = pygame.time.Clock()

FONT_BIG = pygame.font.SysFont("Arial", 64, bold=True)
FONT_SCORE = pygame.font.SysFont("Arial", 42, bold=True)
FONT = pygame.font.SysFont("Arial", 26, bold=True)
FONT_SMALL = pygame.font.SysFont("Arial", 18)
FONT_TINY = pygame.font.SysFont("Arial", 14, bold=True)

# Fruit size, score, color, and relative spawn weight.
FRUITS = [
    ("APPLE", 37, 10, (220, 42, 58), 18),
    ("KIWI", 29, 15, (132, 184, 55), 15),
    ("STRAWBERRY", 27, 20, (232, 45, 75), 13),
    ("ORANGE", 34, 8, (247, 139, 35), 17),
    ("CHERRY", 24, 25, (190, 26, 55), 10),
    ("WATERMELON", 52, 5, (43, 158, 81), 9),
    ("PINEAPPLE", 39, 30, (245, 185, 45), 7),
    ("BANANA", 35, 12, (250, 217, 63), 10),
    ("BONUS", 28, 50, (54, 225, 224), 1),
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
        self.rotation += (self.vx * 0.003 + 1.2) * dt

    def draw(self, surface):
        x, y, r = int(self.x), int(self.y), self.radius
        if self.kind == "bomb":
            pygame.draw.circle(surface, (12, 15, 20), (x, y), r)
            pygame.draw.circle(surface, (210, 218, 224), (x, y), r, 3)
            pygame.draw.circle(surface, (52, 58, 65), (x-r//4, y-r//4), r//3)
            fuse_start = (x + r//2, y - r//2)
            fuse_end = (x + r - 2, y - r - 13)
            pygame.draw.line(surface, (224, 142, 38), fuse_start, fuse_end, 5)
            pygame.draw.circle(surface, (255, 83, 30), fuse_end, 7)
            pygame.draw.circle(surface, (255, 206, 84), fuse_end, 3)
            label = FONT_TINY.render("BOMB", True, (255, 125, 95))
            surface.blit(label, label.get_rect(center=(x, y)))
            return

        angle = self.rotation
        if self.name == "APPLE":
            pygame.draw.ellipse(surface, self.color, (x-r, y-r+4, r*2, int(r*1.8)))
            pygame.draw.ellipse(surface, (247, 104, 105), (x-r//2, y-r//2, r//2, r))
            pygame.draw.line(surface, (95, 63, 34), (x, y-r+5), (x+4, y-r-10), 5)
            pygame.draw.ellipse(surface, (70, 164, 74), (x+2, y-r-10, 19, 9))
        elif self.name == "KIWI":
            pygame.draw.circle(surface, (104, 75, 46), (x, y), r)
            pygame.draw.circle(surface, self.color, (x, y), int(r*.82))
            pygame.draw.circle(surface, (228, 233, 154), (x, y), int(r*.38))
            for index in range(10):
                seed_angle = index * math.tau / 10
                sx = x + int(math.cos(seed_angle) * r * .55)
                sy = y + int(math.sin(seed_angle) * r * .55)
                pygame.draw.ellipse(surface, (45, 57, 32), (sx-2, sy-3, 4, 7))
        elif self.name == "STRAWBERRY":
            points = [(x-r, y-r//3), (x-r//2, y-r), (x+r//2, y-r),
                      (x+r, y-r//3), (x+r//2, y+r//2), (x, y+r),
                      (x-r//2, y+r//2)]
            pygame.draw.polygon(surface, self.color, points)
            pygame.draw.polygon(surface, (255, 112, 126), points, 2)
            for sx, sy in ((-10, -7), (8, -9), (-5, 9), (11, 8), (0, 20)):
                pygame.draw.ellipse(surface, (255, 225, 155), (x+sx, y+sy, 4, 7))
            pygame.draw.polygon(surface, (58, 156, 73),
                                [(x-r//2, y-r+3), (x, y-r-8), (x+2, y-r+4),
                                 (x+r//2, y-r-6), (x+r//3, y-r//2)])
        elif self.name == "ORANGE":
            pygame.draw.circle(surface, self.color, (x, y), r)
            pygame.draw.circle(surface, (255, 197, 80), (x, y), int(r*.77), 3)
            pygame.draw.circle(surface, (255, 222, 133), (x-r//3, y-r//3), r//5)
            pygame.draw.ellipse(surface, (74, 155, 67), (x-3, y-r-7, 15, 8))
        elif self.name == "CHERRY":
            left = (x-r//2, y+r//4)
            right = (x+r//2, y+r//4)
            pygame.draw.line(surface, (67, 133, 67), (x, y-r//2), left, 4)
            pygame.draw.line(surface, (67, 133, 67), (x, y-r//2), right, 4)
            pygame.draw.circle(surface, self.color, left, r//2+3)
            pygame.draw.circle(surface, (239, 90, 105), (left[0]-4, left[1]-5), 5)
            pygame.draw.circle(surface, self.color, right, r//2+3)
            pygame.draw.circle(surface, (239, 90, 105), (right[0]-4, right[1]-5), 5)
        elif self.name == "WATERMELON":
            rect = pygame.Rect(x-int(r*1.35), y-int(r*.72), int(r*2.7), int(r*1.45))
            pygame.draw.ellipse(surface, (31, 111, 62), rect)
            pygame.draw.ellipse(surface, (63, 183, 88), rect.inflate(-9, -8))
            pygame.draw.ellipse(surface, (239, 88, 105), rect.inflate(-19, -17))
            for sx, sy in ((-16, -7), (0, 8), (17, -5), (5, -13)):
                pygame.draw.ellipse(surface, (56, 48, 52), (x+sx, y+sy, 4, 9))
        elif self.name == "PINEAPPLE":
            for leaf_x in (-r//2, -r//5, r//5, r//2):
                pygame.draw.polygon(surface, (49, 151, 70),
                                    [(x, y-r//2), (x+leaf_x-5, y-r-15),
                                     (x+leaf_x+5, y-r-11), (x+8, y-r//3)])
            pygame.draw.ellipse(surface, self.color, (x-r//2, y-r//2, r, int(r*1.55)))
            for offset in range(-r//2, r//2, 12):
                pygame.draw.line(surface, (211, 139, 37), (x-r//2, y+offset),
                                 (x+r//2, y+offset+12), 2)
                pygame.draw.line(surface, (211, 139, 37), (x-r//2, y+offset),
                                 (x+r//2, y+offset-12), 2)
        elif self.name == "BANANA":
            curve = []
            for index in range(9):
                t = index / 8
                px = x + int((t-.5) * r * 2.1)
                py = y + int(math.sin(t * math.pi) * r * .72 - r * .25)
                dx, dy = px-x, py-y
                curve.append((x+int(dx*math.cos(angle)-dy*math.sin(angle)),
                              y+int(dx*math.sin(angle)+dy*math.cos(angle))))
            pygame.draw.lines(surface, (150, 104, 34), False, curve, 16)
            pygame.draw.lines(surface, self.color, False, curve, 12)
            pygame.draw.lines(surface, (255, 239, 137), False, curve, 3)
        else:  # BONUS: luminous star fruit
            pygame.draw.circle(surface, (21, 99, 115), (x, y), r+5)
            points = []
            for index in range(10):
                star_angle = -math.pi/2 + index * math.pi/5 + angle
                star_radius = r if index % 2 == 0 else r*.46
                points.append((x+math.cos(star_angle)*star_radius,
                               y+math.sin(star_angle)*star_radius))
            pygame.draw.polygon(surface, self.color, points)
            pygame.draw.polygon(surface, (220, 255, 245), points, 2)

        label = FONT_TINY.render(str(self.points), True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=(x, y+r+8)))

def ensure_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    print("Downloading MediaPipe hand model...")
    temporary_path = MODEL_PATH + ".download"
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=20) as response:
            with open(temporary_path, "wb") as model_file:
                model_file.write(response.read())
        os.replace(temporary_path, MODEL_PATH)
    except (OSError, urllib.error.URLError) as error:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise RuntimeError(
            "Could not download the MediaPipe model. Place hand_landmarker.task "
            f"at: {MODEL_PATH}"
        ) from error
    print(f"Hand model ready: {MODEL_PATH}")

def create_hand_landmarker():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def open_webcam():
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    for backend in backends:
        camera = cv2.VideoCapture(0, backend)
        if not camera.isOpened():
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for width, height in ((WIDTH, HEIGHT), (960, 540), (640, 480)):
            if (width, height) != (WIDTH, HEIGHT):
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            success, frame = camera.read()
            if success and frame is not None:
                return camera, frame
        camera.release()

    raise RuntimeError(
        "Could not open the MacBook webcam. In System Settings > Privacy & "
        "Security > Camera, allow camera access for VS Code or Terminal, then "
        "close other apps using the camera and restart the game."
    )


def camera_destination(frame_width, frame_height):
    scale = min(WIDTH / frame_width, HEIGHT / frame_height)
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)
    return pygame.Rect((WIDTH-display_width)//2, (HEIGHT-display_height)//2,
                       display_width, display_height)


def show_startup_error(message):
    screen.fill((17, 21, 31))
    title = FONT.render("HAND FRUIT SLASH COULD NOT START", True, (255, 112, 102))
    screen.blit(title, (48, 70))
    lines = []
    for paragraph in message.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=88) or [""])
    for index, line in enumerate(lines):
        screen.blit(FONT_SMALL.render(line, True, (235, 239, 244)), (48, 132+index*30))
    screen.blit(FONT_SMALL.render("Press ESC or close this window to exit.", True,
                                  (163, 184, 205)), (48, HEIGHT-70))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key == pygame.K_ESCAPE):
                return
        clock.tick(30)

# -----------------------------
# Game state
# -----------------------------
objects = []
slash_points = []
particles = []
fruit_splits = []
floating_scores = []
score = 0
combo = 0
best_combo = 0
game_over = False
spawn_timer = 0.0
bomb_timer = 0.0
start_time = time.time()
last_finger = None
smoothed_finger = None
slash_cooldown = 0.0

def reset_game():
    global objects, slash_points, particles, fruit_splits, floating_scores
    global score, combo, best_combo, game_over, spawn_timer, bomb_timer
    global start_time, last_finger, smoothed_finger, slash_cooldown
    objects = []
    slash_points = []
    particles = []
    fruit_splits = []
    floating_scores = []
    score = 0
    combo = 0
    best_combo = 0
    game_over = False
    spawn_timer = 0
    bomb_timer = 0
    start_time = time.time()
    last_finger = None
    smoothed_finger = None
    slash_cooldown = 0

def spawn_fruit(level):
    name, radius, points, color, weight = random.choices(
        FRUITS, weights=[fruit[4] for fruit in FRUITS], k=1
    )[0]
    x = random.randint(radius + 30, WIDTH - radius - 30)
    y = HEIGHT + radius
    vx = random.uniform(-185, 185)
    vy = random.uniform(-820, -675) - (level-1) * 13
    objects.append(FallingObject("fruit", name, x, y, vx, vy, radius, points, color))

def spawn_bomb(level):
    radius = random.randint(27, 38)
    x = random.randint(radius + 30, WIDTH - radius - 30)
    y = HEIGHT + radius
    vx = random.uniform(-155, 155)
    vy = random.uniform(-790, -650) - (level-1) * 10
    objects.append(FallingObject("bomb", "BOMB", x, y, vx, vy, radius, 0, (15, 18, 22)))

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
    for _ in range(min(count, max(0, 220-len(particles)))):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(90, 300)
        particles.append({
            "x": x, "y": y,
            "vx": math.cos(angle)*speed,
            "vy": math.sin(angle)*speed,
            "life": random.uniform(.25, .55),
            "max_life": .55,
            "radius": random.randint(2, 5),
            "color": color
        })

def slice_object(obj):
    global score, combo, best_combo, game_over
    if obj.kind == "bomb":
        game_over = True
        return

    combo += 1
    best_combo = max(best_combo, combo)
    score += obj.points
    floating_scores.append({"x": obj.x, "y": obj.y, "text": f"+{obj.points}",
                            "life": .95, "max_life": .95})
    fruit_splits.append({"x": obj.x, "y": obj.y, "vx": obj.vx,
                         "vy": obj.vy, "radius": obj.radius,
                         "color": obj.color, "life": .34, "max_life": .34})
    add_particles(obj.x, obj.y, obj.color, 20)
    obj.alive = False

def draw_background(surface, level):
    surface.fill((10, 14, 21))

def draw_ui(surface, level, hand_detected, fps):
    panel = pygame.Surface((224, 76), pygame.SRCALPHA)
    panel.fill((8, 13, 20, 190))
    surface.blit(panel, (0, 0))
    score_label = FONT_SMALL.render("SCORE", True, (184, 209, 220))
    score_value = FONT_SCORE.render(str(score), True, (255, 255, 255))
    surface.blit(score_label, (22, 8))
    surface.blit(score_value, (20, 27))

    combo_label = f"COMBO x{combo}" if combo >= 2 else "COMBO"
    combo_color = (255, 211, 103) if combo >= 2 else (195, 206, 214)
    surface.blit(FONT_SMALL.render(combo_label, True, combo_color), (234, 12))

    right_panel = pygame.Surface((150, 60), pygame.SRCALPHA)
    right_panel.fill((8, 13, 20, 175))
    surface.blit(right_panel, (WIDTH-150, 0))
    surface.blit(FONT_SMALL.render(f"LEVEL {level}", True, (205, 225, 237)),
                 (WIDTH-136, 9))
    surface.blit(FONT_SMALL.render(f"{int(fps)} FPS", True, (159, 191, 201)),
                 (WIDTH-136, 33))

    status_color = (110, 239, 178) if hand_detected else (248, 198, 103)
    status_text = "INDEX TIP TRACKED" if hand_detected else "SHOW YOUR HAND TO CAMERA"
    status_panel = pygame.Surface((245, 30), pygame.SRCALPHA)
    status_panel.fill((8, 13, 20, 165))
    surface.blit(status_panel, (0, HEIGHT-30))
    status = FONT_SMALL.render(status_text, True, status_color)
    surface.blit(status, (9, HEIGHT-25))

def main():
    global spawn_timer, bomb_timer, last_finger, smoothed_finger, slash_cooldown
    global combo, game_over, slash_points, particles, fruit_splits, floating_scores
    global objects, score

    reset_game()

    cap = None
    try:
        cap, initial_frame = open_webcam()
        ensure_model()
        landmarker = create_hand_landmarker()
    except Exception as error:
        show_startup_error(str(error))
        if cap is not None:
            cap.release()
        pygame.quit()
        return

    camera_rgb = None
    last_timestamp_ms = 0
    slash_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    with landmarker:
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
                frame = initial_frame

            frame = cv2.flip(frame, 1)
            camera_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=camera_rgb)
            last_timestamp_ms = max(last_timestamp_ms + 1, int(time.monotonic() * 1000))
            result = landmarker.detect_for_video(mp_image, last_timestamp_ms)

            finger = None
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                index_tip = hand[8]
                camera_rect = camera_destination(camera_rgb.shape[1], camera_rgb.shape[0])
                raw_finger = (camera_rect.left + index_tip.x * camera_rect.width,
                              camera_rect.top + index_tip.y * camera_rect.height)
                if smoothed_finger is None:
                    smoothed_finger = raw_finger
                else:
                    smoothing = min(0.72, max(0.28, dt * 18))
                    smoothed_finger = (
                        smoothed_finger[0] + (raw_finger[0]-smoothed_finger[0])*smoothing,
                        smoothed_finger[1] + (raw_finger[1]-smoothed_finger[1])*smoothing,
                    )
                finger = (int(smoothed_finger[0]), int(smoothed_finger[1]))
            else:
                smoothed_finger = None

            # Current difficulty level
            elapsed = time.time() - start_time
            level = min(10, 1 + int(elapsed // 20))

            if not game_over:
                spawn_timer -= dt
                bomb_timer -= dt
                slash_cooldown = max(0, slash_cooldown - dt)

                fruit_interval = max(0.38, 0.94 - (level-1) * 0.065)
                if spawn_timer <= 0:
                    spawn_fruit(level)
                    if level >= 4 and random.random() < min(.45, (level-3)*.09):
                        spawn_fruit(level)
                    spawn_timer = fruit_interval

                bomb_interval = max(1.65, 3.2 - (level-1) * 0.16)
                if bomb_timer <= 0:
                    spawn_bomb(level)
                    bomb_timer = random.uniform(bomb_interval, bomb_interval+0.8)

                # Build slash trail
                if finger is not None:
                    if last_finger is not None:
                        fx, fy = finger
                        lx, ly = last_finger
                        movement = math.hypot(fx-lx, fy-ly)
                        if movement > 2:
                            slash_points.append((fx, fy, .27, (lx, ly)))

                        # A moving index tip sweeps a line segment through targets.
                        if movement > 17 and movement / max(dt, .001) > 210:
                            for obj in sorted(objects, key=lambda target: target.kind != "bomb"):
                                if not obj.alive:
                                    continue
                                distance = distance_point_segment(
                                    obj.x, obj.y, lx, ly, fx, fy
                                )
                                if distance <= obj.radius + 13:
                                    slice_object(obj)
                                    if obj.kind == "bomb":
                                        add_particles(obj.x, obj.y, (255, 103, 49), 32)
                                        break

                    last_finger = finger
                else:
                    last_finger = None

                for obj in objects:
                    obj.update(dt)
                    if obj.y > HEIGHT + obj.radius + 80 or obj.x < -obj.radius-50 or obj.x > WIDTH+obj.radius+50:
                        obj.alive = False
                        if obj.kind == "fruit":
                            combo = 0

                objects[:] = [o for o in objects if o.alive]

            # Update slash trail
            slash_points = [(x, y, life-dt, start) for x, y, life, start in slash_points
                            if life-dt > 0]

            # Update particles
            for p in particles:
                p["x"] += p["vx"] * dt
                p["y"] += p["vy"] * dt
                p["vy"] += 650 * dt
                p["life"] -= dt
            particles[:] = [p for p in particles if p["life"] > 0]

            for split in fruit_splits:
                split["x"] += split["vx"] * dt
                split["y"] += split["vy"] * dt
                split["vy"] += 720 * dt
                split["life"] -= dt
            fruit_splits[:] = [split for split in fruit_splits if split["life"] > 0]

            for floating in floating_scores:
                floating["y"] -= 42 * dt
                floating["life"] -= dt
            floating_scores[:] = [floating for floating in floating_scores
                                  if floating["life"] > 0]

            # Draw
            draw_background(screen, level)
            camera_rect = camera_destination(camera_rgb.shape[1], camera_rgb.shape[0])
            camera_surface = pygame.image.frombuffer(
                camera_rgb.data, (camera_rgb.shape[1], camera_rgb.shape[0]), "RGB"
            )
            camera_surface = pygame.transform.smoothscale(
                camera_surface, (camera_rect.width, camera_rect.height)
            )
            screen.blit(camera_surface, camera_rect)
            for obj in objects:
                obj.draw(screen)

            for p in particles:
                radius = max(1, int(p["radius"] * p["life"] / p["max_life"]))
                pygame.draw.circle(screen, p["color"], (int(p["x"]), int(p["y"])), radius)

            for split in fruit_splits:
                split_radius = max(3, int(split["radius"] * split["life"] /
                                          split["max_life"] * .75))
                spread = int((1-split["life"] / split["max_life"]) * split["radius"])
                half_width = max(3, split_radius)
                pygame.draw.ellipse(
                    screen, split["color"],
                    (int(split["x"]-spread-half_width),
                     int(split["y"]-split_radius), half_width, split_radius*2)
                )
                pygame.draw.ellipse(
                    screen, split["color"],
                    (int(split["x"]+spread), int(split["y"]-split_radius),
                     half_width, split_radius*2)
                )

            for floating in floating_scores:
                alpha = int(255 * floating["life"] / floating["max_life"])
                score_popup = FONT.render(floating["text"], True, (255, 245, 177))
                score_popup.set_alpha(alpha)
                screen.blit(score_popup, score_popup.get_rect(
                    center=(int(floating["x"]), int(floating["y"]))
                ))

            # Draw slash trail
            slash_overlay.fill((0, 0, 0, 0))
            for x, y, life, start in slash_points:
                alpha = int(245 * life / .27)
                width = max(2, int(16 * life / .27))
                pygame.draw.line(slash_overlay, (68, 223, 255, alpha//4),
                                 start, (x, y), width+12)
                pygame.draw.line(slash_overlay, (111, 235, 255, alpha),
                                 start, (x, y), width)
                pygame.draw.line(slash_overlay, (255, 255, 255, alpha),
                                 start, (x, y), max(2, width//3))
            screen.blit(slash_overlay, (0, 0))

            # Finger cursor
            if finger is not None:
                pygame.draw.circle(screen, (58, 224, 255), finger, 18, 2)
                pygame.draw.circle(screen, (255, 255, 255), finger, 11, 3)
                pygame.draw.circle(screen, (255, 255, 255), finger, 4)

            draw_ui(screen, level, finger is not None, clock.get_fps())

            if game_over:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((3, 6, 12, 190))
                screen.blit(overlay, (0, 0))
                text = FONT_BIG.render("GAME OVER", True, (255, 91, 92))
                screen.blit(text, text.get_rect(center=(WIDTH//2, HEIGHT//2-70)))
                sc = FONT.render(f"FINAL SCORE: {score}", True, (255, 255, 255))
                screen.blit(sc, sc.get_rect(center=(WIDTH//2, HEIGHT//2)))
                bc = FONT_SMALL.render(f"BEST COMBO: x{best_combo}", True, (255, 219, 113))
                screen.blit(bc, bc.get_rect(center=(WIDTH//2, HEIGHT//2+43)))
                restart = FONT.render("PRESS R TO RESTART", True, (210, 230, 239))
                screen.blit(restart, restart.get_rect(center=(WIDTH//2, HEIGHT//2+95)))
                quit_text = FONT_SMALL.render("PRESS ESC TO QUIT", True, (179, 200, 211))
                screen.blit(quit_text, quit_text.get_rect(center=(WIDTH//2, HEIGHT//2+132)))

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
