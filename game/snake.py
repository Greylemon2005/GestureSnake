# game/snake.py
import time
import numpy as np
import pygame

from config import CELL, GRID_W, GRID_H, WIN_W, WIN_H, TICK_BASE
from gesture.types import GestureState
from gesture.worker import GestureWorker

DIR_V = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def rand_food(exclude_set):
    while True:
        p = (np.random.randint(0, GRID_W), np.random.randint(0, GRID_H))
        if p not in exclude_set:
            return p

def reset_game():
    snake = [(GRID_W//2, GRID_H//2), (GRID_W//2 - 1, GRID_H//2), (GRID_W//2 - 2, GRID_H//2)]
    direction = "RIGHT"
    food = rand_food(set(snake))
    score = 0
    alive = True
    paused = False
    return snake, direction, food, score, alive, paused

def draw_cell(screen, x, y, color):
    pygame.draw.rect(screen, color, pygame.Rect(x*CELL, y*CELL, CELL, CELL))


def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("GestureSnake (Embedded) - MediaPipe Hands")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    state = GestureState()
    worker = GestureWorker(state)
    worker.start()
    print("[Main] GestureWorker started:", worker.is_alive())

    snake, direction, food, score, alive, paused = reset_game()

    def apply_dir(new_dir: str):
        nonlocal direction
        if not new_dir:
            return
        if direction and OPPOSITE.get(direction) == new_dir:
            return  # 禁止180°掉头
        direction = new_dir

    # Start screen
    start = True
    while start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                worker.stop()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    start = False
                if event.key == pygame.K_ESCAPE:
                    worker.stop()
                    return

        screen.fill((15, 15, 18))
        t1 = font.render("GestureSnake (Embedded)", True, (220, 220, 220))
        t2 = font.render("ENTER to start | ESC to quit", True, (200, 200, 200))
        t3 = font.render("Gestures: Point=Left/Right/Up | V=Down | OK=Pause | OpenHand=Restart", True, (180, 180, 180))
        screen.blit(t1, (20, 40))
        screen.blit(t2, (20, 70))
        screen.blit(t3, (20, 100))
        pygame.display.flip()
        clock.tick(30)

    # Game loop
    move_acc = 0.0
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        if alive and (not paused):
            move_acc += dt
        else:
            move_acc = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                worker.stop()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    worker.stop()
                    return
                # keyboard fallback
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    snake, direction, food, score, alive, paused = reset_game()
                    move_acc = 0.0
                    last_time = time.time()

        # Consume gesture state
        with worker.lock:
            g_dir = state.direction
            g_pause = state.pause_toggle
            g_restart = state.restart
            g_label = state.label
            g_seen = state.hand_seen
            g_cam = state.cam_info
            state.pause_toggle = False
            state.restart = False

        if g_restart:
            snake, direction, food, score, alive, paused = reset_game()
            move_acc = 0.0
            last_time = time.time()

        if g_pause:
            paused = not paused

        if alive and (not paused):
            apply_dir(g_dir)

        step_time = 1.0 / max(1, TICK_BASE)

        if alive and (not paused) and move_acc >= step_time:
            move_acc -= step_time

            vx, vy = DIR_V[direction]
            head = snake[0]
            new_head = (head[0] + vx, head[1] + vy)

            if new_head[0] < 0 or new_head[0] >= GRID_W or new_head[1] < 0 or new_head[1] >= GRID_H:
                alive = False
            elif new_head in snake:
                alive = False
            else:
                snake.insert(0, new_head)
                if new_head == food:
                    score += 1
                    food = rand_food(set(snake))
                else:
                    snake.pop()

        # Render
        screen.fill((12, 12, 14))
        draw_cell(screen, food[0], food[1], (255, 90, 90))

        for i, (x, y) in enumerate(snake):
            c = (90, 220, 140) if i == 0 else (60, 160, 100)
            draw_cell(screen, x, y, c)

        hud1 = font.render(f"Score: {score}", True, (230, 230, 230))
        hud2 = font.render(f"Gesture: {g_label}", True, (200, 200, 200))
        hud3 = font.render(f"Hand: {'YES' if g_seen else 'NO'} | Dir: {g_dir or '-'}", True, (180, 180, 180))
        hud4 = font.render(f"{g_cam}", True, (120, 120, 120))
        screen.blit(hud1, (8, 6))
        screen.blit(hud2, (8, 28))
        screen.blit(hud3, (8, 50))
        screen.blit(hud4, (8, 72))

        if paused:
            msg = font.render("PAUSED (OK pinch to toggle)", True, (255, 255, 120))
            screen.blit(msg, (8, 96))
        if not alive:
            msg = font.render("GAME OVER (Open hand to restart)", True, (255, 160, 160))
            screen.blit(msg, (8, 118))

        pygame.display.flip()
        clock.tick(60)
