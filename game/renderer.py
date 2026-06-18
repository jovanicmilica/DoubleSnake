import pygame
import os
from config.config import BOARD_SIZE

BG          = (15,  15,  20)   # near-black background
GRID        = (28,  28,  38)   # subtle grid lines
FOOD        = (255, 80,  80)   # vivid red food
S1_HEAD     = (80,  220, 120)  # player – bright green head
S1_BODY     = (45,  140,  70)  # player – dark green body
S2_HEAD     = (80,  160, 255)  # AI – bright blue head
S2_BODY     = (40,   90, 180)  # AI – dark blue body
TEXT_MAIN   = (230, 230, 230)
TEXT_DIM    = (120, 120, 140)
OVERLAY_BG  = (10,  10,  15, 210)   # semi-transparent end screen

CELL        = 52          # px per cell
MARGIN      = 12          # px gap between grid cells
SIDEBAR_W   = 220         # right panel width
PADDING     = 24          # outer padding


def cell_rect(row, col):
    """Return pygame.Rect for a board cell."""
    x = PADDING + col * (CELL + MARGIN)
    y = PADDING + row * (CELL + MARGIN)
    return pygame.Rect(x, y, CELL, CELL)


def board_pixel_size():
    """Total pixel size of the board area."""
    side = PADDING * 2 + BOARD_SIZE * CELL + (BOARD_SIZE - 1) * MARGIN
    return side


class Renderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Double Snake")

        board_px = board_pixel_size()
        self.W = board_px + SIDEBAR_W
        self.H = board_px

        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock  = pygame.time.Clock()

        # fonts
        self.font_xl  = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_lg  = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 16)
        self.font_sm  = pygame.font.SysFont("monospace", 13)

        # Load fruit images from assets/fruits/ (optional). Filenames will determine fruit types.
        self.fruit_images = {}
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fruits')
        assets_dir = os.path.normpath(assets_dir)
        if os.path.isdir(assets_dir):
            for fname in os.listdir(assets_dir):
                if not fname.lower().endswith('.png'):
                    continue
                name = os.path.splitext(fname)[0]
                path = os.path.join(assets_dir, fname)
                try:
                    img = pygame.image.load(path).convert_alpha()
                    img = pygame.transform.smoothscale(img, (CELL - 16, CELL - 16))
                    self.fruit_images[name] = img
                except Exception:
                    # ignore loading errors and fall back to drawn fruit
                    pass


    def draw(self, game, scores, fps=10):
        """Render one frame. Call every game tick."""
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_food(game.board.food)
        self._draw_snake(game.snake1, S1_HEAD, S1_BODY)
        self._draw_snake(game.snake2, S2_HEAD, S2_BODY)
        self._draw_sidebar(game, scores)
        pygame.display.flip()
        self.clock.tick(fps)

    def draw_end_screen(self, message, scores):
        """Semi-transparent overlay with result."""
        board_px = board_pixel_size()

        surf = pygame.Surface((board_px, board_px), pygame.SRCALPHA)
        surf.fill(OVERLAY_BG)
        self.screen.blit(surf, (0, 0))

        # result text
        title = self.font_xl.render(message, True, TEXT_MAIN)
        self.screen.blit(title, title.get_rect(center=(board_px // 2, board_px // 2 - 40)))

        hint = self.font_md.render("R  restart      Q  quit", True, TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(board_px // 2, board_px // 2 + 20)))

        score_txt = self.font_md.render(
            f"You {scores['player']}  –  AI {scores['ai']}", True, TEXT_DIM
        )
        self.screen.blit(score_txt, score_txt.get_rect(center=(board_px // 2, board_px // 2 + 55)))

        pygame.display.flip()

    def tick(self, fps=10):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()

    def _draw_grid(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = cell_rect(r, c)
                pygame.draw.rect(self.screen, GRID, rect, border_radius=6)

    def _draw_food(self, food):
        if food is None:
            return
        # food may be (position, type) or just position
        if isinstance(food, tuple) and len(food) >= 1 and isinstance(food[0], tuple):
            pos, ftype = food[0], (food[1] if len(food) > 1 else 'default')
        else:
            pos, ftype = food, 'default'

        rect = cell_rect(*pos)

        # simple fruit palette (can be replaced with images later)
        FRUIT_COLORS = {
            'apple': (220, 60,  60),
            'banana': (255, 220, 80),
            'cherry': (200, 30,  70),
            'orange': (255, 140, 20),
            'default': FOOD,
        }

        color = FRUIT_COLORS.get(ftype, FOOD)

        # Try to draw an image for the fruit if available, otherwise fallback to ellipse
        img = self.fruit_images.get(ftype)
        if img:
            img_rect = img.get_rect(center=rect.center)
            self.screen.blit(img, img_rect)
        else:
            # draw fruit body (fallback)
            pygame.draw.ellipse(self.screen, color, rect.inflate(-10, -10))
            pygame.draw.ellipse(self.screen, (255, 200, 200), rect.inflate(-26, -26))

    def _draw_snake(self, snake, head_col, body_col):
        for i, (r, c) in enumerate(snake.body):
            rect = cell_rect(r, c)
            color = head_col if i == 0 else body_col
            radius = 10 if i == 0 else 7
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)
            # eyes on head
            if i == 0:
                self._draw_eyes(rect, snake.direction)

    def _draw_eyes(self, rect, direction):
        dr, dc = direction
        cx, cy = rect.centerx, rect.centery
        eye_r = 4

        if direction == (-1, 0):   # UP
            e1, e2 = (cx - 9, cy - 8), (cx + 9, cy - 8)
        elif direction == (1, 0):  # DOWN
            e1, e2 = (cx - 9, cy + 8), (cx + 9, cy + 8)
        elif direction == (0, -1): # LEFT
            e1, e2 = (cx - 8, cy - 9), (cx - 8, cy + 9)
        else:                      # RIGHT
            e1, e2 = (cx + 8, cy - 9), (cx + 8, cy + 9)

        for pos in (e1, e2):
            pygame.draw.circle(self.screen, (255, 255, 255), pos, eye_r)
            pygame.draw.circle(self.screen, (10, 10, 10), pos, eye_r - 2)

    def _draw_sidebar(self, game, scores):
        board_px = board_pixel_size()
        x = board_px + 16
        y = PADDING

        def label(text, color=TEXT_DIM, font=None):
            nonlocal y
            f = font or self.font_sm
            surf = f.render(text, True, color)
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 6

        def gap(n=12):
            nonlocal y
            y += n

        label("DOUBLE SNAKE", TEXT_MAIN, self.font_lg)
        label("Player vs AI", TEXT_DIM)
        gap()

        # divider
        pygame.draw.line(self.screen, GRID, (x, y), (x + SIDEBAR_W - 24, y))
        gap(10)

        # player
        pygame.draw.rect(self.screen, S1_HEAD, pygame.Rect(x, y, 14, 14), border_radius=3)
        self.screen.blit(self.font_md.render("  YOU", True, TEXT_MAIN), (x + 18, y - 2))
        y += 22
        label(f"  Length : {len(game.snake1.body)}")
        label(f"  Wins   : {scores['player']}")
        gap()

        # ai
        pygame.draw.rect(self.screen, S2_HEAD, pygame.Rect(x, y, 14, 14), border_radius=3)
        self.screen.blit(self.font_md.render("  AI", True, TEXT_MAIN), (x + 18, y - 2))
        y += 22
        label(f"  Length : {len(game.snake2.body)}")
        label(f"  Wins   : {scores['ai']}")
        gap()

        pygame.draw.line(self.screen, GRID, (x, y), (x + SIDEBAR_W - 24, y))
        gap(10)

        label(f"Steps : {game.steps}")
        gap()

        label("Controls", TEXT_MAIN, self.font_sm)
        label("WASD / Arrows – move")
        label("R – restart")
        label("Q – quit")