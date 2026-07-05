import pygame

from game.renderer import BG, GRID, TEXT_DIM, TEXT_MAIN, board_pixel_size


OPTIONS = [
    ("Player vs RL Agent", "Igraj protiv treniranog DQN agenta"),
    ("RL vs Heuristic", "Posmatraj treniranog DQN agenta"),
    ("Train in Background", "RL agent trenira bez renderera"),
]


class StartScreen:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Double Snake")

        board_px = board_pixel_size()
        self.width = board_px + 240
        self.height = board_px
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("monospace", 42, bold=True)
        self.font_option = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_hint = pygame.font.SysFont("consolas", 16)
        self.selected = 0
        self.option_rects = []

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        return None
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.selected = (self.selected - 1) % len(OPTIONS)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.selected = (self.selected + 1) % len(OPTIONS)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        return self.selected + 1
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        return event.key - pygame.K_0
                if event.type == pygame.MOUSEMOTION:
                    for index, rect in enumerate(self.option_rects):
                        if rect.collidepoint(event.pos):
                            self.selected = index
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for index, rect in enumerate(self.option_rects):
                        if rect.collidepoint(event.pos):
                            return index + 1

            self._draw()
            self.clock.tick(30)

    def _draw(self):
        self.screen.fill(BG)
        self.option_rects = []

        title = self.font_title.render("DOUBLE SNAKE", True, TEXT_MAIN)
        self.screen.blit(title, title.get_rect(center=(self.width // 2, 86)))

        subtitle = self.font_hint.render("Choose mode", True, TEXT_DIM)
        self.screen.blit(subtitle, subtitle.get_rect(center=(self.width // 2, 132)))

        start_y = 190
        button_w = min(560, self.width - 120)
        button_h = 78
        gap = 18
        x = (self.width - button_w) // 2

        for index, (label, hint) in enumerate(OPTIONS):
            y = start_y + index * (button_h + gap)
            rect = pygame.Rect(x, y, button_w, button_h)
            self.option_rects.append(rect)

            selected = index == self.selected
            fill = (38, 54, 48) if selected else GRID
            border = (120, 255, 120) if selected else (45, 45, 58)
            pygame.draw.rect(self.screen, fill, rect, border_radius=8)
            pygame.draw.rect(self.screen, border, rect, width=2, border_radius=8)

            number = self.font_option.render(str(index + 1), True, border)
            self.screen.blit(number, (rect.x + 24, rect.y + 23))

            option_text = self.font_option.render(label, True, TEXT_MAIN)
            self.screen.blit(option_text, (rect.x + 68, rect.y + 15))

            hint_text = self.font_hint.render(hint, True, TEXT_DIM)
            self.screen.blit(hint_text, (rect.x + 70, rect.y + 48))

        footer = self.font_hint.render("Enter / click to start     Q to quit", True, TEXT_DIM)
        self.screen.blit(footer, footer.get_rect(center=(self.width // 2, self.height - 42)))
        pygame.display.flip()
