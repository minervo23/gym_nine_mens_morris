import pygame
from dracarys.constants import DiscreteActions
from dracarys.game import Game

FORCE = 0.5
ROTATION = 0.5


class App:
    def __init__(self):
        pygame.init()
        self._game = Game(params='human_single_player')
        self.p = self._game.params
        self.player = self._game.objects_manager.dragons[0]
        self._display = pygame.display.set_mode(
            (self.p.ui.width, self.p.ui.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self._clock = pygame.time.Clock()
        self._running = True

        self.actions = [[0.0, 0.0, 0.0], 0]

        self.player.policy = lambda **_: self.actions

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False
        self.turn_left = False
        self.turn_right = False
        self.fire_pressed = False
        self.act_pressed = False

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self.up_pressed = True
            elif event.key == pygame.K_s:
                self.down_pressed = True
            elif event.key == pygame.K_a:
                self.left_pressed = True
            elif event.key == pygame.K_d:
                self.right_pressed = True
            elif event.key == pygame.K_q:
                self.turn_left = True
            elif event.key == pygame.K_e:
                self.turn_right = True
            elif event.key == pygame.K_SPACE:
                self.fire_pressed = True
            elif event.key == pygame.K_f:
                self.act_pressed = True

        # User let up on a key
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                self.up_pressed = False
            elif event.key == pygame.K_s:
                self.down_pressed = False
            elif event.key == pygame.K_a:
                self.left_pressed = False
            elif event.key == pygame.K_d:
                self.right_pressed = False
            elif event.key == pygame.K_q:
                self.turn_left = False
            elif event.key == pygame.K_e:
                self.turn_right = False
            elif event.key == pygame.K_SPACE:
                self.fire_pressed = False
            elif event.key == pygame.K_f:
                self.act_pressed = False

    def on_loop(self):
        if self._game.episode_manager.ended:
            self._running = False

        self._set_actions()
        self._game.step()

    def on_render(self):
        self._display.fill((255, 255, 255))
        image = self.player.render()
        image = image.transpose((1, 0, 2))
        self._display.blit(pygame.surfarray.make_surface(image), (0, 0))
        pygame.display.flip()
        pygame.display.update()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        while self._running:
            self._clock.tick(self.p.ui.fps)
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_endgame()
        self.on_cleanup()

    def on_endgame(self):
        self._display.fill((0, 0, 0))
        white = (255, 255, 255)
        blue = (0, 0, 128)
        font = pygame.font.Font('freesansbold.ttf', 20)
        text2 = font.render('Thanks for Playing :)', True, white, blue)
        if self._game.objects_manager.unlocked_gate:
            text = font.render('You have successfully escaped! Congrats!', True, white, blue)
        else:
            text = font.render('You did not make it :(', True, white, blue)
        textRect = text.get_rect()
        textRect2 = text2.get_rect()
        display_surface = pygame.display.set_mode((self._game.params.world.width, self._game.params.world.height // 1.5))
        # set the center of the rectangular object.
        textRect.center = (self._game.params.world.width // 2, self._game.params.world.height // 3)
        textRect2.center = (self._game.params.world.width // 2, self._game.params.world.height // 2)
        display_surface.blit(text, textRect)
        display_surface.blit(text2, textRect2)

        imp = pygame.image.load("objects/images/dragon-1-sitting.png").convert()
        # Using blit to copy content from one surface to other
        display_surface.blit(imp, (self._game.params.world.width // 2.7, 0))
        pygame.display.flip()
        game_over = True
        while game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = False

    def _set_actions(self):
        x, y, r, a = 0.0, 0.0, 0.0, 0
        if self.up_pressed:
            y += FORCE
        if self.down_pressed:
            y -= FORCE
        if self.left_pressed:
            x -= FORCE
        if self.right_pressed:
            x += FORCE
        if self.turn_left:
            r -= ROTATION
        if self.turn_right:
            r += ROTATION
        if self.fire_pressed:
            a = DiscreteActions.FIRE
        if self.act_pressed:
            a = DiscreteActions.ACT

        self.actions = (x, y, r), a


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
