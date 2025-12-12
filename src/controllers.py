# src/controllers.py
import pygame
import random
import time
from typing import Optional, Tuple, List, Dict

from llm_planner import PlannerClient

# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------

# Set to True if you want to use the LLM-based planner (when LangChain + OpenAI/Ollama
# is configured in llm_planner). If False, only local heuristic planner is used.
USE_LLM_PLANNER = False  # flip to True when you want the LLM

# Local planner behavior (these are used even if USE_LLM_PLANNER is False)
#   search_mode: "greedy" (single pass) or "multi" (multiple simulations)
LOCAL_SEARCH_MODE = "multi"  # "greedy" or "multi"
LOCAL_SEARCH_TRIALS = 8  # number of simulations for "multi"
LOCAL_MAX_ITEMS = 20  # max steps in a plan
LLM_BACKEND = "ollama",
# Create planner client with explicit configuration
PLANNER = PlannerClient(
    use_llm=USE_LLM_PLANNER,
    search_mode=LOCAL_SEARCH_MODE,
    num_trials=LOCAL_SEARCH_TRIALS,
    max_items=LOCAL_MAX_ITEMS,
    llm_backend=LLM_BACKEND,
)

# If True, AI will attempt trips even if it cannot guarantee a return-to-base.
# The drone will be allowed to attempt pickup/drop and may become lost if battery drains mid-route.
ALLOW_RISKY_TRIPS = True


class BaseAgentController:
    """Base controller API used by the game loop."""

    def __init__(self, drone, terrain):
        self.drone = drone
        self.terrain = terrain

    def handle_event(self, event):
        pass

    def update(self, dt: float):
        pass


class HumanAgentController(BaseAgentController):
    """Manual control for the drone (keyboard)."""

    def __init__(self, drone, terrain):
        super().__init__(drone, terrain)
        self._last_keys = pygame.key.get_pressed()

    def handle_event(self, event):
        pass

    def update(self, dt: float):
        keys = pygame.key.get_pressed()
        if self.drone.lost:
            return

        # Movement (only set a new target if drone is not already moving)
        if not self.drone.moving:
            dx = 0
            dy = 0
            if keys[pygame.K_LEFT]:
                dx -= 1
            if keys[pygame.K_RIGHT]:
                dx += 1
            if keys[pygame.K_UP]:
                dy -= 1
            if keys[pygame.K_DOWN]:
                dy += 1

            if dx != 0 or dy != 0:
                new_col = self.drone.col + dx
                new_row = self.drone.row + dy
                # ensure energy to move and optionally return home
                station = self.terrain.nearest_station(self.drone.col, self.drone.row)
                if station:
                    home_col = int(station.col + station.w // 2)
                    home_row = int(station.row + station.h // 2)
                else:
                    home_col, home_row = self.drone.col, self.drone.row

                # The human controller still uses a conservative check
                if hasattr(self.drone, "can_reach_and_return") and not self.drone.can_reach_and_return(
                        new_col, new_row, home_col, home_row
                ):
                    # insufficient energy for that move + return -> send home instead
                    self.drone.set_target_cell(home_col, home_row)
                else:
                    self.drone.set_target_cell(new_col, new_row)

        # Edge-detect SPACE for pick/drop (human triggered)
        if keys[pygame.K_SPACE] and not self._last_keys[pygame.K_SPACE]:
            if self.drone.carrying is None:
                p = self.terrain.parcel_at_cell(self.drone.col, self.drone.row)
                if p:
                    success = self.drone.perform_pick(p)
                    if not success:
                        # pick failed due to battery -> pick_failed UI flash
                        self.drone._last_action = ("pick_failed", (self.drone.col, self.drone.row), None)
            else:
                # drop only if target cell has no non-picked parcel (includes delivered)
                if self.terrain.occupied_cell(self.drone.col, self.drone.row):
                    self.drone._last_action = ("drop_failed", (self.drone.col, self.drone.row), None)
                else:
                    self.drone.perform_drop(self.drone.carrying)

        self._last_keys = keys


class ControllerSwitcher:
    """Simple switcher between controllers (Human/AI). TAB cycles."""

    def __init__(self, controllers: List[BaseAgentController]):
        assert controllers, "provide at least one controller"
        self.controllers = controllers
        self.index = 0

    @property
    def current(self) -> BaseAgentController:
        return self.controllers[self.index]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self.index = (self.index + 1) % len(self.controllers)
        else:
            self.current.handle_event(event)

    def update(self, dt: float):
        self.current.update(dt)
