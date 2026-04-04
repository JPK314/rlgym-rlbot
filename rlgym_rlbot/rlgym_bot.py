from __future__ import annotations

import os
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from traceback import print_exc
from typing import Any, Tuple, Dict, Generic, List, Optional, Union

import numpy as np
from rlbot import flat
from rlbot.interface import (
    RLBOT_SERVER_IP,
    RLBOT_SERVER_PORT,
    MsgHandlingResult,
    SocketRelay,
)
from rlbot.managers import Bot
from rlbot.managers.rendering import Renderer
from rlbot.utils import fill_desired_game_state
from rlbot.utils.logging import DEFAULT_LOGGER, get_logger
from rlgym.api import (
    ActionParser,
    ActionType,
    DoneCondition,
    ObsBuilder,
    ObsType,
    RewardFunction,
    RewardType,
    SharedInfoProvider,
    StateMutator,
)
from rlgym_compat import GameState
from rlgym_compat.sim_extra_info import SimExtraInfo

from .util import create_base_state

AgentID = int

USUAL_COUNTDOWN_LENGTH_TICKS = 480


class MissedActionRecoveryStyle(Enum):
    RESET = 0
    IGNORE = 1


class MissedStepTickRecoveryStyle(Enum):
    RESET = 0
    USE_NEAREST = 1


@dataclass
class RLGymBotConfig:
    above_240_fps_mode: bool = False
    action_step_idx_used_to_build_game_state_for_env_step: int = -2
    missed_action_recovery_style: MissedActionRecoveryStyle = (
        MissedActionRecoveryStyle.RESET
    )
    missed_step_tick_recovery_style: MissedStepTickRecoveryStyle = (
        MissedStepTickRecoveryStyle.RESET
    )
    standard_map: bool = True
    sim_extra_info: bool = False


class RLGymBot(Generic[ActionType, ObsType, RewardType]):
    """
    A convenience base class for bots developed using RLGym.
    The base class handles the setup and communication with the rlbot server, along with management of RLGym config objects.
    Subclass from this to override the following methods:
    - initialize
    - retire
    - handle_reward
    - get_action
    - handle_match_comm

    """

    logger = DEFAULT_LOGGER

    team: int = -1
    index: int = -1
    name: str = ""
    player_id: int = 0

    match_config = flat.MatchConfiguration()
    """
    Contains info about what map you're on, game mode, mutators, etc.
    """

    field_info = flat.FieldInfo()
    """
    Contains info about the map, such as the locations of boost pads and goals.
    """

    ball_prediction = flat.BallPrediction()
    """
    A simulated prediction of the ball's trajectory including collisions with field geometry (but not cars).
    """

    _initialized_bot = False
    _has_match_settings = False
    _has_field_info = False
    _has_player_mapping = False

    _latest_prediction = flat.BallPrediction()

    def __init__(
        self,
        obs_builder: ObsBuilder[AgentID, ObsType, GameState, Any],
        action_parser: ActionParser[AgentID, ActionType, np.ndarray, GameState, Any],
        reward_function: Optional[
            RewardFunction[AgentID, GameState, RewardType]
        ] = None,
        termination_condition: Optional[DoneCondition[AgentID, GameState]] = None,
        truncation_condition: Optional[DoneCondition[AgentID, GameState]] = None,
        state_mutator: Optional[StateMutator[GameState]] = None,
        shared_info_provider: Optional[SharedInfoProvider[AgentID, GameState]] = None,
        default_agent_id: Optional[str] = None,
        config=RLGymBotConfig(),
    ):
        # Long term, the below assertion is what we will use. It doesn't hurt to leave it uncommented along with the above.
        assert (
            config.action_step_idx_used_to_build_game_state_for_env_step != -1
            or config.above_240_fps_mode
        ), "action_step_idx_used_to_build_game_state_for_env_step of -1 requires a stable fps of at least 240"
        if config.above_240_fps_mode:
            self.logger.info(
                "above_240_fps_mode enabled. Make sure to have a consistent FPS above 240 in Rocket League for best performance of the bot."
            )
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.reward_function = reward_function
        self.termination_condition = termination_condition
        self.truncation_condition = truncation_condition
        self.state_mutator = state_mutator
        self.shared_info_provider = shared_info_provider
        self.config = config

        self.received_first_packet = False
        self.shared_info = (
            shared_info_provider.create({}) if shared_info_provider is not None else {}
        )
        self._just_handled_countdown_packet = False
        self._match_first_countdown = True
        self._match_first_kickoff = True
        self._last_packet = None
        self._first_countdown_tick = 0
        self._set_first_action = False
        self._early_countdown_finish_adjustment_possible = True
        self._last_env_action_start_tick = 0
        self._last_sent_action_tick = 0
        self._last_sent_action = flat.ControllerState()
        self._unused_packets: List[flat.GamePacket] = []
        self._hist_game_states_and_packets: Dict[
            int, Tuple[GameState, flat.GamePacket]
        ] = {}
        self._future_tick_action_map: Dict[int, flat.ControllerState] = {}

        agent_id = os.environ.get("RLBOT_AGENT_ID") or default_agent_id
        if agent_id is None:
            self.logger.critical(
                "Environment variable RLBOT_AGENT_ID is not set and no default agent id is passed to "
                "the constructor of the bot. If you are starting your bot manually, please set it "
                "manually, e.g. `RLBOT_AGENT_ID=<agent_id> python yourbot.py`"
            )
            exit(1)

        self._game_interface = SocketRelay(agent_id, logger=self.logger)
        self._game_interface.match_config_handlers.append(self._handle_match_config)
        self._game_interface.field_info_handlers.append(self._handle_field_info)
        self._game_interface.match_comm_handlers.append(
            self._handle_match_communication
        )
        self._game_interface.ball_prediction_handlers.append(
            self._handle_ball_prediction
        )
        self._game_interface.controllable_team_info_handlers.append(
            self._handle_controllable_team_info
        )
        self._game_interface.packet_handlers.append(self._handle_packet)

        self.renderer = Renderer(self._game_interface)

    def _try_initialize(self):
        if (
            self._initialized_bot
            or not self._has_match_settings
            or not self._has_field_info
            or not self._has_player_mapping
        ):
            # Not ready to initialize
            return

        # Search match settings for our name
        for player in self.match_config.player_configurations:
            match player.variety:
                case flat.CustomBot(name):
                    if player.player_id == self.player_id:
                        self.name = name
                        self.logger = get_logger(self.name)
                        break
        else:  # else block runs if break was not hit
            self.logger.warning(
                "Bot with agent id '%s' did not find itself in the match configuration",
                self._game_interface.agent_id,
            )

        self.latest_game_state = GameState.create_compat_game_state(
            field_info=self.field_info,
            match_configuration=self.match_config,
            standard_map=self.config.standard_map,
        )
        if self.config.sim_extra_info:
            self.sim_extra_info = SimExtraInfo(self.field_info, self.match_config)
        try:
            self.initialize()
        except Exception as e:
            self.logger.critical(
                "Bot %s failed to initialize due the following error: %s", self.name, e
            )
            print_exc()
            exit()

        self._initialized_bot = True

        self._game_interface.send_msg(flat.InitComplete())

    def _handle_match_config(self, match_config: flat.MatchConfiguration):
        self.match_config = match_config
        self._has_match_settings = True
        self._try_initialize()

    def _handle_field_info(self, field_info: flat.FieldInfo):
        self.field_info = field_info
        self._has_field_info = True
        self._try_initialize()

    def _handle_controllable_team_info(
        self, player_mappings: flat.ControllableTeamInfo
    ):
        self.team = player_mappings.team
        controllable = player_mappings.controllables[0]
        self.player_id = controllable.identifier
        self.index = controllable.index
        self._has_player_mapping = True

        self._try_initialize()

    def _handle_ball_prediction(self, ball_prediction: flat.BallPrediction):
        self._latest_prediction = ball_prediction

    # Update self._future_tick_action_map. The map defines actions where the action self._future_tick_action_map[i] is taken between packets with
    # frame_num i+1 and i+2 if not self.config.above_240_fps_mode and i and i+1 otherwise, i.e., i is when the action is submitted, not when the action is used
    def _update_future_tick_action_map(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        obs: Dict[int, ObsType],
        start_tick: int,
        clear_hist=True,
    ):
        if clear_hist:
            self._future_tick_action_map = {
                tick: action
                for (tick, action) in self._future_tick_action_map.items()
                if tick
                >= self._last_env_action_start_tick
                + min(
                    0, self.config.action_step_idx_used_to_build_game_state_for_env_step
                )
            }
        action = self.get_action(obs[self.player_id], game_state, packet)
        if isinstance(action, List):
            self._last_engine_action_length = len(action)
            controller_states = action
        else:
            engine_action = self.action_parser.parse_actions(
                {self.player_id: action}, game_state, self.shared_info
            )[self.player_id]
            steps = engine_action.shape[0]
            self._last_engine_action_length = steps
            controller_states = []
            for idx in range(steps):
                (
                    throttle,
                    steer,
                    pitch,
                    yaw,
                    roll,
                    jump_float,
                    boost_float,
                    handbrake_float,
                ) = engine_action[idx]
                controller_states.append(
                    flat.ControllerState(
                        throttle=throttle,
                        steer=steer,
                        pitch=pitch,
                        yaw=yaw,
                        roll=roll,
                        jump=jump_float > 0,
                        boost=boost_float > 0,
                        handbrake=handbrake_float > 0,
                        use_item=False,
                    )
                )
        for idx, controller_state in enumerate(controller_states):
            self._future_tick_action_map[start_tick + idx] = controller_state

    @staticmethod
    def _get_agents_list(game_state: GameState):
        return list(game_state.cars.keys())

    def _update_gamestate_ball_touches(self, reset_tick: int):
        if reset_tick not in self._hist_game_states_and_packets:
            self.logger.warning(
                f"Ball touches reset failed - reset requested starting from tick {reset_tick} but there is no game state that was created on this tick and so we can't identify the number of touches to subtract for each player"
            )
            return

        gs, _ = self._hist_game_states_and_packets[reset_tick]
        n_touches_to_sub = {car_id: car.ball_touches for car_id, car in gs.cars.items()}
        for gs, _ in self._hist_game_states_and_packets.values():
            if gs.tick_count < reset_tick:
                continue
            for car_id, car in gs.cars.items():
                car.ball_touches -= n_touches_to_sub[car_id]
                car._last_reset_ball_touches_tick = reset_tick

    def _env_reset(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        start_tick: int,
        set_state=False,
    ):

        self._last_env_action_start_tick = packet.match_info.frame_num
        if self.shared_info_provider is not None:
            self.shared_info = self.shared_info_provider.create(self.shared_info)
        if self.state_mutator is not None:
            # Whether or not we are going to set the state, we should still make this call so that shared_info can get updated by the state mutator (I guess)
            desired_state = create_base_state()
            self.state_mutator.apply(desired_state, self.shared_info)
            if set_state:
                # TODO: set state
                pass
        game_state = deepcopy(game_state)
        self._update_gamestate_ball_touches(game_state.tick_count)
        agents = RLGymBot._get_agents_list(game_state)
        if self.shared_info_provider is not None:
            self.shared_info = self.shared_info_provider.set_state(
                agents, game_state, self.shared_info
            )
        self.obs_builder.reset(agents, game_state, self.shared_info)
        self.action_parser.reset(agents, game_state, self.shared_info)
        if self.termination_condition is not None:
            self.termination_condition.reset(agents, game_state, self.shared_info)
        if self.truncation_condition is not None:
            self.truncation_condition.reset(agents, game_state, self.shared_info)
        if self.reward_function is not None:
            self.reward_function.reset(agents, game_state, self.shared_info)
        obs = self.obs_builder.build_obs(agents, game_state, self.shared_info)
        self._update_future_tick_action_map(packet, game_state, obs, start_tick)

    def _env_step(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        start_tick: int,
        clear_hist=True,
    ) -> Tuple[bool, bool]:
        self._last_env_action_start_tick = packet.match_info.frame_num
        game_state = deepcopy(game_state)
        self._update_gamestate_ball_touches(game_state.tick_count)
        agents = RLGymBot._get_agents_list(game_state)
        if self.shared_info_provider is not None:
            self.shared_info = self.shared_info_provider.step(
                agents, game_state, self.shared_info
            )
        obs = self.obs_builder.build_obs(agents, game_state, self.shared_info)
        is_terminated = {self.player_id: False}
        is_truncated = {self.player_id: False}
        if self.termination_condition is not None:
            is_terminated = self.termination_condition.is_done(
                agents, game_state, self.shared_info
            )
        if self.truncation_condition is not None:
            is_truncated = self.truncation_condition.is_done(
                agents, game_state, self.shared_info
            )
        if self.reward_function is not None:
            rewards = self.reward_function.get_rewards(
                agents,
                game_state,
                is_terminated,
                is_truncated,
                self.shared_info,
            )
            self.handle_reward(rewards[self.player_id])
        self._update_future_tick_action_map(
            packet, game_state, obs, start_tick, clear_hist
        )
        return is_terminated, is_truncated

    def _handle_packet(self, packet: flat.GamePacket):
        self._unused_packets.append(packet)

    def _update_gamestate_using_packets(
        self, packets: List[flat.GamePacket], clean_hist=True
    ):
        # We assume the packets here are only going to be match phase of countdown, kickoff, or active
        for packet in packets:
            extra_info = None
            if self.config.sim_extra_info:
                extra_info = self.sim_extra_info.get_extra_info(packet)
            self.latest_game_state.update(packet, extra_info)
            self._hist_game_states_and_packets[packet.match_info.frame_num] = (
                deepcopy(self.latest_game_state),
                packet,
            )
        if clean_hist:
            self._hist_game_states_and_packets = {
                tick: gsp
                for tick, gsp in self._hist_game_states_and_packets.items()
                if tick >= self._last_env_action_start_tick
            }

    # We last sent an action on some tick, and more than one tick has passed since then. That's ok if the action we sent is the one we were going to send anyway for the ticks since.
    # This method checks to see if that's indeed the case or not
    def _check_last_sent_action_correct_for_ticks_since(self) -> bool:
        cur_tick = self._unused_packets[-1].match_info.frame_num
        correct = False
        if cur_tick in self._future_tick_action_map:
            # If cur_tick is in self._future_tick_action_map, we only need to reset if the action we have taken since then up to the current tick is not equal to
            # self._last_sent_action
            correct = True
            # TODO: switch back to .pack() once Virx adds the functionality back
            last_sent_action_pack = repr(self._last_sent_action)
            idx = self._last_sent_action_tick
            while correct and idx < cur_tick:
                idx += 1
                # TODO: here too
                if repr(self._future_tick_action_map[idx]) != last_sent_action_pack:
                    correct = False
        return correct

    # Handle env resets when they need to happen but we don't necessarily have all the information to do it normally.
    def _unexpected_env_reset_and_update_action_map(
        self, desired_reset_tick, desired_submit_first_action_tick
    ):
        cur_tick = self._unused_packets[-1].match_info.frame_num
        nearest_hist_tick_to_desired_reset_tick = min(
            self._hist_game_states_and_packets,
            key=lambda v: abs(v - desired_reset_tick),
        )
        (nearest_game_state, nearest_packet) = self._hist_game_states_and_packets[
            nearest_hist_tick_to_desired_reset_tick
        ]

        self._env_reset_and_maybe_step(
            packet=nearest_packet,
            game_state=nearest_game_state,
            start_tick=desired_submit_first_action_tick,
        )
        if cur_tick not in self._future_tick_action_map:
            # If the action is only for one tick and not self.config.above_240_fps_mode, it will put it for the previous tick instead of the current tick, which is not workable.
            # If it's more than one tick, starting using the second tick is desirable
            self._future_tick_action_map[cur_tick] = self._future_tick_action_map[
                desired_submit_first_action_tick
            ]

    def _env_reset_and_maybe_step(
        self,
        packet: flat.GamePacket,
        game_state: GameState,
        start_tick: bool,
    ):
        self._env_reset(packet, game_state, start_tick, set_state=False)
        if self.config.action_step_idx_used_to_build_game_state_for_env_step == 0:
            # We calculate the update to the obs etc using the same state as the reset and use this to produce the actions to be used after the reset's obs' actions
            self._env_step(
                packet, game_state, max(self._future_tick_action_map.keys()) + 1
            )

    # Update the future tick action map and internal state based on the unused packets we have collected since the last time this method was called
    def _process_unused_packets(self):
        try:
            # Check if we are past the first countdown of the match so that we can know whether or not we can use the 480 tick thing to line up kickoff actions appropriately (see below)
            if any(
                [
                    p.match_info.match_phase
                    in [flat.MatchPhase.Kickoff, flat.MatchPhase.Active]
                    for p in self._unused_packets
                ]
            ):
                self._match_first_countdown = False

            # If we just received a countdown packet for the first time since the last countdown, store the first countdown packet's frame num for use later
            if (
                self._last_packet is None
                or self._last_packet.match_info.match_phase
                in [
                    flat.MatchPhase.GoalScored,
                    flat.MatchPhase.Replay,
                ]
            ) and any(
                [
                    p.match_info.match_phase == flat.MatchPhase.Countdown
                    for p in self._unused_packets
                ]
            ):
                self._first_countdown_tick = next(
                    p.match_info.frame_num
                    for p in self._unused_packets
                    if p.match_info.match_phase == flat.MatchPhase.Countdown
                )
                self._set_first_action = False
                self._early_countdown_finish_adjustment_possible = True

            latest_packet = self._unused_packets[-1]
            cur_tick = latest_packet.match_info.frame_num

            # If the latest packet is not one of Countdown, Kickoff, or Active, delegate to self.get_other_packet_output
            if latest_packet.match_info.match_phase not in [
                flat.MatchPhase.Countdown,
                flat.MatchPhase.Kickoff,
                flat.MatchPhase.Active,
            ]:
                self._game_interface.send_msg(
                    flat.PlayerInput(
                        self.index, self.get_other_packet_output(latest_packet)
                    )
                )
                return

            # If the latest packet is countdown, update the game state using all the packets sequentially and then handle the latest packet, and then we're done
            if latest_packet.match_info.match_phase == flat.MatchPhase.Countdown:
                self._last_seen_countdown_tick = cur_tick
                # Actually, if this isn't the first kickoff, we can do a little better by assuming we caught the first tick of countdown and countdown lasts for 480 ticks:
                # In that case, we can predict when we are on the last tick of countdown and when this happens we actually don't want to update the future tick action map - we want to leave the
                # actions in place from the previous packet so that we are aligned properly. Of course, we only want to do this if 240 fps mode is disabled, because in that case we
                # want to use the last tick of countdown to reset and determine actions.
                # cur_tick - self._first_countdown_tick == 479 => last countdown packet => want to reset using:
                #  0 case: last tick of countdown (in above 240fps mode) submitting first action on last tick of countdown, 2nd to last tick of countdown (in 120fps mode) submitting first action on 2nd to last tick of countdown
                #  1 case: last tick of countdown (in above 240fps mode) submitting first action on last tick of countdown, 2nd to last tick of countdown (in 120fps mode) submitting first action on 2nd to last tick of countdown
                #  -1 case: last tick of countdown (in above 240fps mode) submitting first action on last tick of countdown
                #  -2 case: 2nd to last tick of countdown (in above 240fps mode) submitting first action on last tick of countdown, 2nd to last tick of countdown (in 120fps mode) submitting first action on 2nd to last tick of countdown
                #  -3 case: 3rd to last tick of countdown (in above 240fps mode) submitting first action on last tick of countdown, 3rd to last tick of countdown (in 120fps mode) submitting first action on 2nd to last tick of countdown
                if not self._match_first_countdown:
                    last_countdown_tick = (
                        cur_tick
                        if cur_tick - self._first_countdown_tick
                        >= USUAL_COUNTDOWN_LENGTH_TICKS - 1
                        else USUAL_COUNTDOWN_LENGTH_TICKS
                        - 1
                        + self._first_countdown_tick
                    )
                    desired_last_reset_tick = (
                        last_countdown_tick - (not self.config.above_240_fps_mode)
                        if self.config.action_step_idx_used_to_build_game_state_for_env_step
                        >= -1
                        else last_countdown_tick
                        + self.config.action_step_idx_used_to_build_game_state_for_env_step
                        + 1
                    )
                    desired_submit_first_action_tick = last_countdown_tick - (
                        not self.config.above_240_fps_mode
                    )
                    if cur_tick < desired_last_reset_tick:
                        # Update the game state using all the packets sequentially and then reset using the latest packet. Let's just say we plan to start the action this tick so that something is submitted for later
                        self._update_gamestate_using_packets(self._unused_packets)
                        self._env_reset_and_maybe_step(
                            packet=latest_packet,
                            game_state=self.latest_game_state,
                            start_tick=cur_tick,
                        )
                        return
                    # cur_tick >= desired_last_reset_tick and so if self._set_first_action is false we need to find the nearest packet tick and use that to reset
                    if not self._set_first_action:
                        packet_ticks = [
                            p.match_info.frame_num for p in self._unused_packets
                        ]
                        nearest_packet_tick = min(
                            packet_ticks, key=lambda v: abs(v - desired_last_reset_tick)
                        )
                        # Update the game state using all the packets up through the one we want to reset on, and then reset using that packet (starting the action in the action map on desired_submit_first_action_tick)
                        last_reset_packet_idx = packet_ticks.index(nearest_packet_tick)
                        self._update_gamestate_using_packets(
                            self._unused_packets[: last_reset_packet_idx + 1]
                        )
                        self._env_reset_and_maybe_step(
                            packet=self._unused_packets[last_reset_packet_idx],
                            game_state=self.latest_game_state,
                            start_tick=desired_submit_first_action_tick,
                        )
                        self._update_gamestate_using_packets(
                            self._unused_packets[last_reset_packet_idx + 1 :],
                            clean_hist=False,
                        )
                        # Just in case the last countdown tick is one tick closer than we expected, have this pre-fired
                        self._future_tick_action_map[cur_tick] = (
                            self._future_tick_action_map[
                                desired_submit_first_action_tick
                            ]
                        )
                        self._set_first_action = True
                        return
                    # At this point we've done all we can do, EXCEPT in the edge case that kickoff is one tick later than we expected and we are in this block having already set the first action
                    if (
                        last_countdown_tick
                        > USUAL_COUNTDOWN_LENGTH_TICKS - 1 + self._first_countdown_tick
                    ):
                        # Just move everything down by len(self._unused_packets)
                        self._future_tick_action_map = {
                            (k + len(self._unused_packets)): v
                            for (k, v) in self._future_tick_action_map.items()
                        }
                    return
                # For first kickoff, we don't know when it will end so we have to assume every latest packet is the last countdown tick
                self._update_gamestate_using_packets(
                    self._unused_packets, clean_hist=False
                )
                desired_reset_tick = (
                    cur_tick - (not self.config.above_240_fps_mode)
                    if self.config.action_step_idx_used_to_build_game_state_for_env_step
                    >= -1
                    else cur_tick
                    + self.config.action_step_idx_used_to_build_game_state_for_env_step
                    + 1
                )
                desired_submit_first_action_tick = cur_tick - (
                    not self.config.above_240_fps_mode
                )
                self._unexpected_env_reset_and_update_action_map(
                    desired_reset_tick, desired_submit_first_action_tick
                )
                return
            if (
                self._unused_packets[0].match_info.match_phase
                == flat.MatchPhase.Countdown
            ):
                self._last_seen_countdown_tick = next(
                    p.match_info.frame_num
                    for p in reversed(self._unused_packets)
                    if p.match_info.match_phase == flat.MatchPhase.Countdown
                )
                # We have some positive number of countdown packets and some positive number of live game packets (either kickoff or active, but probably kickoff unless something crazy has happened).
                if not self._check_last_sent_action_correct_for_ticks_since():
                    # On a previous tick, we took an action we did not expect to have taken initially. Let's reset and start fresh, and then we're done.
                    # I'm not going to use MissedActionRecoveryStyle here because this is an edge case where resetting should always be preferable.
                    self._unexpected_env_reset_and_update_action_map(cur_tick, cur_tick)
                    return
                # OK, at this point we are in the same situation we would be in if all the unused packets were live game ones, so let's handle both at once by finishing this if block here and continuing below

            # If the countdown ended sooner than expected, move up the actions according to when countdown actually ended
            if not self._match_first_kickoff:
                self._match_first_kickoff = False
                countdown_ticks_early = (
                    USUAL_COUNTDOWN_LENGTH_TICKS
                    - 1
                    + self._first_countdown_tick
                    - self._last_seen_countdown_tick
                )
                if (
                    self._early_countdown_finish_adjustment_possible
                    and countdown_ticks_early > 0
                ):
                    self._early_countdown_finish_adjustment_possible = False
                    # Just move everything up by countdown_ticks_early
                    self._future_tick_action_map = {
                        (k - countdown_ticks_early): v
                        for (k, v) in self._future_tick_action_map.items()
                    }

            # Let's say the last action we took defined controller inputs for n ticks, and self._future_tick_action_map defines actions for k more ticks (i.e. k = max(self._future_tick_action_map) - cur_tick).
            # Let's look at when the previous action was created, at tick cur_tick'. We have some k' which is the value of k at the time when the previous action was created. When this happens, the next action
            # will start at cur_tick'+k'+1. After the action gets added, max(self._future_tick_action_map) = cur_tick'+k'+n and so k = cur_tick'+k'+n-cur_tick so k+cur_tick-n+1 = cur_tick'+k'+1.
            # So when k+cur_tick-n+1 = cur_tick (aka n-k=1), we are about to submit the first tick of the last action we generated.
            #
            # In the 0 case, in above_240_fps_mode, we want to generate the new obs when n-k=1 because our most recently decided action is about to start. Otherwise, we want to wait one more tick,
            # so when k+cur_tick-n+2 = cur_tick (aka n-k=2).

            # In the 1 case, in above_240_fps_mode, we want to generate the new obs when n-k=2 because we have then used one tick of the most recently decided action. Otherwise, we want to wait one more tick,
            # so when n-k=3.

            # In the -1 case, which is only achievable in above_240_fps_mode, we want to generate the new obs when k=-1 (i.e. we used the last action in the future tick action map last tick)
            # In the -2 case, in above_240_fps_mode, we want to generate the new obs when k=0, or otherwise when k=-1 (i.e. we used the last action in the future tick action map last tick)

            self._update_gamestate_using_packets(self._unused_packets)
            # We might need to reset if we detect that we have done an incorrect action for any tick we are processing
            if (
                len(self._unused_packets) > 1
                or self._unused_packets[0].match_info.frame_num
                > self._last_packet.match_info.frame_num + 1
            ):
                if not self._check_last_sent_action_correct_for_ticks_since():
                    self.logger.warning(
                        "missed action, recovering using missed action recovery style"
                    )
                    match self.config.missed_action_recovery_style:
                        case MissedActionRecoveryStyle.RESET:
                            self._unexpected_env_reset_and_update_action_map(
                                cur_tick, cur_tick
                            )
                            return
                        case MissedActionRecoveryStyle.IGNORE:
                            # Whatever I guess
                            pass

            # Do the below loop while next_env_step_tick <= cur_tick
            should_continue = True
            while should_continue:
                last_defined_action_tick = max(self._future_tick_action_map)

                # We need to figure out the next tick on which we need to do an env step
                if (
                    self.config.action_step_idx_used_to_build_game_state_for_env_step
                    >= 0
                ):
                    # What's the next tick t we would want to create an obs on? It's when n-k=self.config.action_step_idx_used_to_build_game_state_for_env_step + 1 + (not self.config.above_240_fps_mode)
                    # n = self._last_engine_action_length
                    # k = last_defined_action_tick - t
                    # so self._last_engine_action_length + t - last_defined_action_tick = self.config.action_step_idx_used_to_build_game_state_for_env_step + 1 + (not self.config.above_240_fps_mode)
                    # so t = next_env_step_tick = ...
                    next_env_step_tick = (
                        self.config.action_step_idx_used_to_build_game_state_for_env_step
                        + 1
                        + (not self.config.above_240_fps_mode)
                        + last_defined_action_tick
                        - self._last_engine_action_length
                    )
                else:
                    # Ditto? It's when k = -2 - self.config.action_step_idx_used_to_build_game_state_for_env_step - (not self.config.above_240_fps_mode)
                    # k = last_defined_action_tick - t
                    # so last_defined_action_tick - t = -2 - self.config.action_step_idx_used_to_build_game_state_for_env_step - (not self.config.above_240_fps_mode)
                    # so t = next_env_step_tick = ...
                    next_env_step_tick = (
                        last_defined_action_tick
                        + 2
                        + self.config.action_step_idx_used_to_build_game_state_for_env_step
                        + (not self.config.above_240_fps_mode)
                    )
                if next_env_step_tick < cur_tick:
                    # We were supposed to step the env in the past, which might be fine if we still have actions queued up or the action we've been taking is the intended one
                    # First, find the game state we should use to step
                    step_tick = None
                    if next_env_step_tick not in self._hist_game_states_and_packets:
                        self.logger.warning(
                            "missed step tick %s, recovering using missed step tick recovery style",
                            next_env_step_tick,
                        )
                        # We missed the tick we wanted to use in order to perform the step
                        match self.config.missed_step_tick_recovery_style:
                            case MissedStepTickRecoveryStyle.RESET:
                                self._unexpected_env_reset_and_update_action_map(
                                    cur_tick, cur_tick
                                )
                                continue
                            case MissedStepTickRecoveryStyle.USE_NEAREST:
                                step_tick = min(
                                    self._hist_game_states_and_packets,
                                    key=lambda v: abs(v - next_env_step_tick),
                                )
                    else:
                        step_tick = next_env_step_tick
                    step_gs, step_packet = self._hist_game_states_and_packets[step_tick]

                    # Next see if we are still fine and handle the various scenarios
                    if last_defined_action_tick >= cur_tick:
                        # OK, we can still catch up
                        self._env_step(
                            step_packet,
                            step_gs,
                            last_defined_action_tick + 1,
                        )
                    else:
                        # OK, we still might be fine if the action we have been taking is the intended one, so let's try to step and then see if what we planned aligns with what we actually did
                        # Pass clear_hist as false because the we are calling _check_last_sent_action_correct_for_ticks_since after calling _env_step
                        self._env_step(
                            step_packet,
                            step_gs,
                            last_defined_action_tick + 1,
                            clear_hist=False,
                        )
                        if not self._check_last_sent_action_correct_for_ticks_since():
                            self.logger.warning(
                                "missed action, recovering using missed action recovery style"
                            )
                            match self.config.missed_action_recovery_style:
                                case MissedActionRecoveryStyle.RESET:
                                    self._unexpected_env_reset_and_update_action_map(
                                        cur_tick, cur_tick
                                    )
                                    continue
                                case MissedActionRecoveryStyle.IGNORE:
                                    # Whatever I guess
                                    pass
                else:
                    if next_env_step_tick == cur_tick:
                        self._env_step(
                            latest_packet,
                            self.latest_game_state,
                            last_defined_action_tick + 1,
                        )
                    else:
                        should_continue = False

        except Exception as e:
            self.logger.error(
                "Bot %s encountered an error while processing game packet: %s",
                self.name,
                e,
            )
            print_exc()

    def _run(self):
        running = True
        block_next = False

        while running:
            # If there might be more messages, check for another one with blocking=False
            # if there are no more messages, process the latest packet then wait for the next message with blocking=True
            result = self._game_interface.handle_incoming_messages(blocking=block_next)
            block_next = False
            match result:
                case MsgHandlingResult.TERMINATED:
                    running = False
                case MsgHandlingResult.NO_INCOMING_MSGS:
                    if len(self._unused_packets) > 0:
                        self._process_unused_packets()
                        self._last_packet = self._unused_packets[-1]
                        self._unused_packets.clear()
                        cur_tick = self._last_packet.match_info.frame_num
                        if cur_tick in self._future_tick_action_map:
                            self._last_sent_action_tick = cur_tick
                            self._last_sent_action = self._future_tick_action_map[
                                cur_tick
                            ]
                        else:
                            if self._last_packet.match_info.match_phase in [
                                flat.MatchPhase.Countdown,
                                flat.MatchPhase.Kickoff,
                                flat.MatchPhase.Active,
                            ]:
                                self.logger.warning(
                                    "No controller state set in self._future_tick_action_map for this tick (%d)!",
                                    cur_tick,
                                )
                        self._game_interface.send_msg(
                            flat.PlayerInput(self.index, self._last_sent_action)
                        )
                    block_next = True
                case _:
                    pass

    def run(
        self,
        *,
        wants_match_communications: bool = True,
        wants_ball_predictions: bool = True,
    ):
        """
        Runs the bot. This operation is blocking until the match ends.
        """

        rlbot_server_ip = os.environ.get("RLBOT_SERVER_IP", RLBOT_SERVER_IP)
        rlbot_server_port = int(os.environ.get("RLBOT_SERVER_PORT", RLBOT_SERVER_PORT))

        try:
            self._game_interface.connect(
                wants_match_communications=wants_match_communications,
                wants_ball_predictions=wants_ball_predictions,
                rlbot_server_ip=rlbot_server_ip,
                rlbot_server_port=rlbot_server_port,
            )

            self._run()
        except Exception as e:
            self.logger.error("Unexpected error: %s", e)
            print_exc()
        finally:
            self.retire()
            del self._game_interface

    def _handle_match_communication(self, match_comm: flat.MatchComm):
        self.handle_match_comm(
            match_comm.index,
            match_comm.team,
            match_comm.content,
            match_comm.display,
            match_comm.team_only,
        )

    def handle_match_comm(
        self,
        index: int,
        team: int,
        content: bytes,
        display: Optional[str],
        team_only: bool,
    ):
        """
        Called when a match communication message is received.
        See `send_match_comm`.
        NOTE: Messages from scripts will have `team == 2` and the index will be its index in the match settings.
        """

    def send_match_comm(
        self, content: bytes, display: Optional[str] = None, team_only: bool = False
    ):
        """
        Emits a match communication message to other bots and scripts.

        - `content`: The content of the message containing arbitrary data.
        - `display`: The message to be displayed in the game in "quick chat", or `None` to display nothing.
        - `team_only`: If True, only your team will receive the message.
        """
        self._game_interface.send_match_comm(
            flat.MatchComm(
                self.index,
                self.team,
                team_only,
                display,
                content,
            )
        )

    def set_game_state(
        self,
        balls: dict[int, flat.DesiredBallState] = {},
        cars: dict[int, flat.DesiredCarState] = {},
        match_info: Optional[flat.DesiredMatchInfo] = None,
        commands: list[str] = [],
    ):
        """
        Sets the game to the desired state.
        Through this it is possible to manipulate the position, velocity, and rotations of cars and balls, and more.
        See wiki for a full break down and examples.
        """

        game_state = fill_desired_game_state(balls, cars, match_info, commands)
        self._game_interface.send_game_state(game_state)

    def set_loadout(self, loadout: flat.PlayerLoadout, index: Optional[int] = None):
        """
        Sets the loadout of a bot.
        Can be used to select or generate a loadout for the match when called inside `initialize`.
        Does nothing if called outside `initialize` unless state setting is enabled in which case it
        respawns the car with the new loadout.
        """
        self._game_interface.send_set_loadout(
            flat.SetLoadout(index or self.index, loadout)
        )

    def initialize(self):
        """
        Called when the bot is ready for initialization. Field info, match settings, name, index, and team are
        fully loaded at this point, and will not return garbage data unlike in `__init__`.
        """

    def retire(self):
        """Called when the bot is shut down"""

    def handle_reward(self, reward: RewardType):
        """
        Called if reward_function is not None each time a step of the simulated RLGym environment happens
        """
        pass

    def get_other_packet_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        """
        Called when the packet match phase is NOT one of Countdown, Kickoff, or Active. Useful for cosmetic things, like dancing after winning a match or after a goal is scored.
        """
        return flat.ControllerState()

    @abstractmethod
    def get_action(
        self, obs: ObsType, game_state: GameState, packet: flat.GamePacket
    ) -> Union[ActionType, List[flat.ControllerState]]:
        """
        Called to get the next ActionType to use with the action parser, or to get a list of ControllerStates if hard-coding inputs.
        """
        raise NotImplementedError
