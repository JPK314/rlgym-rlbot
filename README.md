# RLBot v5 - RLGym v2 Wrapper

A high accuracy wrapper interface allowing bots trained with [RLGym v2](https://rlgym.org/) using the [RocketSimEngine transition engine](https://github.com/lucas-emery/rocket-league-gym/blob/main/rlgym/rocket_league/sim/rocketsim_engine.py) to play in [RLBot v5](https://wiki.rlbot.org/v5/) with minimal configuration required by the user.

## Making a bot with this wrapper

Below is a simple snippet from what the main file in the folder of a bot using this wrapper might look like:

```py
from rlgym_rlbot import RLGymBot, RLGymBotConfig

class MyBot(RLGymBot):
    def __init__(self):
        super().__init__(
            obs_builder=MyObsBuilder(),
            action_parser=RepeatAction(LookupTableAction(), 8),
            default_agent_id="jpk314/mybot/v0.1",
            config=RLGymBotConfig(above_240_fps_mode=True),
        )
        self.agent = Agent()

    def get_action(self, obs, game_state, packet):
        return self.agent.act(obs)


if __name__ == "__main__":
    MyBot().run()
```
In the above example, I am using a custom obs builder (implementation not shown) and a standard action parser using imports from the RLGym[rl] Python package. Where you put your configuration objects' implementations so that they can be used in the above snippet is up to you. The other missing definition is the Agent class - this is an abstraction you might have in your learning framework that takes something of type ObsType and converts it to something of type ActionType. Typically this is something that converts a batch of ObsType to a Torch tensor, runs it through your model, and then converts the result to a batch of ActionType. You will need to know what you're using and how to rip it out of the learning framework you're using to use it here instead.

The `RLGymBotConfig` class contains a couple useful knobs which are described in the following section.

Other mandatory configuration files that need to go in your bot's folder are described [here](https://wiki.rlbot.org/v5/botmaking/config-files/) on the RLBot v5 wiki.

### Optional Overridable Methods

#### Mixing with hard-coded actions or action sequences
The method `get_hardcoded_action` allows you to override whatever your model was planning to do and instead do whatever `ControllerState` you return from this method. The default return value is `None` which means the model will continue with whatever it was doing. This is most valuable for hard coding kickoff maneuvers. You can manage the maneuver state in your subclass of RLGymBot. If you ever return a non-`None` value and then return `None` again, the wrapper will simulate an on-the-fly reset of the environment for your config objects so that the model can resume playing.

#### Sending messages
Sending messages to other bots or in the chat can be done during the `get_hardcoded_action` call (which happens every tick during gameplay) or during the `get_other_packet_output` call (which happens the rest of the time) by calling `self.send_match_comm`.

#### Recovering from one or more incorrectly taken actions
The method `decide_missed_action_recovery_style` allows for custom logic to decide which recovery style to use - either `MissedActionRecoveryStyle.RESET` or `MissedActionRecoveryStyle.IGNORE`. The inputs to this method provide a bunch of useful values that you may want in order to determine how to handle this, including
- The current packet and game state
- The number of ticks left in the currently in use action (remember that the output of your action parser is an action which corresponds to a sequence of ControllerStates, one per tick)
- The past ticks where an incorrect `ControllerState` was used and the `ControllerState` that was intended based on the output of your action parser

The default return value is `IGNORE`. There are situations where you may want to perform an on-the-fly reset to generate a new action from the current state instead. You probably do NOT want to perform an on-the-fly reset if you're currently in the middle of a flip, for example, because it's not possible with the RocketSimEngine to start an episode in the middle of a flip.

#### Recovering from a missed tick needed for stepping the environment
The method `decide_missed_step_tick_recovery_style` allows for custom logic to decide which recovery style to use - either `MissedStepTickRecoveryStyle.RESET` or `MissedStepTickRecoveryStyle.USE_NEAREST`. The inputs to this method provide a bunch of useful values that you may want in order to determine how to handle this, including
- The current packet and game state
- The number of ticks left in the currently in use action (remember that the output of your action parser is an action which corresponds to a sequence of ControllerStates, one per tick)
- The last `ControllerState` used by the bot

The default return value is `USE_NEAREST`. Unless your config objects are really sensitive, this is probably good enough for all situations. However, if you come up with a scenario where you think it would be preferable to perform an on-the-fly reset instead of using a slightly inaccurate (or temporally shifted, whatever way you want to think about it) game state for your config objects, you can return `RESET` in that scenario instead.

#### Handling cosmetic actions (kickoff wiggle)
The method `get_other_packet_output` can be used to return `ControllerState`s during times where your actions don't actually matter, in particularly during the Countdown match phase. For the first 440 ticks of countdown, you can do whatever wheel wiggling you desire by returning controller states here. Why 440? Countdown always lasts at least 479 ticks from my testing, and so this is just a safe margin.

#### Handling reward function output
The method `handle_reward` is called if you pass in a `RewardFunction` to the `RLGymBot` constructor on every step. You can decide what to do with this (display it on screen? write it to a file?) by overwriting this method. By default it is a no-op.

#### Various standard Bot python interface methods and fields
Many convenience methods from the `Bot` class in the [python interface](https://github.com/RLBot/python-interface) are available here as well. These include:
- `initialize`
- `retire`
- `set_loadout`
- `handle_match_comm`
- `send_match_comm`
- `set_game_state`

There are convenience fields as well:
- `self.player_id`
- `self.name`
- `self.index`
- `self.team`
- `self.match_config`
- `self.field_info`
- `self.ball_prediction`
- `self.logger`
- `self.renderer`

## Config Options
The `RLGymBotConfig` class contains the following options:

### step_offset
StepOffset dataclass with fields `offset` and `relative_to` - default `offset=-1` and `relative_to=StepOffsetRelativeTo.ACTION_END`. This option determines when during the course of executing the last action a game state is taken to be used for simulating a step of the env (most importantly, for building your observation). See below for specific notes on how this works.
- Use `offset=-1` and `relative_to=StepOffsetRelativeTo.ACTION_END` if you use the [RocketSimEngine transition engine](https://github.com/lucas-emery/rocket-league-gym/blob/main/rlgym/rocket_league/sim/rocketsim_engine.py) with `rlbot_delay=True`
- Use `offset=0` and `relative_to=StepOffsetRelativeTo.ACTION_END` if you use the above with `rlbot_delay=False`
- Use `offset=1` and `relative_to=StepOffsetRelativeTo.ACTION_START` if you are using RLGym v1 or rlgym-sim

Here is a helpful diagram explaining how these values fully work. For simplicity, let's say every action (by this I mean result from the action parser) has a shape of (4,8) i.e. it defines 4 ticks worth of actions sent to Rocket League. `.` represents a regular state, and `-` represents a state transition.

Values `offset=0` and `relative_to=StepOffsetRelativeTo.ACTION_START` would look like the below - in particular, the (n+1)th action is decided using the state that the nth action is about to start from:
```
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
|| | | ||| | | ||| | | └Action 3 tick 4, etc
|| | | ||| | | ||| | └Action 3 tick 3
|| | | ||| | | ||| └Action 3 tick 2
|| | | ||| | | ||└Action 3 tick 1
|| | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
|| | | ||| | | └Action 2 tick 4
|| | | ||| | └Action 2 tick 3
|| | | ||| └Action 2 tick 2
|| | | ||└Action 2 tick 1
|| | | |└Obs 3 created using this state, action 3 decided using obs 3
|| | | └Action 1 tick 4
|| | └Action 1 tick 3
|| └Action 1 tick 2
|└Action 1 tick 1
└Reset occurs, obs 1 and 2 created using this state, actions 1 and 2 decided using obs 1 and 2 respectively
```

Values `offset=1` and `relative_to=StepOffsetRelativeTo.ACTION_START` would look like the below - in particular, an action is decided using the state one tick after the last action's env action ticks began:
```
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
|||| | | ||| | | ||| | └Action 3 tick 4, etc
|||| | | ||| | | ||| └Action 3 tick 3
|||| | | ||| | | ||└Action 3 tick 2
|||| | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
|||| | | ||| | | └Action 3 tick 1
|||| | | ||| | └Action 2 tick 4
|||| | | ||| └Action 2 tick 3
|||| | | ||└Action 2 tick 2
|||| | | |└Obs 3 created using this state, action 3 decided using obs 3
|||| | | └Action 2 tick 1
|||| | └Action 1 tick 4
|||| └Action 1 tick 3
|||└Action 1 tick 2
||└Obs 2 created using this state, action 2 decided using obs 2
|└Action 1 tick 1
└Reset occurs, obs 1 created using this state, action 1 decided using obs 1
```

Values `offset=0` and `relative_to=StepOffsetRelativeTo.ACTION_END` would look like the below - in particular, the next action action is decided using the state after the last action's env action ticks are done:
```
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
|| | | ||| | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4, etc
|| | | ||| | | ||| | | └Action 3 tick 4
|| | | ||| | | ||| | └Action 3 tick 3
|| | | ||| | | ||| └Action 3 tick 2
|| | | ||| | | ||└Action 3 tick 1 
|| | | ||| | | |└Obs 3 created using this state, action 3 decided using obs 3
|| | | ||| | | └Action 2 tick 4
|| | | ||| | └Action 2 tick 3
|| | | ||| └Action 2 tick 2
|| | | ||└Action 2 tick 1
|| | | |└Obs 2 created using this state, action 2 decided using obs 2
|| | | └Action 1 tick 4
|| | └Action 1 tick 3
|| └Action 1 tick 2
|└Action 1 tick 1
└Reset occurs, obs 1 created using this state, action 1 decided using obs 1
```

Values `offset=-1` and `relative_to=StepOffsetRelativeTo.ACTION_END` would look like the below - in particular, an action is decided using the state one tick before the last action's env action ticks are done:
```
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
|| | | ||| | | ||| | | ||└Action 3 tick 4, etc
|| | | ||| | | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
|| | | ||| | | ||| | | └Action 3 tick 3
|| | | ||| | | ||| | └Action 3 tick 2
|| | | ||| | | ||| └Action 3 tick 1
|| | | ||| | | ||└Action 2 tick 4
|| | | ||| | | |└Obs 3 created using this state, action 3 decided using obs 3
|| | | ||| | | └Action 2 tick 3
|| | | ||| | └Action 2 tick 2
|| | | ||| └Action 2 tick 1
|| | | ||└Action 1 tick 4
|| | | |└Obs 2 created using this state, action 2 decided using obs 2
|| | | └Action 1 tick 3
|| | └Action 1 tick 2
|| └Action 1 tick 1
|└"void" state transition where sim is stepped without any cars' controls being set
└Reset occurs, obs 1 created using this state, action 1 decided using obs 1
```
### standard_map
Boolean - default `true`. This option is passed to [rlgym-compat](https://github.com/JPK314/rlgym-compat)'s GameState in order to configure boost pad ordering.

### sim_extra_info
Boolean - default `false`. This option determines whether rlgym-compat's [SimExtraInfo](https://github.com/JPK314/rlgym-compat/blob/main/rlgym_compat/sim_extra_info.py) is used for extra information that otherwise cannot be derived from the RLBot GamePacket.

### agent_ids_fn
Optional Callable - default `None`. This option determines how rlgym-compat assigns `AgentID`s to players when creating the compat GameState from the RLBot v5's `flat.GamePacket`. The input is a `GamePacket` and the output is a map from `PlayerInfo.player_id` to your desired `AgentID` data type and value for your config objects. Normally this would be handled by your `StateMutator`, but that functionality is not implemented yet and this acts as a simple override without the other potential `StateMutator` behavior.


## What does "accurate" mean? What is this wrapper doing?
This wrapper attempts to replicate the training conditions of your bot in RLGym as accurately as possible. This means a lot of consideration is put into tick-perfectly matching everything where possible while maintaining the flow and functionality expected by arbitrary config objects and by proxy your model during training using RLGym v2.

### Tick perfectly matching actions
This wrapper assumes that your computer is running Rocket League in such a way that you have enough time for your actions to always get submitted for use between the latest received packet's physics frame and the next physics frame. The optimal conditions for this are when you play using a capped frame rate that is either 120 or 240, because these frame rates have the largest windows between when the wrapper receives a packet from RLBot Core and when you can no longer submit an action and have it be used during the state transition between the latest received packet's physics tick and the next physics tick. This window is about 6 milliseconds on average with a small standard deviation when the framerate is capped to 120 or 240 (NOT 360!), and is about 4 milliseconds on average with a much wider standard deviation when left uncapped. Other frame rates do very poorly and are not recommended. Possibly if you modify the command line argument RLBot passes for packet send rate from 240 to another number you may be able to get up to a 6 millisecond window using another frame rate, but it's untested and annoying to configure. Just use 120 or 240 capped, or hope for the best uncapped.

### On-the-fly resets and countdown actions
This wrapper tries to tick perfectly match how your config objects and model interact with each other and the sim in RLGym v2. However, there are some situations where this is just not possible. One such situation is when an on-the-fly reset is performed. These can happen if your `decide_*_recovery_style` methods return `RESET` or if you stop taking a hard-coded action. Due to an implementation detail, in RLGym v2's RocketSimEngine with the default constructor parameter (`rlbot_delay=True`), after resetting, all agents will do one tick of an all-zeros action before beginning the action generated due to the reset state's observation. In the language of this wrapper, this means that after resetting, if you have `offset < 0` and `relative_to = StepOffsetRelativeTo.ACTION_END`, in theory you should take `offset` ticks of the zero action in order to perfectly match how everything works during training. This is obviously not desirable against an opponent with no such restrictions, and so instead the observation is generated using the game state from `offset` ticks ago, and the fact that a non-all-zeros action was taken for the ticks since that point until this new action is due to start (this tick) is assumed to come out in the wash.

Due to the duration of the Countdown phase being unreliable, it is impossible to perfectly time the reset so that you start the action as soon as Countdown ends (your actions start getting used on the state transition between the last packet of Countdown and the first packet of Kickoff). During the Countdown match phase, the cars settle onto the ground and cannot move (wheel wiggling is purely cosmetic), so the difference between using the proper tick of Countdown and some older tick of countdown is essentially zero. This wrapper uses the 440th (or next available after) tick after the Countdown phase began as the tick on which to generate the observation that will be used once Countdown ends, and simply sends the first ControllerState of this action every tick from then on until the latest packet is Kickoff instead of Countdown. This should work essentially perfectly UNLESS you intend to jump on the first tick. Because the ControllerState is sent repeatedly during Countdown, the game will think you are just holding jump and so once you can actually move you won't jump, you'll just be holding jump. So... don't do that!
