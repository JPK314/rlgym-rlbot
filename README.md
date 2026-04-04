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
In the above example, I am using a custom obs builder (implementation not shown) and a standard action parser using imports from the RLGym[rl] Python package. Where you put your configuration objects' implementations so that they can be used in the above snippet is up to you. The other missing piece is the implementation of the Actor - this is the piece that takes something of type ObsType and converts it to something of type ActionType. Typically this is something defined in your learning framework of choice that converts a batch of ObsType to a Torch tensor, runs it through your model, and then converts the result to a batch of ActionType. You will need to know what you're using and how to rip it out of the learning framework you're using to use it here instead.

The `RLGymBotConfig` class contains a couple useful knobs which are described in the following section.

Other mandatory configuration files that need to go in your bot's folder are described [here](https://wiki.rlbot.org/v5/botmaking/config-files/) on the RLBot v5 wiki.

## Config Options
The `RLGymBotConfig` class contains the following options:

### above_240_fps_mode
Boolean - default `false`. This option being set to `true` means the bot will play best when the frame rate is 240fps or higher. The option being set to `false` means you should lock your fps to 120 to get the best performance from your bot. The reason for this parameter's existence is due to how controller info being received for use in Rocket League is tied to frame rate. In above 240 fps mode, the action the wrapper sends to the RLBot server (and from there to Rocket League) will come out for the state transition immediately following the most recently received game packet. In 120 fps mode, the action will come out one tick after instead. The wrapper handles this internally in order to make sure your bot plays as expected in either case.

### action_step_idx_used_to_build_game_state_for_env_step
Integer - default `-2`. This option determines when during the course of executing the last action a game state is taken to be used for simulating a step of the env (most importantly, for building your observation). Intuitively, a value of `-1` means you use the state after the last action from the previous sequence has been executed in order to determine the action sequence that will start next. A value of `-2` means you use the state one before the last action from the previous sequence has been executed. A value of `1` means you use the first state after the previous action sequence began, etc. The common values you will want to use are the above 3, but you can use any number. See below for specific notes on how this works.
- Use `-2` if you use the [RocketSimEngine transition engine](https://github.com/lucas-emery/rocket-league-gym/blob/main/rlgym/rocket_league/sim/rocketsim_engine.py) with `rlbot_delay=True`
- Use `-1` if you use the above with `rlbot_delay=False` (note that this requires `above_240_fps_mode=True`)
- Use `1` if you are coming from RLGym v1

Here is a helpful diagram explaining how these values fully work. For simplicity, let's say every action (by this I mean result from the action parser) has a shape of (4,8) i.e. it defines 4 ticks worth of actions sent to Rocket League. `.` represents a regular state, and `-` represents a state transition.

A value of 0 would look like the below - in particular, the (n+1)th action is decided using the state that the nth action is about to start from:
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

A value of 1 would look like the below - in particular, an action is decided using the state one tick after the last action's env action ticks began:
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

A value of -1 would look like the below - in particular, the next action action is decided using the state after the last action's env action ticks are done:
```
.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
|| | | ||| | ||| | | ||└Action 3 tick 4, etc
|| | | ||| | ||| | | |└Obs 4 created using this state, action 4 decided using obs 4
|| | | ||| | ||| | | └Action 3 tick 3
|| | | ||| | ||| | └Action 3 tick 2
|| | | ||| | ||| └Action 3 tick 1
|| | | ||| | ||└Action 2 tick 4
|| | | ||| | |└Obs 3 created using this state, action 3 decided using obs 3
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

A value of -2 would look like the below - in particular, an action is decided using the state one tick before the last action's env action ticks are done:
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

### missed_action_recovery_style
Enum - default RESET. This enum describes how you want the wrapper to recover when it detects that the action your bot intended to take on a given tick was not taken on that tick. This can happen due to fps drops, server lag, slow inference time, etc. Options are either RESET (simulate a reset of the env for your config objects and start a new action beginning as soon as possible) or IGNORE (pretend nothing went wrong). From some testing of a couple bots, RESET seems marginally better, but it's very marginal. You can play around with this to see what plays best, but don't expect any substantial differences either way.

### missed_step_tick_recovery_style
Enum - default RESET. This enum describes how you want the wrapper to recover when it detects that we did not receive a game packet from the server for the tick that should have been used to simulate a step of the env (according to your `action_step_idx_used_to_build_game_state_for_env_step` value). Options are either RESET (simulate a reset of the env for your config objects and start a new action beginning as soon as possible) or USE_NEAREST (use the packet closest in time that we did receive to generate a game state to use for simulating an env step). From some testing of a couple bots, RESET seems marginally better, but it's very marginal. You can play around with this to see what plays best, but don't expect any substantial differences either way.

### standard_map
Boolean - default `true`. This option is passed to [rlgym-compat](https://github.com/JPK314/rlgym-compat)'s GameState in order to configure boost pad ordering.

### sim_extra_info
Boolean - default `false`. This option determines whether rlgym-compat's [SimExtraInfo](https://github.com/JPK314/rlgym-compat/blob/main/rlgym_compat/sim_extra_info.py) is used for extra information that otherwise cannot be derived from the RLBot GamePacket.


## What does "accurate" mean? What is this wrapper doing?
This wrapper attempts to replicate the training conditions of your bot in RLGym as accurately as possible. This means a lot of consideration is put into tick-perfectly matching everything where possible.

### Tick perfectly matching actions
As mentioned in the config option section for `above_240_fps_mode`, due to how RLBot and Rocket League interact, there can be a one tick delay between when an action is submitted to the game and when it is used in the game. This means that we need to track carefully when an action is submitted vs when it is used. Below I describe the thought process behind perfectly tick matching when `above_240_fps_mode=False` (i.e. there is always a one tick delay).

#### 0 case
In RLGym, the reset state is the state directly preceding the tick that action 1 tick 1 is used. However, we can't generate the obs for this and submit it at the correct time, so we use the state during the countdown before the proper reset state as a stand-in since the cars aren't taking any actions anyway (the cars are just settling on the ground)
```
c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
| |||||||||||||||||||||||└Action 3 tick 4 used, etc
| ||||||||||||||||||||||└Action 4 tick 1 submitted
| |||||||||||||||||||||└Action 3 tick 3 used
| ||||||||||||||||||||└Action 3 tick 4 submitted
| |||||||||||||||||||└Action 3 tick 2 used
| ||||||||||||||||||└Action 3 tick 3 submitted
| |||||||||||||||||└Action 3 tick 1 used
| ||||||||||||||||└Obs 4 created using this state, action 4 decided using obs 4, action 3 tick 2 submitted
| |||||||||||||||└Action 2 tick 4 used
| ||||||||||||||└Action 3 tick 1 submitted
| |||||||||||||└Action 2 tick 3 used
| ||||||||||||└Action 2 tick 4 submitted
| |||||||||||└Action 2 tick 2 used
| ||||||||||└Action 2 tick 3 submitted
| |||||||||└Action 2 tick 1 used
| ||||||||└Obs 3 created using this state, action 3 decided using obs 3, action 2 tick 2 submitted
| |||||||└Action 1 tick 4 used
| ||||||└Action 2 tick 1 submitted
| |||||└Action 1 tick 3 used
| ||||└Action 1 tick 4 submitted
| |||└Action 1 tick 2 used
| ||└Action 1 tick 3 submitted
| |└Action 1 tick 1 used
| └Action 1 tick 2 submitted
└Obs 1 and 2 created using this state, actions 1 and 2 decided using obs 1 and 2 respectively. Action 1 tick 1 submitted
```

#### 1 case
Same issue as above.
```
c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
| |||||||||||||||||||||||└Action 3 tick 4 used, etc
| ||||||||||||||||||||||└Action 4 tick 1 submitted
| |||||||||||||||||||||└Action 3 tick 3 used
| ||||||||||||||||||||└Action 3 tick 4 submitted
| |||||||||||||||||||└Action 3 tick 2 used
| ||||||||||||||||||└Obs 4 created using this state, action 4 decided using obs 4, action 3 tick 3 submitted
| |||||||||||||||||└Action 3 tick 1 used
| ||||||||||||||||└Action 3 tick 2 submitted
| |||||||||||||||└Action 2 tick 4 used
| ||||||||||||||└Action 3 tick 1 submitted
| |||||||||||||└Action 2 tick 3 used
| ||||||||||||└Action 2 tick 4 submitted
| |||||||||||└Action 2 tick 2 used
| ||||||||||└Obs 3 created using this state, action 3 decided using obs 3, action 2 tick 3 submitted
| |||||||||└Action 2 tick 1 used
| ||||||||└Action 2 tick 2 submitted
| |||||||└Action 1 tick 4 used
| ||||||└Action 2 tick 1 submitted
| |||||└Action 1 tick 3 used
| ||||└Action 1 tick 4 submitted
| |||└Action 1 tick 2 used
| ||└Obs 2 created using this state, action 2 decided using obs 2, action 1 tick 3 submitted
| |└Action 1 tick 1 used
| └Action 1 tick 2 submitted
└Obs 1 created using this state, action 1 decided using obs 1. Action 1 tick 1 submitted
```

#### -1 case
Impossible to replicate with a 1 tick delay.

#### -2 case
Here the void state transition actually works in our favor. We do the same thing as in the previous cases using the countdown state before the last countdown state to determine the action, but here the void state transition is equivalent to waiting one tick anyway, so this is tick perfect.
```
c-c-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
| |||||||||||||||||||||||└Action 3 tick 4 used, etc
| ||||||||||||||||||||||└Obs 4 created using this state, action 4 decided using obs 4, action 4 tick 1 submitted
| |||||||||||||||||||||└Action 3 tick 3 used
| ||||||||||||||||||||└Action 3 tick 4 submitted
| |||||||||||||||||||└Action 3 tick 2 used
| ||||||||||||||||||└Action 3 tick 3 submitted
| |||||||||||||||||└Action 3 tick 1 used
| ||||||||||||||||└Action 3 tick 2 submitted
| |||||||||||||||└Action 2 tick 4 used
| ||||||||||||||└Obs 3 created using this state, action 3 decided using obs 3, action 3 tick 1 submitted
| |||||||||||||└Action 2 tick 3 used
| ||||||||||||└Action 2 tick 4 submitted
| |||||||||||└Action 2 tick 2 used
| ||||||||||└Action 2 tick 3 submitted
| |||||||||└Action 2 tick 1 used
| ||||||||└Action 2 tick 2 submitted
| |||||||└Action 1 tick 4 used
| ||||||└Obs 2 created using this state, action 2 decided using obs 2, action 2 tick 1 submitted
| |||||└Action 1 tick 3 used
| ||||└Action 1 tick 4 submitted
| |||└Action 1 tick 2 used
| ||└Action 1 tick 3 submitted
| |└Action 1 tick 1 used
| └Action 1 tick 2 submitted
└Obs 1 created using this state, action 1 decided using obs 1. Action 1 tick 1 submitted
```

### Countdown and Kickoffs
There is a tricky edge case during the countdown and on kickoff. As explained in the previous section, when `above_240_fps_mode=False`, we need to start submitting actions on the second to last tick of countdown. Unfortunately, unless you're a time traveller, there's no way to know when you are receiving a packet for the second to last tick of countdown. If we are not on the first kickoff of a match, we can guess at this by assuming the countdown match phase lasts for 480 ticks (4 seconds). This is true about 2/3 of the time in my testing - in the other 1/3 of the time, it lasted for 479 ticks. Based on the values of `above_240_fps_mode` and `action_step_idx_used_to_build_game_state_for_env_step`, the wrapper determines the best tick to perform an env reset on and the best tick to start that action on, under the assumption that the countdown match phase lasts for 480 ticks. Prior to this tick, the wrapper uses the latest game state to simulate a reset of the env and submits the action resulting from this observation so that if a bunch of packets get missed, at least something will be available for the game to start using immediately. There is logic to detect if the countdown phase only lasted 479 ticks and to move up the remaining queued actions by one tick if that happens. On the first countdown of the match, every countdown packet is assumed to be the last tick of countdown and a similar process follows as described above.