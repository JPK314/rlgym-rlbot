from rlbot.flat import (
    DesiredBallState,
    DesiredCarState,
    DesiredGameState,
    DesiredMatchInfo,
    DesiredPhysics,
    GamePacket,
    RotatorPartial,
    Vector3Partial,
)
from rlgym_compat import Car, GameState, PhysicsObject


def physics_rlgym_to_rlbot(physics: PhysicsObject) -> DesiredPhysics:
    return DesiredPhysics(
        location=Vector3Partial(*physics.position),
        rotation=RotatorPartial(*physics.euler_angles),
        velocity=Vector3Partial(*physics.linear_velocity),
        angular_velocity=Vector3Partial(*physics.angular_velocity),
    )


def ball_rlgym_to_rlbot(ball: PhysicsObject) -> DesiredBallState:
    return DesiredBallState(physics=physics_rlgym_to_rlbot(ball))


def car_rlgym_to_rlbot(car: Car) -> DesiredCarState:
    return DesiredCarState(
        physics=physics_rlgym_to_rlbot(car.physics), boost_amount=car.boost_amount * 100
    )


def gamestate_rlgym_to_rlbot(
    game_state: GameState, cur_packet: GamePacket
) -> DesiredGameState:
    assert len(game_state.cars) == len(
        cur_packet.players
    ), "conversion from GameState not possible because it would require changing the number of players in the match"
    assert 1 == len(
        cur_packet.balls
    ), "conversion from GameState not possible because it would require changing the number of balls in the match"

    return DesiredGameState(
        [ball_rlgym_to_rlbot(game_state.ball)],
        [car_rlgym_to_rlbot(game_state.cars[p.spawn_id]) for p in cur_packet.players],
        DesiredMatchInfo(world_gravity_z=game_state.config.gravity * -650),
    )
