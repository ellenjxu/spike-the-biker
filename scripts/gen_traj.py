import numpy as np
import pygame
import sys
from trajectory_generation.floor import Floorplan
from shapely.geometry import Point
from simple_pid import PID
import random

DT = 1/10.0
WHEELBASE = 0.406
ROBOT_WIDTH = 0.15
RENDER_SCALE = 40 # 50px/m

MAX_X, MAX_Y = 28.0, 28.0 # m
LOOKAHED = 1 # m
HUMAN = False
DRAW = True

NB_STEPS = 200_000
EP_LEN = 100
NB_EP = NB_STEPS // EP_LEN
MAX_SPAWN_DIST = 1.0

OUTPUT = "../data/trajectories.txt"


# GitHub: https://github.com/botprof/agv-examples
def rk_four(f, x, u, T):
    """Fourth-order Runge-Kutta numerical integration."""
    k_1 = f(x, u)
    k_2 = f(x + T * k_1 / 2.0, u)
    k_3 = f(x + T * k_2 / 2.0, u)
    k_4 = f(x + T * k_3, u)
    x_new = x + T / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)
    return x_new


def diffdrive_f(x, u):
    """Differential drive kinematic vehicle model."""
    f = np.zeros(3)
    f[0] = 0.5 * (u[0] + u[1]) * np.cos(x[2])
    f[1] = 0.5 * (u[0] + u[1]) * np.sin(x[2])
    f[2] = 1.0 / WHEELBASE * (u[1] - u[0])
    return f


def uni2diff(u):
    """Convert speed and angular rate to wheel speeds."""
    v = u[0]
    omega = u[1]
    v_L = v - WHEELBASE / 2 * omega
    v_R = v + WHEELBASE / 2 * omega
    return np.array([v_L, v_R])


def get_vertices(pos_x, pos_y, theta, width, wheelbase):
    """Calculate the car's rectangle vertices based on its position, orientation and body
    size."""

    half_width = width / 2
    half_wheelbase = wheelbase / 2

    # Define the car's rectangle vertices in the local coordinate system
    local_vertices = [
        (-half_wheelbase, -half_width),
        (-half_wheelbase, half_width),
        (half_wheelbase, half_width),
        (half_wheelbase, -half_width),
    ]

    # Rotate and translate the vertices to the global coordinate system
    cos_yaw = np.cos(theta)
    sin_yaw = np.sin(theta)

    global_vertices = [
        (pos_x + cos_yaw * vx - sin_yaw * vy, pos_y + sin_yaw * vx + cos_yaw * vy)
        for vx, vy in local_vertices
    ]
    return global_vertices


def random_robot_spawn(floorplan):
    # print("spawning robot")
    while True:
        # generate x and y values between 0 and MAX_X, MAX_Y
        x = np.random.uniform(0, MAX_X)
        y = np.random.uniform(0, MAX_Y)
        if floorplan.is_inside(x, y) and floorplan.distance_to_centerline(x, y) < MAX_SPAWN_DIST:
            break

    if x < 6:
        theta = np.pi/2
    elif y < 15:
        theta = np.pi
    else:
        theta = 0
    # generate random heading between -pi/2 and pi/2
    # theta = 0 # np.random.uniform(-np.pi/2, np.pi/2)
    
    # print("robot spawned")
    return np.array([x, y, theta])


def random_robot_spawn_2(floorplan):
    # print("spawning robot")
    # get the centerline length
    centerline_len = floorplan.centerline.length
    
    # generate a value between 0 and centerline_len
    r = random.random()
    while True:
        if r < 0.15:
            dist = 5
        else:
            dist = np.random.uniform(0, centerline_len)
        # get the x and y value of the point along the centerline at dist
        point = floorplan.centerline.interpolate(dist)
        x, y = point.x, point.y
        if floorplan.is_inside(x, y):
            break

    while True:
        dx = np.random.uniform(-MAX_SPAWN_DIST/2, MAX_SPAWN_DIST/2)
        dy = np.random.uniform(-MAX_SPAWN_DIST/2, MAX_SPAWN_DIST/2)
        
        # add them to x and y
        new_x = x + dx
        new_y = y + dy

        # checkif inside the floorplan
        if floorplan.is_inside(new_x, new_y):
            x = new_x
            y = new_y
            break

    if x < 6:
        theta = np.pi/2
    elif y < 15:
        theta = np.pi
    else:
        theta = 0

    # add a lil bit of noise
    theta_noise = np.deg2rad(50) # deg
    dtheta = np.random.uniform(-theta_noise, theta_noise)
    theta += dtheta
    
    # print("robot spawned")
    return np.array([x, y, theta])


if __name__ == "__main__":
    # Constants
    WIDTH, HEIGHT = 1400, 1400

    # Initialize Pygame
    pygame.init()
    clock = pygame.time.Clock()
    if DRAW:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Differential Drive Robot")

    # Initialize model
    # X = np.array([0, 0, 0])  # x, y, theta
    DESIRED_SPEED = 0.5
    MAX_ANGULAR_SPEED = 1/2.

    floorplan = Floorplan(floorplan_fpath="../data/floorplan.dxf", centerline_fpath="../data/centerline_rounded_v2.dxf")

    X = random_robot_spawn_2(floorplan)

    steer_pid = PID(8, 0.2, 14, setpoint=0)

    counter_ep = 0
    trajectories = []

    counter = 0
    trajectory = []

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        distance_error = floorplan.distance_to_centerline(X[0], X[1])
        p = Point(X[0], X[1])
        is_inside_centerline = floorplan.centerline_polygon.contains(p)

        if HUMAN:
            keys = pygame.key.get_pressed()
            angular_action = 0
            if keys[pygame.K_a]:
                angular_action = 1

            if keys[pygame.K_d]:
                angular_action = -1
        else:
            # get distance to centerline
            angular_action = steer_pid(distance_error, dt=DT)
            angular_action = np.clip(angular_action, a_min=-1, a_max=1)
            angular_action = -angular_action if is_inside_centerline else angular_action

        # print(f"distance error: {distance_error}, action: {angular_action}")

        # Exec action
        U = np.array([DESIRED_SPEED, angular_action * MAX_ANGULAR_SPEED])

        # save before executing action, ds is image -> desired action
        trajectory.append([X[0], X[1], X[2], angular_action, False]) # x, y, theta, steer, end_of_seq

        X = rk_four(diffdrive_f, X, uni2diff(U), DT)
        counter += 1

        # reset
        if not floorplan.is_inside(X[0], X[1]) or (counter>=EP_LEN):
            if counter >= EP_LEN:
                trajectory[-1][-1] = True
                trajectories.extend(trajectory)
                trajectory = []
                # save trajectory
                counter_ep +=1
                print(f"{counter_ep}/{NB_EP}")
            
            if counter_ep == NB_EP:
                print("saving trajectories")
                trajectories = np.array(trajectories, dtype=np.float32)
                print(trajectories.shape)
                np.savetxt(OUTPUT, trajectories)
                break
            X = random_robot_spawn_2(floorplan)
            counter = 0
            steer_pid.reset()

        #########################################
        # Draw Robot
        if DRAW:
            x, y, theta  = X
            surf = pygame.Surface((WIDTH, HEIGHT))

            # Draw a rectangle representing the robot
            rect_color = (255, 0, 0)
            rect_size = (int(ROBOT_WIDTH * RENDER_SCALE), int(WHEELBASE * RENDER_SCALE))

            global_vertices = get_vertices(x, y, theta, WHEELBASE, ROBOT_WIDTH)
            global_vertices = [
                (x * RENDER_SCALE, y * RENDER_SCALE) for x, y in global_vertices
            ]
            pygame.draw.polygon(surf, (255, 255, 255), global_vertices)
            
            # draw the floorplan and the line
            for line in [floorplan.centerline, floorplan.floor_outer]:
                # Transform track points to screen points and scale
                points = [(RENDER_SCALE * p[0], RENDER_SCALE * p[1]) for p in line.coords]

                # Draw lines with pygame
                pygame.draw.lines(surf, (255, 255, 255), False, points, 2)

            screen.blit(surf, (0, 0))

            pygame.event.pump()
            clock.tick(int(60))
            pygame.display.flip()
            screen.fill((0, 0, 0))