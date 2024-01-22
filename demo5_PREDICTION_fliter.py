import carla
import time
import cv2
import numpy as np
import pygame
from collections import deque
import random
from pyrsistent import v
import torch
from scipy.spatial import cKDTree
import pickle
import torch
from torch.utils.data import Sampler, DataLoader
from utils import Logger, load_pretrain, gpu
from Net_yaw import Net
import math
import pandas as pd 


def get_traffic_light(vehicle_location):
    # 检查交通灯状态
    traffic_flag, traffic_state = 0, -1

    traffic_light = world.get_traffic_lights_from_waypoint(world.get_map().get_waypoint(vehicle_location), 20)
    if len(traffic_light) > 0:
        traffic_flag = 1
        traffic = traffic_light[0].get_state()
        if traffic == carla.TrafficLightState.Red:
            traffic_state = 0
        elif traffic == carla.TrafficLightState.Yellow:
            traffic_state = 1
        elif traffic == carla.TrafficLightState.Green:
            traffic_state = 2
    else:
        traffic_state = -1
    return traffic_flag, traffic_state 


def update_vehicle_trajectories(target_vehicle, history_trajectory, surrounding_vehicles_queues, detection_radius=20):
    target_location = target_vehicle.get_location()
    traffic_flag, traffic_state = get_traffic_light(target_location)

    # 获取车辆的转换矩阵并提取航向角
    transform = target_vehicle.get_transform()
    yaw = transform.rotation.yaw  # 航向角以度为单位

    # 获取车辆的速度向量
    velocity = target_vehicle.get_velocity()
    # 计算车辆的水平速度（在X-Y平面上）
    speed = math.sqrt(velocity.x**2 + velocity.y**2)

    # 'x', 'y', 'frame_exists', 'yaw', 'speed', 'traffic_flag', 'traffic_state'
    history_trajectory.append((target_location.x, target_location.y, 1, yaw, speed, traffic_flag, traffic_state))

    vehicles = world.get_actors().filter('vehicle.*')
    active_vehicle_ids = []  # 存储在20米范围内的车辆ID
    for vehicle in vehicles:
        vehicle_location = vehicle.get_location()
        traffic_flag, traffic_state = get_traffic_light(vehicle_location)

        # 获取车辆的转换矩阵并提取航向角
        transform = vehicle.get_transform()
        yaw = transform.rotation.yaw  # 航向角以度为单位

        # 获取车辆的速度向量
        velocity = vehicle.get_velocity()
        # 计算车辆的水平速度（在X-Y平面上）
        speed = math.sqrt(velocity.x**2 + velocity.y**2)

        if vehicle.id != target_vehicle.id and vehicle_location.distance(target_location) < detection_radius:
            if vehicle.id not in surrounding_vehicles_queues:
                surrounding_vehicles_queues[vehicle.id] = deque(maxlen=20)

            surrounding_vehicles_queues[vehicle.id].append((vehicle_location.x, vehicle_location.y, 1, yaw, speed, traffic_flag, traffic_state))
            active_vehicle_ids.append(vehicle.id)
    
    # 删除不再活跃的车辆记录
    for vehicle_id in list(surrounding_vehicles_queues):
        if vehicle_id not in active_vehicle_ids:
            del surrounding_vehicles_queues[vehicle_id]

    return history_trajectory, surrounding_vehicles_queues

def create_trajectory_tensor(history_trajectory, surrounding_vehicles_queues):
    trajectories = []
    # 添加目标车辆的轨迹
    target_traj = list(history_trajectory)
    
    # 'x', 'y', 'frame_exists', 'yaw', 'speed', 'traffic_flag', 'traffic_state'
    target_traj += [(*target_traj[-1][:2], 0, 0, 0, -1, -1)] * (20 - len(target_traj))
    trajectories.append(target_traj)

    # 添加周围车辆的轨迹
    for vehicle_id, queue in surrounding_vehicles_queues.items():
        vehicle_traj = list(queue)
        vehicle_traj += [(*vehicle_traj[-1][:2], 0, 0, 0, -1, -1)] * (20 - len(vehicle_traj))
        trajectories.append(vehicle_traj)

    # 转换为tensor
    trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
    return trajectories_tensor


def get_nearby_waypoints(target_vehicle, waypoints_xy, radius=30.0):
    target_location = target_vehicle.get_location()
    indices = tree.query_ball_point([target_location.x, target_location.y], radius)
    nearby_waypoints = [waypoints_xy[i] for i in indices]
    nearby_waypoints_np = np.array(nearby_waypoints)
    nearby_waypoints_tensor = torch.from_numpy(nearby_waypoints_np).float()
    return nearby_waypoints_tensor


def get_topology(target_vehicle, map_dict):
    target_location = target_vehicle.get_location()
    target_coordinateinate = [target_location.x, target_location.y]
    # 获取最近的地图dict
    distances = {key: np.linalg.norm(np.array(target_coordinateinate) - np.array(key)) for key in map_dict.keys()}
    closest_coordinate = min(distances, key=distances.get)
    closest_value = map_dict[closest_coordinate]

    lane_list = [torch.tensor(df[['x', 'y']].values, dtype=torch.float32) for df in closest_value]
    lane_tensor = torch.cat(lane_list, dim=0)
    return lane_tensor

def calculate_heading(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    theta = math.atan2(delta_y, delta_x)  # 弧度
    heading = math.degrees(theta)  # 转换为度
    return heading

def calculate_trajectory_length(trajectory):
    length = 0
    for j in range(len(trajectory) - 1):
        point1 = trajectory[j]
        point2 = trajectory[j + 1]
        length += np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return length

def get_stop_line_location(yaw, target_location, world):
    map_df = generate_map_data(world)
    # 获取车辆当前所在的车道ID和道路ID
    cur_waypoints = world.get_map().get_waypoint(target_location)
    lane_id, road_id = cur_waypoints.lane_id, cur_waypoints.road_id
    road_data = map_df[(map_df['lane_id'] == lane_id) & (map_df['road_id'] == road_id)]
    
    # 获取停止线的坐标
    stop_line_data = road_data.iloc[0]  # 默认选择第一行
    
    # 计算车辆位置与停止线的夹角
    car_to_stop_line = calculate_heading(target_location.x, target_location.y, stop_line_data['x'], stop_line_data['y'])

    # 如果夹角为负，认为车辆朝着相反方向行驶，选择最后一行
    if car_to_stop_line - yaw > 90 or car_to_stop_line - yaw < -90:
        stop_line_data = road_data.iloc[-1]

    # 获取停止线的坐标
    stop_line_x = stop_line_data['x']
    stop_line_y = stop_line_data['y']

    return stop_line_x, stop_line_y

def is_point_before_stop_line(vehicle, point, stop_line_x, stop_line_y):
    # 获取车辆位置和航向角
    vehicle_location = vehicle.get_location()
    transform = vehicle.get_transform()
    vehicle_heading = transform.rotation.yaw

    # 计算轨迹点与车辆位置的相对方向
    dx = point[0] - vehicle_location.x
    dy = point[1] - vehicle_location.y
    angle_to_point = math.degrees(math.atan2(dy, dx))

    # 计算轨迹点与停止线的相对位置
    stop_line_dx = stop_line_x - vehicle_location.x
    stop_line_dy = stop_line_y - vehicle_location.y
    angle_to_stop_line = math.degrees(math.atan2(stop_line_dy, stop_line_dx))

    # 判断轨迹点是否在停止线前
    return abs(angle_to_point - vehicle_heading) < abs(angle_to_stop_line - vehicle_heading)


def is_obstacle_present(world, target_location, your_vehicle_id, your_threshold=2.0):
    # 获取场景中的所有actor
    actors = world.get_actors()

    # 检查是否存在障碍物
    for actor in actors:
        # 排除自己
        if actor.id != your_vehicle_id:
            actor_location = actor.get_location()
            distance = carla.Location.distance(actor_location, target_location)

            # 假设距离小于某个阈值
            if distance < your_threshold:
                return True

    return False

def is_point_in_drivable_area(world, location, threshold_distance=3.0):
    map = world.get_map()
    waypoint = map.get_waypoint(location)
    nearest_waypoint_location = waypoint.transform.location

    distance_to_nearest_waypoint = location.distance(nearest_waypoint_location)

    # 判断距离是否小于阈值
    return distance_to_nearest_waypoint < threshold_distance



def draw_predicted_trajectories(pred_vehicle, traj_pred, world):
    # 定义三种不同的颜色
    colors = [carla.Color(r=255, g=0, b=0),   # 红色
              carla.Color(r=0, g=255, b=0),   # 绿色
              carla.Color(r=0, g=0, b=255)]   # 蓝色

    # 检查轨迹维度是否正确
    if traj_pred.shape[0] != 3 or traj_pred.shape[2] != 2:
        raise ValueError("预测轨迹的形状应为 (3, 30, 2)")

    # 获取预测车辆的位置和朝向
    pred_location = pred_vehicle.get_location()
    pred_vehicle_yaw = pred_vehicle.get_transform().rotation.yaw  # 航向角以度为单位

    # 获取预测车辆前方红绿灯状态
    cur_traffic_state = pred_vehicle.get_traffic_light_state()
    if cur_traffic_state != carla.TrafficLightState.Green:
        stop_line_x, stop_line_y = get_stop_line_location(pred_vehicle_yaw, pred_location, world)


    # 循环绘制每条轨迹的每个点
    for i in range(1):  # 对于每条轨迹
        # 计算预测轨迹的长度
        trajectory_length = calculate_trajectory_length(traj_pred[i])
        # 计算预测轨迹的角度
        heading = calculate_heading(pred_location.x, pred_location.y, traj_pred[i][0][0], traj_pred[i][0][1])

        # 跳过长度大于1m 且 角度过大的预测轨迹
        if trajectory_length > 1 and (heading - pred_vehicle_yaw > 60 or heading - pred_vehicle_yaw < -60):
            continue


        for point in traj_pred[i]:
            # 将轨迹点从numpy数组转换为CARLA的Location对象
            location = carla.Location(x=float(point[0]), y=float(point[1]), z = pred_location.z + 1.5)
            
            # 如果是红灯，则只绘制停止线前的轨迹
            if cur_traffic_state != carla.TrafficLightState.Green and is_point_before_stop_line(pred_vehicle, point, stop_line_x, stop_line_y):
                # stop_line_location = carla.Location(x=stop_line_x, y=stop_line_y, z = pred_vehicle.get_location().z + 1.5)
                # world.debug.draw_string(location=stop_line_location, text='STOP!!!!', color=carla.Color(r=255, g=0, b=0), life_time=10)
                break

            # 检测是否存在碰撞
            if is_obstacle_present(world, location, pred_vehicle.id):
                # world.debug.draw_string(location=location, text='CRASH!!!!', color=carla.Color(r=255, g=0, b=0), life_time=10)
                break

            # 检测是否在道路上
            if not is_point_in_drivable_area(world, location):
                # world.debug.draw_string(location=location, text='Not on road', color=carla.Color(r=0, g=255, b=0), life_time=10)
                break
            
            # 绘制点
            world.debug.draw_point(location, 0.1, colors[i], 0.1)


def spawn_vehicles(world, vehicle_count=50):
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    for _ in range(vehicle_count):
        blueprint = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
    return vehicles


def spawn_target_vehicle(world):
    target_vehicle_bp = world.get_blueprint_library().filter('model3')[0]
    spawn_point = carla.Transform(carla.Location(x=40, y=0, z=3), carla.Rotation(yaw=180))
    target_vehicle = world.spawn_actor(target_vehicle_bp, spawn_point)
    target_vehicle.set_autopilot(True)
    return target_vehicle

def generate_map_data(world):
    world_map = world.get_map()
    data_list = []
    for waypoint in world_map.generate_waypoints(1.0):
        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        road_id = waypoint.road_id
        lane_id = waypoint.lane_id
        s = waypoint.s
        data_list.append([x, y, road_id, lane_id, s])
    map_df = pd.DataFrame(data_list, columns=['x', 'y', 'road_id', 'lane_id', 's'])
    return map_df


def rotate_point(distance, z, angle):
    """Rotate a 2D point counterclockwise by a given angle in degrees."""
    x = distance * math.cos(math.radians(angle)) 
    y = distance * math.sin(math.radians(angle)) 
    return carla.Location(x=x, y=y, z=z)

def draw_surrounding_points(target_vehicle, world, distance=30.0):
    # 获取目标车辆的位置和方向
    target_location = target_vehicle.get_location()

    # 绘制每隔15度的点
    for i in range(24):
        # 计算一个半径为distance的圆上的点
        rotated_location = rotate_point(distance, target_location.z, i * 15.0)
        # 计算相对于车辆的最终位置
        point_location = target_location + rotated_location
        # 绘制点
        world.debug.draw_point(point_location, size=0.3, color=carla.Color(r=0, g=0, b=255), life_time=0.1)


# 初始化Pygame和Carla
# pygame.init()
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town05_Opt')
# world.unload_map_layer(carla.MapLayer.All)

spectator = world.get_spectator()



# 加载地图
waypoints_xy = np.array(pickle.load(open('./waypoints_xy.pkl', 'rb')))
tree = cKDTree(waypoints_xy)
map_dict = pickle.load(open('./map_dict.pkl', 'rb'))

# 初始化轨迹预测网络
net = Net().cuda()
ckpt_path = './yaw_net40.000.ckpt'
ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
load_pretrain(net, ckpt["state_dict"])
net.eval()

# 生成车辆
vehicles = spawn_vehicles(world)  
target_vehicle = spawn_target_vehicle(world) # 生成目标车辆
vehicles.append(target_vehicle)

# 初始化目标车辆的历史轨迹队列和周围车辆队列
history_trajectories = {}
surrounding_vehicles_queues = {}


interval = 0.1
try:
    while True:
        loop_start_time = time.time()
        # 标记目标车辆轨迹
        # world.debug.draw_point(target_vehicle.get_location(), 0.1, carla.Color(r=255, g=0, b=0), 100)
        # draw_surrounding_points(target_vehicle, world, distance=30.0)

        for vehicle in vehicles:
            # 判断是否在预测范围内
            if vehicle.get_location().distance(target_vehicle.get_location()) > 30 or vehicle.id == target_vehicle.id:
                if vehicle.id in history_trajectories:
                    # 删除不在预测范围内的车辆
                    del history_trajectories[vehicle.id]
                    del surrounding_vehicles_queues[vehicle.id]
                continue

            pred_vehicle = vehicle

            # 标记预测车辆
            # draw_surrounding_points(pred_vehicle, world, distance=2.0)
            # world.debug.draw_point(pred_vehicle.get_location(), 0.5, carla.Color(r=0, g=255, b=0), 0.1)
            
            # 更新预测车辆的，历史轨迹队列和周围车辆队列
            if vehicle.id not in history_trajectories:
                history_trajectories[vehicle.id] = deque(maxlen=20)
                surrounding_vehicles_queues[vehicle.id] = {}

            update_vehicle_trajectories(pred_vehicle, history_trajectories[vehicle.id], surrounding_vehicles_queues[vehicle.id])
            pred_length = len(list(history_trajectories[vehicle.id]))

            # 预测车辆出现时间小于10帧时，不进行预测
            if pred_length < 5:
                world.debug.draw_string(vehicle.get_location(), 'Skip!!!', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1)
                continue

            world.debug.draw_string(vehicle.get_location(), 'Pred!!!', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=0.1)

            # 调整数据格式
            trajectories_tensor = create_trajectory_tensor(history_trajectories[vehicle.id], surrounding_vehicles_queues[vehicle.id])
            trajectories_ctrs = trajectories_tensor.clone().detach()[:, -1, :2]
            nearby_waypoints = get_nearby_waypoints(pred_vehicle, waypoints_xy)
            lane_list = get_topology(target_vehicle, map_dict)

            # 构建网络输入
            data = dict()
            data['feat'] = [trajectories_tensor.cuda()]
            data['ctrs'] = [trajectories_ctrs.cuda()]
            data['nbr_waypoints'] = [nearby_waypoints.cuda()]
            data['lane_list'] = [lane_list.cuda()]

            # 预测轨迹并绘制
            output = net(data)
            traj_pred = output['reg'][0][0][:3].cpu().detach().numpy()   # shape = (3, 30, 2)
            draw_predicted_trajectories(pred_vehicle, traj_pred, world)

        loop_end_time = time.time()
        print('loop time:', loop_end_time - loop_start_time)
        loop_elapsed_time = loop_end_time - loop_start_time
        sleep_time = max(0, interval - loop_elapsed_time)

        time.sleep(sleep_time)

except KeyboardInterrupt:
    target_vehicle.destroy()
    for vehicle in vehicles:
        vehicle.destroy()