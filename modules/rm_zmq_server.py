#!/usr/bin/env python3
"""
RM机械臂ZMQ服务器
基于RM Python API实现与arx5相同的ZMQ接口
"""

import sys
import os
import time
import traceback
from typing import Any, Optional, cast
import numpy as np
import numpy.typing as npt
import zmq
import click

# 添加Python库路径
sys.path.append(os.path.join(os.path.dirname(__file__), "Python"))
from Robotic_Arm.rm_robot_interface import *


def echo_exception():
    """捕获并格式化异常信息"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return "".join(tb_lines)


class RMServer:
    def __init__(
        self,
        robot_ip: str,
        robot_port: int,
        zmq_ip: str,
        zmq_port: int,
        home_joint: Optional[list[float]] = None,
        no_cmd_timeout: float = 60.0,
    ):
        """
        初始化RM ZMQ服务器
        
        Args:
            robot_ip: 机械臂IP地址
            robot_port: 机械臂端口
            zmq_ip: ZMQ服务器IP
            zmq_port: ZMQ服务器端口
            home_joint: Home位置的关节角度 (度)，如果为None则使用[0, 0, 0, 0, 0, 0]
            no_cmd_timeout: 无命令超时时间（秒）
        """
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.home_joint = home_joint if home_joint is not None else [0, 20, 70, 0, 90, 0]
        
        # 初始化机械臂连接
        self.robot: Optional[RoboticArm] = None
        self.handle: Optional[rm_robot_handle] = None
        self._connect_robot()
        
        # 初始化ZMQ服务器
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{zmq_ip}:{zmq_port}")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        self.last_cmd_time = time.monotonic()
        self.no_cmd_timeout = no_cmd_timeout
        self.is_reset_to_home = False
        self.last_eef_cmd: Optional[npt.NDArray[np.float64]] = None
        self.home_pose: Optional[npt.NDArray[np.float64]] = None
        
        # 获取home位置的笛卡尔坐标
        self._update_home_pose()
        
        print(f"RM ZMQ Server initialized on {zmq_ip}:{zmq_port}")
        print(f"Connected to RM robot at {robot_ip}:{robot_port}")
        
    def _connect_robot(self):
        """连接机械臂"""
        try:
            self.robot = RoboticArm(rm_thread_mode_e(2))  # 使用三线程模式
            self.handle = self.robot.rm_create_robot_arm(self.robot_ip, self.robot_port, 3)
            
            if self.handle.id == -1:
                raise RuntimeError("Failed to connect to RM robot arm")
            
            print(f"Successfully connected to RM robot arm (handle id: {self.handle.id})")
            
            # 获取机械臂信息
            res, robot_info = self.robot.rm_get_robot_info()
            if res == 0:
                print(f"Robot model: {robot_info.get('arm_model', 'Unknown')}")
                print(f"Robot DOF: {robot_info.get('arm_dof', 'Unknown')}")
        except Exception as e:
            print(f"Error connecting to robot: {e}")
            raise
    
    def _disconnect_robot(self):
        """断开机械臂连接"""
        if self.robot is not None:
            self.robot.rm_delete_robot_arm()
            self.robot = None
            self.handle = None
            print("Disconnected from RM robot arm")
    
    def _update_home_pose(self):
        """更新home位置的笛卡尔坐标"""
        try:
            # 先移动到home位置
            ret = self.robot.rm_movej(self.home_joint, 20, 0, 0, 1)
            if ret != 0:
                print(f"Warning: Failed to move to home position (error: {ret})")
            
            # 获取当前位姿作为home位置
            ret, state = self.robot.rm_get_current_arm_state()
            if ret == 0:
                # state['pose']包含 [x, y, z, rx, ry, rz]
                pose = state.get('pose', [0.0] * 6)
                self.home_pose = np.array(pose[:6], dtype=np.float64)
                print(f"Home pose: {self.home_pose}")
            else:
                print(f"Warning: Failed to get home pose (error: {ret})")
                self.home_pose = np.zeros(6, dtype=np.float64)
        except Exception as e:
            print(f"Error updating home pose: {e}")
            self.home_pose = np.zeros(6, dtype=np.float64)
    
    def _get_current_state(self) -> dict:
        """获取机械臂当前状态"""
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"Failed to get arm state (error: {ret})")
        
        # 将RM的状态数据转换为与arx5兼容的格式
        pose = state.get('pose', [0.0] * 6)
        joint = state.get('joint', [0.0] * 6)
        
        # RM API 返回关节角度（度），需要转换为弧度
        joint_rad = [np.deg2rad(j) for j in joint]
        
        return {
            'timestamp': time.time(),
            'ee_pose': np.array(pose[:6], dtype=np.float64),  # [x, y, z, rx, ry, rz]
            'joint_pos': np.array(joint_rad, dtype=np.float64),  # 弧度
            'joint_vel': np.zeros(len(joint), dtype=np.float64),  # RM API不直接提供速度
            'joint_torque': np.zeros(len(joint), dtype=np.float64),  # RM API不直接提供力矩
            'gripper_pos': 0.0,  # 需要单独查询
            'gripper_vel': 0.0,
            'gripper_torque': 0.0,
        }
    
    def _set_ee_pose(self, target_pose: npt.NDArray[np.float64], gripper_pos: Optional[float] = None):
        """
        设置末端执行器位姿
        
        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz]
            gripper_pos: 夹爪位置（可选）
        """
        # 使用movel进行笛卡尔空间直线运动
        # v=20: 速度百分比
        # r=0: 交融半径
        # connect=0: 不连接下一条轨迹
        # block=0: 非阻塞模式，立即返回
        ret = self.robot.rm_movel(target_pose.tolist(), v=20, r=0, connect=0, block=0)
        if ret != 0:
            raise RuntimeError(f"Failed to set EE pose (error: {ret})")
        
        # 如果指定了夹爪位置，则设置夹爪
        if gripper_pos is not None:
            # RM夹爪位置范围是1-1000
            gripper_position = int(gripper_pos * 1000)
            gripper_position = max(1, min(1000, gripper_position))
            ret = self.robot.rm_set_gripper_position(gripper_position, block=False, timeout=0)
            if ret != 0:
                print(f"Warning: Failed to set gripper position (error: {ret})")
    
    def _reset_to_home(self):
        """重置到home位置"""
        ret = self.robot.rm_movej(self.home_joint, 20, 0, 0, 1)  # block=1，等待运动完成
        if ret != 0:
            raise RuntimeError(f"Failed to reset to home (error: {ret})")
        
        # 更新last_eef_cmd
        ret, state = self.robot.rm_get_current_arm_state()
        if ret == 0:
            pose = state.get('pose', [0.0] * 6)
            self.last_eef_cmd = np.array(pose[:6], dtype=np.float64)
    
    def run(self):
        """运行ZMQ服务器主循环"""
        print(f"RM ZMQ Server is running on {self.zmq_ip}:{self.zmq_port}")
        
        while True:
            try:
                # 等待消息，带超时
                socks = dict(self.poller.poll(int(self.no_cmd_timeout * 1000)))
                
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    msg: dict[str, Any] = self.socket.recv_pyobj()
                    
                    # 如果机械臂连接断开，尝试重新连接
                    if self.robot is None:
                        print("Reestablishing robot connection")
                        self._connect_robot()
                        self._update_home_pose()
                else:
                    # 超时：没有收到命令
                    if self.robot is not None:
                        print(f"Timeout: No command received for {self.no_cmd_timeout} sec. Disconnecting robot.")
                        self._disconnect_robot()
                        self.last_eef_cmd = None
                    continue
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                exception_str = echo_exception()
                print(f"Error in main loop: {exception_str}")
                continue
            
            # 处理收到的消息
            try:
                if not isinstance(msg, dict):
                    print(f"Error: Received invalid message {msg}, ignored")
                    self.socket.send_pyobj({
                        "cmd": "UNKNOWN",
                        "data": f"ERROR: Received invalid message {msg}",
                    })
                    continue
                
                cmd = msg.get("cmd", "UNKNOWN")
                print(f"Received command: {cmd}")
                
                if cmd == "GET_STATE":
                    # 获取机械臂状态
                    state = self._get_current_state()
                    reply_msg = {
                        "cmd": "GET_STATE",
                        "data": state,
                    }
                    self.socket.send_pyobj(reply_msg)
                    
                elif cmd == "SET_EE_POSE":
                    # 设置末端执行器位姿
                    if self.last_eef_cmd is None:
                        error_str = "Error: Cannot set EE pose before RESET_TO_HOME"
                        print(error_str)
                        self.socket.send_pyobj({
                            "cmd": "SET_EE_POSE",
                            "data": error_str,
                        })
                        continue
                    
                    target_ee_pose = cast(np.ndarray, msg["data"]["ee_pose"])
                    
                    # 检查目标位姿与上一次命令的距离
                    if np.linalg.norm(target_ee_pose[:3] - self.last_eef_cmd[:3]) > 0.1:
                        error_str = f"Error: Target pose {target_ee_pose} is too far from last command {self.last_eef_cmd}"
                        print(error_str)
                        self.socket.send_pyobj({
                            "cmd": "SET_EE_POSE",
                            "data": error_str,
                        })
                        continue
                    
                    # 如果刚重置到home，检查是否离home太远
                    if self.is_reset_to_home:
                        if np.linalg.norm(target_ee_pose[:3] - self.home_pose[:3]) > 0.1:
                            error_str = f"Error: Target pose {target_ee_pose} is too far from home {self.home_pose}"
                            print(error_str)
                            self.socket.send_pyobj({
                                "cmd": "SET_EE_POSE",
                                "data": error_str,
                            })
                            continue
                    
                    # 更新last_eef_cmd
                    self.last_eef_cmd = target_ee_pose.copy()
                    
                    # 获取夹爪位置
                    gripper_pos = msg["data"].get("gripper_pos", None)
                    
                    # 设置位姿
                    self._set_ee_pose(target_ee_pose, gripper_pos)
                    
                    # 获取当前状态并返回
                    time.sleep(0.05)  # 等待一小段时间让运动开始
                    state = self._get_current_state()
                    reply_msg = {
                        "cmd": "SET_EE_POSE",
                        "data": state,
                    }
                    self.socket.send_pyobj(reply_msg)
                    self.is_reset_to_home = False
                    
                elif cmd == "RESET_TO_HOME":
                    # 重置到home位置
                    print("Resetting to home position")
                    self._reset_to_home()
                    reply_msg = {
                        "cmd": "RESET_TO_HOME",
                        "data": "OK",
                    }
                    self.socket.send_pyobj(reply_msg)
                    self.is_reset_to_home = True
                    
                elif cmd == "SET_TO_DAMPING":
                    # RM API没有直接的阻尼模式，这里返回成功
                    # 可以考虑停止当前运动
                    print("SET_TO_DAMPING (not fully supported by RM API)")
                    reply_msg = {
                        "cmd": "SET_TO_DAMPING",
                        "data": "OK (not fully supported)",
                    }
                    self.socket.send_pyobj(reply_msg)
                    self.is_reset_to_home = False
                    
                elif cmd == "GET_GAIN":
                    # RM API没有直接的增益接口，返回默认值
                    print("GET_GAIN (not supported by RM API, returning defaults)")
                    reply_msg = {
                        "cmd": "GET_GAIN",
                        "data": {
                            "kp": np.array([100.0] * 6, dtype=np.float64),
                            "kd": np.array([10.0] * 6, dtype=np.float64),
                            "gripper_kp": 100.0,
                            "gripper_kd": 10.0,
                        },
                    }
                    self.socket.send_pyobj(reply_msg)
                    
                elif cmd == "SET_GAIN":
                    # RM API没有直接的增益接口
                    print("SET_GAIN (not supported by RM API)")
                    reply_msg = {
                        "cmd": "SET_GAIN",
                        "data": "OK (not supported)",
                    }
                    self.socket.send_pyobj(reply_msg)
                    
                else:
                    raise ValueError(f"Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                exception_str = echo_exception()
                self.socket.send_pyobj({
                    "cmd": msg.get("cmd", "UNKNOWN"),
                    "data": f"ERROR: {exception_str}"
                })
                print(f"Error processing command: {exception_str}")
                continue
    
    def __del__(self):
        """清理资源"""
        if self.robot is not None:
            self._disconnect_robot()
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        print("RM ZMQ Server terminated")


@click.command()
@click.option("--robot-ip", default="192.168.1.18", help="RM robot IP address")
@click.option("--robot-port", default=8080, help="RM robot port")
@click.option("--zmq-ip", default="0.0.0.0", help="ZMQ server IP")
@click.option("--zmq-port", default=8765, help="ZMQ server port")
@click.option("--home-joint", default="0,20,70,0,90,0", help="Home joint angles (comma-separated)")
def main(robot_ip: str, robot_port: int, zmq_ip: str, zmq_port: int, home_joint: str):
    """启动RM机械臂ZMQ服务器"""
    # 解析home关节角度
    home_joint_list = [float(x.strip()) for x in home_joint.split(",")]
    
    server = RMServer(
        robot_ip=robot_ip,
        robot_port=robot_port,
        zmq_ip=zmq_ip,
        zmq_port=zmq_port,
        home_joint=home_joint_list,
        no_cmd_timeout=60.0,
    )
    server.run()


if __name__ == "__main__":
    main()

