#!/usr/bin/env python3
"""
RM机械臂ZMQ客户端
基于arx5_zmq_client.py接口，与rm_zmq_server.py通信
保持相同的接口和数据结构
"""

from typing import Any, Optional, Union, cast
import zmq
from enum import IntEnum, auto
import numpy.typing as npt
import numpy as np
import sys
import traceback


def echo_exception():
    """捕获并格式化异常信息"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return "".join(tb_lines)


CTRL_DT = 0.005
GRIPPER_WIDTH = 0.08


def rotm2rotvec(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将旋转矩阵转换为旋转向量
    """
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros(3)
    else:
        k = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
        k = k / (2 * np.sin(theta))
        return theta * k


def rotvec2rotm(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将旋转向量转换为旋转矩阵
    """
    theta = np.linalg.norm(rotvec)
    if np.isclose(theta, 0):
        return np.eye(3)
    else:
        k = rotvec / theta
        K = np.array(
            [
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0],
            ]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return R


def rpy2rotm(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将roll-pitch-yaw角转换为旋转矩阵
    """
    roll, pitch, yaw = rpy
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rotm2rpy(R: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将旋转矩阵转换为roll-pitch-yaw角
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def ee2tcp(ee_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将末端执行器位姿转换为TCP位姿
    """
    ee_cartesian = ee_pose[:3]
    ee_rpy = ee_pose[3:]
    ee_rot_mat = rpy2rotm(ee_rpy)
    ee2tcp_rot_mat = np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    )
    tcp_rot_mat = ee_rot_mat @ ee2tcp_rot_mat
    tcp_rotvec = rotm2rotvec(tcp_rot_mat)
    # opposite the rotation vector but keep the same pose
    angle_rad = np.linalg.norm(tcp_rotvec)
    vec = tcp_rotvec / angle_rad
    alternate_angle_rad = 2 * np.pi - angle_rad
    tcp_rotvec = -vec * alternate_angle_rad

    return np.concatenate([ee_cartesian, tcp_rotvec])


def tcp2ee(tcp_pose: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    将TCP位姿转换为末端执行器位姿
    """
    tcp_cartesian = tcp_pose[:3]
    tcp_rotvec = tcp_pose[3:]
    tcp_rot_mat = rotvec2rotm(tcp_rotvec)
    tcp2ee_rot_mat = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )
    ee_rot_mat = tcp_rot_mat @ tcp2ee_rot_mat
    ee_rpy = rotm2rpy(ee_rot_mat)
    return np.concatenate([tcp_cartesian, ee_rpy])


class RMClient:
    """
    RM机械臂ZMQ客户端
    与Arx5Client接口完全一致
    """
    def __init__(
        self,
        zmq_ip: str,
        zmq_port: int,
    ):
        """
        初始化RM客户端
        
        Args:
            zmq_ip: ZMQ服务器IP地址
            zmq_port: ZMQ服务器端口
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{zmq_ip}:{zmq_port}")
        self.zmq_ip = zmq_ip
        self.zmq_port = zmq_port
        self.latest_state: dict[str, Union[npt.NDArray[np.float64], float]]
        print(
            f"RMClient is connected to {self.zmq_ip}:{self.zmq_port}. Fetching state..."
        )
        self.get_state()
        print(f"Initial state fetched")

    def send_recv(self, msg: dict[str, Any]):
        """
        发送消息并接收响应
        
        Args:
            msg: 要发送的消息字典
            
        Returns:
            服务器响应
        """
        try:
            self.socket.send_pyobj(msg)
            reply_msg = self.socket.recv_pyobj()
            return reply_msg
        except KeyboardInterrupt:
            print("RMClient: KeyboardInterrupt. connection is re-established.")
            return {"cmd": msg["cmd"], "data": "KeyboardInterrupt"}
        except zmq.error.ZMQError:
            # Usually happens when the process is interrupted before receiving reply
            print("RMClient: ZMQError. connection is re-established.")
            print(echo_exception())
            self.socket.close()
            del self.socket
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.zmq_ip}:{self.zmq_port}")
            return {"cmd": msg["cmd"], "data": "ZMQError"}

    def get_state(self):
        """
        获取机械臂当前状态
        
        Returns:
            状态字典，包含timestamp, ee_pose, joint_pos, joint_vel, joint_torque,
            gripper_pos, gripper_vel, gripper_torque
        """
        reply_msg = self.send_recv({"cmd": "GET_STATE", "data": None})
        assert reply_msg["cmd"] == "GET_STATE"
        assert isinstance(reply_msg["data"], dict)
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def set_ee_pose(
        self, pose_6d: npt.NDArray[np.float64], gripper_pos: Union[float, None] = None
    ):
        """
        设置末端执行器位姿
        
        Args:
            pose_6d: 目标位姿 [x, y, z, rx, ry, rz] (米/弧度)
            gripper_pos: 夹爪位置 (0-1)，None表示保持当前位置
            
        Returns:
            更新后的状态字典
        """
        reply_msg = self.send_recv(
            {
                "cmd": "SET_EE_POSE",
                "data": {"ee_pose": pose_6d, "gripper_pos": gripper_pos},
            }
        )
        assert reply_msg["cmd"] == "SET_EE_POSE"
        if reply_msg["data"] == "KeyboardInterrupt" or reply_msg["data"] == "ZMQError":
            return self.latest_state
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        state = cast(
            dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"]
        )
        self.latest_state = state
        return state

    def set_tcp_pose(
        self,
        tcp_pose_6d: npt.NDArray[np.float64],
        gripper_pos: Union[float, None] = None,
    ):
        """
        设置TCP位姿（自动转换为EE位姿）
        
        Args:
            tcp_pose_6d: TCP目标位姿 [x, y, z, rx, ry, rz]
            gripper_pos: 夹爪位置 (0-1)
            
        Returns:
            更新后的状态字典
        """
        ee_pose = tcp2ee(tcp_pose_6d)
        return self.set_ee_pose(ee_pose, gripper_pos)

    def reset_to_home(self):
        """
        重置机械臂到home位置
        
        Raises:
            ValueError: 如果重置失败
        """
        reply_msg = self.send_recv({"cmd": "RESET_TO_HOME", "data": None})
        assert reply_msg["cmd"] == "RESET_TO_HOME"
        if reply_msg["data"] != "OK":
            raise ValueError(f"Error: {reply_msg['data']}")
        self.get_state()

    def set_to_damping(self):
        """
        设置机械臂为阻尼模式
        
        注意: RM机械臂不完全支持此功能
        
        Raises:
            ValueError: 如果设置失败
        """
        reply_msg = self.send_recv({"cmd": "SET_TO_DAMPING", "data": None})
        assert reply_msg["cmd"] == "SET_TO_DAMPING"
        if "OK" not in reply_msg["data"]:
            raise ValueError(f"Error: {reply_msg['data']}")
        self.get_state()

    def get_gain(self):
        """
        获取增益参数
        
        注意: RM机械臂返回默认值
        
        Returns:
            增益字典，包含kp, kd, gripper_kp, gripper_kd
        """
        reply_msg = self.send_recv({"cmd": "GET_GAIN", "data": None})
        assert reply_msg["cmd"] == "GET_GAIN"
        if type(reply_msg["data"]) != dict:
            raise ValueError(f"Error: {reply_msg['data']}")
        return cast(dict[str, Union[npt.NDArray[np.float64], float]], reply_msg["data"])

    def set_gain(self, gain: dict[str, Union[npt.NDArray[np.float64], float]]):
        """
        设置增益参数
        
        注意: RM机械臂不支持此功能
        
        Args:
            gain: 增益字典，包含kp, kd, gripper_kp, gripper_kd
            
        Raises:
            ValueError: 如果设置失败
        """
        reply_msg = self.send_recv({"cmd": "SET_GAIN", "data": gain})
        assert reply_msg["cmd"] == "SET_GAIN"
        if "OK" not in reply_msg["data"]:
            raise ValueError(f"Error: {reply_msg['data']}")

    @property
    def timestamp(self):
        """当前状态时间戳"""
        timestamp = self.latest_state["timestamp"]
        return cast(float, timestamp)

    @property
    def ee_pose(self):
        """末端执行器位姿 [x, y, z, rx, ry, rz]"""
        ee_pose = self.latest_state["ee_pose"]
        return cast(npt.NDArray[np.float64], ee_pose)

    @property
    def joint_pos(self):
        """关节位置（弧度）"""
        joint_pos = self.latest_state["joint_pos"]
        return cast(npt.NDArray[np.float64], joint_pos)

    @property
    def joint_vel(self):
        """关节速度（弧度/秒）"""
        joint_vel = self.latest_state["joint_vel"]
        return cast(npt.NDArray[np.float64], joint_vel)

    @property
    def joint_torque(self):
        """关节力矩（N·m）"""
        joint_torque = self.latest_state["joint_torque"]
        return cast(npt.NDArray[np.float64], joint_torque)

    @property
    def gripper_pos(self):
        """夹爪位置 (0-1)"""
        gripper_pos = self.latest_state["gripper_pos"]
        return cast(float, gripper_pos)

    @property
    def gripper_vel(self):
        """夹爪速度"""
        gripper_vel = self.latest_state["gripper_vel"]
        return cast(float, gripper_vel)

    @property
    def gripper_torque(self):
        """夹爪力矩"""
        gripper_torque = self.latest_state["gripper_torque"]
        return cast(float, gripper_torque)

    @property
    def tcp_pose(self):
        """TCP位姿（从EE位姿转换）"""
        tcp_pose = np.zeros(6, dtype=np.float64)
        tcp_pose[:] = ee2tcp(self.ee_pose)
        return tcp_pose

    def __del__(self):
        """清理资源"""
        self.socket.close()
        self.context.term()
        print("RMClient is closed")


# 为了兼容性，创建别名
Arx5Client = RMClient


if __name__ == "__main__":
    """
    测试示例
    """
    import time
    
    print("="*60)
    print("RM ZMQ Client 测试")
    print("="*60)
    
    # 创建客户端连接
    client = RMClient(zmq_ip="127.0.0.1", zmq_port=8765)
    
    try:
        # 1. 重置到home位置
        print("\n1. 重置到home位置...")
        client.reset_to_home()
        print(f"   Home EE位姿: {client.ee_pose}")
        print(f"   Home TCP位姿: {client.tcp_pose}")
        time.sleep(1)
        
        # 2. 获取当前状态
        print("\n2. 获取当前状态...")
        state = client.get_state()
        print(f"   时间戳: {client.timestamp}")
        print(f"   EE位姿: {client.ee_pose}")
        print(f"   关节位置: {client.joint_pos}")
        print(f"   夹爪位置: {client.gripper_pos}")
        
        # 3. 使用EE位姿进行运动
        print("\n3. 使用EE位姿移动...")
        target_ee_pose = client.ee_pose.copy()
        target_ee_pose[0] += 0.05  # X方向移动5cm
        print(f"   目标EE位姿: {target_ee_pose}")
        client.set_ee_pose(target_ee_pose)
        time.sleep(1)
        
        # 4. 使用TCP位姿进行运动
        print("\n4. 使用TCP位姿移动...")
        target_tcp_pose = client.tcp_pose.copy()
        target_tcp_pose[1] += 0.05  # Y方向移动5cm
        print(f"   目标TCP位姿: {target_tcp_pose}")
        client.set_tcp_pose(target_tcp_pose)
        time.sleep(1)
        
        # 5. 带夹爪控制的运动
        print("\n5. 带夹爪控制的运动...")
        target_ee_pose = client.ee_pose.copy()
        target_ee_pose[2] += 0.03  # Z方向移动3cm
        print(f"   目标EE位姿: {target_ee_pose}, 夹爪: 0.5")
        client.set_ee_pose(target_ee_pose, gripper_pos=0.5)
        time.sleep(1)
        
        # 6. 获取增益参数
        print("\n6. 获取增益参数...")
        gain = client.get_gain()
        print(f"   Kp: {gain['kp']}")
        print(f"   Kd: {gain['kd']}")
        
        # 7. 返回home
        print("\n7. 返回home...")
        client.reset_to_home()
        print(f"   当前EE位姿: {client.ee_pose}")
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 客户端会在__del__中自动关闭
        pass

