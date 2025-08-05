# 检查仿真软件路径
import os,sys
import numpy as np
def checkPath():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

# 定义sigmoid函数
def sigmoid(x):
   return 1 / (1 + np.exp(-x))

#  修正实时的占有率或车速值
def correctValue(para):
   if para < 0.0:
     return 0.0
   else:
     return round(para, 2)

from bs4 import BeautifulSoup
def getDataFromDeteFile(dete_file_path, time_interval, para_name):
    deteInfos = BeautifulSoup(open(dete_file_path), "xml").detector
    deteInfos_list = [deteInfo for deteInfo in deteInfos.children if deteInfo != "\n"]
    deteInfos_filtered = [deteInfo for deteInfo in deteInfos_list if (float(deteInfo["begin"]) >=
                                                                      time_interval[0]) & (
                                      float(deteInfo["end"]) <= time_interval[1])]
    para_list = [deteInfo[para_name] for deteInfo in deteInfos_filtered]
    return para_list

#  从车辆出行信息输出文件中获取行程时间
from bs4 import BeautifulSoup
def getIntervalTravelTime(tripinfo_file_path, timeInterval):
    tripinfos = BeautifulSoup(open(tripinfo_file_path), "xml").tripinfos
    tripinfo_list = [tripinfo for tripinfo in tripinfos.children if tripinfo != "\n"]
    tripFiltered = [tripinfo for tripinfo in tripinfo_list if ((float(tripinfo["depart"])) > timeInterval[0]) &
                    ((float(tripinfo["depart"])) <= timeInterval[1]) & (
                                (float(tripinfo["arrival"])) <= timeInterval[1])]
    totalTravelTime = int(sum([float(tripinfo["duration"]) for tripinfo in tripFiltered]))
    totalTravelTime_main = int(sum([float(tripinfo["duration"]) for tripinfo in tripFiltered if tripinfo["id"].startswith('type0' and 'type1')]))  #  从
    totalTimeloss = int(sum([float(tripinfo["timeLoss"]) for tripinfo in tripFiltered]))
    totalTimeloss_main = int(sum([float(tripinfo["timeLoss"]) for tripinfo in tripFiltered if
                                    tripinfo["id"].startswith('type0' and 'type1')]))
    totalVehNum = len(tripFiltered)
    return totalTravelTime,totalVehNum,totalTravelTime_main,totalTimeloss,totalTimeloss_main

#  奖励值获取函数
import traci
def getReward(agent_name):  #这个reward是什么意思
    flow_name = "flow" + agent_name[-1]       # 通过流量名筛选出getArrivedIDList和getDepartedIDList两个函数返回值中的匝道流量车辆或主线流量车辆
    reward = len([x for x in traci.simulation.getArrivedIDList() if flow_name in x]) - len([x for x in traci.simulation.getDepartedIDList() if flow_name in x])
    return reward

# RSU 类（示例）
class RSU(object):
    import traci
    def __init__(self, edge_id):
        self.__edge_id = edge_id

    def getVehicleInfo(self):
        veh_id = traci.edge.getLastStepVehicleIDs(self.__edge_id)     # 获取车辆 id
        veh_xy = [traci.vehicle.getPosition(id) for id in veh_id]     # 获取车辆位置
        veh_speed = [traci.vehicle.getSpeed(id) for id in veh_id]     # 获取车速
        return veh_id, veh_xy, veh_speed

    def getEdgeInfo(self):
        edge_avg_speed = traci.edge.getLastStepMeanSpeed(self.__edge_id)  # 获取平均车速
        edge_avg_occu = traci.edge.getLastStepOccupancy(self.__edge_id) * 100  # 获取平均占有率(%)
        return edge_avg_speed, edge_avg_occu

# TCU 类（示例）
class TCU(object):
    import traci

    def __init__(self, edge_id, kr, optimal_occu):
        self.__edge_id = edge_id
        self.__kr = kr
        self.__optimal_occu = optimal_occu

    def ALINEA(self, rk, real_occu,control_cycle): # ALINEA感觉这里的写法有问题
        rk = rk + int(self.__kr * (self.__optimal_occu - real_occu / 100))  # 计算匝道调节率
        if rk > control_cycle: rk = control_cycle
        elif rk < 0:
            rk = 0
        return rk


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option(
        "- -nogui",
    action="store true",
    default=True,
    help="run the commandline version of sumo",
                        )
    options,args = optParser.parseargs( )
    return options






