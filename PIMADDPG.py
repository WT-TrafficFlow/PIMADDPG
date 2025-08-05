import math
import torch
import pandas as pd
import xml.etree.ElementTree as ET
import random
import os
import traci
import time
import traci.constants as tc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import truncnorm
from DDPG import *
from used_function import *
from bs4 import BeautifulSoup
from onestep_pred import TrafficDetectorMerger, TrafficStatePred
# 设置随机种子
random.seed(0)

origin_df = pd.read_csv("demand/origin.csv")
destination_df = pd.read_csv("demand/destination.csv")

# merged_df = pd.merge(origin_df, destination_df, how='cross') # 由于将python版本将为3.6，改用下面这句话实现how='cross' ↓
origin_df['key'] = 1
destination_df['key'] = 1
merged_df = pd.merge(origin_df, destination_df, on='key').drop('key', axis=1)
# -------------------------------------------------------------------------------------------------------------- ↑

valid_pairs = merged_df[merged_df['location_y'] > merged_df['location_x']]
# print(valid_pairs)

hist_step = 12
detector_merger = TrafficDetectorMerger(hist_step)
wm = TrafficStatePred()
Main_main = [5]
Main_offramp1 = [1]
Main_offramp2 = [2]
Main_offramp3 = [3]
Main_offramp4 = [4]
Onramp1_offramp2 = [6]
Onramp1_offramp3 = [7]
Onramp1_offramp4 = [8]
Onramp2_offramp3 = [10]
Onramp2_offramp4 = [11]
Onramp3_offramp4 = [13]
Onramp1_main = [9]
Onramp2_main = [12]
Onramp3_main = [14]
Onramp4_main = [15]

traffic_light_id = ['J1', 'J2', 'J3', 'J4']

agent1_ddpg = DDPG("agent1", num_state=6, num_action=1, num_other_aciton=1)
agent1_ddpg_target = DDPG("agent1_target", num_state=6, num_action=1, num_other_aciton=1)

agent2_ddpg = DDPG("agent2", num_state=6, num_action=1, num_other_aciton=1)
agent2_ddpg_target = DDPG("agent2_target", num_state=6, num_action=1, num_other_aciton=1)

agent3_ddpg = DDPG("agent3", num_state=6, num_action=1, num_other_aciton=1)
agent3_ddpg_target = DDPG("agent3_target", num_state=6, num_action=1, num_other_aciton=1)

agent4_ddpg = DDPG("agent4", num_state=6, num_action=1, num_other_aciton=1)
agent4_ddpg_target = DDPG("agent4_target", num_state=6, num_action=1, num_other_aciton=1)

agent5_ddpg = DDPG("agent5", num_state=6, num_action=1, num_other_aciton=1)
agent5_ddpg_target = DDPG("agent5_target", num_state=6, num_action=1, num_other_aciton=1)

agent6_ddpg = DDPG("agent6", num_state=6, num_action=1, num_other_aciton=1)
agent6_ddpg_target = DDPG("agent6_target", num_state=6, num_action=1, num_other_aciton=1)

agent7_ddpg = DDPG("agent7", num_state=6, num_action=1, num_other_aciton=1)
agent7_ddpg_target = DDPG("agent7_target", num_state=6, num_action=1, num_other_aciton=1)

agent8_ddpg = DDPG("agent8", num_state=6, num_action=1, num_other_aciton=1)
agent8_ddpg_target = DDPG("agent8_target", num_state=6, num_action=1, num_other_aciton=1)

agent1_actor_target_init,  agent1_actor_target_update  =  create_init_update("agent1_actor", "agent1_target_actor")
agent1_critic_target_init,  agent1_critic_target_update  =  create_init_update("agent1_critic", "agent1_target_critic")

agent2_actor_target_init,  agent2_actor_target_update  =  create_init_update("agent2_actor", "agent2_target_actor")
agent2_critic_target_init,  agent2_critic_target_update  =  create_init_update("agent2_critic", "agent2_target_critic")

agent3_actor_target_init,  agent3_actor_target_update  =  create_init_update("agent3_actor", "agent3_target_actor")
agent3_critic_target_init,  agent3_critic_target_update  =  create_init_update("agent3_critic", "agent3_target_critic")

agent4_actor_target_init,  agent4_actor_target_update  =  create_init_update("agent4_actor", "agent4_target_actor")
agent4_critic_target_init,  agent4_critic_target_update  =  create_init_update("agent4_critic", "agent4_target_critic")

agent5_actor_target_init,  agent5_actor_target_update  =  create_init_update("agent5_actor", "agent5_target_actor")
agent5_critic_target_init,  agent5_critic_target_update  =  create_init_update("agent5_critic", "agent5_target_critic")

agent6_actor_target_init,  agent6_actor_target_update  =  create_init_update("agent6_actor", "agent6_target_actor")
agent6_critic_target_init,  agent6_critic_target_update  =  create_init_update("agent6_critic", "agent6_target_critic")

agent7_actor_target_init,  agent7_actor_target_update  =  create_init_update("agent7_actor", "agent7_target_actor")
agent7_critic_target_init,  agent7_critic_target_update  =  create_init_update("agent7_critic", "agent7_target_critic")

agent8_actor_target_init,  agent8_actor_target_update  =  create_init_update("agent8_actor", "agent8_target_actor")
agent8_critic_target_init,  agent8_critic_target_update  =  create_init_update("agent8_critic", "agent8_target_critic")


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run([agent1_actor_target_init, agent1_critic_target_init,
          agent2_actor_target_init, agent2_critic_target_init,
          agent3_actor_target_init, agent3_critic_target_init,
          agent4_actor_target_init, agent4_critic_target_init])

agent1_memory = ReplayBuffer(1000000)
agent2_memory = ReplayBuffer(1000000)
agent3_memory = ReplayBuffer(1000000)
agent4_memory = ReplayBuffer(1000000)
agent5_memory = ReplayBuffer(1000000)
agent6_memory = ReplayBuffer(1000000)
agent7_memory = ReplayBuffer(1000000)
agent8_memory = ReplayBuffer(1000000)


rm1_list, rm2_list, rm3_list, rm4_list = [], [], [], []
rm1_list_figure, rm2_list_figure, rm3_list_figure, rm4_list_figure = [], [], [], []

vsl1_list, vsl2_list, vsl3_list, vsl4_list= [], [], [], []
vsl1_list_figure, vsl2_list_figure, vsl3_list_figure, vsl4_list_figure= [], [], [], []

avg_bn_occ, avg_bn_veh = [], []
travel_time_epi = []
main_travel_time_epi = []
travelloss_time = []
main_travelloss_time = []
all_ep_r = []
rm1_quene_lenth,rm1_avg_quene_lenth = [],[]
rm2_quene_lenth,rm2_avg_quene_lenth = [],[]
rm3_quene_lenth,rm3_avg_quene_lenth = [],[]
rm4_quene_lenth,rm4_avg_quene_lenth = [],[]
period_start = 0
period_end = 10800
M = 5
I = 2
v0 = 20
control_cycle_rm = 60
dete1_list = ["e1down_0", "e1down_1", "e1ramp", "e1up_0", "e1up_1"]
ramp1_list_0 = ["e1ramp_0"]
ramp1_list_1 = ["e1ramp_1"]

dete2_list = ["e2down_0", "e2down_1", "e2ramp", "e2up_0", "e2up_1"]
ramp2_list_0 = ["e2ramp_0"]
ramp2_list_1 = ["e2ramp_1"]
ramp2_list_2 = ["e2ramp_2"]

dete3_list = ["e3down_0", "e3down_1", "e3ramp", "e3up_0", "e3up_1"]
ramp3_list = ["e3ramp_0"]

dete4_list = ["e4down_0", "e4down_1", "e4ramp", "e4up_0", "e4up_1"]
ramp4_list = ["e4ramp_0"]

dete_list_down = ["e1down_0", "e1down_1","e2down_0", "e2down_1","e3down_0", "e3down_1","e4down_0", "e4down_1"]
dete_list_up = [  "e1up_0","e1up_1",  "e2up_0","e2up_1", "e3up_0", "e3up_1", "e4up_0","e4up_1"]

sumo_bin_path = "E:/sumo/bin"

config_file = f"pimaddpg_control_{0}.sumocfg"

sumo_cmd = [os.path.join(sumo_bin_path, "sumo-gui"), "-c", config_file, "--start",
            "--quit-on-end"]

for i in range(300):

    print(i)


    rm1_list.append([])
    rm2_list.append([])
    rm3_list.append([])
    rm4_list.append([])
    rm1_list_figure.append([])
    rm2_list_figure.append([])
    rm3_list_figure.append([])
    rm4_list_figure.append([])
    vsl1_list.append([])
    vsl2_list.append([])
    vsl3_list.append([])
    vsl4_list.append([])
    vsl1_list_figure.append([])
    vsl2_list_figure.append([])
    vsl3_list_figure.append([])
    vsl4_list_figure.append([])
    rm1_avg_quene_lenth.append([])
    rm2_avg_quene_lenth.append([])
    rm3_avg_quene_lenth.append([])
    rm4_avg_quene_lenth.append([])
    occu, flow = [], []
    avg_bn_occ.append([])
    avg_bn_veh.append([])
    state1_list,state2_list,state3_list,state4_list = [],[],[],[]
    statevsl1_list,statevsl2_list,statevsl3_list,statevsl4_list = [], [], [], []
    state1_list_memory, state2_list_memory,state3_list_memory,state4_list_memory = [], [] ,[] ,[]
    statevsl1_list_memory, statevsl2_list_memory, statevsl3_list_memory, statevsl4_list_memory = [], [], [], []
    action_agent1 = 60
    action_agent2 = 60
    action_agent3 = 60
    action_agent4 = 60
    action_agent5 = 30
    action_agent6 = 30
    action_agent7 = 30
    action_agent8 = 30
    output1 = 1
    output2 = 1
    output3 = 1
    output4 = 1
    rm1_list[i].append(action_agent1)
    rm2_list[i].append(action_agent2)
    rm3_list[i].append(action_agent3)
    rm4_list[i].append(action_agent4)
    rm1_list_figure[i].append(action_agent1)
    rm2_list_figure[i].append(action_agent2)
    rm3_list_figure[i].append(action_agent3)
    rm4_list_figure[i].append(action_agent4)
    vsl1_list[i].append(action_agent5)
    vsl2_list[i].append(action_agent6)
    vsl3_list[i].append(action_agent7)
    vsl4_list[i].append(action_agent8)
    vsl1_list_figure[i].append(action_agent5)
    vsl2_list_figure[i].append(action_agent6)
    vsl3_list_figure[i].append(action_agent7)
    vsl4_list_figure[i].append(action_agent8)

    ep_r = np.array([0])
    reward_list = []

    flow_list = []
    queue1_length_list = []
    queue2_length_list = []
    queue3_length_list = []
    queue4_length_list = []
    down_list_1 = []
    up_list_1 = []
    down_list_2 = []
    up_list_2 = []
    down_list_3 = []
    up_list_3 = []
    down_list_4 = []
    up_list_4 = []
    speed_list_1 = []
    speed_list_2 = []
    speed_list_3 = []
    speed_list_4 = []

    inflow_list = []
    outflow_list = []

    traci.start(sumo_cmd)
    time.sleep(3)
    loop_ids = traci.inductionloop.getIDList()

    noise_rate = 0.3
    step = 1
    simulation_time = 10800
    current_period_vehicles = set()
    index_memory = 0
    period_counter = 0
    while(step<=simulation_time):

        traci.simulationStep()

        if (step > period_start):

            downstream_1 = dete1_list[:2]
            ramp_1 = [dete1_list[2]]
            upstream_1 = dete1_list[3:]

            avg_occ_1 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_1) / len(
                    downstream_1),
                correctValue(traci.inductionloop.getLastStepOccupancy(dete1_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_1) / len(upstream_1),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_1),
                correctValue(traci.inductionloop.getLastStepVehicleNumber(dete1_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_1)]
            state1_list.append(avg_occ_1)

            vsl_avg_occ_1 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_1) / len(
                    downstream_1),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_1) / len(upstream_1),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_1),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_1)]
            statevsl1_list.append(vsl_avg_occ_1)

            downstream_2 = dete2_list[:2]
            ramp_2 = [dete2_list[2]]
            upstream_2 = dete2_list[3:]
            avg_occ_2 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_2) / len(
                    downstream_2),
                correctValue(traci.inductionloop.getLastStepOccupancy(dete2_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_2) / len(upstream_2),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_2),
                correctValue(traci.inductionloop.getLastStepVehicleNumber(dete2_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_2)]
            state2_list.append(avg_occ_2)

            vsl_avg_occ_2 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_2) / len(
                    downstream_2),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_2) / len(upstream_2),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_2),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_2)]
            statevsl2_list.append(vsl_avg_occ_2)

            downstream_3 = dete3_list[:2]
            ramp_3 = [dete3_list[2]]
            upstream_3 = dete3_list[3:]
            avg_occ_3 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_3) / len(
                    downstream_3),
                correctValue(traci.inductionloop.getLastStepOccupancy(dete3_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_3) / len(upstream_3),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_3),
                correctValue(traci.inductionloop.getLastStepVehicleNumber(dete3_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_3)]
            state3_list.append(avg_occ_3)

            vsl_avg_occ_3 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_3) / len(
                    downstream_3),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_3) / len(upstream_3),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_3),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_3)]
            statevsl3_list.append(vsl_avg_occ_3)

            downstream_4 = dete4_list[:2]
            ramp_4 = [dete4_list[2]]
            upstream_4 = dete4_list[3:]
            avg_occ_4 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_4) / len(
                    downstream_4),
                correctValue(traci.inductionloop.getLastStepOccupancy(dete4_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_4) / len(upstream_4),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_4),
                correctValue(traci.inductionloop.getLastStepVehicleNumber(dete4_list[2])),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_4)]
            state4_list.append(avg_occ_4)

            vsl_avg_occ_4 = [
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in downstream_4) / len(
                    downstream_4),
                sum(correctValue(traci.inductionloop.getLastStepOccupancy(x)) for x in upstream_4) / len(upstream_4),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in downstream_4),
                sum(correctValue(traci.inductionloop.getLastStepVehicleNumber(x)) for x in upstream_4)]
            statevsl4_list.append(vsl_avg_occ_4)

            reward_agent1_new_quene_0 = traci.lanearea.getJamLengthMeters("e1ramp_0")
            reward_agent1_new_quene_1 = traci.lanearea.getJamLengthMeters("e1ramp_1")
            reward_agent1_new_quene = reward_agent1_new_quene_0 + reward_agent1_new_quene_1
            queue1_length_list.append(reward_agent1_new_quene)

            reward_agent2_new_quene_0 = traci.lanearea.getJamLengthMeters("e2ramp_0")
            reward_agent2_new_quene_1 = traci.lanearea.getJamLengthMeters("e2ramp_1")
            reward_agent2_new_quene = reward_agent2_new_quene_0 + reward_agent2_new_quene_1
            queue2_length_list.append(reward_agent2_new_quene)

            reward_agent3_new_quene_0 = traci.lanearea.getJamLengthMeters("e3ramp_0")
            reward_agent3_new_quene_1 = traci.lanearea.getJamLengthMeters("e3ramp_1")
            reward_agent3_new_quene = reward_agent3_new_quene_0 + reward_agent3_new_quene_1
            queue3_length_list.append(reward_agent3_new_quene)

            reward_agent4_new_quene_0 = traci.lanearea.getJamLengthMeters("e4ramp_0")
            reward_agent4_new_quene_1 = traci.lanearea.getJamLengthMeters("e4ramp_1")
            reward_agent4_new_quene_2 = traci.lanearea.getJamLengthMeters("e4ramp_2")
            reward_agent4_new_quene = reward_agent4_new_quene_0 + reward_agent4_new_quene_1 + reward_agent4_new_quene_2
            queue4_length_list.append(reward_agent4_new_quene)

            down_list_1.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_down[:2]]))
            up_list_1.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_up[:2]]))
            down_list_2.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_down[2:4]]))
            up_list_2.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_up[2:4]]))
            down_list_3.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_down[4:6]]))
            up_list_3.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_up[4:6]]))
            down_list_4.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_down[6:8]]))
            up_list_4.append(sum([len(traci.inductionloop.getLastStepVehicleIDs(x)) for x in dete_list_up[6:8]]))
            speed_list_1.append(sum([speed if speed != -1 else 40 for speed in (traci.inductionloop.getLastStepMeanSpeed(x) for x in dete_list_up[:2])]) / len(dete_list_up[:2]))
            speed_list_2.append(sum([speed if speed != -1 else 40 for speed in (traci.inductionloop.getLastStepMeanSpeed(x) for x in dete_list_up[2:4])]) / len(dete_list_up[2:4]))
            speed_list_3.append(sum([speed if speed != -1 else 40 for speed in (traci.inductionloop.getLastStepMeanSpeed(x) for x in dete_list_up[4:6])]) / len(dete_list_up[4:6]))
            speed_list_4.append(sum([speed if speed != -1 else 40 for speed in (traci.inductionloop.getLastStepMeanSpeed(x) for x in dete_list_up[6:8])]) / len(dete_list_up[6:8]))

            if (step % control_cycle_rm) == 0:

                reward_queue_1 = sum(queue1_length_list[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                reward_queue_2 = sum(queue2_length_list[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                reward_queue_3 = sum(queue3_length_list[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                reward_queue_4 = sum(queue4_length_list[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                rm1_avg_quene_lenth[i].append(reward_queue_1)
                rm2_avg_quene_lenth[i].append(reward_queue_2)
                rm3_avg_quene_lenth[i].append(reward_queue_3)
                rm4_avg_quene_lenth[i].append(reward_queue_4)

                reword_down_1 = sum(down_list_1[step - control_cycle_rm - period_start:step - period_start])
                reword_up_1 = sum(up_list_1[step - control_cycle_rm - period_start:step - period_start])
                reward_agent_new_flow_1 = (reword_down_1 - reword_up_1) / control_cycle_rm
                reword_down_2 = sum(down_list_2[step - control_cycle_rm - period_start:step - period_start])
                reword_up_2 = sum(up_list_2[step - control_cycle_rm - period_start:step - period_start])
                reward_agent_new_flow_2 = (reword_down_2 - reword_up_2) / control_cycle_rm
                reword_down_3 = sum(down_list_3[step - control_cycle_rm - period_start:step - period_start])
                reword_up_3 = sum(up_list_3[step - control_cycle_rm - period_start:step - period_start])
                reward_agent_new_flow_3 = (reword_down_3 - reword_up_3 ) / control_cycle_rm
                reword_down_4 = sum(down_list_4[step - control_cycle_rm - period_start:step - period_start])
                reword_up_4 = sum(up_list_4[step - control_cycle_rm - period_start:step - period_start])
                reward_agent_new_flow_4 = (reword_down_4 - reword_up_4) / control_cycle_rm

                reward_speed_1 = sum(speed_list_1[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                reward_speed_2 = sum(speed_list_2[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                reward_speed_3 = sum(speed_list_3[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm
                reward_speed_4 = sum(speed_list_4[step - control_cycle_rm - period_start:step - period_start]) / control_cycle_rm

                reward_agent_new = (0.4 * (reward_agent_new_flow_1 + reward_agent_new_flow_2 + reward_agent_new_flow_3 + reward_agent_new_flow_4)
                                    + 0.4 * (reward_speed_1 + reward_speed_2 + reward_speed_3 + reward_speed_4)
                                    - 0.2 * (reward_queue_1 + reward_queue_2 + reward_queue_3 + reward_queue_4))
                reward_list.append([reward_agent_new])
                ep_r = ep_r + np.array([reward_agent_new])

                total_cycle_data_rm1 = [x for x in state1_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_rm1 = [round(x, 2) for x in list(np.array(total_cycle_data_rm1).mean(axis=0))]
                state_rm1 = state_rm1
                state1_list_memory.append(state_rm1)
                output1 = agent1_ddpg.action(np.array([state_rm1]), sess)
                rm1_list[i].append(output1[0][0])
                output1 = output1[0][0] + np.random.randn(1) * noise_rate
                if output1 > 1:
                    output1 = 1
                elif output1 < 0:
                    output1 = 0
                action_agent1 = int(output1 * 60)
                rm1_list_figure[i].append(action_agent1)
                traci.trafficlight.setProgram("J1", "0")
                current_duration = traci.trafficlight.getAllProgramLogics("J1")
                current_duration[0].phases[0].duration =  action_agent1
                current_duration[0].phases[1].duration = 60 - action_agent1
                traci.trafficlight.setProgramLogic("J1", current_duration[0])

                total_cycle_data_rm2 = [x for x in state2_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_rm2 = [round(x, 2) for x in list(np.array(total_cycle_data_rm2).mean(axis=0))]
                state_rm2 = state_rm2
                state2_list_memory.append(state_rm2)
                output2 = agent2_ddpg.action(np.array([state_rm2]), sess)
                rm2_list[i].append(output2[0][0])
                output2 = output2[0][0] + np.random.randn(1) * noise_rate
                if output2 > 1:
                    output2 = 1
                elif output2 < 0:
                    output2 = 0
                action_agent2 = int(output2 * 60)
                rm2_list_figure[i].append(action_agent2)
                traci.trafficlight.setProgram("J2", "0")
                current_duration = traci.trafficlight.getAllProgramLogics("J2")
                current_duration[0].phases[0].duration = action_agent2
                current_duration[0].phases[1].duration = 60 - action_agent2
                traci.trafficlight.setProgramLogic("J2", current_duration[0])

                total_cycle_data_rm3 = [x for x in state3_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_rm3 = [round(x, 2) for x in list(np.array(total_cycle_data_rm3).mean(axis=0))]
                state_rm3 = state_rm3
                state3_list_memory.append(state_rm3)
                output3 = agent3_ddpg.action(np.array([state_rm3]), sess)
                rm3_list[i].append(output3[0][0])
                output3 = output3[0][0] + np.random.randn(1) * noise_rate

                if output3 > 1:
                    output3 = 1
                elif output3 < 0:
                    output3 = 0
                action_agent3 = int(output3 * 60)
                rm3_list_figure[i].append(action_agent3)
                traci.trafficlight.setProgram("J3", "0")
                current_duration = traci.trafficlight.getAllProgramLogics("J3")
                current_duration[0].phases[0].duration = action_agent3
                current_duration[0].phases[1].duration = 60 - action_agent3
                traci.trafficlight.setProgramLogic("J3", current_duration[0])

                total_cycle_data_rm4 = [x for x in state4_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_rm4 = [round(x, 2) for x in list(np.array(total_cycle_data_rm4).mean(axis=0))]
                state_rm4 = state_rm4
                state4_list_memory.append(state_rm4)
                output4 = agent4_ddpg.action(np.array([state_rm4]), sess)
                rm4_list[i].append(output4[0][0])
                output4 = output4[0][0] + np.random.randn(1) * noise_rate
                if output4 > 1:
                    output4 = 1
                elif output4 < 0:
                    output4 = 0
                action_agent4 = int(output4 * 60)
                rm4_list_figure[i].append(action_agent4)
                traci.trafficlight.setProgram("J4", "0")
                current_duration = traci.trafficlight.getAllProgramLogics("J4")
                current_duration[0].phases[0].duration =  action_agent4
                current_duration[0].phases[1].duration = 60 - action_agent4
                traci.trafficlight.setProgramLogic("J4", current_duration[0])

                total_cycle_data_vsl1 = [x for x in statevsl1_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_vsl1 = [round(x, 2) for x in list(np.array(total_cycle_data_vsl1).mean(axis=0))] + [0.0] * 2
                statevsl1_list_memory.append(state_vsl1)
                outputvsl1 = agent5_ddpg.action(np.array([state_vsl1]), sess)
                vsl1_list[i].append(outputvsl1[0][0])
                outputvsl1 = outputvsl1 + np.random.randn(1) * noise_rate
                if outputvsl1 > 1:
                    outputvsl1 = [[1]]
                elif outputvsl1 < 0:
                    outputvsl1 = [[0]]
                action_agent5 = [int(round(outputvsl1[0][0] * M, 0))]
                vsl1_values = [v0 + action_agent5[0] * I]
                traci.lane.setMaxSpeed("-3_0", vsl1_values[0])
                traci.lane.setMaxSpeed("-3_1", vsl1_values[0])
                vsl1_list_figure[i].append(vsl1_values[0])

                total_cycle_data_vsl2 = [x for x in statevsl2_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_vsl2 = [round(x, 2) for x in list(np.array(total_cycle_data_vsl2).mean(axis=0))] + [0.0] * 2
                statevsl2_list_memory.append(state_vsl2)
                outputvsl2 = agent6_ddpg.action(np.array([state_vsl2]), sess)
                vsl2_list[i].append(outputvsl2[0][0])
                outputvsl2 = outputvsl2 + np.random.randn(1) * noise_rate
                if outputvsl2 > 1:
                    outputvsl2 = [[1]]
                elif outputvsl2 < 0:
                    outputvsl2 = [[0]]
                action_agent6 = [int(round(outputvsl2[0][0] * M, 0))]
                vsl2_values = [v0 + action_agent6[0] * I]
                traci.lane.setMaxSpeed("16_0", vsl2_values[0])
                traci.lane.setMaxSpeed("16_1", vsl2_values[0])
                vsl2_list_figure[i].append(vsl2_values[0])

                total_cycle_data_vsl3 = [x for x in statevsl3_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_vsl3 = [round(x, 2) for x in list(np.array(total_cycle_data_vsl3).mean(axis=0))] + [0.0] * 2
                statevsl3_list_memory.append(state_vsl3)
                outputvsl3 = agent7_ddpg.action(np.array([state_vsl3]), sess)
                vsl3_list[i].append(outputvsl3[0][0])
                outputvsl3 = outputvsl3 + np.random.randn(1) * noise_rate
                if outputvsl3 > 1:
                    outputvsl3 = [[1]]
                elif outputvsl3 < 0:
                    outputvsl3 = [[0]]
                action_agent7 = [int(round(outputvsl3[0][0] * M, 0))]
                vsl3_values = [v0 + action_agent7[0] * I]
                traci.lane.setMaxSpeed("35_0", vsl3_values[0])
                traci.lane.setMaxSpeed("35_1", vsl3_values[0])
                vsl3_list_figure[i].append(vsl3_values[0])

                total_cycle_data_vsl4 = [x for x in statevsl4_list[(step - period_start - control_cycle_rm):(step - period_start)]]
                state_vsl4 = [round(x, 2) for x in list(np.array(total_cycle_data_vsl4).mean(axis=0))] + [0.0] * 2
                statevsl4_list_memory.append(state_vsl4)
                outputvsl4 = agent8_ddpg.action(np.array([state_vsl4]), sess) + np.random.randn(1) * noise_rate
                vsl4_list[i].append(outputvsl4[0][0])
                outputvsl4 = outputvsl4 + np.random.randn(1) * noise_rate
                if outputvsl4 > 1:
                    outputvsl4 = [[1]]
                elif outputvsl4 < 0:
                    outputvsl4 = [[0]]
                action_agent8 = [int(round(outputvsl4[0][0] * M, 0))]
                vsl4_values = [v0 + action_agent8[0] * I]
                traci.lane.setMaxSpeed("48_0", vsl4_values[0])
                traci.lane.setMaxSpeed("48_1", vsl4_values[0])
                vsl4_list_figure[i].append(vsl4_values[0])

                duration_last = [action_agent1, action_agent2, action_agent3, action_agent4]
                speed_last = [vsl1_values[0], vsl2_values[0],vsl3_values[0],vsl4_values[0]]
                history = detector_merger.update(loop_ids, duration_last,speed_last)

                if step >= 60:
                    outputs, fd_q = wm.update(history, duration_last, speed_last)  # 调用PhysSTGNN环境模型
                    agents_data_1 = outputs.reshape(4, 3, 2)
                    result_1 = np.zeros((4, 6))
                    result_1_vsl = np.zeros((4, 4))
                    for l in range(4):
                        result_1[l, 0] = agents_data_1[l, 0, 1]  # 下游车辆数
                        result_1_vsl[l, 0] = agents_data_1[l, 0, 1]  # 下游车辆数
                        result_1[l, 1] = agents_data_1[l, 1, 1]  # 匝道车辆数
                        result_1_vsl[l, 1] = agents_data_1[l, 2, 1]  # 上游车辆数
                        result_1[l, 2] = agents_data_1[l, 2, 1]  # 上游车辆数
                        # result_1_vsl[l, 2] = agents_data_1[l, 2, 1]  # 上游车辆数

                        result_1[l, 3] = agents_data_1[l, 0, 0]  # 下游占有率
                        result_1_vsl[l, 2] = agents_data_1[l, 0, 0]  # 下游占有率
                        result_1[l, 4] = agents_data_1[l, 1, 0]  # 匝道占有率
                        # result_1_vsl[l, 4] = 0  # 匝道占有率
                        result_1[l, 5] = agents_data_1[l, 2, 0]  # 上游占有率
                        result_1_vsl[l, 3] = agents_data_1[l, 2, 0]  # 上游占有率

                    agents_data_2 = agents_data_1[[1, 0, 2, 3]]
                    result_2 = np.zeros((4, 6))
                    result_2_vsl = np.zeros((4, 6))
                    for m in range(4):
                        result_2[m, 0] = agents_data_2[m, 0, 1]  # 下游车辆数
                        result_2_vsl[m, 0] = agents_data_2[m, 0, 1]  # 下游车辆数
                        result_2[m, 1] = agents_data_2[m, 1, 1]  # 匝道车辆数
                        result_2_vsl[m, 1] = agents_data_2[m, 2, 1]  # 上游车辆数
                        result_2[m, 2] = agents_data_2[m, 2, 1]  # 上游车辆数
                        # result_2_vsl[m, 2] = agents_data_2[m, 2, 1]  # 上游车辆数

                        result_2[m, 3] = agents_data_2[m, 0, 0]  # 下游占有率
                        result_2_vsl[m, 2] = agents_data_2[m, 0, 0]  # 下游占有率
                        result_2[m, 4] = agents_data_2[m, 1, 0]  # 匝道占有率
                        # result_2_vsl[m, 4] = 0  # 匝道占有率
                        result_2[m, 5] = agents_data_2[m, 2, 0]  # 上游占有率
                        result_2_vsl[m, 3] = agents_data_2[m, 2, 0]  # 上游占有率

                    agents_data_3 = agents_data_1[[2, 0, 1, 3]]
                    result_3 = np.zeros((4, 6))
                    result_3_vsl = np.zeros((4, 6))
                    for n in range(4):
                        result_3[n, 0] = agents_data_3[n, 0, 1]  # 下游车辆数
                        result_3_vsl[n, 0] = agents_data_3[n, 0, 1]  # 下游车辆数
                        result_3[n, 1] = agents_data_3[n, 1, 1]  # 匝道车辆数
                        result_3_vsl[n, 1] = agents_data_3[n, 2, 1]  # 上游车辆数
                        result_3[n, 2] = agents_data_3[n, 2, 1]  # 上游车辆数
                        # result_3_vsl[n, 2] = agents_data_3[n, 2, 1]  # 上游车辆数

                        result_3[n, 3] = agents_data_3[n, 0, 0]  # 下游占有率
                        result_3_vsl[n, 2] = agents_data_3[n, 0, 0]  # 下游占有率
                        result_3[n, 4] = agents_data_3[n, 1, 0]  # 匝道占有率
                        # result_3_vsl[n, 4] = 0  # 匝道占有率
                        result_3[n, 5] = agents_data_3[n, 2, 0]  # 上游占有率
                        result_3_vsl[n, 3] = agents_data_3[n, 2, 0]  # 上游占有率

                    agents_data_4 = agents_data_1[[3, 0, 1, 2]]
                    result_4 = np.zeros((4, 6))
                    result_4_vsl = np.zeros((4, 6))
                    for p in range(4):
                        result_4[p, 0] = agents_data_4[p, 0, 1]  # 下游车辆数
                        result_4_vsl[p, 0] = agents_data_4[p, 0, 1]  # 下游车辆数
                        result_4[p, 1] = agents_data_4[p, 1, 1]  # 匝道车辆数
                        result_4_vsl[p, 1] = agents_data_4[p, 2, 1]  # 上游车辆数
                        result_4[p, 2] = agents_data_4[p, 2, 1]  # 上游车辆数
                        # result_4_vsl[p, 2] = agents_data_4[p, 2, 1]  # 上游车辆数

                        result_4[p, 3] = agents_data_4[p, 0, 0]  # 下游占有率
                        result_4_vsl[p, 2] = agents_data_4[p, 0, 0]  # 下游占有率
                        result_4[p, 4] = agents_data_4[p, 1, 0]  # 匝道占有率
                        # result_4_vsl[p, 4] = 0  # 匝道占有率
                        result_4[p, 5] = agents_data_4[p, 2, 0]  # 上游占有率
                        result_4_vsl[p, 3] = agents_data_4[p, 2, 0]  # 上游占有率

            index = index_memory - period_start
            if index >= 0:
            #    ----------------------------------------------  预测的经验池  -------------------------------------------------------------
                if (step + 1) % 60 == 0 and step != 59: # 60预测了120步的状态，因此需要119步时候的状态
                    print('result_1[index]', result_1[index], 'index', index)
                    print('result_1_vsl[index]',result_1_vsl[index],'index',index)
                    agent1_memory.add(np.vstack(
                        [state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],
                         state4_list_memory[index - 1], statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1], rm4_list[i][index - 1],
                         vsl1_list[i][index - 1], vsl2_list[i][index - 1], vsl3_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_1[index], result_2[index], result_3[index], result_4[index], result_1_vsl[index],
                         result_2_vsl[index], result_3_vsl[index], result_4_vsl[index]]), False)

                    agent2_memory.add(np.vstack(
                        [state2_list_memory[index - 1], state1_list_memory[index - 1], state3_list_memory[index - 1],
                         state4_list_memory[index - 1], statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [rm2_list[i][index - 1], rm1_list[i][index - 1], rm3_list[i][index - 1], rm4_list[i][index - 1],
                         vsl1_list[i][index - 1], vsl2_list[i][index - 1], vsl3_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_2[index], result_1[index], result_3[index], result_4[index], result_1_vsl[index],
                         result_2_vsl[index], result_3_vsl[index], result_4_vsl[index]]), False)

                    agent3_memory.add(np.vstack(
                        [state3_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1],
                         state4_list_memory[index - 1], statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [rm3_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm4_list[i][index - 1],
                         vsl1_list[i][index - 1], vsl2_list[i][index - 1], vsl3_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_3[index], result_1[index], result_2[index], result_4[index], result_1_vsl[index],
                         result_2_vsl[index], result_3_vsl[index], result_4_vsl[index]]), False)

                    agent4_memory.add(np.vstack(
                        [state4_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1],
                         state3_list_memory[index - 1], statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [rm4_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                         vsl1_list[i][index - 1], vsl2_list[i][index - 1], vsl3_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_4[index], result_1[index], result_2[index], result_3[index], result_1_vsl[index],
                         result_2_vsl[index], result_3_vsl[index], result_4_vsl[index]]), False)

                    agent5_memory.add(np.vstack(
                        [statevsl1_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1],
                         state3_list_memory[index - 1], state4_list_memory[index - 1], statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [vsl1_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                         rm4_list[i][index - 1], vsl2_list[i][index - 1], vsl3_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_1_vsl[index], result_1[index], result_2[index], result_3[index], result_4[index],
                         result_2_vsl[index], result_3_vsl[index], result_4_vsl[index]]), False)

                    agent6_memory.add(np.vstack(
                        [statevsl2_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1],
                         state3_list_memory[index - 1], state4_list_memory[index - 1], statevsl1_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [vsl2_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                         rm4_list[i][index - 1], vsl1_list[i][index - 1], vsl3_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_2_vsl[index], result_1[index], result_2[index], result_3[index], result_4[index],
                         result_1_vsl[index], result_3_vsl[index], result_4_vsl[index]]), False)

                    agent7_memory.add(np.vstack(
                        [statevsl3_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1],
                         state3_list_memory[index - 1], state4_list_memory[index - 1], statevsl1_list_memory[index - 1],
                         statevsl2_list_memory[index - 1], statevsl4_list_memory[index - 1]]), np.vstack(
                        [vsl3_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                         rm4_list[i][index - 1], vsl1_list[i][index - 1], vsl2_list[i][index - 1],
                         vsl4_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_3_vsl[index], result_1[index], result_2[index], result_3[index], result_4[index],
                         result_1_vsl[index], result_2_vsl[index], result_4_vsl[index]]), False)

                    agent8_memory.add(np.vstack(
                        [statevsl4_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1],
                         state3_list_memory[index - 1], state4_list_memory[index - 1], statevsl1_list_memory[index - 1],
                         statevsl2_list_memory[index - 1], statevsl3_list_memory[index - 1]]), np.vstack(
                        [vsl4_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                         rm4_list[i][index - 1], vsl1_list[i][index - 1], vsl2_list[i][index - 1],
                         vsl3_list[i][index - 1]]), reward_agent_new, np.vstack(
                        [result_4_vsl[index], result_1[index], result_2[index], result_3[index], result_4[index],
                         result_1_vsl[index], result_2_vsl[index], result_3_vsl[index]]), False)

                    # ---------------------------------------------- 仿真的经验池 ---------------------------------------------------------------
                    agent1_memory.add(np.vstack([state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],state4_list_memory[index - 1],
                                                 statevsl1_list_memory[index - 1] , statevsl2_list_memory[index - 1],statevsl3_list_memory[index - 1],statevsl4_list_memory[index - 1]]),
                                      np.vstack([rm1_list[i][index-1], rm2_list[i][index-1], rm3_list[i][index-1],rm4_list[i][index-1],
                                                 vsl1_list[i][index - 1],vsl2_list[i][index - 1],vsl3_list[i][index - 1],vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([state1_list_memory[index], state2_list_memory[index], state3_list_memory[index], state4_list_memory[index],
                                                 statevsl1_list_memory[index], statevsl2_list_memory[index],statevsl3_list_memory[index],
                                                 statevsl4_list_memory[index]]), False)

                    agent2_memory.add(np.vstack([state2_list_memory[index - 1], state1_list_memory[index - 1], state3_list_memory[index - 1],state4_list_memory[index - 1],
                                                 statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],statevsl3_list_memory[index - 1],
                                                 statevsl4_list_memory[index - 1]]),
                                      np.vstack([rm2_list[i][index - 1], rm1_list[i][index - 1], rm3_list[i][index - 1],rm4_list[i][index - 1],
                                                 vsl1_list[i][index - 1],vsl2_list[i][index - 1],vsl3_list[i][index - 1],vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([state2_list_memory[index], state1_list_memory[index], state3_list_memory[index], state4_list_memory[index],
                                                 statevsl1_list_memory[index], statevsl2_list_memory[index],statevsl3_list_memory[index],
                                                 statevsl4_list_memory[index]]), False)

                    agent3_memory.add(np.vstack([state3_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1], state4_list_memory[index - 1],
                                                 statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],statevsl3_list_memory[index - 1],
                                                 statevsl4_list_memory[index - 1]]),
                                      np.vstack([rm3_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1],rm4_list[i][index - 1],
                                                 vsl1_list[i][index - 1],vsl2_list[i][index - 1],vsl3_list[i][index - 1],vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([state3_list_memory[index], state1_list_memory[index], state2_list_memory[index], state4_list_memory[index],
                                                 statevsl1_list_memory[index], statevsl2_list_memory[index],statevsl3_list_memory[index],
                                                 statevsl4_list_memory[index]]),False)

                    agent4_memory.add(np.vstack([state4_list_memory[index - 1], state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],
                                                 statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],statevsl3_list_memory[index - 1],
                                                 statevsl4_list_memory[index - 1]]),
                                      np.vstack([rm4_list[i][index - 1], rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                                                 vsl1_list[i][index - 1],vsl2_list[i][index - 1],vsl3_list[i][index - 1],vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([state4_list_memory[index], state1_list_memory[index], state2_list_memory[index], state3_list_memory[index],
                                                 statevsl1_list_memory[index], statevsl2_list_memory[index],statevsl3_list_memory[index],
                                                 statevsl4_list_memory[index]]),False)

                    agent5_memory.add(np.vstack(
                        [statevsl1_list_memory[index - 1],state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],
                         state4_list_memory[index - 1],
                         statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]),
                                      np.vstack([vsl1_list[i][index - 1],rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                                                 rm4_list[i][index - 1],
                                                 vsl2_list[i][index - 1],
                                                 vsl3_list[i][index - 1], vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([statevsl1_list_memory[index],state1_list_memory[index], state2_list_memory[index],
                                                 state3_list_memory[index], state4_list_memory[index],
                                                 statevsl2_list_memory[index],
                                                 statevsl3_list_memory[index], statevsl4_list_memory[index]]), False)

                    agent6_memory.add(np.vstack(
                        [statevsl2_list_memory[index - 1],state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],
                         state4_list_memory[index - 1],
                         statevsl1_list_memory[index - 1],
                         statevsl3_list_memory[index - 1], statevsl4_list_memory[index - 1]]),
                                      np.vstack([vsl2_list[i][index - 1],rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                                                 rm4_list[i][index - 1],
                                                 vsl1_list[i][index - 1],
                                                 vsl3_list[i][index - 1], vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([statevsl2_list_memory[index],state1_list_memory[index], state2_list_memory[index],
                                                 state3_list_memory[index], state4_list_memory[index],
                                                 statevsl1_list_memory[index],
                                                 statevsl3_list_memory[index], statevsl4_list_memory[index]]), False)

                    agent7_memory.add(np.vstack(
                        [statevsl3_list_memory[index - 1],state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],
                         state4_list_memory[index - 1],
                         statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],
                          statevsl4_list_memory[index - 1]]),
                                      np.vstack([vsl3_list[i][index - 1],rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                                                 rm4_list[i][index - 1],
                                                 vsl1_list[i][index - 1], vsl2_list[i][index - 1],
                                                  vsl4_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([statevsl3_list_memory[index],state1_list_memory[index], state2_list_memory[index],
                                                 state3_list_memory[index], state4_list_memory[index],
                                                 statevsl1_list_memory[index], statevsl2_list_memory[index],
                                                  statevsl4_list_memory[index]]), False)
                    agent8_memory.add(np.vstack(
                        [statevsl4_list_memory[index - 1],state1_list_memory[index - 1], state2_list_memory[index - 1], state3_list_memory[index - 1],
                         state4_list_memory[index - 1],
                         statevsl1_list_memory[index - 1], statevsl2_list_memory[index - 1],
                         statevsl3_list_memory[index - 1] ]),
                                      np.vstack([vsl4_list[i][index - 1],rm1_list[i][index - 1], rm2_list[i][index - 1], rm3_list[i][index - 1],
                                                 rm4_list[i][index - 1],
                                                 vsl1_list[i][index - 1], vsl2_list[i][index - 1],
                                                 vsl3_list[i][index - 1]]),
                                      reward_agent_new,
                                      np.vstack([statevsl4_list_memory[index],state1_list_memory[index], state2_list_memory[index],
                                                 state3_list_memory[index], state4_list_memory[index],
                                                 statevsl1_list_memory[index], statevsl2_list_memory[index],
                                                 statevsl3_list_memory[index]]), False)
                    index_memory += 1

        step += 1
    traci.close()

    if i >=100:
        train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                    agent1_critic_target_update, sess, agent2_ddpg_target,agent3_ddpg_target,agent4_ddpg_target,
                    agent5_ddpg_target,agent6_ddpg_target,agent7_ddpg_target,agent8_ddpg_target)
        train_agent(agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                    agent2_critic_target_update, sess, agent1_ddpg_target,agent3_ddpg_target,agent4_ddpg_target,
                    agent5_ddpg_target,agent6_ddpg_target,agent7_ddpg_target,agent8_ddpg_target)
        train_agent(agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                    agent3_critic_target_update, sess, agent1_ddpg_target, agent2_ddpg_target, agent4_ddpg_target,
                    agent5_ddpg_target,agent6_ddpg_target,agent7_ddpg_target,agent8_ddpg_target)
        train_agent(agent4_ddpg, agent4_ddpg_target, agent4_memory, agent4_actor_target_update,
                    agent4_critic_target_update, sess, agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target,
                    agent5_ddpg_target,agent6_ddpg_target,agent7_ddpg_target,agent8_ddpg_target)
        train_agent(agent5_ddpg, agent5_ddpg_target, agent5_memory, agent5_actor_target_update,
                    agent5_critic_target_update, sess, agent1_ddpg_target,agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target,
                     agent6_ddpg_target, agent7_ddpg_target, agent8_ddpg_target)
        train_agent(agent6_ddpg, agent6_ddpg_target, agent6_memory, agent6_actor_target_update,
                    agent6_critic_target_update, sess, agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target,
                    agent4_ddpg_target,
                    agent5_ddpg_target, agent7_ddpg_target, agent8_ddpg_target)
        train_agent(agent7_ddpg, agent7_ddpg_target, agent7_memory, agent7_actor_target_update,
                    agent7_critic_target_update, sess, agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target,
                    agent4_ddpg_target,
                    agent5_ddpg_target, agent6_ddpg_target, agent8_ddpg_target)
        train_agent(agent8_ddpg, agent8_ddpg_target, agent8_memory, agent8_actor_target_update,
                    agent8_critic_target_update, sess, agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target,
                    agent4_ddpg_target,
                    agent5_ddpg_target, agent6_ddpg_target, agent7_ddpg_target)


    tripinfo_file_path = f"output_pimaddpg_control/pimaddpg_control_tripinfo_{0}.xml"
    totalTravelTime_output, totalVehNum_output, totalTravelTime_main,totalTimeloss,totalTimeloss_main = getIntervalTravelTime(tripinfo_file_path,[period_start, 10800])
    print('totalTravelTime_output:', totalTravelTime_output, '\n totalVehNum_output:', totalVehNum_output,
          '\n totalTimeloss:',totalTimeloss, '\n totalTimeloss_main:',totalTimeloss_main)
    travel_time_epi.append(totalTravelTime_output)
    main_travel_time_epi.append(totalTravelTime_main)
    travelloss_time.append(totalTimeloss)
    main_travelloss_time.append(totalTimeloss_main)
    all_ep_r.append(ep_r)
    print("all_ep_r",all_ep_r)
sess.close()

# # ---------- 存储奖励函数 ------------------
reward_list = np.array(all_ep_r)
np.save(f"output_pimaddpg_control/reward_list.npy",reward_list)
# ---------- 存储总行程时间\主路行程时间\总延误时间\主路延误时间 ----------------
pimaddpg_control_travel_time_epi = np.array(travel_time_epi)
np.save(f'output_pimaddpg_control/pimaddpg_control_travel_time_epi.npy',pimaddpg_control_travel_time_epi)
pimaddpg_control_main_travel_time_epi = np.array(main_travel_time_epi)
np.save(f'output_pimaddpg_control/pimaddpg_control_main_travel_time_epi.npy',pimaddpg_control_main_travel_time_epi)
pimaddpg_control_travelloss_time_epi = np.array(travelloss_time)
np.save(f'output_pimaddpg_control/pimaddpg_control_travelloss_time_epi.npy',pimaddpg_control_travelloss_time_epi)
pimaddpg_control_main_travelloss_time_epi = np.array(main_travelloss_time)
np.save(f'output_pimaddpg_control/pimaddpg_control_main_travelloss_time_epi.npy',pimaddpg_control_main_travelloss_time_epi)
# # --------- 存储智能体的动作集合 ------------
rm1_action_list = np.array(rm1_list_figure)
rm2_action_list = np.array(rm2_list_figure)
rm3_action_list = np.array(rm3_list_figure)
rm4_action_list = np.array(rm4_list_figure)
np.save(f'output_pimaddpg_control/rm1_action_list.npy',rm1_action_list)
np.save(f'output_pimaddpg_control/rm2_action_list.npy',rm2_action_list)
np.save(f'output_pimaddpg_control/rm3_action_list.npy',rm3_action_list)
np.save(f'output_pimaddpg_control/rm4_action_list.npy',rm4_action_list)
# # --------- 存储智能体的动作集合 ------------
vsl1_action_list = np.array(vsl1_list_figure)
vsl2_action_list = np.array(vsl2_list_figure)
vsl3_action_list = np.array(vsl3_list_figure)
vsl4_action_list = np.array(vsl4_list_figure)
np.save(f'output_pimaddpg_control/vsl1_action_list.npy',vsl1_action_list)
np.save(f'output_pimaddpg_control/vsl2_action_list.npy',vsl2_action_list)
np.save(f'output_pimaddpg_control/vsl3_action_list.npy',vsl3_action_list)
np.save(f'output_pimaddpg_control/vsl4_action_list.npy',vsl4_action_list)
# # ------- 存储匝道排队长度 -------------------
pimaddpg_control_avg_quene_lenth1 = np.array(rm1_avg_quene_lenth)
pimaddpg_control_avg_quene_lenth2 = np.array(rm2_avg_quene_lenth)
pimaddpg_control_avg_quene_lenth3 = np.array(rm3_avg_quene_lenth)
pimaddpg_control_avg_quene_lenth4 = np.array(rm4_avg_quene_lenth)
np.save(f'output_pimaddpg_control/pimaddpg_control_avg_quene_lenth1.npy',pimaddpg_control_avg_quene_lenth1)
np.save(f'output_pimaddpg_control/pimaddpg_control_avg_quene_lenth2.npy',pimaddpg_control_avg_quene_lenth2)
np.save(f'output_pimaddpg_control/pimaddpg_control_avg_quene_lenth3.npy',pimaddpg_control_avg_quene_lenth3)
np.save(f'output_pimaddpg_control/pimaddpg_control_avg_quene_lenth4.npy',pimaddpg_control_avg_quene_lenth4)




