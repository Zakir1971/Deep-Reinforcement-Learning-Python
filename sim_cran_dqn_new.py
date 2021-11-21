import sys
sys.path.append("../Agent")
#from Agent.dqnet import DQNetwork
#from Agent.agent_dqn import per_antenna_power,DQNAgent, user_demand, total_power, average_slot_power, episode_power, episode_rewards
#from Agent.agent_duelingdqn import DDQNAgent, us_demand, ttl_power, avg_slt_pwr, ep_power
#from Agent.agent_doubledqn import DoubleDQNAgent, u_demand, t_power, a_s_p, e_power, doubledqn_episode_rewards
#from Agent.agent_doubleqn import DoubleDQNAgent, cost_his
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import math
import numpy as np
import nUserEfficiency as allModnUser
import nNodesEfficency as allModnNodes
from nUserEfficiency import user_demand, total_power,u_demand,t_power,us_demand,ttl_power
from nNodesEfficency import user_demandN, total_powerN,u_demandN,t_powerN,us_demandN,ttl_powerN
#p_watt = 1e-3 * (10. **((p_max[i] / n_rrh)/ 10))


if __name__ == '__main__':
    # note n_usr<=n_rrh
    #to get banwidth
    '''parser = CRANParser()
    config = parser.parse_args()
    env = Env('master', config)
    Bandwidth=env._BAND'''
    Bandwidth=10000000.0

    min_power = 10 #10 watts per antenna min. power
    max_power = 60 #60 watts per antenna max. power
    #min_power = 10*n_rrh
    #max_power = 60*n_rrh
    #min_power = 1*n_rrh#0.01
    #max_power = 10000*n_rrh#1000
    n_epochs = 10
    p_min = list()#data rate
    p_max = list()

    #for plotting graph bw total power and user demand
    #-------------------------------------------------
    slot_range = int(max_power/min_power)
    #slot_range = int(np.log10(max_power/min_power)+1)
    #d_min = [min_demand-10.e6]
    #d_min.append(min_demand-10.e6)
    #d_max = [min_demand] #user demand (MBps)
    #d_max.append(max_demand-10.e6)
    for i in range(slot_range):
        p_max.append(min_power*(i+1))
        p_min.append(min_power*(i))
        #p_max.append(min_power*10**(i+1))
        #p_min.append(min_power*10**(i))

    p_max=20*np.ones(6)#per antenna max power same



    opt=input("To plot No. of users vs Efficiency, press: U\n To plot No. of nodes vs Efficeny press: N\n")
    if opt=='U' or opt=='u':

        n_rrh = 14
        n_usr = 8
        #n_userList=[2,4,6,8,10,12]
        p_max=20*np.ones(n_usr)#per antenna max power same
        n_userList=range(0,n_usr)
        #n_userList=[1,2,3,4,5,6]
        #n_userList=[1,1,1,1,1,1]

        '''total_dqn_power1 = list()
        total_ddqn_power1 = list()
        total_doubleqn_power1 = list()
        ddqn_user_demand1 = list()
        doubleqn_user_demand1 = list()
        dqn_user_demand1 = list()'''
        #args1 = p_min, p_max, n_rrh, n_userList, n_epochs

        #allMod=AllDQN(args1)
        #total_dqn_power = copy.deepcopy(total_power)
        #dqn_user_demand = copy.deepcopy(user_demand)
        #for i in range(len(n_userList)):
        ########## for plotting number of users and user demand and total power ########
        #args1 = p_min, p_max, n_rrh, n_userList,n_epochs
        args1 = p_min, p_max, n_rrh, n_usr,n_epochs
        allModnUser.DQN_Start(args1)
        allModnUser.DoubleDQN_Start(args1)
        '''allModnUser.DDQN_Start(args1)
        dqn_user_demand1.append(user_demand)
        ddqn_user_demand1.append(us_demand)
        doubleqn_user_demand1.append(u_demand)'''
        print("user_demand DQN",user_demand)
        print("total_power DQN",total_power)
        print("user_demand DoubleDQN",u_demand)
        print("total_power DoubleDQN",t_power)
        DQNEE = 100*(np.array(user_demand) / np.array(total_power))
        DQNSE = 100*(np.array(user_demand)/Bandwidth)
        DoubleDQNEE = 100*(np.array(u_demand)/np.array(t_power))
        DoubleDQNSE = 100*(np.array(u_demand)/Bandwidth)
        print("DQNEE",DQNEE)
        print("DoubleDQNEE",DoubleDQNEE)
        ################################################################
        #graph for ENERGY EFFICIENCY VS No. of Users

        fig1 = plt.figure()
        plt.plot(n_userList, DoubleDQNEE, label='Double Deep Q Network')
        #plt.plot(n_userList, DQNEE, label='Deep Q Network')
        plt.plot(n_userList, DQNEE, label='Deep Q Network')
        plt.xlabel('Number Of Users')
        plt.ylabel('Energy Efficiency')
        #plt.xlim(10, 60)
        banner = 'Energy Efficiency vs No. of users'
        plt.title(banner)
        plt.grid(True)
        plt.figtext(0.5,0.7,"n_rrh="+str(n_rrh),horizontalalignment ="center",
                    verticalalignment ="center",fontsize = 10,color ="blue")
        plt.legend(loc='best')
        fig1.savefig('EEvsN_users.png')
        plt.show()
        ################################################################
        #graph for SPECTRAL EFFICIENCY VS No. of Users

        fig1 = plt.figure()
        plt.plot(n_userList, DoubleDQNSE, label='Double Deep Q Network')
        #plt.plot(n_userList, DQNEE, label='Deep Q Network')
        plt.plot(n_userList, DQNSE, label='Deep Q Network')
        plt.xlabel('Number Of Users')
        plt.ylabel('Spectral Efficiency')
        #plt.xlim(10, 60)
        plt.figtext(0.5,0.7,"n_rrh="+str(n_rrh),horizontalalignment ="center",
                    verticalalignment ="center",fontsize = 10,color ="blue")
        banner = 'Spectral Efficiency vs No. of users'
        plt.title(banner)
        plt.grid(True)

        plt.legend(loc='best')
        fig1.savefig('SEvsN_users.png')
        plt.show()
    elif opt=='N' or opt=='n':
        n_usr=4
        n_rrhList=[6,8,10,12,14,16]
        p_max=30 #per antnna max.power in watts

        ########## for plotting number of users and user demand and total power ########
        args2 = p_min, p_max, n_rrhList, n_usr,n_epochs
        allModnNodes.DQN_Start(args2)
        allModnNodes.DoubleDQN_Start(args2)
        '''allModnNodes.DDQN_Start(args2)
        dqn_user_demand1.append(user_demand)
        ddqn_user_demand1.append(us_demand)
        doubleqn_user_demand1.append(u_demand)'''
        print("user_demand DQN",user_demandN)
        print("total_power DQN",total_powerN)
        print("user_demand DoubleDQN",u_demandN)
        print("total_power DoubleDQN",t_powerN)
        DQNEE = 100*(np.array(user_demandN) / np.array(total_powerN))
        DQNSE = 100*(np.array(user_demandN)/Bandwidth)
        DoubleDQNEE = 10*(np.array(u_demandN)/np.array(t_powerN))
        DoubleDQNSE = 100*(np.array(u_demandN)/Bandwidth)
        print("DQNEE",DQNEE)
        print("DQNSE",DQNSE)
        print("DoubleDQNEE",DoubleDQNEE)
        print("DoubleDQNSE",DoubleDQNSE)
        ################################################################
        #graph for ENERGY EFFICIENCY VS No. of Users

        fig1 = plt.figure()
        plt.plot(n_rrhList, DoubleDQNEE, label='Double Deep Q Network')
        #plt.plot(n_userList, DQNEE, label='Deep Q Network')
        plt.plot(n_rrhList, DQNEE, label='Deep Q Network')
        plt.xlabel('Number Of Nodes')
        plt.ylabel('Energy Efficiency')
        #plt.xlim(10, 60)
        banner = 'Energy Efficiency vs No. of Nodes'
        plt.title(banner)
        plt.grid(True)
        plt.figtext(0.5,0.8,"Number of Users:"+str(n_usr)+", Per antenna max. power:"+str(p_max),horizontalalignment ="center",
                    wrap=True,verticalalignment ="center",fontsize = 10,color ="blue")
        plt.legend(loc='upper right')
        fig1.savefig('EEvsN_Nodes.png')
        plt.show()
        ################################################################
        #graph for SPECTRAL EFFICIENCY VS No. of Users

        fig1 = plt.figure()
        plt.plot(n_rrhList, DoubleDQNSE, label='Double Deep Q Network')
        #plt.plot(n_userList, DQNEE, label='Deep Q Network')
        plt.plot(n_rrhList, DQNSE, label='Deep Q Network')
        plt.xlabel('Number Of Nodes')
        plt.ylabel('Spectral Efficiency')
        #plt.xlim(10, 60)
        plt.figtext(0.5,0.7,"Number of Users:"+str(n_usr)+", per antenna max. power:"+str(p_max),horizontalalignment ="center",
                    wrap=True,verticalalignment ="center",fontsize = 10,color ="blue")
        banner = 'Spectral Efficiency vs No. of Nodes'
        plt.title(banner)
        plt.grid(True)

        plt.legend(loc='upper right')
        fig1.savefig('SEvsN_Nodes.png')
        plt.show()

