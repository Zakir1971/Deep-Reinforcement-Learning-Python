import sys
sys.path.append("../Agent")
from Agent.dqnet import DQNetwork
from Agent.agent_dqn import DQNAgent, user_demand, total_power, average_slot_power, episode_power
from Agent.agent_duelingdqn import DDQNAgent, us_demand, ttl_power, avg_slt_pwr, ep_power
from Agent.agent_doubledqn import DoubleDQNAgent, u_demand, t_power, a_s_p, e_power
 #from Agent.agent_doubleqn import DoubleDQNAgent, cost_his
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import numpy as np
 
 
def DQN_Start(args):    
     d_min, d_max, n_rrh, n_usr, n_epochs = args
     print("DQNStart")
     for i in range(len(d_min)):
         
         parser = CRANParser()
         parser.set_defaults(demand_min=d_min[i], demand_max=d_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
         config = parser.parse_args()
        # print("Eenie")
         env = Env('master', config)
        # print("Meenie")
         #gains = env._get_gains(n_rrh, n_usr) 
         #print (gains)
         agent = DQNAgent(env, config)       
        # print("Minie")                      
         tf.reset_default_graph()              
         agent.work()
         agent.save()
         print("DQNEND")
 
def DDQN_Start(args):    
     d_min, d_max, n_rrh, n_usr, n_epochs = args
     print("DDQNStart")
     for i in range(len(d_min)):
         
         parser = CRANParser()
         parser.set_defaults(demand_min=d_min[i], demand_max=d_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
         config = parser.parse_args()
       #  print("Eenie")
         env = Env('master', config)
      #   print("Meenie")
         #gains = env._get_gains(n_rrh, n_usr) 
         #print (gains)
         agent = DDQNAgent(env, config)       
       #  print("Minie")                      
         tf.reset_default_graph()              
         agent.work()
         agent.save()
         print("DDQNEND")
         
def DoubleQN_Start(args):    
     d_min, d_max, n_rrh, n_usr, n_epochs = args
     print("DoubleDQNStart")
     for i in range(len(d_min)):        
         parser = CRANParser()
         parser.set_defaults(demand_min=d_min[i], demand_max=d_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
         config = parser.parse_args()
        # print("Eenie")
         env = Env('master', config)
       #  print("Meenie")
         #gains = env._get_gains(n_rrh, n_usr) 
         #print (gains)
         agent = DoubleDQNAgent(env, config) 
        # print("Minie")                      
         tf.reset_default_graph()              
         agent.work()
         agent.save()
         print("DoubleDQNEND")
 
if __name__ == '__main__':
     
     min_demand = 10.e6# user demand in bps
     max_demand = 60.e6
     n_rrh = 8
     n_usr = 4
     n_epochs = 10
     d_min = list()
     d_max = list()  
     parser = CRANParser()
     parser.set_defaults(demand_min=min_demand, demand_max=max_demand, num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
     config = parser.parse_args()
     env = Env('master', config)
     Bandwidth=env._BAND
     #for plotting graph bw total power and user demand
     #-------------------------------------------------
     slot_range = int(max_demand/10.e6)-1 
     #d_min = [min_demand-10.e6]
     d_min.append(min_demand-10.e6)
     #d_max = [min_demand]
     d_max.append(min_demand)
     for i in range(slot_range):
         d_max.append(d_max[i]+10.e6)
         d_min.append(d_min[i]+10.e6)
         
 #    print(len(d_min))
 #    print("KKK")
     args = d_min, d_max, n_rrh, n_usr, n_epochs
     total_ddqn_power = list()
     ddqn_user_demand = list()
     total_doubleqn_power = list()
     doubleqn_user_demand = list()
     total_dqn_power = list()
     dqn_user_demand = list()
     
     DQN_Start(args)    
     total_dqn_power = copy.deepcopy(total_power)
     dqn_user_demand = copy.deepcopy(user_demand)
    
     DDQN_Start(args)
     total_ddqn_power = copy.deepcopy(ttl_power)
    #print(total_ddqn_power)
     ddqn_user_demand = copy.deepcopy(us_demand)

     DoubleQN_Start(args)
     total_doubleqn_power = copy.deepcopy(t_power)
     doubleqn_user_demand = copy.deepcopy(u_demand)
     #-------------------------------------------------
     
     
     #for plotting graph bw average power and time
     #-------------------------------------------------
     #d_min = [min_demand]
     #d_max = [max_demand]
     #args = d_min, d_max, n_rrh, n_usr, n_epochs 
     
     
     #DQN_Start(args)
     #DDQN_Start(args)
     #DoubleQN_Start(args)
     #-------------------------------------------------
     
     
     #graph bw total power and user demand
     #-------------------------------------------------
 #    print(total_ddqn_power)
 #    print(ddqn_user_demand)
     dqn_user_list = list()
     dqn_power_list = list()
     ddqn_user_list = list()
     ddqn_power_list = list()
     doubleqn_user_list = list()
     doubleqn_power_list = list()
     for i in range(len(dqn_user_demand)):
         dqn_user_list.append(dqn_user_demand[i])
         dqn_power_list.append(total_dqn_power[i])
         ddqn_user_list.append(ddqn_user_demand[i])
         ddqn_power_list.append(total_ddqn_power[i])
         doubleqn_user_list.append(doubleqn_user_demand[i])
         doubleqn_power_list.append(total_doubleqn_power[i])
     print("dqn_user_list",dqn_user_list)
     print("ddqn_user_list",ddqn_user_list)
     print("doubleqn_user_list",dqn_user_list)
     fig1 = plt.figure()
     plt.plot(ddqn_user_list, ddqn_power_list, 'b-*', linestyle=':', marker='*', label='Dueling Deep Q Network')
     plt.plot(dqn_user_list, dqn_power_list, 'r-h', linestyle=':', marker='p', label='Deep Q Network')
     plt.plot(doubleqn_user_list, doubleqn_power_list, 'g-h', linestyle=':', marker='d', label='Double Deep Q Network')

     plt.xlabel('User Demand (Mbps)')
     plt.ylabel('Total Power Consumption (Watts)')
     plt.xlim(10, 60)
#     if len(total_ddqn_power)==0:
#         l1 =0 
#     else:
#         l1=  min(total_ddqn_power)
#     if len(total_ddqn_power)==0:
#         l2 =0 
#     else:
#         l2= (min(total_ddqn_power)%5)
#     if len(total_dqn_power)==0:
#         l3 =0 
#     else:
#         l3= max(total_dqn_power)%5
#     if len(total_dqn_power)==0:
#         l4 =0 
#     else:
#         l4= (max(total_dqn_power))
#     plt.ylim(l1-(l2), 5-(l3)+ l4)
#     banner = 'Scenario:', n_rrh , ' RRHs and' , n_usr , ' users'
#     plt.title(banner)
#     plt.grid(True)
#     fig1.savefig('Power saving.png')
#     plt.legend(loc='upper left')
#     plt.show()
#     #-------------------------------------------------
 #  print("total_ddqn_power:", total_ddqn_power)
  #  print("ddqn_user_demand:", ddqn_user_demand)
     DuelEE = 100*(np.array(ddqn_user_demand) / np.array(total_ddqn_power))
     DuelSE = 100*(np.array(ddqn_user_demand) / Bandwidth)
     print("Dueling DQN Energy efficiency:", DuelEE)
     print("Dueling DQN Spectral efficiency:", DuelSE)

#     print("total_doubleqn_power:", total_doubleqn_power)
#     print("doubleqn_user_demand:", doubleqn_user_demand)
     DoubleDQNEE = 100*(np.array(doubleqn_user_demand)/np.array(total_doubleqn_power))
     DoubleDQNSE = 100*(np.array(doubleqn_user_demand)/Bandwidth)
     print("Double DQN Energy efficiency:", DoubleDQNEE)
     print("Double DQN Spectral efficiency:", DoubleDQNSE)
#     print("total_dqn_power:", total_dqn_power)
#     print("dqn_user_demand:", dqn_user_demand)
     DQNEE = 100*(np.array(dqn_user_demand) / np.array(total_dqn_power))
     DQNSE = 100*(np.array(dqn_user_demand) /Bandwidth)

     print("DQN Energy efficiency:", DQNEE)
     print("DQN Spectral efficiency:", DQNSE)
# =============================================================================
    ## figure between SE VS EE Trade-off######
     fig1 = plt.figure()
     plt.plot(DuelSE, DuelEE,'b-h', linestyle=':', marker='*', label='Dueling Deep Q Network')
     plt.plot(DoubleDQNSE, DoubleDQNEE,'k',linestyle=':', marker='D', label='Double Deep Q Network')
     plt.plot(DQNSE, DQNEE, 'r-h', linestyle=':', marker='p',label='Deep Q Network')
     plt.xlabel('SE')
     plt.ylabel('EE')
     #plt.xlim(10, 60)
     banner = 'Energy Efficiency vs Spectral Efficiency'
     plt.title(banner)
     plt.grid(True)
     plt.legend(loc='upper left')
     fig1.savefig('EEvsSE.png')
     plt.show()
# =============================================================================
    #graph bw average power and time
    #-------------------------------------------------
# =============================================================================
#
     fig1 = plt.figure()

     plt.plot(ddqn_user_list, DuelEE, 'k' , marker = '*',label='Dueling Deep Q Network')
     plt.plot(doubleqn_user_list, DoubleDQNEE,'k', marker= 's',label='Double Deep Q Network')
     plt.plot(dqn_user_list, DQNEE,'k', marker='D' ,label='Deep Q Network')
     plt.xlabel('User Demand')
     plt.ylabel('Energy Efficiency')
     plt.xlim(10, 60)
     banner = 'Energy Efficiency per User Demand'
     plt.title(banner)
     plt.grid(True)
     plt.legend(loc='upper left')
     fig1.savefig('EEvsUserDemand.png')
     plt.show()
# =============================================================================
    #------------
    
     fig1=plt.figure()
     time_slot_DQN = range(0, len(average_slot_power))
     time_slot = range(0, len(avg_slt_pwr))
     plt.plot(time_slot, avg_slt_pwr, 'b-*', linestyle=':', marker='*', label='Dueling Deep Q Network')
     plt.plot(time_slot_DQN, average_slot_power, 'r-h', linestyle=':', marker='p', label='Deep Q Network')
     plt.plot(time_slot, a_s_p, 'g-h', linestyle=':', marker='d', label='Double Deep Q Network')
     plt.xlabel('Time Slot')
     plt.ylabel('Average Total Power Consumption')
     plt.title('Average total power consumption in time varying user demand scenario')
     plt.grid(True)
     #plt.xlim(0, 20)
     #plt.ylim(min(avg_slt_pwr)-(min(avg_slt_pwr)%5), 5-(max(average_slot_power)%5)+max(average_slot_power))
     plt.legend(loc='lower right')
     fig1.savefig("time vs power")
     plt.grid()
     plt.show()
 #    #-------------------------------------------------
 #    
 #    #graph bw episode power and episodes
 #    #-------------------------------------------------
 #    episode_slot = np.array(range(20, len(ep_power)*20+20, 20), dtype=object)
 #    
 #    fig, ax = plt.subplots()
 #    #plotting DDQN
 #    ep_power = np.array(ep_power, dtype=object)
 #    ax.plot((episode_slot[ep_power != 'I']), ep_power[ep_power != 'I'], 'b-', label = "DDQN")    
 #    flag_positions = episode_slot[ep_power == 'I']
 #    ax.plot(flag_positions, np.zeros_like(flag_positions),'bx', 
 #            clip_on=False, mew=2, label = "DDQN inf")
 #    #plotting DQN
 #    episode_power = np.array(episode_power, dtype=object)
 #    ax.plot(episode_slot[episode_power != 'I'], episode_power[episode_power != 'I'], 'r-', label = "DQN")
 #    inf_power = episode_slot[episode_power == 'I']
 #    ax.plot(inf_power, np.zeros_like(inf_power),'rx', 
 #            clip_on=False, mew=2, label = "DQN inf")
 #    #plotting DoubleQN
 #    e_power = np.array(e_power, dtype=object)
 #    ax.plot(episode_slot[e_power != 'I'], e_power[e_power != 'I'], 'r-', label = "DoubleQN")
 #    inf_power = episode_slot[e_power == 'I']
 #    ax.plot(inf_power, np.zeros_like(inf_power),'rx', 
 #            clip_on=False, mew=2, label = "DoubleQN inf")
 #        
 #    plt.xlabel('Episodes')
 #    plt.ylabel('Convergence of Power')
 #    plt.title('Algorithm Convergence')
 #    plt.grid(False)
 #    plt.legend(loc='upper right')
 #    plt.xticks(np.arange(0, len(ep_power)*20+20, step=((len(ep_power)*20)/10)))
 #    limits = np.array(ax.axis())
 #    ax.axis(limits + [0, 0, -0.5, 20])
 #    plt.grid()
 #    plt.show()
     #-------------------------------------------------
     
     '''parser = CRANParser()
     parser.set_defaults(demand_min=min_demand, demand_max=max_demand, num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
     config = parser.parse_args()'''
 #    print("Eenie")
 #    env = Env('master', config)
 #    print("Meenie")
 #    ch_gain = env._get_gains(n_rrh, n_usr)
 #    print ('channel gain  = ', ch_gain)
# =============================================================================
