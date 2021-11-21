import sys
sys.path.append("../Agent")
#from Agent.dqnet import DQNetwork
from Agent.agent_dqn import DQNAgent, user_demand, total_power, average_slot_power, episode_power#,episode_rewards
from Agent.agent_duelingdqn import DDQNAgent, us_demand, ttl_power, avg_slt_pwr, ep_power
from Agent.agent_doubledqn import DoubleDQNAgent, u_demand, t_power, a_s_p, e_power#,doubledqn_episode_rewards
#from Agent.agent_doubleqn import DoubleDQNAgent, cost_his
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import math
import numpy as np

'''user_demand=[]
total_power=[]
u_demand=[]
t_power=[]'''
#p_watt = 1e-3 * (10. **((p_max[i] / n_rrh)/ 10))

def DQN_Start(args):
    p_min, p_max, n_rrh, n_userList, n_epochs = args
    #global agent2
    print("DQNStart")
    for i in range(len(p_min)):

        parser = CRANParser()
        #parser.set_defaults(power_min=p_min[i], power_max=p_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm = p_max[i]/n_rrh)
        parser.set_defaults(num_rrh=n_rrh, num_usr=n_userList[i], epochs=n_epochs, pow_tsm=p_max[i] / n_rrh)
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=1e-3 * (10. **((p_max[i] / n_rrh)/ 10)))
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))# covert to dBm
        config = parser.parse_args()
        #print("Eenie")
        env = Env('master', config)
        #print("Meenie")
        #gains = env._get_gainps(n_rrh, n_usr)
        #print (gains)
        agent = DQNAgent(env, config)
        #print("Minie")
        tf.reset_default_graph()
        #tpower,usrdemd,episodrewd=agent.work()
        agent.work()
        '''meanuserdmd=sum(usrdemd)/len(usrdemd)
        user_demand.append(meanuserdmd)
        meantotalpowr=sum(tpower)/len(tpower)
        total_power.append(meantotalpowr)'''
        agent.save()
        #agent.printWeights("DQN")
    print("DQNEND")
    #print("user demand DQNNNNNN:",user_demand)
    #breakpoint()


def DDQN_Start(args):
    p_min, p_max, n_rrh, n_userList, n_epochs = args
    print("DDQNStart")
    for i in range(len(p_min)):

        parser = CRANParser()
        #parser.set_defaults(power_min=p_min[i], power_max=p_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm = p_max[i]/n_rrh)
        parser.set_defaults(num_rrh=n_rrh, num_usr=n_userList[i], epochs=n_epochs, pow_tsm=p_max[i] / n_rrh)
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=1e-3 * (10. ** ((p_max[i] / n_rrh) / 10)))
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))# covert to dBm
        config = parser.parse_args()
        #print("Eenie")
        env = Env('master', config)
        #print("Meenie")
        #gains = env._get_gains(n_rrh, n_usr)
        #print (gains)
        agent = DDQNAgent(env, config)
        #print("Minie")
        tf.reset_default_graph()
        agent.work()
        #breakpoint()
        #temp = agent.work()
        agent.save()
        #agent.printWeights("DDQN")
    print("DDQNEND")


def DoubleDQN_Start(args):
   p_min, p_max, n_rrh, n_userList, n_epochs = args
   print("DoubleDQNStart")

   for i in range(len(p_min)):
        parser = CRANParser()
        #parser.set_defaults(power_min=p_min[i], power_max=p_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
        parser.set_defaults(num_rrh=n_rrh, num_usr=n_userList[i], epochs=n_epochs, pow_tsm=p_max[i] / n_rrh)
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=1e-3 * (10. ** ((p_max[i] / n_rrh) / 10)))
        #parser.set_defaults(num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs, pow_tsm=10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))# covert to dBm
        config = parser.parse_args()
        #print("Eenie")
        env = Env('master', config)
        #print("Meenie")
        #gains = env._get_gains(n_rrh, n_usr)
        #print (gains)
        agent = DoubleDQNAgent(env, config)
        #print("Minie")
        #tf.reset_default_graph()
        tf.compat.v1.reset_default_graph()
        #tpower,usrdemd,episodrewd=agent.work()
        agent.work()

        '''meanuserdmd=sum(usrdemd)/len(usrdemd)
        u_demand.append(meanuserdmd)
        meantotalpowr=sum(tpower)/len(tpower)
        t_power.append(meantotalpowr)'''
        agent.save()
        #agent.printWeights("DoubleQN")

        print("DoubleDQNEND")

if __name__ == '__main__':
    # note n_usr<=n_rrh

    n_rrh = 14
    #n_usr = 3
    #n_userList=[2,4,6,8,10,12]
    n_userList=[1,2,3,4,5,6]
    min_power = 10*n_rrh
    max_power = 60*n_rrh
    #min_power = 0.01*n_rrh
    #max_power = 1000*n_rrh
    n_epochs = 10
    p_min = list()#data rate
    p_max = list()
    #to get banwidth
    '''parser = CRANParser()
    config = parser.parse_args()
    env = Env('master', config)
    Bandwidth=env._BAND'''
    Bandwidth=10000000.0


    #for plotting graph bw total power and user demand
    #-------------------------------------------------
    slot_range = int(max_power/min_power)
    #slot_range = int(np.log10(max_power/min_power)+1)
    #d_min = [min_demand-10.e6]
    #d_min.append(min_demand-10.e6)
    #d_max = [min_demand] #user demand (MBps)
    #d_max.append(max_demand-10.e6)
    for i in range(slot_range):#slot_range=6
        p_max.append(min_power*(i+1))
        p_min.append(min_power*(i))
        #p_max.append(min_power*10**(i+1))#in watts
        #p_min.append(min_power*10**(i))
    p_max=20*n_rrh*np.ones(6)#per antenna max power same

   # print(len(d_min))
    #print("KKK")

    args = p_min, p_max, n_rrh, n_userList, n_epochs
    total_ddqn_power = list()
    ddqn_user_demand = list()
    total_doubleqn_power = list()
    doubleqn_user_demand = list()
    total_dqn_power = list()
    dqn_user_demand = list()
    dqn_episode_rewards = list()
    double_dqn_episode_rewards = list()
    dqn_per_antenna_power=list()


    for i in range(slot_range):
        dqn_per_antenna_power.append(10 *math.log10 ((10 **3)* (p_max[i] / n_rrh)))#power in dBm
    print("dqn_per_antenna_power",dqn_per_antenna_power)
    #breakpoint()

    #print("\n\nargs: ",args,"\n\n")
    DQN_Start(args)

    #print('total power is ', total_power)
    total_dqn_power = copy.deepcopy(total_power)
    dqn_user_demand = copy.deepcopy(user_demand)
    #dqn_episode_rewards = copy.deepcopy(episode_rewards)
    #dqn_per_antenna_power=copy.deepcopy(per_antenna_power)
    #dqn_Bandwidth=copy.deepcopy(Bandwidth)
    print("DQN bandwidth",Bandwidth)
    print("DQN total power",total_dqn_power)
    print("DQN per_antenna_power",dqn_per_antenna_power)
    print("DQN user demand",dqn_user_demand)
    #breakpoint()


    DDQN_Start(args)
   # print('total power is ', ttl_power)
    total_ddqn_power = copy.deepcopy(ttl_power)
    #print(total_ddqn_power)
    ddqn_user_demand = copy.deepcopy(us_demand)

    DoubleDQN_Start(args)
   # print('total power is ', t_power)
    total_doubleqn_power = copy.deepcopy(t_power)
    doubleqn_user_demand = copy.deepcopy(u_demand)
    #double_dqn_episode_rewards = copy.deepcopy(doubledqn_episode_rewards)


#    print("\n\n\n",total_power,user_demand,ttl_power,us_demand,t_power,u_demand,"\n\n\n")
#
#    print("\n\n")
#

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

    '''print("total_ddqn_power:", total_ddqn_power)
    print("ddqn_user_demand:", ddqn_user_demand)
    DuelEE = 100*(np.array(ddqn_user_demand) / np.array(total_ddqn_power))
    DuelSE = 1000*(np.array(ddqn_user_demand) / Bandwidth)
    print("Dueling DQN Energy efficiency:", DuelEE)
    print("Dueling DQN Spectral efficiency:", DuelSE)'''

    print("total_doubleqn_power:", total_doubleqn_power)
    print("doubleqn_user_demand:", doubleqn_user_demand)
    DoubleDQNEE = 100*(np.array(doubleqn_user_demand)/np.array(total_doubleqn_power))
    DoubleDQNSE = 100*(np.array(doubleqn_user_demand)/Bandwidth)
    print("Double DQN Energy efficiency:", DoubleDQNEE)
    print("Double DQN Spectral efficiency:", DoubleDQNSE)

    print("total_dqn_power:", total_dqn_power)
    print("dqn_user_demand:", dqn_user_demand)# R(t) 1.pdf
    DQNEE = 100*(np.array(dqn_user_demand) / np.array(total_dqn_power))
    DQNSE = 100*(np.array(dqn_user_demand) /Bandwidth)

    print("DQN Energy efficiency:", DQNEE)
    print("DQN Spectral efficiency:", DQNSE)

    #np.array(dqn_power_list) / n_rrh

    #print("dqn episode rewards:", dqn_episode_rewards)
    #print("double dqn episode rewards", double_dqn_episode_rewards)




    dqn_user_list = list()
    dqn_power_list = list()
    ddqn_user_list = list()
    ddqn_power_list = list()
    doubleqn_user_list = list()
    doubleqn_power_list = list()
   # print("\n\n\nLenght: ", len(dqn_user_demand),"\n\n\n")
    '''for i in range(len(dqn_user_demand)):
        dqn_user_list.append(dqn_user_demand[i])
        dqn_power_list.append(total_dqn_power[i])
        #ddqn_user_list.append(ddqn_user_demand[i])
        #ddqn_power_list.append(total_ddqn_power[i])
        doubleqn_user_list.append(doubleqn_user_demand[i])
        doubleqn_power_list.append(total_doubleqn_power[i])'''

    '''
    fig = plt.figure()
    #plt.plot(ddqn_user_list, ddqn_power_list, 'b-*', linestyle=':', marker='*', label='Dueling Deep Q Network')
    plt.plot(dqn_user_list, dqn_power_list, 'r-h', linestyle=':', marker='p', label='Deep Q Network')
    #plt.plot(doubleqn_user_list, doubleqn_power_list, 'g-h', linestyle=':', marker='d', label='Double Deep Q Network')

    plt.xlabel('User Demand (Mbps)')
    plt.ylabel('Total Power Consumption (Watts)')
    plt.xlim(10, 60)
    '''
    '''
    if len(total_ddqn_power)==0:
        l1 =0
    else:
        l1=  min(total_ddqn_power)
    if len(total_ddqn_power)==0:
        l2 =0
    else:
        l2= (min(total_ddqn_power)%5)
    if len(total_dqn_power)==0:
        l3 =0
    else:
        l3= max(total_dqn_power)%5
    if len(total_dqn_power)==0:
        l4 =0
    else:
        l4= (max(total_dqn_power))
    plt.ylim(l1-(l2), 5-(l3)+ l4)
    '''
    '''
    banner = 'Scenario:', n_rrh , ' RRHs and' , n_usr , ' users'
    plt.title(banner)
    plt.grid(True)
    plt.legend(loc='lower right')
    fig.savefig('Total_Power_Consumption_graph.png')
    plt.show()
    #-------------------------------------------------
    '''
    '''
    #-------------------------------------------------
    #graph for ENERGY EFFICIENCY VS USER DEMAND

    fig1 = plt.figure()

    #plt.plot(ddqn_user_list, DuelEE, label='Dueling Deep Q Network')
    #plt.plot(doubleqn_user_list, DoubleDQNEE, label='Double Deep Q Network')
    plt.plot(dqn_user_list, DQNEE, label='Deep Q Network')
    plt.xlabel('User Demand')
    plt.ylabel('Energy Efficiency')
    plt.xlim(10, 60)
    banner = 'Energy Efficiency per User Demand'
    plt.title(banner)
    plt.grid(True)
    plt.legend(loc='upper right')
    fig1.savefig('EEvsUserDemand.png')
    plt.show()
    #-------------------------------------------------
    '''



    #-------------------------------------------------
    #graph for ENERGY EFFICIENCY VS PER ANTENNA POWER

    fig1 = plt.figure()

    #plt.plot(ddqn_user_list, DuelEE, label='Dueling Deep Q Network')
    #plt.plot(dqn_per_antenna_power, DuelEE, label='Dueling Deep Q Network')
    #plt.plot(np.array([10,20,30,40,50,60]), DoubleDQNEE, label='Double Deep Q Network')
    plt.plot(n_userList, DoubleDQNEE, label='Double Deep Q Network')
    #plt.plot(np.array([10,20,30,40,50,60]), DQNEE, label='Deep Q Network')
    plt.plot(n_userList, DQNEE, label='Deep Q Network')
    plt.xlabel('Number of users')
    plt.ylabel('Energy Efficiency')
    #plt.xlim(10, 60)
    banner = 'Energy Efficiency vs No. of users'
    plt.title(banner)
    plt.grid(True)
    plt.legend(loc='upper right')
    fig1.savefig('EEvsuserInv.png')
    plt.show()
    #-------------------------------------------------
    #graph for Spectral EFFICIENCY VS PER ANTENNA POWER

    fig1 = plt.figure()

    #plt.plot(ddqn_user_list, DuelEE, label='Dueling Deep Q Network')
    #plt.plot(np.array([10,20,30,40,50,60]), DoubleDQNEE, label='Double Deep Q Network')

    #plt.plot(np.array([10,20,30,40,50,60]), DQNEE, label='Deep Q Network')
    #plt.plot(dqn_per_antenna_power, DuelSE, label='Dueling Deep Q Network')
    plt.plot(n_userList, DoubleDQNSE, label='Double Deep Q Network')
    plt.plot(n_userList, DQNSE, label='Deep Q Network')
    plt.xlabel('No. of users')
    plt.ylabel('Spectral Efficiency')
    #plt.xlim(10, 60)
    banner = 'Spectral Efficiency per Antenna Power'
    plt.title(banner)
    plt.grid(True)
    plt.legend(loc='upper right')
    fig1.savefig('SEvsusersInv.png')
    plt.show()

    #-------------------------------------------------
    #graph for Energy EFFICIENCY VS Spectral EFFICIENCY

    fig1 = plt.figure()

    #plt.plot(ddqn_user_list, DuelEE, label='Dueling Deep Q Network')
    #plt.plot(np.array([10,20,30,40,50,60]), DoubleDQNEE, label='Double Deep Q Network')

    #plt.plot(np.array([10,20,30,40,50,60]), DQNEE, label='Deep Q Network')
    #plt.plot(DuelSE, DuelEE, label='Dueling Deep Q Network')
    plt.plot(DoubleDQNSE, DoubleDQNEE, label='Double Deep Q Network')
    plt.plot(DQNSE, DQNEE, label='Deep Q Network')
    plt.xlabel('SE')
    plt.ylabel('EE')
    #plt.xlim(10, 60)
    banner = 'Energy Efficiency vs Spectral Efficiency'
    plt.title(banner)
    plt.grid(True)
    plt.legend(loc='upper right')
    fig1.savefig('EEvsSEInv.png')
    plt.show()
    #-------------------------------------------------
    #graph for episode rewards

    # fig1 = plt.figure()
    # plt.plot(dqn_episode_rewards, label='Deep Q Network')
    # plt.plot(double_dqn_episode_rewards, label='Double Deep Q Network')
    # plt.xlabel('Episode')
    # plt.ylabel('Episode Reward')
    # banner = 'Episode Rewards Graph'
    # plt.title(banner)
    # plt.grid(True)
    # plt.legend(loc='upper left')
    # fig1.savefig('Episode Rewards Graph.png')
    # plt.show()
    #-------------------------------------------------



    #graph bw average power and time
#    #-------------------------------------------------
    '''time_slot = range(0, len(average_slot_power))
    #plt.plot(time_slot, avg_slt_pwr, 'b-*', linestyle=':', marker='*', label='Dueling Deep Q Network')
    plt.plot(time_slot, average_slot_power, 'r-h', linestyle=':', marker='p', label='Deep Q Network')
    plt.plot(time_slot, a_s_p, 'g-h', linestyle=':', marker='d', label='Double Deep Q Network')
    plt.xlabel('Time Slot')
    plt.ylabel('Average Total Power Consumption')
    plt.title('Average total power consumption in time varying user demand scenario')
    plt.grid(True)
    plt.xlim(0, 20)
    plt.ylim(min(avg_slt_pwr)-(min(avg_slt_pwr)%5), 5-(max(average_slot_power)%5)+max(average_slot_power))
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()'''
##    #-------------------------------------------------
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
#    ax.plot(episode_slot[episode_power != 'I'], episode_power[episode_power != 'I'], 'k-', label = "DQN")
#    inf_power = episode_slot[episode_power == 'I']
#    ax.plot(inf_power, np.zeros_like(inf_power),'rx',
#            clip_on=False, mew=2, label = "DQN inf")
#    #plotting DoubleQN
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
    #parser.set_defaults(demand_min=min_demand, demand_max=max_demand, num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
    parser.set_defaults(power_min = min_power, power_max = max_power, num_rrh = n_rrh, num_usr = n_usr, epochs = n_epochs)
    config = parser.parse_args()
    #print("Eenie")
    env = Env('master', config)
   # print("Meenie")
    ch_gain = env._get_gains(n_rrh, n_usr)
    print ('channel gain  = ', ch_gain)'''
