import time
import numpy as np
import math
import random

import torch
import torch.nn as nn
def evoluation_algorithm(
        trainer,
        training_strategy,
        supernet,
        val_loader,
        lookup_table,
        target_hc,
        logger,
        generation_num=20,
        population=60,
        parent_num=30,
        info_metric="flops"):
    # Population initialization
    new_population = []
    population_info = []
    best_arch = []

    #QEA population
    PI = math.pi
    theta = 0.03*PI
    delta_theta = theta
    state_len = 4
    Genome=21*state_len  #21*4
    genomeLength=Genome
    qpv = np.empty([population, genomeLength, 2])
    nqpv = np.empty([population, genomeLength, 2])  
    chromosome = np.empty([population, genomeLength],dtype=np.int) #obversed qpv
    #Initialization Qpop
    Init_Qpopulation(population,genomeLength,qpv)
    Measure(0.5,chromosome,qpv,population,genomeLength)
    Arch = toArch(chromosome,population,genomeLength,state_len)
    #Fitness
    Arch = np.array(Arch)
    print("start QEA")
    pop_fitness = get_population_accuracy(Arch,trainer,supernet,val_loader,info_metric)

    # Generation start
    global_best_fitness = 0
    start_time = time.time()
    for g in range(generation_num):
        logger.info(
            "Generation_QEA : {}, Time : {}".format(
                g, time.time() - start_time))
        cur_best_fitness = np.max(pop_fitness)

        if global_best_fitness < cur_best_fitness:
            global_best_fitness = cur_best_fitness
            logger.info(
                "New global best fitness : {}".format(global_best_fitness))
        if g==generation_num-1:
            break
        best_index = np.argmax(pop_fitness)
        best_arch = Arch[best_index]
        rotation(qpv,nqpv,chromosome,pop_fitness,population,genomeLength,best_index,theta)
        Measure(0.5,chromosome,qpv,population,genomeLength)
        Arch = toArch(chromosome,population,genomeLength,state_len)
        #Fitness
        Arch = np.array(Arch)
        pop_fitness = get_population_accuracy(Arch,trainer,supernet,val_loader,info_metric)
    
    best_index_final = np.argmax(pop_fitness)
    logger.info("Best fitness : {}".format(np.max(pop_fitness)))
    logger.info("Best global fitness: {}".format(global_best_fitness))

    #11.8 add
    #print("11.8_best_arch:",best_arch,"\n best_fitness",global_best_fitness)
    print("new population best:",Arch[best_index_final])
    #return new_population[best_match_index]
    return best_arch

def balance_evolution_algorithm(
        trainer,
        training_strategy,
        supernet,
        val_loader,
        logger,
        b_pool_fit,
        b_pool_arch,
        generation_num=20,
        population=60,
        parent_num=30,
        info_metric="flops"):
    # Population initialization
    new_population = []
    population_info = []
    best_arch = []
    # QEA population
    PI = math.pi
    theta = 0.03 * PI
    delta_theta = theta
    state_len = 4
    Genome = 21 * state_len  # 21*4
    genomeLength = Genome
    qpv = np.empty([population, genomeLength, 2])
    nqpv = np.empty([population, genomeLength, 2])
    chromosome = np.empty([population, genomeLength], dtype=np.int)  # obversed qpv
    # Initialization Qpop
    Init_Qpopulation(population, genomeLength, qpv)
    Measure(0.5, chromosome, qpv, population, genomeLength)
    Arch = toArch(chromosome, population, genomeLength, state_len)
    # Fitness
    Arch = np.array(Arch)
    pop_fitness = get_population_accuracy(Arch, trainer, supernet, val_loader, info_metric)

    # Generation start
    global_best_fitness = 0
    start_time = time.time()
    for g in range(generation_num):
        logger.info(
            "Generation : {}, Time : {}".format(
                g, time.time() - start_time))
        cur_best_fitness = np.max(pop_fitness)

        if global_best_fitness < cur_best_fitness:
            global_best_fitness = cur_best_fitness
            logger.info(
                "New global best fitness : {}".format(global_best_fitness))
        # 11.8 add
        if g == generation_num - 1:
            break
        best_index = np.argmax(pop_fitness)
        best_arch = Arch[best_index]
        rotation(qpv, nqpv, chromosome, pop_fitness, population, genomeLength, best_index, theta)
        Measure(0.5, chromosome, qpv, population, genomeLength)
        Arch = toArch(chromosome, population, genomeLength, state_len)
        # Fitness
        Arch = np.array(Arch)
        pop_fitness = get_population_accuracy(Arch, trainer, supernet, val_loader, info_metric)

    best_index_final = np.argmax(pop_fitness)
    logger.info("Best fitness : {}".format(np.max(pop_fitness)))
    logger.info("Best global fitness: {}".format(global_best_fitness))

    sort_id = np.argsort(pop_fitness)
    for i in range(3):  # top-k add
        if len(b_pool_fit) < 5:
            b_pool_fit.append(pop_fitness[sort_id[-1 - i]])  # -1为最大值
            b_pool_arch.append(Arch[sort_id[-1 - i]])
        else:
            min_id = np.argmin(b_pool_fit)
            if b_pool_fit[min_id] < pop_fitness[sort_id[-1 - i]]:
                b_pool_fit[min_id] = pop_fitness[sort_id[-1 - i]]
                b_pool_arch[min_id] = Arch[sort_id[-1 - i]]
    print("balance_pool_upgraded!")

    # 11.8 add
    # print("11.8_best_arch:",best_arch,"\n best_fitness",global_best_fitness)
    #print("new population best:", Arch[best_index_final])
    # return new_population[best_match_index]
    return b_pool_fit, b_pool_arch

#---------------------------------------------------------------------------------------EA
    """
    print("entray!")
    for p in range(population):
        architecture = training_strategy.generate_training_architecture()
        architecture_info = lookup_table.get_model_info(
            architecture, info_metric=info_metric)
        while architecture_info > target_hc:
            architecture = training_strategy.generate_training_architecture()
            architecture_info = lookup_table.get_model_info(
                architecture, info_metric=info_metric)
        #print("entray_2")
        new_population.append(architecture.tolist())
        population_info.append(architecture_info)
    #print("entray_3")
    new_population = np.array(new_population)
    population_fitness = get_population_accuracy(
        new_population, trainer, supernet, val_loader, info_metric)
    population_info = np.array(population_info)

    # Generation start
    global_best_fitness = 0
    start_time = time.time()
    for g in range(generation_num):
        logger.info(
            "Generation : {}, Time : {}".format(
                g, time.time() - start_time))
        cur_best_fitness = np.max(population_fitness)

        if global_best_fitness < cur_best_fitness:
            global_best_fitness = cur_best_fitness
            logger.info(
                "New global best fitness : {}".format(global_best_fitness))

            #11.8 add
            best_arch = new_population[np.argmax(population_fitness)]

        parents, parents_fitness = select_mating_pool(
            new_population, population_fitness, parent_num)
        #print("select_mating_pool over")
        offspring_size = population - parent_num

        evoluation_id = 0
        offspring = []
        while evoluation_id < offspring_size:
            # Evolve for each offspring
            offspring_evolution = crossover(parents)
            #print('1')
            offspring_evolution = mutation(
                offspring_evolution, training_strategy)
            #print('2')

            offspring_hc = lookup_table.get_model_info(
                offspring_evolution, info_metric=info_metric)
            #print('3')

            if offspring_hc <= target_hc:
                offspring.append(offspring_evolution)
                evoluation_id += 1
            #print('4')

        offspring_evolution = np.array(offspring_evolution)
        offspring_fittness = get_population_accuracy(
            offspring_evolution, trainer, supernet, val_loader, info_metric)

        new_population[:parent_num, :] = parents
        new_population[parent_num:, :] = offspring_evolution

        population_fitness[:parent_num] = parents_fitness
        population_fitness[parent_num:] = offspring_fittness

    best_match_index = np.argmax(population_fitness)
    logger.info("Best fitness : {}".format(np.max(population_fitness)))
    logger.info("Best global fitness: {}".format(global_best_fitness))

    #11.8 add
    #print("11.8_best_arch:",best_arch,"\n best_fitness",global_best_fitness)
    print("new population best:",new_population[best_match_index])
    #return new_population[best_match_index]
    return best_arch
    """



def select_mating_pool(population, population_fitness, parent_num):
    pf_sort_indexs = population_fitness.argsort()[::-1]
    pf_indexs = pf_sort_indexs[:parent_num]

    parents = population[pf_indexs]
    parents_fitness = population_fitness[pf_indexs]

    return parents, parents_fitness


def crossover(parents):
    parents_size = parents.shape[0]
    architecture_len = parents.shape[1]

    offspring_evolution = np.empty((1, architecture_len), dtype=np.int32)

    crossover_point = np.random.randint(low=0, high=architecture_len)
    parent1_idx = np.random.randint(low=0, high=parents_size)
    parent2_idx = np.random.randint(low=0, high=parents_size)

    offspring_evolution[0,
                        :crossover_point] = parents[parent1_idx,
                                                    :crossover_point]
    offspring_evolution[0,
                        crossover_point:] = parents[parent2_idx,
                                                    crossover_point:]
    return offspring_evolution


def mutation(offspring_evolution, training_strategy):
    architecture_len = offspring_evolution.shape[1]

    for l in range(architecture_len):
        mutation_p = np.random.choice([0, 1], p=[0.9, 0.1])

        if mutation_p == 1:
            # Mutation activate
            micro_len = training_strategy.get_block_len()
            random_mutation = np.random.randint(low=0, high=micro_len)

            offspring_evolution[0, l] = random_mutation
    return offspring_evolution


def get_population_accuracy(
        population,
        trainer,
        supernet,
        val_loader,
        info_metric):
    architectures_top1_acc = []
    for architecture in population:
        supernet.module.set_activate_architecture(architecture) if isinstance(
            supernet, nn.DataParallel) else supernet.set_activate_architecture(architecture)
        architectures_top1_acc.append(
            trainer.validate(
                supernet, val_loader, 0))

    return np.array(architectures_top1_acc)

def Init_Qpopulation(popSize,genomeLength,qpv):
    QuBitZero = np.array([[1],[0]])
    QuBitOne = np.array([[0],[1]])
    AlphaBeta = np.empty([2])
    # Hadamard gate
    r2=math.sqrt(2.0)
    h=np.array([[1/r2,1/r2],[1/r2,-1/r2]])
    # Rotation Q-gate
    theta=0
    rot =np.empty([2,2])
    # Initial population array (individual x chromosome)
    for i in range(0,popSize):
        for j in range(0,genomeLength):
            theta=np.random.uniform(0,1)*90
            theta=math.radians(theta)
            rot[0,0]=math.cos(theta); rot[0,1]=-math.sin(theta)
            rot[1,0]=math.sin(theta); rot[1,1]=math.cos(theta)
            AlphaBeta[0]=rot[0,0]*(h[0][0]*QuBitZero[0])+rot[0,1]*(h[0][1]*QuBitZero[1])
            AlphaBeta[1]=rot[1,0]*(h[1][0]*QuBitZero[0])+rot[1,1]*(h[1][1]*QuBitZero[1])
        # alpha squared          
            qpv[i,j,0]=np.around(2*pow(AlphaBeta[0],2),3) 
        # beta squared
            qpv[i,j,1]=np.around(2*pow(AlphaBeta[1],2),3) 

def Measure(p_alpha,chromosome,qpv,popSize,genomeLength):#Obverse
    for i in range(0,popSize):
        #p_init = random.random()
        #print()
        #if i==best_chrom[generation-1]:
            #continue
        for j in range(0,genomeLength):
            if p_alpha<=qpv[i, j, 0]**2:
                chromosome[i,j]=0
            else:
                chromosome[i,j]=1

def toArch(chromosome,popSize,genomeLength,state_len):
    arch_i_j = 0
    Arch=[] 
    for i in range(0,popSize):
        Arch_i=[]
        for j in range(0,genomeLength):
            arch_i_j = arch_i_j + chromosome[i][j]*(2**(j%state_len))#per_arch+=4pers
            if j!=0 and j%state_len==state_len-1:
                Arch_i.append(arch_i_j)
                arch_i_j = 0 #re 0
        Arch.append(Arch_i)
    return Arch

def rotation(qpv,nqpv,chromosome,fitness,popSize,genomeLength,best_index,theta):
    rot=np.empty([2,2])
    # Lookup table of the rotation angle
    for i in range(0,popSize):
        for j in range(0,genomeLength):
            if fitness[i]<fitness[int(best_index)]:

# if chromosome[i,j]==0 and chromosome[best_chrom[generation],j]==0:

                if chromosome[i,j]==0 and chromosome[int(best_index),j]==1:

# 旋转角0.03pi

                    if qpv[i,j,0]*qpv[i,j,1]>0:
                        delta_theta=-theta
                    if qpv[i,j,0]*qpv[i,j,1]<0: 
                        delta_theta=theta
                    if qpv[i,j,1]==0:
                        delta_theta=0
                    if qpv[i,j,0]==0:
                        p_dir = random.random()
                        if p_dir<0.5:
                            delta_theta=theta
                        else:
                            delta_theta=-theta

                    rot[0,0]=math.cos(delta_theta)
                    rot[0,1]=-math.sin(delta_theta)
                    rot[1,0]=math.sin(delta_theta)
                    rot[1,1]=math.cos(delta_theta)

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                    qpv[i,j,0]=round(nqpv[i,j,0],3)
                    qpv[i,j,1]=round(nqpv[i,j,1],3)

                if chromosome[i,j]==1 and chromosome[int(best_index),j]==0:

                    if qpv[i,j,0]*qpv[i,j,1]>0:
                        delta_theta=theta
                    if qpv[i,j,0]*qpv[i,j,1]<0: 
                        delta_theta=-theta
                    if qpv[i,j,0]==0:
                        delta_theta=0
                    if qpv[i,j,1]==0:
                        p_dir = random.random()
                        if p_dir<0.5:
                            delta_theta=theta
                        else:
                            delta_theta=-theta

                    rot[0,0]=math.cos(delta_theta)
                    rot[0,1]=-math.sin(delta_theta)
                    rot[1,0]=math.sin(delta_theta)
                    rot[1,1]=math.cos(delta_theta)

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                    qpv[i,j,0]=round(nqpv[i,j,0],3)
                    qpv[i,j,1]=round(nqpv[i,j,1],3)

                if chromosome[i,j]==1 and chromosome[int(best_index),j]==1:
                    
                    if qpv[i,j,0]*qpv[i,j,1]>0:
                        delta_theta=theta
                    if qpv[i,j,0]*qpv[i,j,1]<0: 
                        delta_theta=-theta
                    if qpv[i,j,0]==0:
                        delta_theta=0
                    if qpv[i,j,1]==0:
                        p_dir = random.random()
                        if p_dir<0.5:
                            delta_theta=theta
                        else:
                            delta_theta=-theta

                    rot[0,0]=math.cos(delta_theta)
                    rot[0,1]=-math.sin(delta_theta)
                    rot[1,0]=math.sin(delta_theta)
                    rot[1,1]=math.cos(delta_theta)

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                    qpv[i,j,0]=round(nqpv[i,j,0],3)
                    qpv[i,j,1]=round(nqpv[i,j,1],3)
            else:
                if chromosome[i,j]==0 and chromosome[int(best_index),j]==1:

# 旋转角0.03pi      
                    delta_theta=0
                    rot[0,0]=math.cos(delta_theta)
                    rot[0,1]=-math.sin(delta_theta)
                    rot[1,0]=math.sin(delta_theta)
                    rot[1,1]=math.cos(delta_theta)

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                    qpv[i,j,0]=round(nqpv[i,j,0],3)
                    qpv[i,j,1]=round(nqpv[i,j,1],3)

                if chromosome[i,j]==1 and chromosome[int(best_index),j]==0:

                    if qpv[i,j,0]*qpv[i,j,1]>0:
                        delta_theta=-theta
                    if qpv[i,j,0]*qpv[i,j,1]<0: 
                        delta_theta=theta
                    if qpv[i,j,1]==0:
                        delta_theta=0
                    if qpv[i,j,0]==0:
                        p_dir = random.random()
                        if p_dir<0.5:
                            delta_theta=theta
                        else:
                            delta_theta=-theta

                    rot[0,0]=math.cos(delta_theta)
                    rot[0,1]=-math.sin(delta_theta)
                    rot[1,0]=math.sin(delta_theta)
                    rot[1,1]=math.cos(delta_theta)

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                    qpv[i,j,0]=round(nqpv[i,j,0],3)
                    qpv[i,j,1]=round(nqpv[i,j,1],3)

                if chromosome[i,j]==1 and chromosome[int(best_index),j]==1:
                    if qpv[i,j,0]*qpv[i,j,1]>0:
                        delta_theta=theta
                    if qpv[i,j,0]*qpv[i,j,1]<0: 
                        delta_theta=-theta
                    if qpv[i,j,0]==0:
                        delta_theta=0
                    if qpv[i,j,1]==0:
                        p_dir = random.random()
                        if p_dir<0.5:
                            delta_theta=theta
                        else:
                            delta_theta=-theta

                    rot[0,0]=math.cos(delta_theta) 
                    rot[0,1]=-math.sin(delta_theta)
                    rot[1,0]=math.sin(delta_theta) 
                    rot[1,1]=math.cos(delta_theta)

                    nqpv[i,j,0]=(rot[0,0]*qpv[i,j,0])+(rot[0,1]*qpv[i,j,1])
                    nqpv[i,j,1]=(rot[1,0]*qpv[i,j,0])+(rot[1,1]*qpv[i,j,1])
                    qpv[i,j,0]=round(nqpv[i,j,0],3)
                    qpv[i,j,1]=round(nqpv[i,j,1],3)