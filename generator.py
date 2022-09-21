import numpy as np
import sys

# Calcular Fitness da população atual
def fitness(equation_input, population):
  #Calcular a soma dos produtos de w.x (população x inputs)
  return np.sum(population * equation_input, axis=1)

# Pega os melhores
def selection(population, fitness, num_parents):
  parents = np.empty((num_parents, population.shape[1]))

  for idx in range(num_parents):
    max_fitness_idx = np.where(fitness == np.max(fitness))
    max_fitness_idx = max_fitness_idx[0][0]
    
    parents[idx, :] = population[max_fitness_idx, :]
    fitness[max_fitness_idx] = -sys.maxsize - 1
    
    
  return parents

# Realizar Crossover
def crossover(parents, generation_size):
  
  #Criar o vetor
  offspring = np.empty(generation_size)
  
  ''' obter o indice que é a metade do tamanho do cromossomo
  (metade do número de colunas)'''
  crossover_point = np.uint8(generation_size[1]/2)
  
  for idx in range(generation_size[0]):
    p1_idx = idx % parents.shape[0]
    p2_idx = (idx + 1) % parents.shape[0]
    
    offspring[idx, 0:crossover_point] = parents[p1_idx, 0:crossover_point]
    
    offspring[idx, crossover_point:] = parents[p2_idx, crossover_point:]
    
  return offspring

# Fazer mutação nos gerados
def mutation(offspring):
  for idx in range(offspring.shape[0]):
    random_value = np.random.uniform(-1.0, 1.0, 1)
    random_idx = np.random.randint(offspring.shape[1])
    
    offspring[idx, random_idx] = offspring[idx, random_idx] + random_value
  
  return offspring

# Criar novo clã UCHIHA