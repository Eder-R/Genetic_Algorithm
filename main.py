'''
Y = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6
'''

import numpy as np
import generator

def main():
  
  equation_inputs = [ 4, -2, 3.5, 5, -11, -4.7] #*
  num_weights = 6                               #**
  solutions_per_population = 8                  # ***
  
  population_size = (solutions_per_population, num_weights)
  
  population = np.random.uniform(low=-4.0, high=4.0, size= population_size)
  
  print(f'Populacao inicial: \n {population}\n')
  
  num_generations = 100
  
  num_parents_crossover = 5
  
  for generation in range(num_generations):
    print(f'Geracao: {generation}\n')
    # Calcular Fitness
    fitness = generator.fitness(equation_inputs, population)
    print(f'Fitness: \n{fitness}\n')
    # Pega os melhores
    selected_parents = generator.selection(population, fitness, num_parents_crossover)
    print(f'Genitores selecionados: \n {selected_parents}\n')
    
    # Realizar Crossover
    offspring_crossover = generator.crossover(selected_parents, (solutions_per_population - num_parents_crossover, num_weights))
    print(f'Filhos gerados por crossover: \n {offspring_crossover}\n')
    
    # Fazer mutação nos gerados
    offspring_mutation = generator.mutation(offspring_crossover)
    print(f'Filhos pos radiacao: \n {offspring_mutation}\n')
    
    # Criar novo clã UCHIHA
    # ELITISMO
    population[0:selected_parents.shape[0],:] = selected_parents
    # CROSSOVER + MUTAÇÃO
    population[selected_parents.shape[0]:,:] = offspring_mutation
    
    print(f'\nNova Populacao: \n {population}\n')
    print('MELHOR RESULTADO: \n ', np.max(generator.fitness(equation_inputs, population)))
    
  fitness = generator.fitness(equation_inputs, population)
  best_fit_idx = np.where(fitness == np.max(fitness))
  print('MELHOR RESULTADO: \n ',population[best_fit_idx, :])
  print('FITNESS DO MELHOR : \n ',  fitness[best_fit_idx])
if __name__ == '__main__':
  main()