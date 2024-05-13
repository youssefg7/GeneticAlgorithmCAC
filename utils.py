import numpy as np

def generatePopulation(population_size : int, block_height_chromosome_length : int, block_width_chromosome_length : int, possible_heights : list, possible_widths : list) -> list:
    population = []
    for _ in range(population_size):
        block_height_index = np.random.randint(0, len(possible_heights))
        block_width_index = np.random.randint(0, len(possible_widths))
        chromosome = np.concatenate((decimalToBinary(block_height_index, block_height_chromosome_length), decimalToBinary(block_width_index, block_width_chromosome_length)))
        population.append(chromosome)
    return population

def get_block_type(image_array, block_width, block_height, w_start, h_start):
    has_white = False
    has_black = False
    for w in range(w_start, w_start + block_width):
        for h in range(h_start, h_start + block_height):
            if image_array[h][w]:
                has_white = True
            else:
                has_black = True
            if has_white and has_black:
                return 'M'
    if has_white:
        return 'W'
    elif has_black:
        return 'B'
    
def blocks_counter_encoder(blocks_array):
    unique, counts = np.unique(blocks_array, return_counts=True)
    counter = dict(zip(unique, counts))
    codes = counter.copy()
    max_count = max(counter, key=counter.get)
    codes[max_count] = '0'
    
    types_count = len(counter)
    i = 0
    for key in counter.keys():
        if key is not max_count:
            if types_count == 2:
                codes[key] = '1'
            elif types_count == 3:
                if i == 0:
                    codes[key] = '01'
                    i = i + 1
                elif i == 1:
                    codes[key] = '11'
    return counter, codes

def CAC(original_image : np.array, block_width : int, block_height : int) -> float:
    image_height, image_width = original_image.shape
    x_steps = int(image_width / block_width)
    y_steps = int(image_height / block_height)
    blocks_array = []
    for y in range(y_steps):
        blocks_array.append([])
        for x in range(x_steps):
            x_start = x * block_width
            y_start = y * block_height
            block_type = get_block_type(original_image, block_width, block_height, x_start, y_start)
            blocks_array[y].append(block_type)
    blocks_array = np.array(blocks_array)
    
    counter, codes = blocks_counter_encoder(blocks_array)
    N2 = 0
    for key, value in counter.items():
        if key == 'M':
            N2 = N2 + value * (len(codes[key]) + block_width * block_height) # why block_width * block_height? answer: because we need to store the pixel values of the block
        else:
            N2 = N2 + (value * len(codes[key]))
    return N2

def getDivisors(n : int) -> list:
    divisors = []
    for i in range(1, n+1):
        if n % i == 0:
            divisors.append(i)
    return divisors

def fitnessFunction(original_size : int, compressed_size : int) -> int:
    return original_size / compressed_size # compression ratio

def evaluatePopulation(population : list, original : np.array, block_height_chromosome_length : int, block_width_chromosome_length : int, possible_heights : list, possible_widths : list) -> list:
    original_size = original.size
    fitnessValues = []
    for chromosome in population:
        block_height_index = binaryToDecimal(chromosome[:block_height_chromosome_length])
        block_width_index = binaryToDecimal(chromosome[block_height_chromosome_length:])
        if block_height_index >= len(possible_heights) or block_width_index >= len(possible_widths):
            fitnessValues.append(0)
            continue
        block_height = possible_heights[block_height_index]
        block_width = possible_widths[block_width_index]
        compressed_size = CAC(original, block_width, block_height)
        fitness = fitnessFunction(original_size, compressed_size)
        fitnessValues.append(fitness)
    return fitnessValues


def matchParents(survivors : list) -> list:
    available_parents = np.arange(len(survivors))
    parents_tuples = []
    for _ in range(len(survivors)//2):
        parent1_index = np.random.choice(available_parents)
        available_parents = np.delete(available_parents, np.where(available_parents == parent1_index))
        parent2_index = np.random.choice(available_parents)
        available_parents = np.delete(available_parents, np.where(available_parents == parent2_index))
        parents_tuples.append((survivors[parent1_index], survivors[parent2_index]))
    return parents_tuples

def selectSurvivors(population : list, fitness_values : list) -> list:
    survivors_indices = np.argsort(fitness_values)[::-1][:int(len(population) / 2)]
    survivors = [population[i] for i in survivors_indices]
    return survivors

def mutate(chromosome : np.array) -> np.array:
    mutation = np.random.randint(0, 2, chromosome.size)
    mutated_chromosome = np.logical_xor(chromosome, mutation)
    return mutated_chromosome

def crossover(chromosome1 : np.array, chromosome2 : np.array, block_height_chromosome_length : int, block_width_chromosome_length : int) -> np.array:
    height_crossover_point = np.random.randint(0, block_height_chromosome_length)
    width_crossover_point = np.random.randint(0, block_width_chromosome_length)
    child1 = np.concatenate((chromosome1[:height_crossover_point], 
                             chromosome2[height_crossover_point:block_height_chromosome_length], 
                             chromosome1[block_height_chromosome_length:block_height_chromosome_length+width_crossover_point], 
                             chromosome2[block_height_chromosome_length+width_crossover_point:]))
    child2 = np.concatenate((chromosome2[:height_crossover_point], 
                             chromosome1[height_crossover_point:block_height_chromosome_length], 
                             chromosome2[block_height_chromosome_length:block_height_chromosome_length+width_crossover_point], 
                             chromosome1[block_height_chromosome_length+width_crossover_point:]))
    return [child1, child2]

def generateOffsprings(parents_tuples, block_height_chromosome_length, block_width_chromosome_length):
    offsprings = []
    for parent1, parent2 in parents_tuples:
        offsprings.append(parent1)
        offsprings.append(parent2)
        children = crossover(parent1, parent2, block_height_chromosome_length, block_width_chromosome_length)
        children = [mutate(child) for child in children]
        offsprings.extend(children)
    return offsprings
    

def binaryToDecimal(binary : np.array) -> int:
    decimal = 0
    for i in range(binary.size):
        decimal += binary[i] * 2 ** (binary.size - i - 1)
    return int(decimal)

def decimalToBinary(decimal : int, chromosomeSize : int) -> np.array:
    binary = np.zeros(chromosomeSize)
    for i in range(chromosomeSize):
        binary[chromosomeSize - i - 1] = decimal % 2
        decimal = decimal // 2
    return binary