from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple


Genome = List[int]
Population = List[Genome]
Thing = namedtuple("Thing", ["name", "value", "weight"])
FitnessFunc = Callable[[Thing], int]
PopulateFunc = Callable[[], Population]  # Takes in nothing and spits out new solutions
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]


example_1 = [
    Thing("Laptop", 500, 2200),
    Thing("Headphones", 150, 160),
    Thing("Coffee mug", 60, 350),
    Thing("Notepad", 40, 333),
    Thing("Water bottle", 30, 192),
]

example_2 = [
    Thing("Mints", 5, 25),
    Thing("Socks", 10, 38),
    Thing("Tissues", 15, 80),
    Thing("Phone", 500, 200),
    Thing("Baseball cap", 100, 70),
]


def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:

    if len(genome) != len(things):
        raise ValueError("Genome and things must have the same length")

    weight, value = 0, 0

    for i, thing in enumerate(things):
        if genome[i] == 1:  # Item exists in genome
            weight += thing.weight
            value += thing.value

            if weight > weight_limit:
                # Weight has been exceeded and the solution is invalid
                return 0

    # If solution is valid return score
    return value


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    """
    Selects a pair of solutions that will be the parents for the generation of the next solution.
    """
    return choices(
        population=population,
        weights=[fitness_func(gene) for gene in population],
        k=2
    )


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    """
    Mutates the 2 genomes by mixing them up. Genomes must have the same length
    """
    if len(a) != len(b):
        raise ValueError("Genomes must be of the same length")

    length = len(a)
    # Genomes must be of at least len 2
    if length < 2:
        return a, b

    p = randint(1, length - 1)  # Select random integer to split genome
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    """
    Takes a genome and randomly mutates it by changing some 1s to 0s or 0s to 1s with
    certain probability.
    """

    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def population_fitness(population: Population, fitness_func: FitnessFunc) -> int:
    return sum([fitness_func(genome) for genome in population])


def sort_population(population: Population, fitness_func: FitnessFunc) -> Population:
    return sorted(population, key=fitness_func, reverse=True)


def genome_to_string(genome: Genome) -> str:
    return "".join(map(str, genome))


def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    print("GENERATION %02d" % generation_id)
    print("=============")
    print("Population: [%s]" % ", ".join([genome_to_string(gene) for gene in population]))
    print("Avg. Fitness: %f" % (population_fitness(population, fitness_func) / len(population)))
    sorted_population = sort_population(population, fitness_func)
    print(
        "Best: %s (%f)" % (genome_to_string(sorted_population[0]), fitness_func(sorted_population[0])))
    print("Worst: %s (%f)" % (genome_to_string(sorted_population[-1]),
                              fitness_func(sorted_population[-1])))
    print("")

    return sorted_population[0]


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:

    population = populate_func()

    for i in range(generation_limit):
        population = sorted(population, key=lambda genome: fitness_func(genome), reverse=True)

        if printer is not None:
            printer(population, i, fitness_func)

        if fitness_func(population[0]) >= fitness_limit:
            break

        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]

        population = next_generation

    return population, i


def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]
    return result


population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=len(example_2)
    ),
    fitness_func=partial(fitness, example_2, weight_limit=3000),
    fitness_limit=1310,
    generation_limit=100
)

print(f"Number of generations: {generations}")
print(f"Best solution: {genome_to_things(population[0], example_2)}")
