from functools import partial
import knapsack
from bruteforce import bruteforce
import genetic
from analyze import timer

things = knapsack.generate_things(22)
things = knapsack.second_example

weight_limit = 3000

print("Weight Limit: %dkg" % weight_limit)
print("")
print("Bruteforce Algorithm")
print("----------")

with timer():
	result = bruteforce(things, weight_limit)

knapsack.print_stats(result[1])

print("")
print("Genetic Algorithm")
print("----------")

with timer():
	population, generations = genetic.run_evolution(
		populate_func=partial(genetic.generate_population, size=10, genome_length=len(things)),
		fitness_func=partial(knapsack.fitness, things=things, weight_limit=weight_limit),
		fitness_limit=result[0],
		generation_limit=100
	)

sack = knapsack.from_genome(population[0], things)
knapsack.print_stats(sack)