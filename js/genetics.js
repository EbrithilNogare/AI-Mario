"use strict";

function createGeneticsModule(NeuralNetwork) {
  // Children stay close to their parents: crossover mixes two good parents
  // per-weight, then a light gaussian nudge touches only a few weights.
  // A slice of the population are pure fine-tuning mutants of the elites,
  // so proven networks get polished instead of replaced.
  const ELITE_FRACTION = 0.1;
  const REFINE_FRACTION = 0.2;
  const TOURNAMENT_SIZE = 3;
  const MUTATION_RATE = 0.1;
  const MUTATION_SIGMA = 0.12;
  const REFINE_SIGMA = 0.05;
  const RESET_RATE = 0.002;

  function gaussian() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  function createPopulation(size, shape) {
    return Array.from({ length: size }, () => NeuralNetwork.randomGenome(shape));
  }

  function tournament(scored) {
    let best = scored[(Math.random() * scored.length) | 0];
    for (let i = 1; i < TOURNAMENT_SIZE; i++) {
      const other = scored[(Math.random() * scored.length) | 0];
      if (other.fitness > best.fitness) best = other;
    }
    return best.genome;
  }

  function mutateInPlace(genome, sigma) {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < MUTATION_RATE) genome[i] += gaussian() * sigma;
      if (Math.random() < RESET_RATE) genome[i] = (Math.random() * 2 - 1) * 0.8;
    }
  }

  function evolve(population, fitnesses, targetSize) {
    const scored = population
      .map((genome, index) => ({ genome, fitness: fitnesses[index] }))
      .sort((a, b) => b.fitness - a.fitness);
    const eliteCount = Math.min(scored.length, Math.max(2, Math.round(targetSize * ELITE_FRACTION)));
    const next = scored.slice(0, eliteCount).map(entry => entry.genome.slice());

    // Fine-tuning mutants: tiny nudges on copies of the elites.
    const refineCount = Math.round(targetSize * REFINE_FRACTION);
    for (let i = 0; i < refineCount && next.length < targetSize; i++) {
      const child = scored[i % eliteCount].genome.slice();
      mutateInPlace(child, REFINE_SIGMA);
      next.push(child);
    }

    while (next.length < targetSize) {
      const parentA = tournament(scored);
      const parentB = tournament(scored);
      const child = new Float32Array(parentA.length);
      for (let i = 0; i < child.length; i++) {
        child[i] = Math.random() < 0.5 ? parentA[i] : parentB[i];
      }
      mutateInPlace(child, MUTATION_SIGMA);
      next.push(child);
    }
    return {
      population: next.slice(0, targetSize),
      bestGenome: scored[0].genome.slice(),
      bestFitness: scored[0].fitness,
      averageFitness: scored.reduce((sum, entry) => sum + entry.fitness, 0) / scored.length
    };
  }

  return { createPopulation, evolve };
}
