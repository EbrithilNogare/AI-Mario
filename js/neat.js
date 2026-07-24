"use strict";

// NEAT (NeuroEvolution of Augmenting Topologies), feedforward-only.
// Genomes are plain JSON objects so they can be sent to workers as-is.
function createNeatModule() {
  const WEIGHT_MUTATE_RATE = 0.8;
  const WEIGHT_PERTURB_SIGMA = 0.15; // gaussian nudge — children stay close to parents
  const WEIGHT_RESET_RATE = 0.02;
  const ADD_CONNECTION_RATE = 0.2;
  const ADD_NODE_RATE = 0.08;
  const CROSSOVER_RATE = 0.75;
  const STAGNATION_LIMIT = 15;
  const SURVIVAL_RATIO = 0.5;

  function gaussian() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  let innovationCounter = 0;
  let nodeCounter = 0;
  let connectionCache = new Map();
  let splitCache = new Map();
  let speciesList = [];
  let speciesCounter = 0;
  let compatThreshold = 3;

  function resetInnovations() {
    innovationCounter = 0;
    nodeCounter = 0;
    connectionCache.clear();
    splitCache.clear();
    speciesList = [];
    speciesCounter = 0;
    compatThreshold = 3;
  }

  function connectionInnovation(inNode, outNode) {
    const key = inNode + ">" + outNode;
    if (!connectionCache.has(key)) connectionCache.set(key, innovationCounter++);
    return connectionCache.get(key);
  }

  // Node ids: 0 = bias, 1..inputCount = inputs, then outputs, then hidden.
  function createGenome(inputCount, outputCount) {
    const nodes = [{ id: 0, type: "bias" }];
    for (let i = 1; i <= inputCount; i++) nodes.push({ id: i, type: "input" });
    for (let o = 0; o < outputCount; o++) nodes.push({ id: inputCount + 1 + o, type: "output" });
    const connections = [];
    for (const from of nodes) {
      if (from.type === "output") continue;
      for (let o = 0; o < outputCount; o++) {
        const outId = inputCount + 1 + o;
        connections.push({
          inNode: from.id, outNode: outId,
          weight: Math.random() * 2 - 1, enabled: true,
          innovation: connectionInnovation(from.id, outId)
        });
      }
    }
    return { inputCount, outputCount, nodes, connections };
  }

  function createPopulation(size, inputCount, outputCount) {
    resetInnovations();
    nodeCounter = inputCount + outputCount + 1;
    return Array.from({ length: size }, () => createGenome(inputCount, outputCount));
  }

  function cloneGenome(genome) {
    return {
      inputCount: genome.inputCount,
      outputCount: genome.outputCount,
      nodes: genome.nodes.map(node => ({ ...node })),
      connections: genome.connections.map(connection => ({ ...connection }))
    };
  }

  function createsCycle(genome, fromId, toId) {
    if (fromId === toId) return true;
    const outgoing = new Map();
    for (const connection of genome.connections) {
      if (!outgoing.has(connection.inNode)) outgoing.set(connection.inNode, []);
      outgoing.get(connection.inNode).push(connection.outNode);
    }
    const stack = [toId];
    const visited = new Set();
    while (stack.length) {
      const id = stack.pop();
      if (id === fromId) return true;
      if (visited.has(id)) continue;
      visited.add(id);
      for (const next of outgoing.get(id) || []) stack.push(next);
    }
    return false;
  }

  function mutateAddConnection(genome) {
    const targets = genome.nodes.filter(node => node.type === "hidden" || node.type === "output");
    for (let attempt = 0; attempt < 20; attempt++) {
      const from = genome.nodes[(Math.random() * genome.nodes.length) | 0];
      const to = targets[(Math.random() * targets.length) | 0];
      if (!to || from.type === "output" || from.id === to.id) continue;
      if (genome.connections.some(c => c.inNode === from.id && c.outNode === to.id)) continue;
      if (createsCycle(genome, from.id, to.id)) continue;
      genome.connections.push({
        inNode: from.id, outNode: to.id,
        weight: Math.random() * 2 - 1, enabled: true,
        innovation: connectionInnovation(from.id, to.id)
      });
      return;
    }
  }

  function mutateAddNode(genome) {
    const enabled = genome.connections.filter(connection => connection.enabled);
    if (!enabled.length) return;
    const connection = enabled[(Math.random() * enabled.length) | 0];
    let split = splitCache.get(connection.innovation);
    if (!split) {
      const nodeId = nodeCounter++;
      split = {
        nodeId,
        inInnovation: connectionInnovation(connection.inNode, nodeId),
        outInnovation: connectionInnovation(nodeId, connection.outNode)
      };
      splitCache.set(connection.innovation, split);
    }
    if (genome.nodes.some(node => node.id === split.nodeId)) return;
    connection.enabled = false;
    genome.nodes.push({ id: split.nodeId, type: "hidden" });
    genome.connections.push({
      inNode: connection.inNode, outNode: split.nodeId,
      weight: 1, enabled: true, innovation: split.inInnovation
    });
    genome.connections.push({
      inNode: split.nodeId, outNode: connection.outNode,
      weight: connection.weight, enabled: true, innovation: split.outInnovation
    });
  }

  function mutate(genome) {
    if (Math.random() < WEIGHT_MUTATE_RATE) {
      for (const connection of genome.connections) {
        if (Math.random() < WEIGHT_RESET_RATE) connection.weight = Math.random() * 2 - 1;
        else connection.weight += gaussian() * WEIGHT_PERTURB_SIGMA;
      }
    }
    if (Math.random() < ADD_CONNECTION_RATE) mutateAddConnection(genome);
    if (Math.random() < ADD_NODE_RATE) mutateAddNode(genome);
    return genome;
  }

  // fitParent must be the fitter one: disjoint/excess genes come from it.
  function crossover(fitParent, weakParent) {
    const weakByInnovation = new Map(weakParent.connections.map(c => [c.innovation, c]));
    const child = cloneGenome(fitParent);
    child.connections = fitParent.connections.map(connection => {
      const other = weakByInnovation.get(connection.innovation);
      const source = other && Math.random() < 0.5 ? other : connection;
      let enabled = connection.enabled && (!other || other.enabled);
      if (!enabled) enabled = Math.random() < 0.25;
      return {
        inNode: connection.inNode, outNode: connection.outNode,
        weight: source.weight, enabled, innovation: connection.innovation
      };
    });
    return child;
  }

  function compatibility(a, b) {
    const bByInnovation = new Map(b.connections.map(c => [c.innovation, c]));
    let matching = 0;
    let weightDifference = 0;
    for (const connection of a.connections) {
      const other = bByInnovation.get(connection.innovation);
      if (other) {
        matching++;
        weightDifference += Math.abs(connection.weight - other.weight);
      }
    }
    const mismatched = a.connections.length + b.connections.length - 2 * matching;
    const largest = Math.max(a.connections.length, b.connections.length, 1);
    const normalizer = largest < 20 ? 1 : largest / 20;
    return mismatched / normalizer + 0.4 * (matching ? weightDifference / matching : 0);
  }

  function evolve(population, fitnesses, config) {
    const targetSize = config.targetSize;
    const scored = population.map((genome, index) => ({ genome, fitness: fitnesses[index] }));

    for (const species of speciesList) species.members = [];
    for (const entry of scored) {
      let placed = false;
      for (const species of speciesList) {
        if (compatibility(entry.genome, species.representative) < compatThreshold) {
          species.members.push(entry);
          placed = true;
          break;
        }
      }
      if (!placed) {
        speciesList.push({
          id: ++speciesCounter, representative: entry.genome,
          members: [entry], bestFitness: -Infinity, stagnation: 0
        });
      }
    }
    speciesList = speciesList.filter(species => species.members.length > 0);
    if (speciesList.length > config.speciesTarget) compatThreshold += 0.3;
    else if (speciesList.length < config.speciesTarget) compatThreshold = Math.max(0.5, compatThreshold - 0.3);

    let globalBest = scored[0];
    for (const entry of scored) if (entry.fitness > globalBest.fitness) globalBest = entry;

    for (const species of speciesList) {
      let best = -Infinity;
      for (const member of species.members) best = Math.max(best, member.fitness);
      if (best > species.bestFitness) {
        species.bestFitness = best;
        species.stagnation = 0;
      } else {
        species.stagnation++;
      }
    }
    const surviving = speciesList.filter(species =>
      species.stagnation < STAGNATION_LIMIT || species.members.includes(globalBest));
    if (surviving.length) speciesList = surviving;

    let minFitness = Infinity;
    for (const entry of scored) minFitness = Math.min(minFitness, entry.fitness);
    let totalScore = 0;
    for (const species of speciesList) {
      // Mean shifted fitness = sum of sharing-adjusted fitness; big species get no size bonus.
      species.score = species.members.reduce((sum, member) =>
        sum + (member.fitness - minFitness + 1), 0) / species.members.length;
      totalScore += species.score;
    }

    const next = [cloneGenome(globalBest.genome)];
    for (const species of speciesList) {
      species.members.sort((a, b) => b.fitness - a.fitness);
      let quota = Math.max(0, Math.round(species.score / totalScore * (targetSize - 1)));
      if (species.members.length >= 5 && quota > 1) {
        next.push(cloneGenome(species.members[0].genome));
        quota--;
      }
      const parents = species.members.slice(0, Math.max(1, Math.ceil(species.members.length * SURVIVAL_RATIO)));
      const pick = () => parents[(Math.random() * parents.length) | 0];
      for (let i = 0; i < quota && next.length < targetSize; i++) {
        let child;
        if (parents.length > 1 && Math.random() < CROSSOVER_RATE) {
          let parentA = pick();
          let parentB = pick();
          if (parentB.fitness > parentA.fitness) [parentA, parentB] = [parentB, parentA];
          child = crossover(parentA.genome, parentB.genome);
        } else {
          child = cloneGenome(pick().genome);
        }
        next.push(mutate(child));
      }
      species.representative = species.members[(Math.random() * species.members.length) | 0].genome;
    }
    while (next.length < targetSize) {
      const species = speciesList[(Math.random() * speciesList.length) | 0];
      next.push(mutate(cloneGenome(species.members[0].genome)));
    }
    if (next.length > targetSize) next.length = targetSize;

    return {
      population: next,
      bestGenome: cloneGenome(globalBest.genome),
      bestFitness: globalBest.fitness,
      averageFitness: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
      speciesCount: speciesList.length,
      compatThreshold
    };
  }

  // Decode a genome into a runnable feedforward network.
  // `values` holds the latest activation of every node (indexed like genome.nodes).
  function buildNetwork(genome) {
    const nodes = genome.nodes;
    const indexById = new Map(nodes.map((node, index) => [node.id, index]));
    const incoming = nodes.map(() => []);
    for (const connection of genome.connections) {
      if (!connection.enabled) continue;
      const from = indexById.get(connection.inNode);
      const to = indexById.get(connection.outNode);
      if (from === undefined || to === undefined) continue;
      incoming[to].push({ from, weight: connection.weight });
    }
    const depth = new Array(nodes.length).fill(0);
    let changed = true;
    let guard = 0;
    while (changed && guard++ <= nodes.length) {
      changed = false;
      for (let to = 0; to < nodes.length; to++) {
        for (const edge of incoming[to]) {
          if (depth[edge.from] + 1 > depth[to]) {
            depth[to] = depth[edge.from] + 1;
            changed = true;
          }
        }
      }
    }
    const order = nodes.map((node, index) => index)
      .filter(index => nodes[index].type === "hidden" || nodes[index].type === "output")
      .sort((a, b) => depth[a] - depth[b]);
    const outputIndices = nodes.map((node, index) => index)
      .filter(index => nodes[index].type === "output")
      .sort((a, b) => nodes[a].id - nodes[b].id);
    const values = new Float32Array(nodes.length);

    function activate(input) {
      for (let index = 0; index < nodes.length; index++) {
        if (nodes[index].type === "bias") values[index] = 1;
        else if (nodes[index].type === "input") values[index] = input[nodes[index].id - 1];
      }
      for (const index of order) {
        let sum = 0;
        for (const edge of incoming[index]) sum += values[edge.from] * edge.weight;
        values[index] = Math.tanh(sum);
      }
      const output = new Float32Array(outputIndices.length);
      for (let o = 0; o < outputIndices.length; o++) output[o] = values[outputIndices[o]];
      return output;
    }

    return { activate, values, depth, nodes };
  }

  // Layout description for rendering: node depths with outputs pushed to the last column.
  function toGraph(genome) {
    const network = buildNetwork(genome);
    const nodes = genome.nodes;
    const depth = network.depth.slice();
    let maxDepth = 1;
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].type !== "output") maxDepth = Math.max(maxDepth, depth[i]);
    }
    maxDepth++;
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].type === "output") depth[i] = maxDepth;
    }
    return {
      nodes, depth, maxDepth,
      inputCount: genome.inputCount,
      links: genome.connections.filter(connection => connection.enabled)
    };
  }

  function networkStats(genome) {
    return {
      nodes: genome.nodes.length,
      hidden: genome.nodes.filter(node => node.type === "hidden").length,
      connections: genome.connections.filter(connection => connection.enabled).length
    };
  }

  return { resetInnovations, createPopulation, evolve, buildNetwork, toGraph, networkStats, cloneGenome };
}
