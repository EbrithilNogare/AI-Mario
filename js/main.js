"use strict";

const NeuralNetwork = createNeuralNetworkModule();
const Simulation = createSimulationModule(NeuralNetwork);
const Level = createLevelModule();
const Genetics = createGeneticsModule(NeuralNetwork);
const NEAT = createNeatModule();
const RL = createRlModule(NeuralNetwork);
const Trainer = createTrainerModule(Simulation, NeuralNetwork, NEAT);
const Rendering = createRenderingModule(Simulation);

const DEFAULT_INPUTS = [
  "obstacleDistance", "obstacleHeight", "pitDistance", "pitWidth", "spikeDistance", "springDistance",
  "rayForward", "rayUp", "enemy1", "coin1", "velocityY"
];

const INPUT_TIPS = {
  velocityX: "current horizontal speed — tells the AI whether it is actually moving or pushed against a wall",
  velocityY: "current vertical speed — tells the AI whether it is rising, falling or standing still",
  onGround: "1 when standing on something, 0 while mid-air — helps time jumps",
  playerHeight: "how high above ground level it currently is — useful for judging jump arcs",
  obstacleDistance: "distance to the next pipe ahead — stays 0 while over/against it, 1 = nothing in sight",
  obstacleHeight: "how tall the next pipe ahead is — decides whether one jump clears it",
  pitDistance: "distance to the next pit ahead — stays 0 the whole time it is above the pit",
  pitWidth: "how wide the next pit ahead is — decides when to commit to the jump",
  spikeDistance: "distance to the next spike patch ahead — stays 0 while over it",
  springDistance: "distance to the next spring ahead — springs bounce extra high",
  enemy1: "relative x/y position of the nearest enemy ahead",
  enemy2: "relative x/y position of the second nearest enemy ahead",
  enemy1Velocity: "velocity of the nearest enemy ahead — is it approaching or retreating, rising or diving",
  enemy2Velocity: "velocity of the second nearest enemy ahead",
  enemy1Type: "1 if the nearest enemy can be stomped, -1 for a spiny (deadly from above), 0 if none",
  coin1: "relative x/y position of the nearest uncollected coin ahead",
  coin2: "relative x/y position of the second nearest coin ahead",
  coin3: "relative x/y position of the third nearest coin ahead",
  rayForward: "how far it can see straight ahead before hitting something solid",
  rayUp: "how far it can see straight up before hitting something solid — useful under platforms",
  rayDown: "how far below the ground is — while airborne this tells it whether a landing spot exists (a pit reads as bottomless)",
  flagDistance: "remaining distance to the flag (1 = at the start, 0 = at the flag) — a sense of overall progress",
  tileGrid: "a width x height grid of tiles that follows the player (row 0 = under the feet, column 0 = one cell behind), each marked solid / hazard / coin — the richest (and biggest) input"
};

const OUTPUT_TIPS = [
  "do nothing this frame — cheap way to wait for an enemy to pass",
  "run left — rarely needed since the goal is to the right, but allows backing off",
  "run right — the main way forward",
  "jump straight up (only works while standing on something)",
  "jump while running right — convenience combo; off by default so the AI must learn to chain jump and right itself"
];

const BENCHMARK_SEED = 20260713;

const state = {
  algorithm: "ga",
  levelSeed: 12345,
  level: null,
  levels: [],
  mapWindow: 10,
  benchmarkLevel: null,
  inputConfig: { selected: DEFAULT_INPUTS.slice(), gridWidth: 4, gridHeight: 3 },
  actionMap: [0, 1, 2, 3], // enabled global action indices; default: all but jump+right
  sensorReader: null,
  hiddenLayers: 3,
  layerSize: 10,
  shape: null,
  populationSize: 60,
  threadCount: Math.min(4, navigator.hardwareConcurrency || 4),
  rewards: { ...Simulation.DEFAULT_REWARDS },
  neat: { speciesTarget: 8 },
  rl: { learningRate: 0.002, batchSize: 64, epochsPerRun: 4, dropFrames: 45, autoStrategy: "imitation" },
  rlMode: "human", // "human" = you demonstrate, "auto" = self-training
  evalLevels: [], // fixed suite of held-out maps: scores stay comparable run-to-run
  training: false,
  watching: false,
  watchRun: null
};

// Each learning method keeps its own model + history, so switching preserves progress.
const algoState = {
  ga: { population: [], bestGenome: null, generation: 0, history: [], levelMarkers: [] },
  neat: { population: [], bestGenome: null, bestNetwork: null, speciesCount: 1, generation: 0, history: [], levelMarkers: [] },
  rl: {
    agent: null, bestGenome: null, bestScore: -Infinity, generation: 0, history: [], levelMarkers: [],
    epsilon: 0.3, lastBest: 0, lastAverage: 0, suiteMeasured: false
  }
};

const ALGORITHM_INFO = {
  ga: { generationLabel: "gen" },
  neat: { generationLabel: "gen" },
  rl: { generationLabel: "gen" }
};

let trainerPool = null;
let trainToken = 0;
let watchToken = 0;

const element = id => document.getElementById(id);

const controllers = {
  ga: {
    reset() {
      const A = algoState.ga;
      A.population = Genetics.createPopulation(state.populationSize, state.shape);
      A.bestGenome = A.population[0].slice();
      A.generation = 0;
      A.history = [];
      A.levelMarkers = [];
    },
    async iterate() {
      const A = algoState.ga;
      const fitnesses = await trainerPool.evaluatePopulation(A.population);
      const result = Genetics.evolve(A.population, fitnesses, state.populationSize);
      A.population = result.population;
      A.bestGenome = result.bestGenome;
      A.generation++;
      const benchmark = Simulation.evaluate(
        result.bestGenome, state.shape, state.benchmarkLevel, state.sensorReader, state.rewards, state.actionMap
      );
      A.history.push({ best: result.bestFitness, average: result.averageFitness, benchmark });
    },
    act(sensorVector) {
      const result = NeuralNetwork.forward(algoState.ga.bestGenome, state.shape, sensorVector, true);
      const chosenIndex = NeuralNetwork.argmax(result.output);
      return { actionIndex: state.actionMap[chosenIndex], chosenIndex, outputs: result.output, activations: result.activations };
    },
    drawNetwork(actResult) {
      Rendering.drawNetwork(state.shape, algoState.ga.bestGenome,
        actResult ? actResult.activations : null, state.sensorReader.labels, outputLabels());
    },
    statsText() {
      return "";
    }
  },

  neat: {
    reset() {
      const A = algoState.neat;
      A.population = NEAT.createPopulation(state.populationSize, state.sensorReader.size, state.actionMap.length);
      A.bestGenome = NEAT.cloneGenome(A.population[0]);
      A.bestNetwork = NEAT.buildNetwork(A.bestGenome);
      A.speciesCount = 1;
      A.generation = 0;
      A.history = [];
      A.levelMarkers = [];
    },
    async iterate() {
      const A = algoState.neat;
      const fitnesses = await trainerPool.evaluatePopulation(A.population);
      const result = NEAT.evolve(A.population, fitnesses, {
        targetSize: state.populationSize,
        speciesTarget: state.neat.speciesTarget
      });
      A.population = result.population;
      A.bestGenome = result.bestGenome;
      A.bestNetwork = NEAT.buildNetwork(A.bestGenome);
      A.speciesCount = result.speciesCount;
      A.generation++;
      const network = A.bestNetwork;
      const benchmark = Simulation.evaluateWith(
        vector => state.actionMap[NeuralNetwork.argmax(network.activate(vector))],
        state.benchmarkLevel, state.sensorReader, state.rewards
      );
      A.history.push({ best: result.bestFitness, average: result.averageFitness, benchmark });
    },
    act(sensorVector) {
      const A = algoState.neat;
      const outputs = A.bestNetwork.activate(sensorVector);
      const chosenIndex = NeuralNetwork.argmax(outputs);
      return { actionIndex: state.actionMap[chosenIndex], chosenIndex, outputs };
    },
    drawNetwork(actResult) {
      const A = algoState.neat;
      Rendering.drawNeatNetwork(NEAT.toGraph(A.bestGenome),
        actResult ? A.bestNetwork.values : null, state.sensorReader.labels, outputLabels());
    },
    statsText() {
      const A = algoState.neat;
      const stats = NEAT.networkStats(A.bestGenome);
      return "species " + A.speciesCount + " · hidden " + stats.hidden;
    }
  },

  rl: {
    reset() {
      const A = algoState.rl;
      A.agent = RL.createAgent(state.shape, state.rl);
      A.bestGenome = null;
      A.bestScore = -Infinity;
      A.generation = 0;
      A.history = [];
      A.levelMarkers = [];
      A.epsilon = 0.3;
      A.lastBest = 0;
      A.lastAverage = 0;
      A.suiteMeasured = false;
    },
    act(sensorVector) {
      const A = algoState.rl;
      const result = NeuralNetwork.forward(A.bestGenome || A.agent.genome, state.shape, sensorVector, true, true, true);
      let maxAbs = 1;
      for (const value of result.output) maxAbs = Math.max(maxAbs, Math.abs(value));
      const display = Float32Array.from(result.output, value => value / maxAbs); // logits are unbounded
      const chosenIndex = NeuralNetwork.argmax(result.output);
      return { actionIndex: state.actionMap[chosenIndex], chosenIndex, outputs: display, activations: result.activations };
    },
    drawNetwork(actResult) {
      const A = algoState.rl;
      Rendering.drawNetwork(state.shape, A.bestGenome || A.agent.genome,
        actResult ? actResult.activations : null, state.sensorReader.labels, outputLabels());
    },
    statsText() {
      const A = algoState.rl;
      return A.agent
        ? "demos " + RL.datasetSize(A.agent) + " frames · " + A.agent.episodes + " runs · ε " + A.epsilon.toFixed(2)
        : "";
    }
  }
};

// ---- imitation training: human demonstrations or ε-greedy self-play ----

const IDLE_WEIGHT = 0.3; // idle frames count less, so "do nothing" can't dominate
const AUTO_STEPS_PER_FRAME = 6; // auto runs play accelerated
const GENERATION_MS = 1000;
const EVAL_SEEDS = Array.from({ length: 10 }, (_, i) => BENCHMARK_SEED + 7919 * (i + 1));
const HILLCLIMB_SIGMA = 0.05;
const HILLCLIMB_RATE = 0.2;

const keyState = { left: false, right: false, jump: false };
let playRun = null;
let playFrames = null;
let playCommitted = 0; // frames already fed into the dataset this run
let playSeenInput = false;
let lastGenerationTime = 0;
let autoCtx = null; // per-run state of the active auto strategy

function bindKeys() {
  const keyMap = {
    KeyA: "left", ArrowLeft: "left",
    KeyD: "right", ArrowRight: "right",
    KeyW: "jump", ArrowUp: "jump", Space: "jump"
  };
  const onKey = (event, down) => {
    const key = keyMap[event.code];
    if (!key) return;
    keyState[key] = down;
    if (state.training && state.algorithm === "rl") event.preventDefault();
  };
  window.addEventListener("keydown", event => onKey(event, true));
  window.addEventListener("keyup", event => onKey(event, false));
}

// Composes the pressed keys into the closest enabled action. Jump only fires
// on the ground (like the physics); mid-air the horizontal keys keep steering,
// which is exactly how the AI has to chain jump and right too.
function humanAction(run) {
  const enabled = action => state.actionMap.includes(action);
  const jump = keyState.jump && run.onGround;
  if (jump && keyState.right && enabled(4)) return 4;
  if (jump && enabled(3)) return 3;
  if (keyState.right && enabled(2)) return 2;
  if (keyState.left && enabled(1)) return 1;
  return state.actionMap[0];
}

function commitFrame(agent, frame, pool) {
  RL.addFrame(agent, frame, frame.action === state.actionMap.indexOf(0) ? IDLE_WEIGHT : 1, pool);
}

// Greedy score of a genome on the fixed held-out suite — deterministic, so
// before/after comparisons (rollback, hill-climb adoption) are noise-free.
async function evalOnSuite(genome) {
  let best = -Infinity;
  let sum = 0;
  for (const level of state.evalLevels) {
    const score = rlGreedyScore(genome, level);
    best = Math.max(best, score);
    sum += score;
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return { best, average: sum / state.evalLevels.length };
}

function beginAutoRun(A) {
  const strategy = state.rl.autoStrategy;
  if (strategy === "hillclimb") {
    return { strategy, candidate: RL.perturb(A.agent.genome, HILLCLIMB_SIGMA, HILLCLIMB_RATE) };
  }
  if (strategy === "dqn") {
    RL.initDqn(A.agent);
    return { strategy, pending: [], lastVector: null, lastAction: 0, lastFitness: 0, sinceDecision: 0 };
  }
  return {
    strategy: "imitation",
    baseline: rlGreedyScore(A.agent.genome, state.level), // what greedy play scores on THIS level
    snapshot: RL.snapshot(A.agent)
  };
}

// Close the current DQN decision: n-step reward from the fitness delta.
function finalizeDqnDecision(A, c, run, nextVector) {
  if (!c.lastVector) return;
  const fitnessAfter = Simulation.fitness(run, state.level, state.rewards);
  c.pending.push({
    state: c.lastVector, action: c.lastAction,
    reward: (fitnessAfter - c.lastFitness) / RL.DQN.REWARD_SCALE
  });
  const flush = done => {
    let nStepReward = 0;
    for (let i = c.pending.length - 1; i >= 0; i--) {
      nStepReward = c.pending[i].reward + RL.DQN.GAMMA * nStepReward;
    }
    const first = c.pending.shift();
    RL.dqnRemember(A.agent, first.state, first.action, nStepReward, nextVector, done);
  };
  if (run.dead) {
    // timeout is a time limit, not an outcome — only real deaths/wins are terminal
    const terminal = run.killedByHazard || run.won;
    while (c.pending.length) flush(terminal);
  } else if (c.pending.length === RL.DQN.N_STEP) {
    flush(false);
  }
  c.lastVector = null;
}

// One "generation" per second of training: score the current policy on the
// reference level (blue line); best/average carry the latest 10-map results.
function rlGenerationPoint() {
  const A = algoState.rl;
  const benchmark = rlGreedyScore(A.agent.genome, state.benchmarkLevel);
  if (benchmark > A.bestScore) {
    A.bestScore = benchmark;
    A.bestGenome = A.agent.genome.slice();
  }
  A.generation++;
  A.history.push({ best: A.lastBest, average: A.lastAverage, benchmark });
  updateStats();
  Rendering.drawChart(A.history, A.levelMarkers);
}

function rlLoop(token) {
  if (!state.training || token !== trainToken) return;
  const A = algoState.rl;
  const auto = state.rlMode === "auto";
  if (!playRun) {
    playRun = Simulation.newRun(state.level);
    playFrames = [];
    playCommitted = 0;
    playSeenInput = false;
    lastGenerationTime = performance.now();
    autoCtx = auto ? beginAutoRun(A) : null;
    setMode(auto
      ? "auto · " + state.rl.autoStrategy + (state.rl.autoStrategy === "hillclimb" ? "" : " · ε " + A.epsilon.toFixed(2))
      : "YOU play — WASD / space (learning live)");
  }
  const run = playRun;

  // a human run waits for the first key press — no idle frames are recorded
  // and no time budget burns while you get ready
  if (!auto && !playSeenInput) {
    if (!keyState.left && !keyState.right && !keyState.jump) {
      Rendering.drawGame(run, state.level, null, null, {
        chosenIndex: -1, actionLabel: "press a key to start",
        outputs: new Float32Array(state.actionMap.length), labels: outputLabels(),
        fitness: 0
      });
      lastGenerationTime = performance.now();
      requestAnimationFrame(() => rlLoop(token));
      return;
    }
    playSeenInput = true;
  }

  let sensorVector = null;
  let networkIndex = 0;
  let action = 0;
  const steps = auto ? AUTO_STEPS_PER_FRAME : 1;
  for (let s = 0; s < steps && !run.dead; s++) {
    sensorVector = state.sensorReader.read(run, state.level);
    if (!auto) {
      action = humanAction(run);
      networkIndex = Math.max(0, state.actionMap.indexOf(action));
      playFrames.push({ state: sensorVector, action: networkIndex });
    } else if (autoCtx.strategy === "hillclimb") {
      // the candidate plays greedily; no recording, no backprop — evaluation
      // at the end decides whether its weights replace the current ones
      networkIndex = NeuralNetwork.argmax(
        NeuralNetwork.forward(autoCtx.candidate, state.shape, sensorVector, false, true, true).output);
      action = state.actionMap[networkIndex];
    } else if (autoCtx.strategy === "dqn") {
      const c = autoCtx;
      if (c.sinceDecision === 0) {
        finalizeDqnDecision(A, c, run, sensorVector);
        c.lastVector = sensorVector;
        c.lastFitness = Simulation.fitness(run, state.level, state.rewards);
        c.lastAction = Math.random() < A.epsilon
          ? (Math.random() * state.actionMap.length) | 0
          : RL.act(A.agent, sensorVector);
        if (A.agent.dqn.replay.size >= RL.DQN.WARMUP_TRANSITIONS) {
          RL.dqnTrainBatch(A.agent);
          RL.dqnTrainBatch(A.agent);
        }
      }
      c.sinceDecision = (c.sinceDecision + 1) % RL.DQN.ACTION_REPEAT;
      networkIndex = c.lastAction;
      action = state.actionMap[networkIndex];
    } else { // self-imitation
      const explore = Math.random() < A.epsilon;
      networkIndex = explore
        ? (Math.random() * state.actionMap.length) | 0
        : RL.act(A.agent, sensorVector);
      action = state.actionMap[networkIndex];
      // exploration frames are recorded for display but never imitated
      playFrames.push({ state: sensorVector, action: networkIndex, explore });
    }
    Simulation.step(run, state.level, action);
  }

  // Human demos stream into the dataset once survived by the drop window, so
  // a death's fatal tail is never learned, and train one batch per frame.
  // Auto strategies train at run end (imitation/hill-climb) or per decision (dqn).
  if (!auto) {
    while (playCommitted < playFrames.length - state.rl.dropFrames) {
      commitFrame(A.agent, playFrames[playCommitted++], "human");
    }
    if (RL.datasetSize(A.agent) >= state.rl.batchSize) RL.trainBatch(A.agent);
  }

  if (run.steps % 15 === 0) {
    Rendering.drawNetwork(state.shape, A.agent.genome, null, state.sensorReader.labels, outputLabels());
  }
  const outputs = new Float32Array(state.actionMap.length);
  outputs[networkIndex] = 1;
  Rendering.drawGame(run, state.level, state.sensorReader, sensorVector, {
    chosenIndex: networkIndex,
    actionLabel: Simulation.OUTPUT_LABELS[action],
    outputs,
    labels: outputLabels(),
    fitness: Simulation.fitness(run, state.level, state.rewards)
  });

  if (performance.now() - lastGenerationTime >= GENERATION_MS) {
    lastGenerationTime = performance.now();
    rlGenerationPoint();
  }
  if (run.dead) {
    finishRlRun(token, run);
    return;
  }
  requestAnimationFrame(() => rlLoop(token));
}

async function finishRlRun(token, run) {
  const A = algoState.rl;
  const auto = state.rlMode === "auto";
  const c = autoCtx;
  const runScore = Simulation.fitness(run, state.level, state.rewards);
  // the fatal tail of a hazard death is never trained on
  const keepUntil = run.killedByHazard && !run.won
    ? Math.max(0, playFrames.length - state.rl.dropFrames)
    : playFrames.length;
  A.agent.episodes++;
  const frames = playFrames;
  playRun = null;
  playFrames = null;
  setMode("evaluating on " + state.evalLevels.length + " held-out maps…");

  // establish the baseline suite score once, before any auto decisions use it
  if (!A.suiteMeasured) {
    const initial = await evalOnSuite(A.agent.genome);
    if (!state.training || token !== trainToken) return;
    A.lastBest = initial.best;
    A.lastAverage = initial.average;
    A.suiteMeasured = true;
  }

  if (!auto) {
    while (playCommitted < keepUntil) commitFrame(A.agent, frames[playCommitted++], "human");
    if (run.won) for (const frame of frames) commitFrame(A.agent, frame, "human"); // wins count double
    await RL.train(A.agent, state.rl.epochsPerRun);
    if (!state.training || token !== trainToken) return;
    const result = await evalOnSuite(A.agent.genome);
    if (!state.training || token !== trainToken) return;
    A.lastBest = result.best;
    A.lastAverage = result.average;
  } else if (c.strategy === "imitation") {
    // gate against greedy play on this very level: only imitate the run if
    // exploration actually found something better here
    if (run.won || runScore > c.baseline) {
      for (let i = 0; i < keepUntil; i++) {
        if (!frames[i].explore) commitFrame(A.agent, frames[i], "auto");
      }
      await RL.train(A.agent, state.rl.epochsPerRun);
      if (!state.training || token !== trainToken) return;
      const result = await evalOnSuite(A.agent.genome);
      if (!state.training || token !== trainToken) return;
      if (result.average >= A.lastAverage) {
        A.lastAverage = result.average;
        A.lastBest = result.best;
      } else {
        RL.restore(A.agent, c.snapshot); // the run's training made it worse — undo
      }
    }
    A.epsilon = Math.max(0.02, A.epsilon * 0.98);
  } else if (c.strategy === "hillclimb") {
    const result = await evalOnSuite(c.candidate);
    if (!state.training || token !== trainToken) return;
    if (result.average > A.lastAverage) {
      A.agent.genome.set(c.candidate);
      A.agent.adamM.fill(0); // fresh optimizer state for the new weights
      A.agent.adamV.fill(0);
      A.agent.adamT = 0;
      A.lastAverage = result.average;
      A.lastBest = result.best;
    }
  } else { // dqn
    finalizeDqnDecision(A, c, run, state.sensorReader.read(run, state.level));
    A.epsilon = Math.max(0.02, A.epsilon * 0.98);
    const result = await evalOnSuite(A.agent.genome);
    if (!state.training || token !== trainToken) return;
    A.lastBest = result.best;
    A.lastAverage = result.average;
  }

  autoCtx = null;
  rlGenerationPoint();
  updateNetworkInfo();
  activeController().drawNetwork(null);
  addMap();
  requestAnimationFrame(() => rlLoop(token));
}

function activeController() {
  return controllers[state.algorithm];
}

function activeState() {
  return algoState[state.algorithm];
}

function readInputConfig() {
  const selected = Simulation.INPUT_DEFINITIONS
    .map(definition => definition.id)
    .filter(id => element("input_" + id).checked);
  return {
    selected: selected.length ? selected : ["obstacleDistance"],
    gridWidth: Math.max(1, Math.min(8, +element("gridWidth").value || 4)),
    gridHeight: Math.max(1, Math.min(6, +element("gridHeight").value || 3))
  };
}

function rebuildShared() {
  state.sensorReader = Simulation.createSensorReader(state.inputConfig);
  state.shape = NeuralNetwork.buildShape(
    state.sensorReader.size, state.hiddenLayers, state.layerSize, state.actionMap.length
  );
}

function outputLabels() {
  return state.actionMap.map(index => Simulation.OUTPUT_LABELS[index]);
}

function rlGreedyScore(genome, level) {
  return Simulation.evaluateWith(vector => {
    const q = NeuralNetwork.forward(genome, state.shape, vector, false, true, true).output;
    return state.actionMap[NeuralNetwork.argmax(q)];
  }, level, state.sensorReader, state.rewards);
}

function rebuildTrainerPool() {
  if (trainerPool) trainerPool.terminate();
  trainerPool = new Trainer.TrainerPool(state.threadCount);
  configureTrainerPool();
}

function configureTrainerPool() {
  trainerPool.configure(state.levels, state.shape, state.inputConfig, state.rewards, state.algorithm, state.actionMap);
}

function addMap() {
  state.levelSeed = (Math.random() * 1e9) | 0;
  state.level = Level.buildLevel(state.levelSeed);
  state.levels.push(state.level);
  while (state.levels.length > state.mapWindow) state.levels.shift();
}

function setMode(mode) {
  element("modeValue").textContent = mode;
}

function updateStats() {
  const A = activeState();
  element("genLabel").textContent = ALGORITHM_INFO[state.algorithm].generationLabel;
  element("generationValue").textContent = A.generation;
  element("bestValue").textContent = A.history.length
    ? A.history[A.history.length - 1].best | 0 : 0;
  element("algoStats").textContent = activeController().statsText();
}

function updateNetworkInfo() {
  element("inputCount").textContent = state.sensorReader.size;
  element("outputCount").textContent = state.actionMap.length;
  if (state.algorithm === "neat") {
    element("weightLabel").textContent = "connections";
    element("weightCount").textContent = NEAT.networkStats(algoState.neat.bestGenome).connections;
  } else {
    element("weightLabel").textContent = "weights";
    element("weightCount").textContent = NeuralNetwork.weightCount(state.shape);
  }
}

function trainButtonLabel() {
  return state.algorithm === "rl" ? "TRAIN BY HUMAN" : "TRAIN";
}

function stopAll() {
  state.training = false;
  state.watching = false;
  playRun = null;
  playFrames = null;
  trainToken++;
  watchToken++;
  element("buttonTrain").textContent = trainButtonLabel();
  element("buttonTrainAuto").textContent = "TRAIN AUTO";
  setMode("idle");
}

function redrawAll() {
  updateStats();
  updateNetworkInfo();
  const A = activeState();
  Rendering.drawChart(A.history, A.levelMarkers);
  activeController().drawNetwork(null);
  Rendering.drawGame(Simulation.newRun(state.level), state.level, null, null);
  layoutHelpPanel(); // panel heights may have changed
}

async function runIteration() {
  addMap();
  configureTrainerPool();
  await activeController().iterate();
}

async function trainLoop() {
  const token = ++trainToken;
  while (state.training && token === trainToken) {
    await runIteration();
    if (token !== trainToken) return;
    updateStats();
    updateNetworkInfo();
    const A = activeState();
    Rendering.drawChart(A.history, A.levelMarkers);
    activeController().drawNetwork(null);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
}

function watchLoop(token) {
  if (!state.watching || token !== watchToken) return;
  if (!state.watchRun || state.watchRun.dead) {
    if (state.watchRun && state.watchRun.dead) {
      state.watching = false;
      setMode("idle");
      return;
    }
    state.watchRun = Simulation.newRun(state.level);
  }
  const controller = activeController();
  const sensorVector = state.sensorReader.read(state.watchRun, state.level);
  const actResult = controller.act(sensorVector);
  Simulation.step(state.watchRun, state.level, actResult.actionIndex);
  const debug = {
    chosenIndex: actResult.chosenIndex,
    actionLabel: Simulation.OUTPUT_LABELS[actResult.actionIndex],
    outputs: actResult.outputs,
    labels: outputLabels(),
    fitness: Simulation.fitness(state.watchRun, state.level, state.rewards)
  };
  Rendering.drawGame(state.watchRun, state.level, state.sensorReader, sensorVector, debug);
  controller.drawNetwork(actResult);
  requestAnimationFrame(() => watchLoop(token));
}

function applyAlgorithmUi() {
  for (const key of ["ga", "neat", "rl"]) {
    element("algo_" + key).classList.toggle("active", key === state.algorithm);
  }
  document.querySelectorAll("[data-algo]").forEach(node => {
    node.classList.toggle("algo-hidden", !node.getAttribute("data-algo").split(" ").includes(state.algorithm));
  });
  document.querySelectorAll("[data-help-algo]").forEach(node => {
    node.classList.toggle("help-active", node.getAttribute("data-help-algo") === state.algorithm);
  });
  if (!state.training) element("buttonTrain").textContent = trainButtonLabel();
  layoutHelpPanel(); // panel heights change when per-method rows show/hide
}

function setHelpVisible(visible) {
  element("helpPanel").classList.toggle("help-hidden", !visible);
  element("helpOpen").classList.toggle("help-hidden", visible);
  if (visible) layoutHelpPanel();
}

// Docked mode: the dialog sits right of the content as one bordered block,
// each section top-aligned with (and never taller than) the panel it explains.
function layoutHelpPanel() {
  const help = element("helpPanel");
  if (help.classList.contains("help-hidden")) return;
  const firstPanel = element("panelMethod");
  const lastPanel = element("panelProgress");
  const contentRight = firstPanel.getBoundingClientRect().right + window.scrollX;
  const available = document.documentElement.clientWidth - contentRight - 8 - 16;
  const docked = available >= 220;
  help.classList.toggle("help-docked", docked);
  const sections = help.querySelectorAll(".help-section");
  if (!docked) {
    help.style.left = help.style.top = help.style.height = help.style.width = "";
    sections.forEach(section => {
      section.style.top = "";
      section.style.maxHeight = "";
    });
    return;
  }
  const top = document.querySelector("h1").getBoundingClientRect().top + window.scrollY;
  const firstTop = firstPanel.getBoundingClientRect().top + window.scrollY;
  const bottom = lastPanel.getBoundingClientRect().bottom + window.scrollY;
  help.style.left = (contentRight + 8) + "px";
  help.style.top = top + "px";
  help.style.width = Math.min(400, available) + "px";
  help.style.height = (bottom - top) + "px";
  for (const section of sections) {
    const targetId = section.getAttribute("data-help-for");
    if (!targetId) { // intro block: fills the space beside the page header
      section.style.top = "4px";
      section.style.maxHeight = (firstTop - top - 2) + "px";
      continue;
    }
    const rect = element(targetId).getBoundingClientRect();
    section.style.top = (rect.top + window.scrollY - top + 4) + "px";
    section.style.maxHeight = (rect.height - 8) + "px";
  }
}

function setAlgorithm(id) {
  if (state.algorithm === id) return;
  stopAll();
  state.algorithm = id;
  state.watchRun = null;
  applyAlgorithmUi();
  configureTrainerPool();
  redrawAll();
}

function resetAllAlgorithms() {
  rebuildShared();
  controllers.ga.reset();
  controllers.neat.reset();
  controllers.rl.reset();
}

function onInputConfigChanged() {
  stopAll();
  state.inputConfig = readInputConfig();
  resetAllAlgorithms();
  configureTrainerPool();
  redrawAll();
}

function onShapeChanged() {
  stopAll();
  rebuildShared();
  controllers.ga.reset();
  controllers.rl.reset();
  configureTrainerPool();
  redrawAll();
}

function readOutputConfig() {
  const selected = Simulation.OUTPUT_LABELS
    .map((_, index) => index)
    .filter(index => element("output_" + index).checked);
  return selected.length ? selected : [2]; // never empty: fall back to "right"
}

function onOutputConfigChanged() {
  stopAll();
  state.actionMap = readOutputConfig();
  resetAllAlgorithms();
  configureTrainerPool();
  redrawAll();
}

function buildOutputCheckboxes() {
  const container = element("outputList");
  Simulation.OUTPUT_LABELS.forEach((outputLabel, index) => {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "output_" + index;
    checkbox.checked = state.actionMap.includes(index);
    checkbox.onchange = onOutputConfigChanged;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(outputLabel));
    const tip = document.createElement("sup");
    tip.className = "tip";
    tip.setAttribute("data-tip", OUTPUT_TIPS[index]);
    tip.textContent = "?";
    label.appendChild(tip);
    container.appendChild(label);
  });
}

function buildInputCheckboxes() {
  for (const definition of Simulation.INPUT_DEFINITIONS) {
    const container = element("inputGroup_" + definition.group);
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "input_" + definition.id;
    checkbox.checked = DEFAULT_INPUTS.includes(definition.id);
    checkbox.onchange = onInputConfigChanged;
    label.appendChild(checkbox);
    label.appendChild(document.createTextNode(definition.label));
    if (INPUT_TIPS[definition.id]) {
      const tip = document.createElement("sup");
      tip.className = "tip";
      tip.setAttribute("data-tip", INPUT_TIPS[definition.id]);
      tip.textContent = "?";
      label.appendChild(tip);
    }
    container.appendChild(label);
  }
}

function bindControls() {
  element("helpOpen").onclick = () => setHelpVisible(true);
  element("helpClose").onclick = () => setHelpVisible(false);

  element("algo_ga").onclick = () => setAlgorithm("ga");
  element("algo_neat").onclick = () => setAlgorithm("neat");
  element("algo_rl").onclick = () => setAlgorithm("rl");

  element("gridWidth").onchange = onInputConfigChanged;
  element("gridHeight").onchange = onInputConfigChanged;

  element("hiddenLayers").oninput = event => {
    state.hiddenLayers = +event.target.value;
    element("hiddenLayersValue").textContent = state.hiddenLayers;
    onShapeChanged();
  };
  element("layerSize").oninput = event => {
    state.layerSize = +event.target.value;
    element("layerSizeValue").textContent = state.layerSize;
    onShapeChanged();
  };
  element("populationSize").oninput = event => {
    state.populationSize = +event.target.value;
    element("populationSizeValue").textContent = state.populationSize;
  };
  element("threadCount").oninput = event => {
    state.threadCount = +event.target.value;
    element("threadCountValue").textContent = state.threadCount;
    rebuildTrainerPool();
  };

  const bindNumber = (id, target, key, min, max) => {
    element(id).onchange = event => {
      const value = +event.target.value;
      if (Number.isFinite(value)) target[key] = Math.min(max, Math.max(min, value));
      event.target.value = target[key];
    };
  };
  bindNumber("speciesTarget", state.neat, "speciesTarget", 1, 20);
  bindNumber("rlLearningRate", state.rl, "learningRate", 0.00001, 0.1);
  bindNumber("rlBatchSize", state.rl, "batchSize", 8, 256);
  bindNumber("rlEpochs", state.rl, "epochsPerRun", 1, 20);
  bindNumber("rlDropFrames", state.rl, "dropFrames", 0, 180);
  element("rlAutoStrategy").onchange = event => {
    if (state.training && state.rlMode === "auto") stopAll();
    state.rl.autoStrategy = event.target.value;
  };

  const startRlTraining = mode => {
    state.rlMode = mode;
    state.watching = false;
    state.training = true;
    element(mode === "auto" ? "buttonTrainAuto" : "buttonTrain").textContent = "STOP";
    const token = ++trainToken;
    playRun = null;
    rlLoop(token);
  };

  element("buttonTrain").onclick = () => {
    if (state.training) {
      stopAll();
      activeController().drawNetwork(null);
      return;
    }
    if (state.algorithm === "rl") {
      startRlTraining("human");
      return;
    }
    state.watching = false;
    state.training = true;
    element("buttonTrain").textContent = "STOP";
    setMode("training on " + state.threadCount + " thread" + (state.threadCount > 1 ? "s" : ""));
    trainLoop();
  };

  element("buttonTrainAuto").onclick = () => {
    if (state.training) {
      stopAll();
      activeController().drawNetwork(null);
      return;
    }
    startRlTraining("auto");
  };

  element("buttonWatch").onclick = () => {
    stopAll();
    state.watching = true;
    state.watchRun = null;
    setMode("watching best");
    watchLoop(watchToken);
  };

  element("buttonLevel").onclick = () => {
    addMap();
    state.watchRun = null;
    configureTrainerPool();
    if (!state.training && !state.watching) redrawAll();
  };

  element("mapWindow").onchange = event => {
    state.mapWindow = Math.max(1, +event.target.value || 1);
    event.target.value = state.mapWindow;
    while (state.levels.length > state.mapWindow) state.levels.shift();
    configureTrainerPool();
  };

  const bindReward = (id, key) => {
    element(id).onchange = event => {
      const value = +event.target.value;
      if (Number.isFinite(value)) state.rewards[key] = value;
      event.target.value = state.rewards[key];
      configureTrainerPool();
    };
  };
  bindReward("rewardCoin", "coin");
  bindReward("rewardKill", "enemyKill");
  bindReward("rewardDeath", "death");
  bindReward("rewardJump", "jump");
  bindReward("rewardDistance", "distance");
  bindReward("rewardTime", "time");
}

function initialize() {
  state.level = Level.buildLevel(state.levelSeed);
  state.levels = [state.level];
  state.benchmarkLevel = Level.buildLevel(BENCHMARK_SEED);
  state.evalLevels = EVAL_SEEDS.map(seed => Level.buildLevel(seed));
  buildInputCheckboxes();
  buildOutputCheckboxes();
  bindControls();
  bindKeys();
  element("threadCount").value = state.threadCount;
  element("threadCountValue").textContent = state.threadCount;
  resetAllAlgorithms();
  rebuildTrainerPool();
  applyAlgorithmUi();
  redrawAll();
  layoutHelpPanel();
  window.addEventListener("resize", layoutHelpPanel);
}

initialize();
