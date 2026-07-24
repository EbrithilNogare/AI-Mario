"use strict";

// Imitation learning core plus auto-training support:
// - two demonstration pools: human demos are never evicted by self-play, and
//   fill half of every training batch while both pools have data
// - snapshot/restore lets auto-training undo a run that made the policy worse
// - perturb() supports weight-space hill climbing
// - a compact double-DQN (n-step returns, soft target updates) warm-started
//   from the imitated network for reward-driven fine-tuning
// The policy plays greedily: argmax of the action logits / Q-values (both use
// ReLU hidden layers and a linear output, so all strategies share one genome).
function createRlModule(NeuralNetwork) {
  const POOL_CAPACITY = 30000;
  const MAX_UPDATES_PER_RUN = 800;
  const ADAM_BETA1 = 0.9;
  const ADAM_BETA2 = 0.999;
  const ADAM_EPSILON = 1e-8;

  // DQN fine-tuning
  const REPLAY_CAPACITY = 20000;
  const WARMUP_TRANSITIONS = 300;
  const REWARD_SCALE = 100;
  const GAMMA = 0.99;
  const N_STEP = 3;
  const ACTION_REPEAT = 4;
  const TARGET_TAU = 0.01;
  const ERROR_CLIP = 1;

  function gaussian() {
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
  }

  function newPool() {
    return { states: [], actions: [], weights: [], index: 0, size: 0 };
  }

  function createAgent(shape, config) {
    const genome = NeuralNetwork.heGenome(shape);
    return {
      shape, config, genome,
      gradients: new Float32Array(genome.length),
      adamM: new Float32Array(genome.length),
      adamV: new Float32Array(genome.length),
      adamT: 0,
      episodes: 0,
      updates: 0,
      pools: { human: newPool(), auto: newPool() },
      dqn: null
    };
  }

  function datasetSize(agent) {
    return agent.pools.human.size + agent.pools.auto.size;
  }

  // frame: { state, action } with action as a network output index.
  function addFrame(agent, frame, weight, poolName) {
    const pool = agent.pools[poolName || "human"];
    if (pool.size < POOL_CAPACITY) {
      pool.states.push(frame.state);
      pool.actions.push(frame.action);
      pool.weights.push(weight || 1);
      pool.size++;
    } else {
      const i = pool.index;
      pool.states[i] = frame.state;
      pool.actions[i] = frame.action;
      pool.weights[i] = weight || 1;
      pool.index = (i + 1) % POOL_CAPACITY;
    }
  }

  function samplePool(agent) {
    const { human, auto } = agent.pools;
    if (!auto.size) return human;
    if (!human.size) return auto;
    return Math.random() < 0.5 ? human : auto;
  }

  function adamStep(agent) {
    agent.adamT++;
    const correction1 = 1 - Math.pow(ADAM_BETA1, agent.adamT);
    const correction2 = 1 - Math.pow(ADAM_BETA2, agent.adamT);
    for (let i = 0; i < agent.genome.length; i++) {
      const gradient = agent.gradients[i];
      agent.adamM[i] = ADAM_BETA1 * agent.adamM[i] + (1 - ADAM_BETA1) * gradient;
      agent.adamV[i] = ADAM_BETA2 * agent.adamV[i] + (1 - ADAM_BETA2) * gradient * gradient;
      agent.genome[i] -= agent.config.learningRate * (agent.adamM[i] / correction1)
        / (Math.sqrt(agent.adamV[i] / correction2) + ADAM_EPSILON);
    }
    agent.updates++;
  }

  function trainBatch(agent) {
    const { config, shape } = agent;
    const batchSize = Math.min(Math.max(1, config.batchSize | 0), datasetSize(agent));
    const outputCount = shape[shape.length - 1];
    agent.gradients.fill(0);
    for (let n = 0; n < batchSize; n++) {
      const pool = samplePool(agent);
      const i = (Math.random() * pool.size) | 0;
      const activations = NeuralNetwork.forwardTrain(agent.genome, shape, pool.states[i], true, true);
      const logits = activations[activations.length - 1];
      let maxLogit = logits[0];
      for (let j = 1; j < outputCount; j++) maxLogit = Math.max(maxLogit, logits[j]);
      let expSum = 0;
      const outputGradient = new Float32Array(outputCount);
      for (let j = 0; j < outputCount; j++) {
        outputGradient[j] = Math.exp(logits[j] - maxLogit);
        expSum += outputGradient[j];
      }
      const scale = pool.weights[i] / batchSize;
      for (let j = 0; j < outputCount; j++) {
        outputGradient[j] = (outputGradient[j] / expSum - (j === pool.actions[i] ? 1 : 0)) * scale;
      }
      NeuralNetwork.backward(agent.genome, shape, activations, outputGradient, agent.gradients, true, true);
    }
    adamStep(agent);
  }

  // Runs `epochs` passes over the dataset, yielding to the UI between chunks.
  async function train(agent, epochs) {
    const size = datasetSize(agent);
    if (!size) return;
    const batchSize = Math.max(1, agent.config.batchSize | 0);
    const updates = Math.min(MAX_UPDATES_PER_RUN,
      Math.max(1, Math.ceil(size / batchSize) * Math.max(1, epochs | 0)));
    for (let u = 0; u < updates; u++) {
      trainBatch(agent);
      if (u % 20 === 19) await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  function act(agent, input) {
    return NeuralNetwork.argmax(
      NeuralNetwork.forward(agent.genome, agent.shape, input, false, true, true).output);
  }

  function snapshot(agent) {
    return {
      genome: agent.genome.slice(),
      adamM: agent.adamM.slice(),
      adamV: agent.adamV.slice(),
      adamT: agent.adamT
    };
  }

  function restore(agent, snap) {
    agent.genome.set(snap.genome);
    agent.adamM.set(snap.adamM);
    agent.adamV.set(snap.adamV);
    agent.adamT = snap.adamT;
  }

  // Small gaussian nudge on a fraction of the weights — a hill-climb candidate.
  function perturb(genome, sigma, rate) {
    const candidate = genome.slice();
    for (let i = 0; i < candidate.length; i++) {
      if (Math.random() < rate) candidate[i] += gaussian() * sigma;
    }
    return candidate;
  }

  // ---- DQN fine-tuning (warm-started: target net = current policy net) ----

  function initDqn(agent) {
    if (agent.dqn) return;
    agent.dqn = {
      target: agent.genome.slice(),
      replay: { states: [], actions: [], rewards: [], nextStates: [], dones: [], index: 0, size: 0 }
    };
  }

  function dqnRemember(agent, state, action, reward, nextState, done) {
    const replay = agent.dqn.replay;
    if (replay.size < REPLAY_CAPACITY) {
      replay.states.push(state);
      replay.actions.push(action);
      replay.rewards.push(reward);
      replay.nextStates.push(nextState);
      replay.dones.push(done);
      replay.size++;
    } else {
      const i = replay.index;
      replay.states[i] = state;
      replay.actions[i] = action;
      replay.rewards[i] = reward;
      replay.nextStates[i] = nextState;
      replay.dones[i] = done;
      replay.index = (i + 1) % REPLAY_CAPACITY;
    }
  }

  function qValues(genome, shape, input) {
    return NeuralNetwork.forward(genome, shape, input, false, true, true).output;
  }

  function dqnTrainBatch(agent) {
    const { config, shape } = agent;
    const replay = agent.dqn.replay;
    const batchSize = Math.min(Math.max(1, config.batchSize | 0), replay.size);
    const outputCount = shape[shape.length - 1];
    const bootstrapGamma = Math.pow(GAMMA, N_STEP); // stored rewards span N_STEP decisions
    agent.gradients.fill(0);
    for (let n = 0; n < batchSize; n++) {
      const i = (Math.random() * replay.size) | 0;
      const activations = NeuralNetwork.forwardTrain(agent.genome, shape, replay.states[i], true, true);
      const q = activations[activations.length - 1];
      let targetValue = replay.rewards[i];
      if (!replay.dones[i]) {
        // Double DQN: online net chooses, target net evaluates.
        const onlineNext = qValues(agent.genome, shape, replay.nextStates[i]);
        const targetNext = qValues(agent.dqn.target, shape, replay.nextStates[i]);
        targetValue += bootstrapGamma * targetNext[NeuralNetwork.argmax(onlineNext)];
      }
      const action = replay.actions[i];
      const error = Math.max(-ERROR_CLIP, Math.min(ERROR_CLIP, q[action] - targetValue));
      const outputGradient = new Float32Array(outputCount);
      outputGradient[action] = error / batchSize;
      NeuralNetwork.backward(agent.genome, shape, activations, outputGradient, agent.gradients, true, true);
    }
    adamStep(agent);
    for (let i = 0; i < agent.genome.length; i++) {
      agent.dqn.target[i] += TARGET_TAU * (agent.genome[i] - agent.dqn.target[i]);
    }
  }

  return {
    createAgent, datasetSize, addFrame, train, trainBatch, act,
    snapshot, restore, perturb,
    initDqn, dqnRemember, dqnTrainBatch,
    DQN: { N_STEP, ACTION_REPEAT, WARMUP_TRANSITIONS, REWARD_SCALE, GAMMA }
  };
}
