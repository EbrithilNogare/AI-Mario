"use strict";

function createLevelModule() {
  const GROUND_Y = 200;
  const LEVEL_LENGTH = 6000;

  const SAFE_RISE_FROM_GROUND = 56;
  const SAFE_RISE_STEP = 44;
  const SAFE_GAP = 72;
  const SAFE_PIT = 76;
  const MAX_PIPE_HEIGHT = 64;

  function mulberry32(seed) {
    return function () {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Follows a jump arc: low at both ends, peaking in the middle — so every
  // coin lies on the path of a normal jump across the gap.
  function coinArc(level, fromX, toX, peakY, count) {
    const edgeY = GROUND_Y - 30;
    for (let i = 0; i < count; i++) {
      const t = count === 1 ? 0.5 : i / (count - 1);
      const arc = 1 - (2 * t - 1) * (2 * t - 1);
      level.coins.push({ x: fromX + (toX - fromX) * t, y: edgeY + (peakY - edgeY) * arc });
    }
  }

  function coinLine(level, fromX, fromY, toX, toY, count) {
    for (let i = 0; i < count; i++) {
      const t = count === 1 ? 0 : i / (count - 1);
      level.coins.push({ x: fromX + (toX - fromX) * t, y: fromY + (toY - fromY) * t });
    }
  }

  function addFlat(level, x, difficulty, random) {
    const coinCount = 2 + (random() * 3 | 0);
    const height = random() < 0.5 ? 55 : 80; // max jump height is ~82px
    for (let i = 0; i < coinCount; i++) {
      level.coins.push({ x: x + i * 24, y: GROUND_Y - height });
    }
    return x + coinCount * 24;
  }

  function addPipe(level, x, difficulty, random) {
    const height = 32 + (random() < 0.4 + difficulty * 0.4 ? 16 : 0) + (random() < difficulty * 0.5 ? 16 : 0);
    level.obstacles.push({ x, w: 32, h: Math.min(MAX_PIPE_HEIGHT, height) });
    if (random() < 0.5) level.coins.push({ x: x + 16, y: GROUND_Y - height - 30 });
    return x + 32;
  }

  function addPipeIsland(level, x, difficulty, random) {
    const pipeHeight = 32 + (random() < 0.5 ? 16 : 0);
    const pipeTop = GROUND_Y - pipeHeight;
    level.obstacles.push({ x, w: 32, h: pipeHeight });

    const islandRise = Math.min(SAFE_RISE_FROM_GROUND, 40 + (random() * 16 | 0));
    const islandY = pipeTop - islandRise;
    const islandX = x + 40 + (random() * 24 | 0);
    const islandWidth = 64 + (random() * 2 | 0) * 16;
    level.platforms.push({ x: islandX, y: islandY, w: islandWidth });

    const coinCount = islandWidth / 24 | 0;
    for (let i = 0; i < coinCount; i++) level.coins.push({ x: islandX + 12 + i * 24, y: islandY - 22 });
    if (random() < 0.4 + difficulty * 0.4) {
      level.enemies.push({ x: islandX + islandWidth / 2, y: islandY, range: islandWidth / 2 - 12, type: "walker" });
    }
    return islandX + islandWidth;
  }

  function addTwoTierIslands(level, x, difficulty, random) {
    const lowRise = Math.min(SAFE_RISE_FROM_GROUND, 40 + (random() * 12 | 0));
    const lowY = GROUND_Y - lowRise;
    const lowWidth = 64;
    level.platforms.push({ x, y: lowY, w: lowWidth });

    const highRise = Math.min(SAFE_RISE_STEP, 36 + (random() * 8 | 0));
    const highY = lowY - highRise;
    const highX = x + lowWidth + Math.min(SAFE_GAP, 8 + (random() * 12 | 0));
    const highWidth = 64 + (random() * 2 | 0) * 16;
    level.platforms.push({ x: highX, y: highY, w: highWidth });

    coinLine(level, x + 16, lowY - 22, highX + 8, highY - 22, 4);
    for (let i = 0; i < (highWidth / 24 | 0); i++) level.coins.push({ x: highX + 12 + i * 24, y: highY - 22 });
    if (random() < 0.3 + difficulty * 0.5) {
      level.enemies.push({ x: highX + highWidth / 2, y: highY, range: highWidth / 2 - 12, type: "walker" });
    }
    return highX + highWidth;
  }

  function addPit(level, x, difficulty, random) {
    const width = 48 + (random() * 3 | 0) * 8;
    level.pits.push({ x, w: Math.min(SAFE_PIT, width) });
    coinArc(level, x + 6, x + width - 6, GROUND_Y - 78, 3);
    return x + Math.min(SAFE_PIT, width);
  }

  function addPitWithStones(level, x, difficulty, random) {
    const stoneCount = 1 + (random() < 0.4 + difficulty * 0.4 ? 1 : 0);
    const gap = Math.min(SAFE_GAP, 44 + (random() * 12 | 0));
    const stoneWidth = 40;
    const stoneRise = 18 + (random() * 14 | 0);
    const stoneY = GROUND_Y - stoneRise;

    let cursor = x;
    for (let i = 0; i < stoneCount; i++) {
      const stoneX = cursor + gap;
      level.platforms.push({ x: stoneX, y: stoneY, w: stoneWidth });
      for (let c = 0; c < 2; c++) level.coins.push({ x: stoneX + 8 + c * 20, y: stoneY - 22 });
      cursor = stoneX + stoneWidth;
    }
    const pitWidth = cursor + gap - x;
    level.pits.push({ x, w: pitWidth });
    return x + pitWidth;
  }

  function addGapToLedge(level, x, difficulty, random) {
    // a single jump must clear the pit AND land on the ledge, so both stay small
    const pitWidth = Math.min(56, 40 + (random() * 3 | 0) * 8);
    level.pits.push({ x, w: pitWidth });
    const ledgeHeight = 32;
    const ledgeX = x + pitWidth;
    level.obstacles.push({ x: ledgeX, w: 48 + (random() * 2 | 0) * 16, h: ledgeHeight });
    coinArc(level, x + 6, ledgeX - 6, GROUND_Y - ledgeHeight - 40, 3);
    const ledgeWidth = level.obstacles[level.obstacles.length - 1].w;
    if (random() < 0.4) level.coins.push({ x: ledgeX + ledgeWidth / 2, y: GROUND_Y - ledgeHeight - 26 });
    return ledgeX + ledgeWidth;
  }

  function addPipeStairs(level, x, difficulty, random) {
    const steps = 2 + (random() * 2 | 0);
    let stepX = x;
    for (let i = 0; i < steps; i++) {
      const height = 24 + i * 20;
      level.obstacles.push({ x: stepX, w: 28, h: height });
      level.coins.push({ x: stepX + 14, y: GROUND_Y - height - 26 });
      stepX += 28;
    }
    return stepX + 20;
  }

  function addWalkers(level, x, difficulty, random) {
    const count = 1 + (random() < difficulty * 0.7 ? 1 : 0);
    for (let i = 0; i < count; i++) {
      const type = random() < difficulty * 0.4 ? "spiny" : "walker";
      level.enemies.push({ x: x + 50 + i * 90, y: GROUND_Y, range: 55 + random() * (55 + difficulty * 55), type });
    }
    coinArc(level, x + 20, x + 40 + count * 90, GROUND_Y - 84, 2 + count);
    return x + 60 + (count - 1) * 90;
  }

  function addFlyers(level, x, difficulty, random) {
    const count = 1 + (random() < difficulty * 0.6 ? 1 : 0);
    for (let i = 0; i < count; i++) {
      level.enemies.push({
        x: x + 50 + i * 110, y: GROUND_Y - 50 - random() * 30,
        range: 45 + random() * (45 + difficulty * 50), type: "flyer"
      });
    }
    for (let i = 0; i < 3; i++) level.coins.push({ x: x + 30 + i * 24, y: GROUND_Y - 80 });
    return x + 70 + (count - 1) * 110;
  }

  function addSpikes(level, x, difficulty, random) {
    // jump carries ~90px; keep spikes clearly shorter than that
    const width = 24 + (random() < 0.4 + difficulty * 0.4 ? 16 : 0);
    level.spikes.push({ x, w: width });
    coinArc(level, x + 6, x + width - 6, GROUND_Y - 74, 3);
    return x + width;
  }

  function addSpring(level, x, difficulty, random) {
    level.springs.push({ x });
    const platformY = GROUND_Y - 120;
    level.platforms.push({ x: x + 40, y: platformY, w: 64 });
    coinLine(level, x, GROUND_Y - 60, x + 40, platformY - 20, 4);
    for (let i = 0; i < 2; i++) level.coins.push({ x: x + 52 + i * 24, y: platformY - 22 });
    return x + 110;
  }

  // Weighted by difficulty: near the start almost only flats, pipes and lone
  // pits appear; the tricky combinations take over toward the flag.
  const SEGMENTS = [
    { weight: d => 1.8 - d * 1.0, build: addFlat },
    { weight: d => 1.2 - d * 0.4, build: addPipe },
    { weight: d => 0.3 + d * 1.0, build: addPipeIsland },
    { weight: d => 0.1 + d * 1.2, build: addTwoTierIslands },
    { weight: d => 0.5 + d * 0.7, build: addPit },
    { weight: d => 0.1 + d * 1.2, build: addPitWithStones },
    { weight: d => 0.1 + d * 1.0, build: addGapToLedge },
    { weight: d => 0.1 + d * 1.0, build: addPipeStairs },
    { weight: d => 0.4 + d * 1.0, build: addWalkers },
    { weight: d => 0.1 + d * 1.1, build: addFlyers },
    { weight: d => 0.2 + d * 1.0, build: addSpikes },
    { weight: () => 0.6, build: addSpring }
  ];

  function pickSegment(difficulty, random) {
    const weights = SEGMENTS.map(segment => segment.weight(difficulty));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    let roll = random() * total;
    for (let i = 0; i < SEGMENTS.length; i++) {
      roll -= weights[i];
      if (roll <= 0) return SEGMENTS[i];
    }
    return SEGMENTS[SEGMENTS.length - 1];
  }

  function buildLevel(seed) {
    const random = mulberry32(seed);
    const level = {
      length: LEVEL_LENGTH,
      flagX: LEVEL_LENGTH - 160,
      obstacles: [], pits: [], coins: [], enemies: [],
      platforms: [], spikes: [], springs: []
    };
    let x = 420;
    while (x < level.flagX - 320) {
      const difficulty = Math.min(1, x / (LEVEL_LENGTH - 800));
      x = pickSegment(difficulty, random).build(level, x, difficulty, random);
      // always leave at least ~80px of flat ground: room to land after a jump
      // before the next hazard, even at full difficulty
      x += 80 + (1 - difficulty) * 60 + random() * 100;
    }
    return level;
  }

  return { buildLevel, LEVEL_LENGTH, GROUND_Y };
}
