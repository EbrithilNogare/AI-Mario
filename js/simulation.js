"use strict";

function createSimulationModule(NeuralNetwork) {
  const TILE = 16;
  const GROUND_Y = 200;
  const SIGHT = 300;
  const VERTICAL_SIGHT = 150;
  const GRID_CELL = 32;
  const GRAVITY = 0.55;
  const JUMP_VELOCITY = -9.5;
  const SPRING_VELOCITY = -14;
  const RUN_SPEED = 2.6;
  const MAX_STEPS = 4800;
  // Time budget: a base allowance plus extra time earned per pixel of progress
  // (2x the optimal pace). Standing still runs out fast; slow-but-moving agents
  // can finish, and the time reward makes fast agents score higher.
  const TIME_BASE = 360;
  const TIME_STEPS_PER_PIXEL = 2 / 2.6;
  const PLAYER_START_X = 60;
  const OUTPUT_COUNT = 5;
  const OUTPUT_LABELS = ["idle", "left", "right", "jump", "jmp+R"];

  function newRun(level) {
    return {
      x: PLAYER_START_X, y: GROUND_Y, velocityX: 0, velocityY: 0,
      onGround: true, dead: false, won: false, killedByHazard: false,
      distance: 0, coinsCollected: 0, enemiesKilled: 0, jumps: 0, steps: 0, bestX: PLAYER_START_X,
      collectedCoins: new Set(),
      enemies: level.enemies.map(enemy => ({
        x: enemy.x, y: enemy.y, baseX: enemy.x, baseY: enemy.y,
        range: enemy.range, type: enemy.type,
        direction: -1, phase: 0, alive: true
      }))
    };
  }

  function groundAt(level, x) {
    for (const pit of level.pits) if (x > pit.x && x < pit.x + pit.w) return false;
    return true;
  }

  function solidAt(level, x, y) {
    if (y >= GROUND_Y) return groundAt(level, x);
    for (const obstacle of level.obstacles) {
      if (x >= obstacle.x && x <= obstacle.x + obstacle.w && y >= GROUND_Y - obstacle.h) return true;
    }
    for (const platform of level.platforms) {
      if (x >= platform.x && x <= platform.x + platform.w && y >= platform.y && y <= platform.y + 8) return true;
    }
    return false;
  }

  function hazardAt(level, state, x, y) {
    for (const spike of level.spikes) {
      if (x >= spike.x && x <= spike.x + spike.w && y >= GROUND_Y - GRID_CELL) return true;
    }
    for (const enemy of state.enemies) {
      if (enemy.alive && Math.abs(enemy.x - x) < GRID_CELL / 2 && Math.abs(enemy.y - y) < GRID_CELL / 2 + 4) return true;
    }
    if (y > GROUND_Y - GRID_CELL && !groundAt(level, x)) return true;
    return false;
  }

  function enemySpeed(enemy) {
    return enemy.type === "flyer" ? 1.1 : enemy.type === "spiny" ? 1.0 : 0.8;
  }

  function step(state, level, action) {
    let velocityX = 0;
    if (action === 1) velocityX = -RUN_SPEED;
    if (action === 2 || action === 4) velocityX = RUN_SPEED;
    if ((action === 3 || action === 4) && state.onGround) {
      state.velocityY = JUMP_VELOCITY;
      state.onGround = false;
      state.jumps++;
    }

    state.x += velocityX;
    state.velocityX = velocityX;
    if (state.x < 20) state.x = 20;
    const previousY = state.y;
    state.velocityY += GRAVITY;
    state.y += state.velocityY;
    state.onGround = false;

    if (state.y >= GROUND_Y) {
      if (groundAt(level, state.x)) {
        state.y = GROUND_Y;
        state.velocityY = 0;
        state.onGround = true;
      } else if (state.y > GROUND_Y + 60) {
        state.dead = true;
        state.killedByHazard = true;
      }
    }

    for (const platform of level.platforms) {
      if (state.velocityY >= 0 && previousY <= platform.y && state.y >= platform.y
          && state.x + 8 > platform.x && state.x - 8 < platform.x + platform.w) {
        state.y = platform.y;
        state.velocityY = 0;
        state.onGround = true;
      }
    }

    for (const obstacle of level.obstacles) {
      const top = GROUND_Y - obstacle.h;
      if (state.x + 10 > obstacle.x && state.x - 10 < obstacle.x + obstacle.w && state.y > top) {
        if (state.velocityY > 0 && previousY <= top + 8) {
          state.y = top;
          state.velocityY = 0;
          state.onGround = true;
        } else {
          state.x = velocityX >= 0 ? obstacle.x - 10 : obstacle.x + obstacle.w + 10;
        }
      }
    }

    for (const spring of level.springs) {
      if (state.onGround && state.y >= GROUND_Y - 2 && Math.abs(state.x - (spring.x + 8)) < 16) {
        state.velocityY = SPRING_VELOCITY;
        state.onGround = false;
      }
    }

    for (const spike of level.spikes) {
      if (state.x + 6 > spike.x && state.x - 6 < spike.x + spike.w && state.y >= GROUND_Y - 4) {
        state.dead = true;
        state.killedByHazard = true;
      }
    }

    for (const enemy of state.enemies) {
      if (!enemy.alive) continue;
      enemy.x += enemy.direction * enemySpeed(enemy);
      if (enemy.x < enemy.baseX - enemy.range || enemy.x > enemy.baseX + enemy.range) enemy.direction *= -1;
      if (enemy.type === "flyer") {
        enemy.phase += 0.06;
        enemy.y = enemy.baseY + Math.sin(enemy.phase) * 26;
      }
      if (Math.abs(enemy.x - state.x) < 12 && Math.abs(enemy.y - state.y) < 14) {
        // predictable stomp rule: any downward motion while above the enemy kills it
        if (enemy.type !== "spiny" && state.velocityY > 0 && state.y < enemy.y - 2) {
          enemy.alive = false;
          state.enemiesKilled++;
          state.velocityY = JUMP_VELOCITY * 0.6;
        } else {
          state.dead = true;
          state.killedByHazard = true;
        }
      }
    }

    level.coins.forEach((coin, index) => {
      if (!state.collectedCoins.has(index)
          && Math.abs(coin.x - state.x) < 14 && Math.abs(coin.y - (state.y - 8)) < 20) {
        state.collectedCoins.add(index);
        state.coinsCollected++;
      }
    });

    if (state.x >= level.flagX) {
      state.won = true;
      state.dead = true;
    }

    state.steps++;
    if (state.x > state.bestX) state.bestX = state.x;
    const allowedSteps = Math.min(MAX_STEPS,
      TIME_BASE + (state.bestX - PLAYER_START_X) * TIME_STEPS_PER_PIXEL);
    if (state.steps > allowedSteps) state.dead = true;
    state.distance = state.bestX - PLAYER_START_X;
  }

  const DEFAULT_REWARDS = { coin: 20, enemyKill: 5, death: -10, jump: -1, distance: 1000, time: -1 };
  const STEPS_PER_SECOND = 60;

  function fitness(state, level, rewards) {
    const progress = state.distance / (level.flagX - PLAYER_START_X);
    let score = progress * rewards.distance
      + state.coinsCollected * rewards.coin
      + state.enemiesKilled * rewards.enemyKill
      + state.jumps * (rewards.jump || 0)
      + (state.steps / STEPS_PER_SECOND) * (rewards.time || 0);
    if (state.killedByHazard) score += rewards.death;
    if (state.won) score += 2000 + Math.max(0, MAX_STEPS - state.steps) * 0.3;
    return score;
  }

  // Distance to the leading edge of the nearest item ahead; stays 0 the whole
  // time the player is over/inside the item (an item only disappears from the
  // sensor once its far edge is passed).
  function nearestAheadDistance(state, items, getX, getW) {
    let distance = SIGHT;
    for (const item of items) {
      const start = getX(item);
      if (start + getW(item) < state.x) continue; // fully passed
      const d = Math.max(0, start - state.x);
      if (d < distance) distance = d;
    }
    return distance / SIGHT;
  }

  function enemiesAhead(state) {
    return state.enemies
      .filter(enemy => enemy.alive && enemy.x - state.x > -16 && enemy.x - state.x < SIGHT)
      .sort((a, b) => a.x - b.x);
  }

  function coinsAhead(state, level) {
    const ahead = [];
    level.coins.forEach((coin, index) => {
      if (state.collectedCoins.has(index)) return;
      const d = coin.x - state.x;
      if (d >= 0 && d < SIGHT) ahead.push(coin);
    });
    return ahead.sort((a, b) => a.x - b.x);
  }

  function writeEntity(state, item, output, offset) {
    if (!item) {
      output[offset] = 1;
      output[offset + 1] = 0;
      return;
    }
    output[offset] = Math.max(0, item.x - state.x) / SIGHT;
    output[offset + 1] = Math.max(-1, Math.min(1, (item.y - state.y) / VERTICAL_SIGHT));
  }

  // Writes an entity's velocity (vx, vy) normalized to roughly [-1, 1].
  function writeEntityVelocity(enemy, output, offset) {
    if (!enemy) {
      output[offset] = 0;
      output[offset + 1] = 0;
      return;
    }
    output[offset] = enemy.direction * enemySpeed(enemy) / 1.5;
    // flyers bob vertically: y = baseY + sin(phase) * 26, phase += 0.06 per step
    output[offset + 1] = enemy.type === "flyer" ? Math.max(-1, Math.min(1, Math.cos(enemy.phase) * 26 * 0.06 / 2)) : 0;
  }

  // Shared by the tileGrid sensor and its rendering overlay: the grid snaps to
  // a lattice aligned with the ground and follows the player in whole cells.
  function gridAnchor(state) {
    return {
      baseX: Math.floor(state.x / GRID_CELL) * GRID_CELL - GRID_CELL,
      baseY: GROUND_Y + Math.round((state.y - GROUND_Y) / GRID_CELL) * GRID_CELL
    };
  }

  const INPUT_DEFINITIONS = [
    {
      id: "velocityX", label: "horizontal speed", group: "player",
      size: () => 1, labels: () => ["vx"],
      read(state, level, config, output, offset) {
        output[offset] = Math.max(-1, Math.min(1, state.velocityX / RUN_SPEED));
      }
    },
    {
      id: "velocityY", label: "vertical speed", group: "player",
      size: () => 1, labels: () => ["vy"],
      read(state, level, config, output, offset) {
        output[offset] = Math.max(-1, Math.min(1, state.velocityY / 12));
      }
    },
    {
      id: "onGround", label: "on ground", group: "player",
      size: () => 1, labels: () => ["ground"],
      read(state, level, config, output, offset) {
        output[offset] = state.onGround ? 1 : 0;
      }
    },
    {
      id: "playerHeight", label: "height", group: "player",
      size: () => 1, labels: () => ["height"],
      read(state, level, config, output, offset) {
        output[offset] = Math.max(0, Math.min(1, (GROUND_Y - state.y) / VERTICAL_SIGHT));
      }
    },
    {
      id: "obstacleDistance", label: "obstacle distance", group: "terrain",
      size: () => 1, labels: () => ["obst"],
      read(state, level, config, output, offset) {
        output[offset] = nearestAheadDistance(state, level.obstacles, obstacle => obstacle.x, obstacle => obstacle.w);
      }
    },
    {
      id: "pitDistance", label: "pit distance", group: "terrain",
      size: () => 1, labels: () => ["pit"],
      read(state, level, config, output, offset) {
        output[offset] = nearestAheadDistance(state, level.pits, pit => pit.x, pit => pit.w);
      }
    },
    {
      id: "spikeDistance", label: "spike distance", group: "terrain",
      size: () => 1, labels: () => ["spike"],
      read(state, level, config, output, offset) {
        output[offset] = nearestAheadDistance(state, level.spikes, spike => spike.x, spike => spike.w);
      }
    },
    {
      id: "springDistance", label: "spring distance", group: "terrain",
      size: () => 1, labels: () => ["spring"],
      read(state, level, config, output, offset) {
        output[offset] = nearestAheadDistance(state, level.springs, spring => spring.x, () => 16);
      }
    },
    {
      id: "enemy1", label: "enemy 1 x/y", group: "enemies",
      size: () => 2, labels: () => ["en1 dx", "en1 dy"],
      read(state, level, config, output, offset) {
        writeEntity(state, enemiesAhead(state)[0], output, offset);
      }
    },
    {
      id: "enemy2", label: "enemy 2 x/y", group: "enemies",
      size: () => 2, labels: () => ["en2 dx", "en2 dy"],
      read(state, level, config, output, offset) {
        writeEntity(state, enemiesAhead(state)[1], output, offset);
      }
    },
    {
      id: "enemy1Velocity", label: "enemy 1 vx/vy", group: "enemies",
      size: () => 2, labels: () => ["en1 vx", "en1 vy"],
      read(state, level, config, output, offset) {
        writeEntityVelocity(enemiesAhead(state)[0], output, offset);
      }
    },
    {
      id: "enemy2Velocity", label: "enemy 2 vx/vy", group: "enemies",
      size: () => 2, labels: () => ["en2 vx", "en2 vy"],
      read(state, level, config, output, offset) {
        writeEntityVelocity(enemiesAhead(state)[1], output, offset);
      }
    },
    {
      id: "enemy1Type", label: "enemy 1 stompable", group: "enemies",
      size: () => 1, labels: () => ["en1 st"],
      read(state, level, config, output, offset) {
        const enemy = enemiesAhead(state)[0];
        output[offset] = !enemy ? 0 : enemy.type === "spiny" ? -1 : 1;
      }
    },
    {
      id: "coin1", label: "coin 1 x/y", group: "coins",
      size: () => 2, labels: () => ["co1 dx", "co1 dy"],
      read(state, level, config, output, offset) {
        writeEntity(state, coinsAhead(state, level)[0], output, offset);
      }
    },
    {
      id: "coin2", label: "coin 2 x/y", group: "coins",
      size: () => 2, labels: () => ["co2 dx", "co2 dy"],
      read(state, level, config, output, offset) {
        writeEntity(state, coinsAhead(state, level)[1], output, offset);
      }
    },
    {
      id: "coin3", label: "coin 3 x/y", group: "coins",
      size: () => 2, labels: () => ["co3 dx", "co3 dy"],
      read(state, level, config, output, offset) {
        writeEntity(state, coinsAhead(state, level)[2], output, offset);
      }
    },
    {
      id: "rayForward", label: "sight front", group: "terrain",
      size: () => 1, labels: () => ["ray fw"],
      read(state, level, config, output, offset) {
        let distance = SIGHT;
        for (let t = 0; t < SIGHT; t += 8) {
          if (solidAt(level, state.x + 12 + t, state.y - 12)) { distance = t; break; }
        }
        output[offset] = distance / SIGHT;
      }
    },
    {
      id: "rayUp", label: "sight up", group: "terrain",
      size: () => 1, labels: () => ["ray up"],
      read(state, level, config, output, offset) {
        let distance = VERTICAL_SIGHT;
        for (let t = 0; t < VERTICAL_SIGHT; t += 8) {
          if (solidAt(level, state.x, state.y - 26 - t)) { distance = t; break; }
        }
        output[offset] = distance / VERTICAL_SIGHT;
      }
    },
    {
      id: "rayDown", label: "sight down", group: "terrain",
      size: () => 1, labels: () => ["ray dn"],
      read(state, level, config, output, offset) {
        let distance = VERTICAL_SIGHT;
        for (let t = 0; t < VERTICAL_SIGHT; t += 8) {
          if (solidAt(level, state.x, state.y + t)) { distance = t; break; }
        }
        output[offset] = distance / VERTICAL_SIGHT;
      }
    },
    {
      id: "obstacleHeight", label: "obstacle height", group: "terrain",
      size: () => 1, labels: () => ["obst h"],
      read(state, level, config, output, offset) {
        let nearest = null;
        for (const obstacle of level.obstacles) {
          if (obstacle.x + obstacle.w < state.x) continue; // fully passed
          if (obstacle.x - state.x < SIGHT && (!nearest || obstacle.x < nearest.x)) nearest = obstacle;
        }
        output[offset] = nearest ? Math.min(1, nearest.h / 100) : 0;
      }
    },
    {
      id: "pitWidth", label: "pit width", group: "terrain",
      size: () => 1, labels: () => ["pit w"],
      read(state, level, config, output, offset) {
        let nearest = null;
        for (const pit of level.pits) {
          const d = pit.x + pit.w - state.x;
          if (d >= 0 && pit.x - state.x < SIGHT && (!nearest || pit.x < nearest.x)) nearest = pit;
        }
        output[offset] = nearest ? Math.min(1, nearest.w / 100) : 0;
      }
    },
    {
      id: "flagDistance", label: "flag distance", group: "terrain",
      size: () => 1, labels: () => ["flag"],
      read(state, level, config, output, offset) {
        output[offset] = Math.max(0, Math.min(1, (level.flagX - state.x) / (level.flagX - PLAYER_START_X)));
      }
    },
    {
      id: "tileGrid", label: "tile grid (WxH around player)", group: "grid",
      size: config => config.gridWidth * config.gridHeight,
      labels: config => {
        const labels = [];
        for (let row = 0; row < config.gridHeight; row++)
          for (let col = 0; col < config.gridWidth; col++) labels.push(`g${col},${row}`);
        return labels;
      },
      // The grid follows the player: column 0 is one cell behind, row 0 is the
      // cell under the feet (so pits and the ground itself are visible), rows
      // grow upward. Anchored to a ground-aligned lattice so it doesn't flicker.
      read(state, level, config, output, offset) {
        const anchor = gridAnchor(state);
        for (let row = 0; row < config.gridHeight; row++) {
          for (let col = 0; col < config.gridWidth; col++) {
            const centerX = anchor.baseX + col * GRID_CELL + GRID_CELL / 2;
            const centerY = anchor.baseY + GRID_CELL / 2 - row * GRID_CELL;
            let value = 0;
            for (let coinIndex = 0; coinIndex < level.coins.length; coinIndex++) {
              const coin = level.coins[coinIndex];
              if (!state.collectedCoins.has(coinIndex)
                  && Math.abs(coin.x - centerX) < GRID_CELL / 2 && Math.abs(coin.y - centerY) < GRID_CELL / 2) {
                value = 0.5;
                break;
              }
            }
            // two samples per cell so half-height solids (pipe tops, platforms) register
            if (solidAt(level, centerX, centerY) || solidAt(level, centerX, centerY + 12)) value = 1;
            if (hazardAt(level, state, centerX, centerY)) value = -1;
            output[offset + row * config.gridWidth + col] = value;
          }
        }
      }
    }
  ];

  function createSensorReader(config) {
    const active = INPUT_DEFINITIONS.filter(definition => config.selected.includes(definition.id));
    const layout = [];
    let total = 0;
    for (const definition of active) {
      const size = definition.size(config);
      layout.push({ id: definition.id, offset: total, size });
      total += size;
    }
    const labels = active.flatMap(definition => definition.labels(config));
    return {
      size: total,
      labels,
      layout,
      config,
      read(state, level) {
        const output = new Float32Array(total);
        for (let i = 0; i < active.length; i++) {
          active[i].read(state, level, config, output, layout[i].offset);
        }
        return output;
      }
    };
  }

  // actionMap translates a network output index to a global action index
  // (the output layer only has neurons for the enabled actions).
  function evaluate(genome, shape, level, sensorReader, rewards, actionMap) {
    const state = newRun(level);
    while (!state.dead) {
      const result = NeuralNetwork.forward(genome, shape, sensorReader.read(state, level), false);
      const choice = NeuralNetwork.argmax(result.output);
      step(state, level, actionMap ? actionMap[choice] : choice);
    }
    return fitness(state, level, rewards);
  }

  // Same as evaluate, but for any policy: act(sensorVector) -> actionIndex.
  function evaluateWith(act, level, sensorReader, rewards) {
    const state = newRun(level);
    while (!state.dead) {
      step(state, level, act(sensorReader.read(state, level)));
    }
    return fitness(state, level, rewards);
  }

  return {
    TILE, GROUND_Y, SIGHT, VERTICAL_SIGHT, GRID_CELL,
    OUTPUT_COUNT, OUTPUT_LABELS,
    INPUT_DEFINITIONS, DEFAULT_REWARDS,
    newRun, step, fitness, groundAt, solidAt, gridAnchor,
    createSensorReader, evaluate, evaluateWith
  };
}
