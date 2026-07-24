"use strict";

function createRenderingModule(Simulation) {
  const gameCanvas = document.getElementById("gameCanvas");
  const gameContext = gameCanvas.getContext("2d");
  const networkCanvas = document.getElementById("networkCanvas");
  const networkContext = networkCanvas.getContext("2d");
  const chartCanvas = document.getElementById("chartCanvas");
  const chartContext = chartCanvas.getContext("2d");

  const { TILE, GROUND_Y, SIGHT, VERTICAL_SIGHT, GRID_CELL, OUTPUT_LABELS } = Simulation;

  const PALETTE = {
    r: "#c62828", b: "#1565c0", s: "#f2c197", k: "#3e2723",
    g: "#2e7d32", y: "#f9a825", w: "#fff", o: "#8d4e10", p: "#7b1fa2", _: null
  };
  const SPRITE_MARIO = [
    "__rrrr__", "_rrrrrr_", "_kkss_s_", "kskssss_", "kskkssss", "_ssssss_",
    "_rbbbbr_", "rrbbbbrr", "ssbybybs", "_bbbbbb_", "_bb__bb_", "_kk__kk_"
  ];
  const SPRITE_GOOMBA = [
    "__oooo__", "_oooooo_", "owokkowo", "oooooooo", "_oooooo_", "_kk__kk_"
  ];
  const SPRITE_FLYER = [
    "w_pppp_w", "wppppppw", "pwpkkpwp", "pppppppp", "_pppppp_", "__p__p__"
  ];
  const SPRITE_SPINY = [
    "r_r__r_r", "_rrrrrr_", "rwrkkrwr", "rrrrrrrr", "_rrrrrr_", "_kk__kk_"
  ];
  const SPRITE_COIN = [
    "_yyyy_", "yywwyy", "yywyyy", "yywyyy", "yywwyy", "_yyyy_"
  ];

  const SENSOR_COLORS = {
    obstacleDistance: "#aa0000", pitDistance: "#884400", spikeDistance: "#cc2277",
    springDistance: "#008877", rayForward: "#555555",
    enemy1: "#770077", enemy2: "#aa44aa",
    coin1: "#bb8800", coin2: "#cc9922", coin3: "#ddaa44"
  };

  function drawSprite(context, sprite, x, y, pixelSize) {
    for (let row = 0; row < sprite.length; row++) {
      for (let column = 0; column < sprite[row].length; column++) {
        const color = PALETTE[sprite[row][column]];
        if (!color) continue;
        context.fillStyle = color;
        context.fillRect(x + column * pixelSize, y + row * pixelSize, pixelSize, pixelSize);
      }
    }
  }

  function drawBackground(cameraX) {
    const skyGradient = gameContext.createLinearGradient(0, 0, 0, GROUND_Y);
    skyGradient.addColorStop(0, "#8fc7f5");
    skyGradient.addColorStop(1, "#d7ecff");
    gameContext.fillStyle = skyGradient;
    gameContext.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

    const hillParallax = cameraX * 0.55;
    gameContext.fillStyle = "#a8d8a0";
    for (let i = Math.floor(hillParallax / 260) - 1; i * 260 - hillParallax < gameCanvas.width + 130; i++) {
      const hillX = i * 260 - hillParallax;
      const hillHeight = 46 + (i * 7919 % 3) * 16;
      gameContext.beginPath();
      gameContext.arc(hillX + 130, GROUND_Y, hillHeight, Math.PI, 0);
      gameContext.fill();
    }

    const cloudParallax = cameraX * 0.3;
    gameContext.fillStyle = "rgba(255,255,255,0.9)";
    for (let i = Math.floor(cloudParallax / 240) - 1; i * 240 - cloudParallax < gameCanvas.width + 120; i++) {
      const cloudX = i * 240 - cloudParallax + (i * 104729 % 90);
      const cloudY = 26 + (i * 7919 % 44);
      gameContext.beginPath();
      gameContext.arc(cloudX, cloudY, 12, 0, 7);
      gameContext.arc(cloudX + 14, cloudY - 6, 10, 0, 7);
      gameContext.arc(cloudX + 28, cloudY, 12, 0, 7);
      gameContext.fill();
    }
  }

  function drawGround(level, cameraX) {
    for (let groundX = Math.floor(cameraX / TILE) * TILE; groundX < cameraX + gameCanvas.width; groundX += TILE) {
      if (!Simulation.groundAt(level, groundX + TILE / 2)) continue;
      const x = groundX - cameraX;
      gameContext.fillStyle = "#8a5a1a";
      gameContext.fillRect(x, GROUND_Y, TILE, gameCanvas.height - GROUND_Y);
      gameContext.fillStyle = "#6b4210";
      gameContext.fillRect(x + 2, GROUND_Y + 10, 5, 4);
      gameContext.fillRect(x + 9, GROUND_Y + 22, 5, 4);
      gameContext.fillStyle = "#3c8d2f";
      gameContext.fillRect(x, GROUND_Y, TILE, 6);
      gameContext.fillStyle = "#57b344";
      gameContext.fillRect(x, GROUND_Y, TILE, 2);
    }
  }

  function drawProgressBar(state, level) {
    const barX = 150, barWidth = gameCanvas.width - 300, barY = 8, barHeight = 8;
    const progress = Math.max(0, Math.min(1, (state.x - 60) / (level.flagX - 60)));
    gameContext.fillStyle = "rgba(255,255,255,0.7)";
    gameContext.fillRect(barX, barY, barWidth, barHeight);
    gameContext.fillStyle = "#3c8d2f";
    gameContext.fillRect(barX, barY, barWidth * progress, barHeight);
    gameContext.strokeStyle = "#111";
    gameContext.strokeRect(barX + 0.5, barY + 0.5, barWidth, barHeight);
    gameContext.fillStyle = "#c62828";
    gameContext.fillRect(barX + barWidth * progress - 2, barY - 2, 4, barHeight + 4);
    gameContext.fillStyle = "#111";
    gameContext.font = "10px Courier New";
    gameContext.fillText((progress * 100 | 0) + "%", barX + barWidth + 6, barY + barHeight);
  }

  function drawSensors(state, sensorReader, sensorVector, cameraX) {
    const playerX = state.x - cameraX;
    const playerY = state.y - 14;
    let rayIndex = 0;
    for (const entry of sensorReader.layout) {
      const color = SENSOR_COLORS[entry.id];
      if (entry.id === "rayUp") {
        const distance = sensorVector[entry.offset] * VERTICAL_SIGHT;
        gameContext.strokeStyle = "#555555";
        gameContext.lineWidth = 2;
        gameContext.beginPath();
        gameContext.moveTo(playerX, state.y - 26);
        gameContext.lineTo(playerX, state.y - 26 - distance);
        gameContext.stroke();
      } else if (entry.id === "rayDown") {
        const distance = sensorVector[entry.offset] * VERTICAL_SIGHT;
        if (distance <= 0) continue;
        gameContext.strokeStyle = "#557755";
        gameContext.lineWidth = 2;
        gameContext.beginPath();
        gameContext.moveTo(playerX, state.y);
        gameContext.lineTo(playerX, state.y + distance);
        gameContext.stroke();
      } else if (entry.size === 1 && color) {
        const value = sensorVector[entry.offset];
        if (value >= 1) { rayIndex++; continue; }
        const y = playerY - rayIndex * 3;
        gameContext.strokeStyle = color;
        gameContext.lineWidth = 2;
        gameContext.beginPath();
        gameContext.moveTo(playerX, y);
        gameContext.lineTo(playerX + value * SIGHT, y);
        gameContext.stroke();
        gameContext.fillStyle = color;
        gameContext.fillRect(playerX + value * SIGHT - 2, y - 2, 4, 4);
        rayIndex++;
      } else if (entry.size === 2 && color) {
        const dx = sensorVector[entry.offset];
        if (dx >= 1) continue;
        const dy = sensorVector[entry.offset + 1];
        gameContext.strokeStyle = color;
        gameContext.lineWidth = 1;
        gameContext.beginPath();
        gameContext.moveTo(playerX, playerY);
        gameContext.lineTo(playerX + dx * SIGHT, state.y + dy * VERTICAL_SIGHT);
        gameContext.stroke();
      } else if (entry.id === "tileGrid") {
        const { gridWidth, gridHeight } = sensorReader.config;
        const anchor = Simulation.gridAnchor(state);
        for (let row = 0; row < gridHeight; row++) {
          for (let column = 0; column < gridWidth; column++) {
            const value = sensorVector[entry.offset + row * gridWidth + column];
            const x = anchor.baseX + column * GRID_CELL - cameraX;
            const y = anchor.baseY - row * GRID_CELL;
            if (value === 1) gameContext.fillStyle = "rgba(0,120,0,0.18)";
            else if (value === -1) gameContext.fillStyle = "rgba(200,0,0,0.18)";
            else if (value === 0.5) gameContext.fillStyle = "rgba(220,170,0,0.25)";
            else gameContext.fillStyle = "rgba(0,0,0,0.03)";
            gameContext.fillRect(x, y, GRID_CELL, GRID_CELL);
            gameContext.strokeStyle = "rgba(0,0,0,0.15)";
            gameContext.strokeRect(x + 0.5, y + 0.5, GRID_CELL, GRID_CELL);
          }
        }
      }
    }
  }

  function drawDebugPanel(state, debug) {
    const labels = debug.labels || OUTPUT_LABELS;
    gameContext.fillStyle = "rgba(255,255,255,0.75)";
    gameContext.fillRect(4, 22, 210, 14 + labels.length * 12);
    gameContext.fillStyle = "#111";
    gameContext.font = "10px Courier New";
    gameContext.fillText("fitness " + (debug.fitness | 0) + "   action: " + debug.actionLabel, 8, 33);
    for (let i = 0; i < labels.length; i++) {
      const value = debug.outputs[i];
      const y = 40 + i * 12;
      gameContext.fillStyle = i === debug.chosenIndex ? "#111" : "#777";
      gameContext.fillText(labels[i].padEnd(5), 8, y + 8);
      const barOrigin = 100;
      const barLength = value * 50;
      gameContext.fillStyle = value >= 0 ? "#2e7d32" : "#aa0000";
      if (barLength >= 0) gameContext.fillRect(barOrigin, y + 1, barLength, 8);
      else gameContext.fillRect(barOrigin + barLength, y + 1, -barLength, 8);
      gameContext.strokeStyle = "#999";
      gameContext.beginPath();
      gameContext.moveTo(barOrigin, y);
      gameContext.lineTo(barOrigin, y + 10);
      gameContext.stroke();
    }
  }

  function drawGame(state, level, sensorReader, sensorVector, debug) {
    const cameraX = Math.max(0, state.x - 200);
    drawBackground(cameraX);
    drawGround(level, cameraX);

    for (const obstacle of level.obstacles) {
      const x = obstacle.x - cameraX;
      if (x < -80 || x > gameCanvas.width) continue;
      const pipeGradient = gameContext.createLinearGradient(x, 0, x + obstacle.w, 0);
      pipeGradient.addColorStop(0, "#2ea02e");
      pipeGradient.addColorStop(0.35, "#7fd07f");
      pipeGradient.addColorStop(1, "#1b7f1b");
      gameContext.fillStyle = pipeGradient;
      gameContext.fillRect(x, GROUND_Y - obstacle.h, obstacle.w, obstacle.h);
      gameContext.fillRect(x - 3, GROUND_Y - obstacle.h, obstacle.w + 6, 10);
      gameContext.strokeStyle = "#0b4d0b";
      gameContext.strokeRect(x, GROUND_Y - obstacle.h, obstacle.w, obstacle.h);
      gameContext.strokeRect(x - 3, GROUND_Y - obstacle.h, obstacle.w + 6, 10);
    }

    for (const platform of level.platforms) {
      const x = platform.x - cameraX;
      if (x + platform.w < 0 || x > gameCanvas.width) continue;
      for (let brickX = 0; brickX < platform.w; brickX += 16) {
        gameContext.fillStyle = "#b0603a";
        gameContext.fillRect(x + brickX, platform.y, 15, 10);
        gameContext.fillStyle = "#8a4526";
        gameContext.fillRect(x + brickX, platform.y + 7, 15, 3);
      }
      gameContext.strokeStyle = "#5c2f16";
      gameContext.strokeRect(x, platform.y, platform.w, 10);
    }

    for (const spike of level.spikes) {
      const x = spike.x - cameraX;
      if (x + spike.w < 0 || x > gameCanvas.width) continue;
      for (let spikeX = 0; spikeX < spike.w; spikeX += 8) {
        gameContext.fillStyle = "#9aa0a6";
        gameContext.beginPath();
        gameContext.moveTo(x + spikeX, GROUND_Y);
        gameContext.lineTo(x + spikeX + 4, GROUND_Y - 12);
        gameContext.lineTo(x + spikeX + 8, GROUND_Y);
        gameContext.fill();
        gameContext.strokeStyle = "#5f6368";
        gameContext.stroke();
      }
    }

    for (const spring of level.springs) {
      const x = spring.x - cameraX;
      if (x < -20 || x > gameCanvas.width) continue;
      gameContext.fillStyle = "#cc2222";
      gameContext.fillRect(x, GROUND_Y - 6, 16, 6);
      gameContext.fillStyle = "#dddddd";
      gameContext.fillRect(x + 3, GROUND_Y - 12, 10, 6);
      gameContext.strokeStyle = "#111";
      gameContext.strokeRect(x + 3, GROUND_Y - 12, 10, 6);
    }

    level.coins.forEach((coin, index) => {
      if (state.collectedCoins.has(index)) return;
      const x = coin.x - cameraX;
      if (x < -20 || x > gameCanvas.width) return;
      const spin = Math.abs(Math.sin(state.steps * 0.12 + index));
      gameContext.save();
      gameContext.translate(x, coin.y);
      gameContext.scale(0.35 + 0.65 * spin, 1);
      drawSprite(gameContext, SPRITE_COIN, -6, -6, 2);
      gameContext.restore();
    });

    for (const enemy of state.enemies) {
      if (!enemy.alive) continue;
      const x = enemy.x - cameraX;
      if (x < -30 || x > gameCanvas.width) continue;
      const sprite = enemy.type === "flyer" ? SPRITE_FLYER
        : enemy.type === "spiny" ? SPRITE_SPINY : SPRITE_GOOMBA;
      drawSprite(gameContext, sprite, x - 8, enemy.y - 12, 2);
    }

    const flagX = level.flagX - cameraX;
    if (flagX > -50 && flagX < gameCanvas.width) {
      gameContext.fillStyle = "#333";
      gameContext.fillRect(flagX, GROUND_Y - 120, 4, 120);
      gameContext.fillStyle = "#ddd";
      gameContext.beginPath();
      gameContext.arc(flagX + 2, GROUND_Y - 122, 4, 0, 7);
      gameContext.fill();
      const wave = Math.sin(state.steps * 0.1) * 4;
      gameContext.fillStyle = "#1b7f1b";
      gameContext.beginPath();
      gameContext.moveTo(flagX + 4, GROUND_Y - 118);
      gameContext.lineTo(flagX + 40, GROUND_Y - 106 + wave);
      gameContext.lineTo(flagX + 4, GROUND_Y - 90);
      gameContext.fill();
    }

    if (sensorReader && sensorVector) drawSensors(state, sensorReader, sensorVector, cameraX);

    drawSprite(gameContext, SPRITE_MARIO, state.x - 8 - cameraX, state.y - 24, 2);

    drawProgressBar(state, level);

    gameContext.fillStyle = "#111";
    gameContext.font = "12px Courier New";
    gameContext.fillText(
      `dist ${state.distance | 0}  coins ${state.coinsCollected}  kills ${state.enemiesKilled}  jumps ${state.jumps}`,
      8, 16
    );

    if (debug) drawDebugPanel(state, debug);

    if (state.dead) {
      gameContext.fillStyle = state.won ? "rgba(46,125,50,0.85)" : "rgba(170,0,0,0.85)";
      gameContext.font = "bold 34px Courier New";
      const text = state.won ? "LEVEL COMPLETE!" : "DEAD";
      const textWidth = gameContext.measureText(text).width;
      gameContext.fillText(text, (gameCanvas.width - textWidth) / 2, 110);
    }
  }

  function drawNetwork(shape, genome, activations, inputLabels, outputLabels) {
    outputLabels = outputLabels || OUTPUT_LABELS;
    networkContext.fillStyle = "#fff";
    networkContext.fillRect(0, 0, networkCanvas.width, networkCanvas.height);
    const layerCount = shape.length;
    const layerX = layer => 60 + layer * (networkCanvas.width - 130) / (layerCount - 1);
    const neuronY = (layer, index) =>
      networkCanvas.height / 2 + (index - (shape[layer] - 1) / 2) * Math.min(20, (networkCanvas.height - 30) / shape[layer]);

    let k = 0;
    for (let layer = 1; layer < layerCount; layer++) {
      for (let j = 0; j < shape[layer]; j++) {
        k++;
        for (let i = 0; i < shape[layer - 1]; i++) {
          const weight = genome ? genome[k++] : 0;
          if (Math.abs(weight) < 0.35) continue;
          networkContext.strokeStyle = weight > 0 ? "rgba(0,100,0,.35)" : "rgba(170,0,0,.35)";
          networkContext.lineWidth = Math.min(2.5, Math.abs(weight));
          networkContext.beginPath();
          networkContext.moveTo(layerX(layer - 1), neuronY(layer - 1, i));
          networkContext.lineTo(layerX(layer), neuronY(layer, j));
          networkContext.stroke();
        }
      }
    }

    const outputLayer = layerCount - 1;
    let chosenOutput = -1;
    if (activations) {
      chosenOutput = 0;
      for (let j = 1; j < shape[outputLayer]; j++) {
        if (activations[outputLayer][j] > activations[outputLayer][chosenOutput]) chosenOutput = j;
      }
    }

    const radius = shape[0] > 24 ? 4 : 6;
    for (let layer = 0; layer < layerCount; layer++) {
      for (let j = 0; j < shape[layer]; j++) {
        const activation = activations ? activations[layer][j] : 0;
        const value = Math.max(-1, Math.min(1, activation));
        networkContext.fillStyle = value > 0
          ? `rgb(${230 - 180 * value | 0},230,${230 - 180 * value | 0})`
          : `rgb(230,${230 + 180 * value | 0},${230 + 180 * value | 0})`;
        networkContext.beginPath();
        networkContext.arc(layerX(layer), neuronY(layer, j), radius, 0, 7);
        networkContext.fill();
        networkContext.strokeStyle = "#111";
        networkContext.lineWidth = 1;
        networkContext.stroke();
        if (layer === outputLayer && j === chosenOutput) {
          networkContext.strokeStyle = "#c62828";
          networkContext.lineWidth = 2;
          networkContext.beginPath();
          networkContext.arc(layerX(layer), neuronY(layer, j), radius + 3, 0, 7);
          networkContext.stroke();
        }
      }
    }

    networkContext.fillStyle = "#111";
    networkContext.font = shape[0] > 20 ? "8px Courier New" : "11px Courier New";
    for (let j = 0; j < shape[0]; j++) {
      networkContext.fillText(inputLabels[j] || "", 2, neuronY(0, j) + 3);
    }
    networkContext.font = "11px Courier New";
    for (let j = 0; j < shape[outputLayer]; j++) {
      networkContext.fillStyle = j === chosenOutput ? "#c62828" : "#111";
      networkContext.fillText(outputLabels[j] || "", layerX(outputLayer) + 10, neuronY(outputLayer, j) + 4);
    }
  }

  // Draws a NEAT genome as a free-form graph: columns by node depth.
  // `values` (per-node activations, indexed like graph.nodes) may be null.
  function drawNeatNetwork(graph, values, inputLabels, outputLabels) {
    outputLabels = outputLabels || OUTPUT_LABELS;
    networkContext.fillStyle = "#fff";
    networkContext.fillRect(0, 0, networkCanvas.width, networkCanvas.height);
    const { nodes, depth, maxDepth, links, inputCount } = graph;

    const columns = new Map();
    for (let i = 0; i < nodes.length; i++) {
      if (!columns.has(depth[i])) columns.set(depth[i], []);
      columns.get(depth[i]).push(i);
    }
    const positions = new Array(nodes.length);
    for (const [columnDepth, column] of columns) {
      column.sort((a, b) => nodes[a].id - nodes[b].id);
      const spacing = Math.min(20, (networkCanvas.height - 30) / column.length);
      column.forEach((nodeIndex, row) => {
        positions[nodeIndex] = {
          x: 60 + (networkCanvas.width - 130) * columnDepth / maxDepth,
          y: networkCanvas.height / 2 + (row - (column.length - 1) / 2) * spacing
        };
      });
    }
    const indexById = new Map(nodes.map((node, index) => [node.id, index]));

    for (const link of links) {
      const from = positions[indexById.get(link.inNode)];
      const to = positions[indexById.get(link.outNode)];
      if (!from || !to) continue;
      networkContext.strokeStyle = link.weight > 0 ? "rgba(0,100,0,.4)" : "rgba(170,0,0,.4)";
      networkContext.lineWidth = Math.max(0.5, Math.min(2.5, Math.abs(link.weight)));
      networkContext.beginPath();
      networkContext.moveTo(from.x, from.y);
      networkContext.lineTo(to.x, to.y);
      networkContext.stroke();
    }

    let chosenOutput = -1;
    if (values) {
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].type !== "output") continue;
        if (chosenOutput === -1 || values[i] > values[chosenOutput]) chosenOutput = i;
      }
    }

    const radius = nodes.length > 24 ? 4 : 6;
    for (let i = 0; i < nodes.length; i++) {
      const activation = values ? values[i] : 0;
      const value = Math.max(-1, Math.min(1, activation));
      networkContext.fillStyle = value > 0
        ? `rgb(${230 - 180 * value | 0},230,${230 - 180 * value | 0})`
        : `rgb(230,${230 + 180 * value | 0},${230 + 180 * value | 0})`;
      networkContext.beginPath();
      networkContext.arc(positions[i].x, positions[i].y, radius, 0, 7);
      networkContext.fill();
      networkContext.strokeStyle = nodes[i].type === "hidden" ? "#7b1fa2" : "#111";
      networkContext.lineWidth = 1;
      networkContext.stroke();
      if (i === chosenOutput) {
        networkContext.strokeStyle = "#c62828";
        networkContext.lineWidth = 2;
        networkContext.beginPath();
        networkContext.arc(positions[i].x, positions[i].y, radius + 3, 0, 7);
        networkContext.stroke();
      }
    }

    networkContext.fillStyle = "#111";
    networkContext.font = nodes.length > 20 ? "8px Courier New" : "11px Courier New";
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].type === "bias") networkContext.fillText("bias", 2, positions[i].y + 3);
      else if (nodes[i].type === "input") networkContext.fillText(inputLabels[nodes[i].id - 1] || "", 2, positions[i].y + 3);
    }
    networkContext.font = "11px Courier New";
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].type !== "output") continue;
      networkContext.fillStyle = i === chosenOutput ? "#c62828" : "#111";
      networkContext.fillText(outputLabels[nodes[i].id - 1 - inputCount] || "", positions[i].x + 10, positions[i].y + 4);
    }
  }

  let chartData = { history: [], levelMarkers: [] };

  function drawChart(history, levelMarkers) {
    chartData = { history, levelMarkers };
    renderChart(null);
  }

  function renderChart(hover) {
    const { history, levelMarkers } = chartData;
    chartContext.fillStyle = "#fff";
    chartContext.fillRect(0, 0, chartCanvas.width, chartCanvas.height);
    chartContext.strokeStyle = "#111";
    chartContext.lineWidth = 1;
    chartContext.strokeRect(0.5, 0.5, chartCanvas.width - 1, chartCanvas.height - 1);
    if (!history.length) {
      chartContext.fillStyle = "#999";
      chartContext.font = "12px Courier New";
      chartContext.fillText("no data yet - press TRAIN", 10, 20);
      return;
    }
    let minFitness = 0;
    let maxFitness = 1;
    for (const entry of history) {
      minFitness = Math.min(minFitness, entry.best, entry.average, entry.benchmark);
      maxFitness = Math.max(maxFitness, entry.best, entry.average, entry.benchmark);
    }
    const range = Math.max(1, maxFitness - minFitness);
    const chartX = index => 10 + (chartCanvas.width - 20) * index / Math.max(1, history.length - 1);
    const chartY = value => chartCanvas.height - 12 - (chartCanvas.height - 30) * (value - minFitness) / range;

    chartContext.setLineDash([4, 4]);
    chartContext.strokeStyle = "#777";
    for (const marker of levelMarkers) {
      if (marker <= 0 || marker >= history.length) continue;
      chartContext.beginPath();
      chartContext.moveTo(chartX(marker), 5);
      chartContext.lineTo(chartX(marker), chartCanvas.height - 5);
      chartContext.stroke();
    }
    chartContext.setLineDash([]);

    const drawLine = (key, color, width) => {
      chartContext.strokeStyle = color;
      chartContext.lineWidth = width;
      chartContext.beginPath();
      history.forEach((entry, index) => {
        if (index) chartContext.lineTo(chartX(index), chartY(entry[key]));
        else chartContext.moveTo(chartX(index), chartY(entry[key]));
      });
      chartContext.stroke();
    };
    drawLine("average", "#999", 1.5);
    drawLine("benchmark", "#1565c0", 2);
    drawLine("best", "#111", 2);

    chartContext.fillStyle = "#111";
    chartContext.font = "11px Courier New";
    chartContext.fillText(maxFitness | 0, 12, 14);
    if (minFitness < 0) chartContext.fillText(minFitness | 0, 12, chartCanvas.height - 6);
    chartContext.fillText("gen " + history.length, chartCanvas.width - 70, chartCanvas.height - 6);

    if (!hover) return;
    const index = Math.max(0, Math.min(history.length - 1,
      Math.round((hover.x - 10) / (chartCanvas.width - 20) * Math.max(1, history.length - 1))));
    const cursorScore = minFitness + (chartCanvas.height - 12 - hover.y) / (chartCanvas.height - 30) * range;
    const entry = history[index];
    const x = chartX(index);

    chartContext.strokeStyle = "#c62828";
    chartContext.lineWidth = 1;
    chartContext.beginPath();
    chartContext.moveTo(x, 2);
    chartContext.lineTo(x, chartCanvas.height - 2);
    chartContext.stroke();
    for (const [key, color] of [["best", "#111"], ["average", "#999"], ["benchmark", "#1565c0"]]) {
      chartContext.fillStyle = color;
      chartContext.beginPath();
      chartContext.arc(x, chartY(entry[key]), 3, 0, 7);
      chartContext.fill();
    }

    const lines = [
      "gen " + (index + 1) + " · y " + (cursorScore | 0),
      "best " + (entry.best | 0) + "  avg " + (entry.average | 0) + "  bench " + (entry.benchmark | 0)
    ];
    chartContext.font = "11px Courier New";
    const boxWidth = Math.max(...lines.map(line => chartContext.measureText(line).width)) + 12;
    const boxX = x + boxWidth + 14 > chartCanvas.width ? x - boxWidth - 8 : x + 8;
    const boxY = Math.max(4, Math.min(chartCanvas.height - 36, hover.y - 16));
    chartContext.fillStyle = "rgba(255,255,255,0.92)";
    chartContext.fillRect(boxX, boxY, boxWidth, 32);
    chartContext.strokeStyle = "#111";
    chartContext.strokeRect(boxX + 0.5, boxY + 0.5, boxWidth, 32);
    chartContext.fillStyle = "#111";
    chartContext.fillText(lines[0], boxX + 6, boxY + 13);
    chartContext.fillText(lines[1], boxX + 6, boxY + 26);
  }

  chartCanvas.addEventListener("mousemove", event => {
    const rect = chartCanvas.getBoundingClientRect();
    renderChart({
      x: (event.clientX - rect.left) * chartCanvas.width / rect.width,
      y: (event.clientY - rect.top) * chartCanvas.height / rect.height
    });
  });
  chartCanvas.addEventListener("mouseleave", () => renderChart(null));

  return { drawGame, drawNetwork, drawNeatNetwork, drawChart };
}
