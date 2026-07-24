# ai-mario

Watch a tiny artificial "brain" teach itself to play a Mario-style platformer, right in your browser.

**▶️ [Open the live demo](https://ebrithilnogare.github.io/AI-Mario/)**

## What is this?

Little characters keep trying to run through a level full of pipes, pits, spikes, enemies and coins. At first they fail almost immediately, but over time they learn to jump, dodge and reach the flag — no one programs how, they figure it out by trial and error.

You can pick **how** they learn, and compare three classic approaches on the same game:

- **Genetic algorithm** — a population of fixed-size brains; each round the best are kept, crossed and mutated.
- **NEAT** — like the genetic algorithm, but the brain's *structure* evolves too: it starts minimal and grows neurons and connections, with speciation protecting new ideas.
- **Imitation learning (play & teach)** — *you* play the level with WASD / space and a single brain learns to copy you by backpropagation, live while you play: each move is trained on as soon as you survive it, moves right before a death are discarded so it never imitates fatal mistakes, and winning runs count double.

## How to use it

1. Open the [demo](https://ebrithilnogare.github.io/AI-Mario/).
2. Pick a **learning method** at the top (each keeps its own progress).
3. Press **TRAIN** and watch the graph climb as the brains improve.
4. Press **WATCH BEST** any time to see the current best player run a level.
5. Press **NEW LEVEL** to throw a fresh, unseen level at them.

## Things you can tweak

- **Inputs** — what the character is allowed to "see" (nearby enemies, coins, gaps, walls, and more).
- **Network** — how big the brain is (for NEAT the topology grows on its own).
- **Method settings** — population size and species for the evolutionary methods; learning rate, batch size and training epochs for imitation.
- **Rewards** — what counts as good or bad: reaching the flag, grabbing coins, stomping enemies, dying, jumping, wasting time. All three methods optimize the same score.
- **Maps & threads** — how many levels each player is tested on, and how much of your computer to use for faster learning.

## Reading the graph

- **Black line** — the best score so far.
- **Grey line** — the average of everyone.
- **Blue line** — how the current best does on a secret level it never practices on, showing whether it's truly learning or just memorizing.

No installation, no accounts, nothing to download — it all runs in your browser.
