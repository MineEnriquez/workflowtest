# python
# Starsim-inspired minimal ABM implementation (SIR) in Python
# This is a self-contained, production-ready, runnable script with a concise README.
# Assumptions and missing values are clearly labeled in code comments.

"""
Starsim-inspired minimal ABM: SIR on a random network

Overview:
- Disease-modeling approach: agent-based SIR with stochastic transmission, recovery, and deaths (optional)
- Compartment/state variables per agent: Susceptible (S), Infected (I), Recovered (R), Alive (A)
- Network: Random network ( Erdős–Rényi ) for potential contacts
- Time discretization: yearly steps with fractional dt support via simulation loop
- Key parameters (with sensible defaults, marked as assumptions if not data-driven)
- Simulation method: vectorized updates where possible; per-step infection attempts are drawn stochastically

Assumptions and missing values:
- The model uses an SIR framework. It does not model births/migration in detail by default.
- Infectiousness is uniform across infected agents; no stage-based infectivity.
- Deaths are optional and independent; default only recoveries mark transition out of infection.
- If you need multi-disease co-infection, this skeleton can be extended.
- Parameters not provided in the document are given reasonable defaults and clearly marked.

Usage:
- Run the script directly. It will run a sample scenario and print basic results.
- Modify parameters in the main() function or via command-line-like injections in code.

Output:
- Console summary per year
- Final state counts for S, I, R, and Alive

"""

import numpy as np
from typing import NamedTuple, Optional
import math


class Params(NamedTuple):
    n_agents: int
    beta: float           # transmission probability per contact per time step
    gamma: float          # recovery probability per infected agent per time step
    dt: float               # time step (years)
    end_years: float        # simulation duration in years
    avg_degree: float       # average contacts per agent (ER network)
    seed: int
    verbose: int
    import_network_only_once: bool = True
    external_infection_start: int = 0  # year to initialize infections
    initial_infected_frac: float = 0.01  # initial prevalence
    death_rate_given_infection: float = 0.0  # optional infection-specific death; if >0, infections can die
    enable_deaths: bool = False


class StarsimSIR:
    def __init__(self, params: Params):
        self.params = params
        rng = np.random.default_rng(params.seed)
        self.rng = rng

        self.n = int(params.n_agents)
        self.dt = float(params.dt)
        self.years = float(params.end_years)

        # Create random network: Erdős–Rényi G(n, p) with p = avg_degree/(n-1)
        p = min(1.0, max(0.0, params.avg_degree / max(1.0, self.n - 1)))
        # Adjacency represented as a boolean matrix for vectorized infection checks is memory-heavy for large n.
        # We'll store neighbor lists for each node to sample contacts.
        self.neighbors = self._generate_random_contacts(p, rng)

        # State arrays
        # Alive mask
        self.alive = np.ones(self.n, dtype=bool)
        # Disease compartments
        self.S = np.ones(self.n, dtype=bool)  # susceptible flag
        self.I = np.zeros(self.n, dtype=bool)
        self.R = np.zeros(self.n, dtype=bool)

        # Initialize infections
        init_infected = int(self.n * max(0.0, min(1.0, params.initial_infected_frac)))
        if init_infected > 0:
            idx = rng.choice(self.n, size=init_infected, replace=False)
            self.S[idx] = False
            self.I[idx] = True
        self._update_counts()

        self.year = 0.0

        # History
        self.history = {
            "year": [],
            "n_s": [],
            "n_i": [],
            "n_r": [],
            "n_alive": [],
            "new_infections": [],
        }

        # Diagnostics
        self.verbose = bool(params.verbose)

    def _generate_random_contacts(self, p, rng: np.random.Generator):
        # For memory efficiency, build a neighbor list: for each node, a numpy array of neighbor indices
        neigh = []
        # We sample a random subset of potential partners for each agent
        for i in range(self.n):
            # sample a random set of potential neighbors
            # Number of potential contacts follows Binomial(n-1, p) with mean (n-1)*p ~ avg_degree
            degree_i = rng.binomial(self.n - 1, p)
            if degree_i <= 0:
                neigh.append(np.array([], dtype=int))
            else:
                # draw without replacement
                # ensure we don't pick the node itself
                candidates = rng.choice(self.n, size=degree_i, replace=False)
                candidates = candidates[candidates != i]
                neigh.append(candidates)
        return neigh

    def _step(self, year_step: float):
        rng = self.rng
        # Pre-steps: identify alive agents
        alive_idx = np.where(self.alive)[0]

        # Infections: for each susceptible, check if any neighbor is infected
        # We'll sample per contact: probability of transmission = beta * dt
        beta_dt = self.params.beta * self.dt

        # Gather infected neighbors for each susceptible
        new_infections = []
        for i in alive_idx:
            if not self.S[i]:
                continue
            neigh = self.neighbors[i]
            if neigh.size == 0:
                continue
            # check if any neighbor is infected
            infected_neighbors = self.I[neigh]
            if not infected_neighbors.any():
                continue
            # compute probability of transmission from any infected neighbor
            # approximate by 1 - product(1 - beta_dt) over infected contacts
            k = infected_neighbors.sum()
            if k == 0:
                continue
            # number of infected contacts
            # For performance, assume independent transmission per contact
            p_infection = 1.0 - (1.0 - beta_dt) ** float(k)
            if rng.random() < p_infection:
                new_infections.append(i)

        # Apply new infections
        for i in new_infections:
            if self.S[i]:
                self.S[i] = False
                self.I[i] = True

        # Recovery: each infected has probability gamma * dt to recover
        infected_idx = alive_idx[self.I[alive_idx]]
        if infected_idx.size > 0:
            recover_prob = min(1.0, self.params.gamma * self.dt)
            recovering = infected_idx[self.rng.random(infected_idx.size) < recover_prob]
            if recovering.size > 0:
                self.I[recovering] = False
                self.R[recovering] = True

        # Optional death process (not typical in simple SIR). If enabled, apply deaths to infected or alive overall.
        if self.params.enable_deaths and self.params.death_rate_given_infection > 0:
            # naive implementation: each infected has a small death prob
            death_prob = self.params.death_rate_given_infection * self.dt
            to_die = infected_idx[self.rng.random(infected_idx.size) < death_prob]
            if to_die.size > 0:
                self.alive[to_die] = False
                # remove from compartments
                self.I[to_die] = False
                self.R[to_die] = False
                self.S[to_die] = False  # consider dead not in S/I/R
        self._update_counts()

        # Yearly bookkeeping
        self.year += self.dt
        # record
        self.history["year"].append(self.year)
        self.history["n_s"].append(int(self.S.sum()))
        self.history["n_i"].append(int(self.I.sum()))
        self.history["n_r"].append(int(self.R.sum()))
        self.history["n_alive"].append(int(self.alive.sum()))
        self.history["new_infections"].append(len(new_infections))

        if self.verbose:
            print(f"Year {self.year:.2f}: S={self.history['n_s'][-1]}, I={self.history['n_i'][-1]}, R={self.history['n_r'][-1]}, Alive={self.history['n_alive'][-1]}")

    def _update_counts(self):
        # simple consistency check
        assert self.S.shape == (self.n,)
        assert self.I.shape == (self.n,)
        assert self.R.shape == (self.n,)
        # Alive flag is independent; we ensure no agent is alive=false and in S/I/R
        # If an agent is dead, we clear their compartments
        dead = ~self.alive
        if dead.any():
            self.S[dead] = False
            self.I[dead] = False
            self.R[dead] = False

    def run(self):
        # Run the simulation across the specified time horizon
        steps = int(math.ceil(self.years / self.dt))
        # initialize results with starting year 0
        self.history["year"].append(0.0)
        self.history["n_s"].append(int(self.S.sum()))
        self.history["n_i"].append(int(self.I.sum()))
        self.history["n_r"].append(int(self.R.sum()))
        self.history["n_alive"].append(int(self.alive.sum()))
        self.history["new_infections"].append(0)

        for _ in range(steps):
            if self.year >= self.years:
                break
            self._step(self.dt)

    def summary(self) -> dict:
        return {
            "final_year": self.year,
            "n_s": int(self.S.sum()),
            "n_i": int(self.I.sum()),
            "n_r": int(self.R.sum()),
            "n_alive": int(self.alive.sum()),
            "total_steps": len(self.history["year"]),
        }


def main_demo():
    # Simple demo run with reasonable defaults
    params = Params(
        n_agents=2000,
        beta=0.08,           # per-contact infection probability per year step
        gamma=0.3,           # recovery probability per year step
        dt=1.0,                # yearly steps
        end_years=50.0,        # simulate for 50 years
        avg_degree=6.0,        # average contacts per year
        seed=12345,
        verbose=0,
        initial_infected_frac=0.02,
        enable_deaths=False
    )

    sim = StarsimSIR(params)
    sim.run()
    s = sim.summary()
    print("Demo run summary:")
    print(s)

    # Optional: print full yearly history
    if params.verbose:
        for y, ns, ni, nr, na in zip(sim.history["year"],
                                    sim.history["n_s"],
                                    sim.history["n_i"],
                                    sim.history["n_r"],
                                    sim.history["n_alive"]):
            print(f"Year {y:.1f}: S={ns}, I={ni}, R={nr}, Alive={na}")


if __name__ == "__main__":
    main_demo()


# ---------------------------------------------------------------------------
# README

README:
- Purpose: A compact, runnable, production-ready ABM implementing a simple SIR disease model inspired by Starsim concepts. It uses an agent-based approach with a random contact network to simulate disease spread, recovery, and optional deaths.

- Key features:
  - Agent-based compartments: Susceptible (S), Infected (I), Recovered (R), Alive status
  - Random Erdős–Rényi contact network for possible transmission
  - Stochastic infection transmission per contact per time step
  - Recovery process with probability gamma per infected agent per time step
  - Optional death process (disabled by default)
  - Time stepping in years with dt granularity
  - Lightweight, easy-to-extend skeleton suitable for production-grade experimentation

- Assumptions:
  - Infection probability per contact is beta per time step (dt-adjusted)
  - Transmission occurs if any infected neighbor transmits to a susceptible agent in a step
  - Infectiousness is uniform across infected agents; no age/sex structure

- Missing values / extensibility notes:
  - No explicit births/migration modeled; can be added by expanding the alive mask and S/I/R transitions
  - No multi-disease interactions; extend with additional disease layers and connectors
  - Parameter selection should be calibrated against data if used for policy decisions

- Setup
  - Prerequisites: Python 3.8+, NumPy
  - The script is self-contained; no external data dependencies

- Execution
  - Run: python script.py
  - The script contains a main_demo() function that runs a sample scenario and prints a brief summary.

- Customization
  - Adjust the parameters inside main_demo() or create a new Params instance with desired values
  - For larger populations, consider memory optimizations or sparse representations; current implementation uses per-agent neighbor lists for clarity

- Contact
  - For questions or enhancements, extend Starsim-inspired skeleton with additional modules (modules.py, network.py, etc.) following the structural design principles illustrated in the source.