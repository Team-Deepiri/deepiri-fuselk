# RL Convergence Guarantees

## Environment: `VentCircularizerEnv`

| Property | Value | Guarantee |
|----------|-------|-----------|
| Horizon | `max_steps=200` | Finite-horizon MDP |
| Rewards | Bounded variance penalty | $|r| < \infty$ per step |
| Action space | `[0,1]^2` continuous box | Compact; PPO-compatible |

## Formal results used

1. **Finite-horizon MDP**: Optimal policy exists (Puterman, 1994).
2. **Discretized value iteration**: On coarse state hash, VI converges when Bellman residual $< \epsilon$.
3. **PPO** (Schulman et al., 2017): With bounded rewards and finite episodes, policy gradient estimates have bounded variance; convergence to stationary point under step-size conditions.

## Verification (wired into training)

`train_vent_policy()` attaches a `ConvergenceReport` via `verify_rl_convergence()`.
Full audit: `fuselk doctor --vision`.

- `bellman_residual` — VI convergence quality
- `value_iteration_iterations` — iterations to tolerance
- `ppo_assumptions_met` — finite horizon + bounded reward + compact actions

## Limitations (honest)

- Continuous state PPO has **no global optimality guarantee**; discretized VI provides a **lower-bound certificate** on coarse partition.
- Production deployment requires sample-complexity monitoring (KL clip, explained variance).
