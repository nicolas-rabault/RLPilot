# RL Training Configuration

## Robot
- Name: <robot name>
- Type: <biped, quadruped, arm, etc.>
- Specificities: <key physical traits, constraints>
- Actuators: [<joint1>, <joint2>, ...]
- Special mechanics: <any special mechanisms, conversions>

## Task
- Name: <task identifier>
- Simulator: <MuJoCo, IsaacSim, PyBullet, etc.>
- Framework: <mjlab, IsaacLab, etc.>
- Algorithm: <PPO, SAC, etc.>
- Objective: <what the robot should learn>

## Training
- Command: <full training command>
- Execution: <remote or local>
- Env count: <number of parallel environments>

## Hosts
Order: [<host1>, <host2>, ...]

## Monitoring
- Tool: <wandb, tensorboard, local>
- Task monitoring: <task-name>
- Metric categories: [<prefix1/>, <prefix2/>, ...]
- Key metrics: [<metric1>, <metric2>, ...]
- Kill threshold: 2
- Max iterations: 10

## Evaluation
- Scenarios:
  - <name>: <params>
- Metrics: [<metric1>, <metric2>, ...]
- Video: true

## Decision Criteria
- KEEP: <when to keep training>
- BAD: <when training is going wrong>
- FINISH: <when training is done>

## Notifications
- Enabled: true
- Method: script
- When: [training_started, monitor_update, eval_complete, training_killed, iteration_started, blocker]

## Source Files
- Task config: <path>
- Rewards: <path>
- Observations: <path>
