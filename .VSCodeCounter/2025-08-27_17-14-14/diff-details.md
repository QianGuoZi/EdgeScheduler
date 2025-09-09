# Diff Details

Date : 2025-08-27 17:14:14

Directory /home/qianguo/Edge-Scheduler

Total : 53 files,  15282 codes, 0 comments, 3368 blanks, all 18650 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [Controller/base/algorithm/PPO/constraint_manager.py](/Controller/base/algorithm/PPO/constraint_manager.py) | Python | -225 | 0 | -49 | -274 |
| [Controller/base/algorithm/PPO/network_scheduler.py](/Controller/base/algorithm/PPO/network_scheduler.py) | Python | -737 | 0 | -163 | -900 |
| [Controller/base/algorithm/PPO/physical_environment_test.py](/Controller/base/algorithm/PPO/physical_environment_test.py) | Python | -1,073 | 0 | -216 | -1,289 |
| [Controller/base/algorithm/PPO/plot_utils.py](/Controller/base/algorithm/PPO/plot_utils.py) | Python | -195 | 0 | -36 | -231 |
| [Controller/base/algorithm/PPO/ppo_vs_random_test.py](/Controller/base/algorithm/PPO/ppo_vs_random_test.py) | Python | -853 | 0 | -197 | -1,050 |
| [Controller/base/algorithm/PPO/replay_buffer.py](/Controller/base/algorithm/PPO/replay_buffer.py) | Python | -262 | 0 | -49 | -311 |
| [Controller/base/algorithm/PPO/requirements.txt](/Controller/base/algorithm/PPO/requirements.txt) | pip requirements | -7 | 0 | 0 | -7 |
| [Controller/base/algorithm/PPO/train_two_stage_ppo.py](/Controller/base/algorithm/PPO/train_two_stage_ppo.py) | Python | -752 | 0 | -139 | -891 |
| [Controller/base/algorithm/PPO/two_stage_actor_design.py](/Controller/base/algorithm/PPO/two_stage_actor_design.py) | Python | -1,193 | 0 | -252 | -1,445 |
| [Controller/base/algorithm/PPO/two_stage_environment.py](/Controller/base/algorithm/PPO/two_stage_environment.py) | Python | -800 | 0 | -181 | -981 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/constraint_manager.py](/Controller/base/algorithm/PPO_example/PPO_code_example/constraint_manager.py) | Python | 228 | 0 | 47 | 275 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/heuristic_algorithm.py](/Controller/base/algorithm/PPO_example/PPO_code_example/heuristic_algorithm.py) | Python | 298 | 0 | 84 | 382 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/heuristic_comparison_test.py](/Controller/base/algorithm/PPO_example/PPO_code_example/heuristic_comparison_test.py) | Python | 517 | 0 | 129 | 646 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/multi_virtual_nodes_test.py](/Controller/base/algorithm/PPO_example/PPO_code_example/multi_virtual_nodes_test.py) | Python | 453 | 0 | 100 | 553 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/network_scheduler.py](/Controller/base/algorithm/PPO_example/PPO_code_example/network_scheduler.py) | Python | 1,035 | 0 | 225 | 1,260 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/original_problem_config.py](/Controller/base/algorithm/PPO_example/PPO_code_example/original_problem_config.py) | Python | 128 | 0 | 22 | 150 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/original_reward.py](/Controller/base/algorithm/PPO_example/PPO_code_example/original_reward.py) | Python | 236 | 0 | 48 | 284 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/physical_environment_test.py](/Controller/base/algorithm/PPO_example/PPO_code_example/physical_environment_test.py) | Python | 1,104 | 0 | 222 | 1,326 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/plot_utils.py](/Controller/base/algorithm/PPO_example/PPO_code_example/plot_utils.py) | Python | 356 | 0 | 64 | 420 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/ppo_vs_random_test.py](/Controller/base/algorithm/PPO_example/PPO_code_example/ppo_vs_random_test.py) | Python | 883 | 0 | 204 | 1,087 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/replay_buffer.py](/Controller/base/algorithm/PPO_example/PPO_code_example/replay_buffer.py) | Python | 262 | 0 | 49 | 311 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/requirements.txt](/Controller/base/algorithm/PPO_example/PPO_code_example/requirements.txt) | pip requirements | 7 | 0 | 0 | 7 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/sequential_agent.py](/Controller/base/algorithm/PPO_example/PPO_code_example/sequential_agent.py) | Python | 299 | 0 | 70 | 369 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/sequential_environment.py](/Controller/base/algorithm/PPO_example/PPO_code_example/sequential_environment.py) | Python | 624 | 0 | 155 | 779 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/sequential_ppo_vs_random_test.py](/Controller/base/algorithm/PPO_example/PPO_code_example/sequential_ppo_vs_random_test.py) | Python | 231 | 0 | 60 | 291 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/test_original_config.py](/Controller/base/algorithm/PPO_example/PPO_code_example/test_original_config.py) | Python | 135 | 0 | 38 | 173 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/test_sequential_original_simple.py](/Controller/base/algorithm/PPO_example/PPO_code_example/test_sequential_original_simple.py) | Python | 196 | 0 | 38 | 234 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/test_sequential_simple.py](/Controller/base/algorithm/PPO_example/PPO_code_example/test_sequential_simple.py) | Python | 204 | 0 | 53 | 257 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/three_algorithms_comparison_test.py](/Controller/base/algorithm/PPO_example/PPO_code_example/three_algorithms_comparison_test.py) | Python | 682 | 0 | 141 | 823 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/train_sequential_original.py](/Controller/base/algorithm/PPO_example/PPO_code_example/train_sequential_original.py) | Python | 416 | 0 | 74 | 490 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/train_two_stage_ppo.py](/Controller/base/algorithm/PPO_example/PPO_code_example/train_two_stage_ppo.py) | Python | 844 | 0 | 145 | 989 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/two_stage_actor_design.py](/Controller/base/algorithm/PPO_example/PPO_code_example/two_stage_actor_design.py) | Python | 1,435 | 0 | 292 | 1,727 |
| [Controller/base/algorithm/PPO_example/PPO_code_example/two_stage_environment.py](/Controller/base/algorithm/PPO_example/PPO_code_example/two_stage_environment.py) | Python | 747 | 0 | 158 | 905 |
| [Controller/base/algorithm/PPO_mapping/mapping_agent.py](/Controller/base/algorithm/PPO_mapping/mapping_agent.py) | Python | 324 | 0 | 75 | 399 |
| [Controller/base/algorithm/PPO_mapping/mapping_environment.py](/Controller/base/algorithm/PPO_mapping/mapping_environment.py) | Python | 530 | 0 | 122 | 652 |
| [Controller/base/algorithm/PPO_mapping/network_scheduler.py](/Controller/base/algorithm/PPO_mapping/network_scheduler.py) | Python | 789 | 0 | 174 | 963 |
| [Controller/base/algorithm/PPO_mapping/original_problem_config.py](/Controller/base/algorithm/PPO_mapping/original_problem_config.py) | Python | 54 | 0 | 6 | 60 |
| [Controller/base/algorithm/PPO_mapping/test_mapping.py](/Controller/base/algorithm/PPO_mapping/test_mapping.py) | Python | 210 | 0 | 53 | 263 |
| [Controller/base/algorithm/PPO_mapping/train_mapping.py](/Controller/base/algorithm/PPO_mapping/train_mapping.py) | Python | 400 | 0 | 78 | 478 |
| [Controller/base/algorithm/PPO_my/four_algorithms_comparison_test.py](/Controller/base/algorithm/PPO_my/four_algorithms_comparison_test.py) | Python | 1,163 | 0 | 247 | 1,410 |
| [Controller/base/algorithm/PPO_my/heuristic_algorithm.py](/Controller/base/algorithm/PPO_my/heuristic_algorithm.py) | Python | 431 | 0 | 119 | 550 |
| [Controller/base/algorithm/PPO_my/lightweight_heuristic_integration_environment.py](/Controller/base/algorithm/PPO_my/lightweight_heuristic_integration_environment.py) | Python | 444 | 0 | 127 | 571 |
| [Controller/base/algorithm/PPO_my/network_scheduler.py](/Controller/base/algorithm/PPO_my/network_scheduler.py) | Python | 789 | 0 | 174 | 963 |
| [Controller/base/algorithm/PPO_my/new_heuristic_environment.py](/Controller/base/algorithm/PPO_my/new_heuristic_environment.py) | Python | 406 | 0 | 118 | 524 |
| [Controller/base/algorithm/PPO_my/original_problem_config.py](/Controller/base/algorithm/PPO_my/original_problem_config.py) | Python | 54 | 0 | 6 | 60 |
| [Controller/base/algorithm/PPO_my/original_reward.py](/Controller/base/algorithm/PPO_my/original_reward.py) | Python | 236 | 0 | 48 | 284 |
| [Controller/base/algorithm/PPO_my/sequential_agent.py](/Controller/base/algorithm/PPO_my/sequential_agent.py) | Python | 276 | 0 | 66 | 342 |
| [Controller/base/algorithm/PPO_my/sequential_environment.py](/Controller/base/algorithm/PPO_my/sequential_environment.py) | Python | 628 | 0 | 145 | 773 |
| [Controller/base/algorithm/PPO_my/three_algorithms_comparison_test.py](/Controller/base/algorithm/PPO_my/three_algorithms_comparison_test.py) | Python | 787 | 0 | 170 | 957 |
| [Controller/base/algorithm/PPO_my/train_lightweight_ppo.py](/Controller/base/algorithm/PPO_my/train_lightweight_ppo.py) | Python | 580 | 0 | 113 | 693 |
| [Controller/base/algorithm/PPO_my/train_new_heuristic_ppo.py](/Controller/base/algorithm/PPO_my/train_new_heuristic_ppo.py) | Python | 580 | 0 | 113 | 693 |
| [Controller/base/algorithm/PPO_my/train_sequential_original.py](/Controller/base/algorithm/PPO_my/train_sequential_original.py) | Python | 482 | 0 | 86 | 568 |
| [Controller/base/algorithm/PPO_my/two_ppo_comparison_test.py](/Controller/base/algorithm/PPO_my/two_ppo_comparison_test.py) | Python | 896 | 0 | 192 | 1,088 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details