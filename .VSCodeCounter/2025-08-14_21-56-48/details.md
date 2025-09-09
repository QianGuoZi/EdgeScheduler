# Details

Date : 2025-08-14 21:56:48

Directory /home/qianguo/Edge-Scheduler

Total : 119 files,  12935 codes, 0 comments, 2788 blanks, all 15723 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [Controller/__init__.py](/Controller/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/base/__init__.py](/Controller/base/__init__.py) | Python | 15 | 0 | 4 | 19 |
| [Controller/base/algorithm/GA.py](/Controller/base/algorithm/GA.py) | Python | 119 | 0 | 25 | 144 |
| [Controller/base/algorithm/PPO/constraint_manager.py](/Controller/base/algorithm/PPO/constraint_manager.py) | Python | 225 | 0 | 49 | 274 |
| [Controller/base/algorithm/PPO/network_scheduler.py](/Controller/base/algorithm/PPO/network_scheduler.py) | Python | 737 | 0 | 163 | 900 |
| [Controller/base/algorithm/PPO/physical_environment_test.py](/Controller/base/algorithm/PPO/physical_environment_test.py) | Python | 1,073 | 0 | 216 | 1,289 |
| [Controller/base/algorithm/PPO/plot_utils.py](/Controller/base/algorithm/PPO/plot_utils.py) | Python | 195 | 0 | 36 | 231 |
| [Controller/base/algorithm/PPO/ppo_vs_random_test.py](/Controller/base/algorithm/PPO/ppo_vs_random_test.py) | Python | 853 | 0 | 197 | 1,050 |
| [Controller/base/algorithm/PPO/replay_buffer.py](/Controller/base/algorithm/PPO/replay_buffer.py) | Python | 262 | 0 | 49 | 311 |
| [Controller/base/algorithm/PPO/requirements.txt](/Controller/base/algorithm/PPO/requirements.txt) | pip requirements | 7 | 0 | 0 | 7 |
| [Controller/base/algorithm/PPO/train_two_stage_ppo.py](/Controller/base/algorithm/PPO/train_two_stage_ppo.py) | Python | 752 | 0 | 139 | 891 |
| [Controller/base/algorithm/PPO/two_stage_actor_design.py](/Controller/base/algorithm/PPO/two_stage_actor_design.py) | Python | 1,193 | 0 | 252 | 1,445 |
| [Controller/base/algorithm/PPO/two_stage_environment.py](/Controller/base/algorithm/PPO/two_stage_environment.py) | Python | 800 | 0 | 181 | 981 |
| [Controller/base/algorithm/Rand.py](/Controller/base/algorithm/Rand.py) | Python | 66 | 0 | 16 | 82 |
| [Controller/base/controller.py](/Controller/base/controller.py) | Python | 657 | 0 | 95 | 752 |
| [Controller/base/link.py](/Controller/base/link.py) | Python | 12 | 0 | 5 | 17 |
| [Controller/base/manager.py](/Controller/base/manager.py) | Python | 328 | 0 | 41 | 369 |
| [Controller/base/nfs.py](/Controller/base/nfs.py) | Python | 5 | 0 | 1 | 6 |
| [Controller/base/node.py](/Controller/base/node.py) | Python | 157 | 0 | 22 | 179 |
| [Controller/base/scheduler.py](/Controller/base/scheduler.py) | Python | 144 | 0 | 25 | 169 |
| [Controller/base/task.py](/Controller/base/task.py) | Python | 86 | 0 | 17 | 103 |
| [Controller/base/taskManger.py](/Controller/base/taskManger.py) | Python | 111 | 0 | 14 | 125 |
| [Controller/base/utils.py](/Controller/base/utils.py) | Python | 28 | 0 | 6 | 34 |
| [Controller/dml_app/1/dml_utils.py](/Controller/dml_app/1/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Controller/dml_app/1/gl_peer.py](/Controller/dml_app/1/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Controller/dml_app/1/nns/__init__.py](/Controller/dml_app/1/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/dml_app/1/nns/nn_cifar10.py](/Controller/dml_app/1/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Controller/dml_app/1/nns/nn_fashion_mnist.py](/Controller/dml_app/1/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Controller/dml_app/1/nns/nn_mnist.py](/Controller/dml_app/1/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Controller/dml_app/1/nns/nn_mnist_lenet.py](/Controller/dml_app/1/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Controller/dml_app/1/worker_utils.py](/Controller/dml_app/1/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Controller/dml_app/2/dml_utils.py](/Controller/dml_app/2/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Controller/dml_app/2/gl_peer.py](/Controller/dml_app/2/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Controller/dml_app/2/nns/__init__.py](/Controller/dml_app/2/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/dml_app/2/nns/nn_cifar10.py](/Controller/dml_app/2/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Controller/dml_app/2/nns/nn_fashion_mnist.py](/Controller/dml_app/2/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Controller/dml_app/2/nns/nn_mnist.py](/Controller/dml_app/2/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Controller/dml_app/2/nns/nn_mnist_lenet.py](/Controller/dml_app/2/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Controller/dml_app/2/worker_utils.py](/Controller/dml_app/2/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Controller/dml_app/3/dml_utils.py](/Controller/dml_app/3/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Controller/dml_app/3/gl_peer.py](/Controller/dml_app/3/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Controller/dml_app/3/nns/__init__.py](/Controller/dml_app/3/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/dml_app/3/nns/nn_cifar10.py](/Controller/dml_app/3/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Controller/dml_app/3/nns/nn_fashion_mnist.py](/Controller/dml_app/3/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Controller/dml_app/3/nns/nn_mnist.py](/Controller/dml_app/3/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Controller/dml_app/3/nns/nn_mnist_lenet.py](/Controller/dml_app/3/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Controller/dml_app/3/worker_utils.py](/Controller/dml_app/3/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Controller/dml_tool/1/conf_utils.py](/Controller/dml_tool/1/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Controller/dml_tool/1/dataset_conf.py](/Controller/dml_tool/1/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Controller/dml_tool/1/gl_structure_conf.py](/Controller/dml_tool/1/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Controller/dml_tool/2/conf_utils.py](/Controller/dml_tool/2/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Controller/dml_tool/2/dataset_conf.py](/Controller/dml_tool/2/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Controller/dml_tool/2/gl_structure_conf.py](/Controller/dml_tool/2/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Controller/dml_tool/3/conf_utils.py](/Controller/dml_tool/3/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Controller/dml_tool/3/dataset_conf.py](/Controller/dml_tool/3/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Controller/dml_tool/3/gl_structure_conf.py](/Controller/dml_tool/3/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Controller/scheduling_logs/plot_loads.py](/Controller/scheduling_logs/plot_loads.py) | Python | 55 | 0 | 14 | 69 |
| [Controller/task_file/1/dml_app/dml_utils.py](/Controller/task_file/1/dml_app/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Controller/task_file/1/dml_app/gl_peer.py](/Controller/task_file/1/dml_app/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Controller/task_file/1/dml_app/nns/__init__.py](/Controller/task_file/1/dml_app/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/task_file/1/dml_app/nns/nn_cifar10.py](/Controller/task_file/1/dml_app/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Controller/task_file/1/dml_app/nns/nn_fashion_mnist.py](/Controller/task_file/1/dml_app/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Controller/task_file/1/dml_app/nns/nn_mnist.py](/Controller/task_file/1/dml_app/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Controller/task_file/1/dml_app/nns/nn_mnist_lenet.py](/Controller/task_file/1/dml_app/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Controller/task_file/1/dml_app/worker_utils.py](/Controller/task_file/1/dml_app/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Controller/task_file/1/dml_tool/conf_utils.py](/Controller/task_file/1/dml_tool/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Controller/task_file/1/dml_tool/dataset_conf.py](/Controller/task_file/1/dml_tool/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Controller/task_file/1/dml_tool/gl_structure_conf.py](/Controller/task_file/1/dml_tool/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Controller/task_file/1/task_manager.py](/Controller/task_file/1/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_file/2/dml_app/dml_utils.py](/Controller/task_file/2/dml_app/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Controller/task_file/2/dml_app/gl_peer.py](/Controller/task_file/2/dml_app/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Controller/task_file/2/dml_app/nns/__init__.py](/Controller/task_file/2/dml_app/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/task_file/2/dml_app/nns/nn_cifar10.py](/Controller/task_file/2/dml_app/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Controller/task_file/2/dml_app/nns/nn_fashion_mnist.py](/Controller/task_file/2/dml_app/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Controller/task_file/2/dml_app/nns/nn_mnist.py](/Controller/task_file/2/dml_app/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Controller/task_file/2/dml_app/nns/nn_mnist_lenet.py](/Controller/task_file/2/dml_app/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Controller/task_file/2/dml_app/worker_utils.py](/Controller/task_file/2/dml_app/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Controller/task_file/2/dml_tool/conf_utils.py](/Controller/task_file/2/dml_tool/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Controller/task_file/2/dml_tool/dataset_conf.py](/Controller/task_file/2/dml_tool/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Controller/task_file/2/dml_tool/gl_structure_conf.py](/Controller/task_file/2/dml_tool/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Controller/task_file/2/task_manager.py](/Controller/task_file/2/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_file/3/dml_app/dml_utils.py](/Controller/task_file/3/dml_app/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Controller/task_file/3/dml_app/gl_peer.py](/Controller/task_file/3/dml_app/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Controller/task_file/3/dml_app/nns/__init__.py](/Controller/task_file/3/dml_app/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Controller/task_file/3/dml_app/nns/nn_cifar10.py](/Controller/task_file/3/dml_app/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Controller/task_file/3/dml_app/nns/nn_fashion_mnist.py](/Controller/task_file/3/dml_app/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Controller/task_file/3/dml_app/nns/nn_mnist.py](/Controller/task_file/3/dml_app/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Controller/task_file/3/dml_app/nns/nn_mnist_lenet.py](/Controller/task_file/3/dml_app/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Controller/task_file/3/dml_app/worker_utils.py](/Controller/task_file/3/dml_app/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Controller/task_file/3/dml_tool/conf_utils.py](/Controller/task_file/3/dml_tool/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Controller/task_file/3/dml_tool/dataset_conf.py](/Controller/task_file/3/dml_tool/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Controller/task_file/3/dml_tool/gl_structure_conf.py](/Controller/task_file/3/dml_tool/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Controller/task_file/3/task_manager.py](/Controller/task_file/3/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/1/task_manager.py](/Controller/task_manager/1/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/10/task_manager.py](/Controller/task_manager/10/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/11/task_manager.py](/Controller/task_manager/11/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/2/task_manager.py](/Controller/task_manager/2/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/3/task_manager.py](/Controller/task_manager/3/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/4/task_manager.py](/Controller/task_manager/4/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/5/task_manager.py](/Controller/task_manager/5/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/6/task_manager.py](/Controller/task_manager/6/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/7/task_manager.py](/Controller/task_manager/7/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/8/task_manager.py](/Controller/task_manager/8/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/task_manager/9/task_manager.py](/Controller/task_manager/9/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Controller/test.py](/Controller/test.py) | Python | 21 | 0 | 4 | 25 |
| [Task/dml_app/dml_utils.py](/Task/dml_app/dml_utils.py) | Python | 101 | 0 | 30 | 131 |
| [Task/dml_app/gl_peer.py](/Task/dml_app/gl_peer.py) | Python | 124 | 0 | 41 | 165 |
| [Task/dml_app/nns/__init__.py](/Task/dml_app/nns/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [Task/dml_app/nns/nn_cifar10.py](/Task/dml_app/nns/nn_cifar10.py) | Python | 38 | 0 | 8 | 46 |
| [Task/dml_app/nns/nn_fashion_mnist.py](/Task/dml_app/nns/nn_fashion_mnist.py) | Python | 24 | 0 | 8 | 32 |
| [Task/dml_app/nns/nn_mnist.py](/Task/dml_app/nns/nn_mnist.py) | Python | 14 | 0 | 8 | 22 |
| [Task/dml_app/nns/nn_mnist_lenet.py](/Task/dml_app/nns/nn_mnist_lenet.py) | Python | 18 | 0 | 8 | 26 |
| [Task/dml_app/worker_utils.py](/Task/dml_app/worker_utils.py) | Python | 62 | 0 | 14 | 76 |
| [Task/dml_tool/conf_utils.py](/Task/dml_tool/conf_utils.py) | Python | 27 | 0 | 11 | 38 |
| [Task/dml_tool/dataset_conf.py](/Task/dml_tool/dataset_conf.py) | Python | 33 | 0 | 6 | 39 |
| [Task/dml_tool/gl_structure_conf.py](/Task/dml_tool/gl_structure_conf.py) | Python | 92 | 0 | 18 | 110 |
| [Task/task_manager.py](/Task/task_manager.py) | Python | 65 | 0 | 7 | 72 |
| [Worker/agent.py](/Worker/agent.py) | Python | 328 | 0 | 39 | 367 |
| [__init__.py](/__init__.py) | Python | 0 | 0 | 1 | 1 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)