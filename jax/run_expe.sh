#!/bin/bash

python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 1
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 100

python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 1 --agent 2 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 10 --agent 2 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 100 --agent 2 --rep 10

python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 1 --agent 10 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 10 --agent 10 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 100 --agent 10 --rep 10

python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 1 --agent 50 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 10 --agent 50 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 100 --agent 50 --rep 10

python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 1 --agent 100 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 10 --agent 100 --rep 10
python3 sarsa_federated_parallel.py --env cartpole --seed 1 --local 100 --agent 100 --rep 10
