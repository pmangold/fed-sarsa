# fed-sarsa

This code is the code that was used to generate the experiments of the ICML submission "Convergence Guarantees for Federated SARSA with Local Training and Heterogeneous Agents".

To run the experiments on Garnet, go to the `garnet` folder and run:

	python expe_convergence.py
	
Then, to make the plots, run
	
	python plot_convergence.py
	
The code is self-contained, and only depends on usual libraries `numpy`, `matplotlib`, `seaborn`, `pickle` and `os`.


To run the experiments on CartPole, go to the `jax` folder and run:
	
	./run_expe.sh
	
Then, to make the plots run

	python plot.py
