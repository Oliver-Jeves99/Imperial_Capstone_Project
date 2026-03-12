# Imperial_Capstone_Project
**Black-Box Optimisation Capstone Project**
Non-Technical Explanation of the Project

This project explores how machine learning can be used to optimise unknown systems. Participants were given eight hidden functions and a limited number of opportunities to test input values. Each week a new set of inputs was submitted, and the resulting outputs were returned. By analysing previous results and building surrogate models, the goal was to gradually discover input values that produce the highest possible outputs. The project demonstrates how optimisation algorithms learn from small amounts of data and improve decisions over time.

Data

The dataset consists of query points submitted to eight unknown functions during the capstone challenge. Each query contains input variables and the resulting output value returned by the evaluation system.

The dataset was generated through the capstone portal as part of the optimisation process.

Model

The optimisation strategy uses Bayesian optimisation with a Gaussian Process surrogate model. This approach models the unknown function using previously observed data and selects new query points that balance exploration and exploitation.

This method is well suited to problems where evaluations are expensive and only a small number of observations are available.

Hyperparameter Optimisation

Key parameters in the optimisation process include the Gaussian Process kernel settings and the acquisition function used to select new queries. These parameters influence how aggressively the model explores new regions versus exploiting known high-performing areas.

Strategies were adjusted over time based on observed results.

Results

Over multiple optimisation rounds, the strategy gradually identified promising regions for several functions. Later rounds focused on refining these regions to improve results further.

The project illustrates how iterative optimisation strategies can progressively improve solutions even when the underlying function is completely unknown.
