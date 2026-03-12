Model Description

Input

The model takes numerical input vectors representing candidate query points for the optimisation problem. Each function has a different dimensionality ranging from two to eight variables.

Output

The model predicts the expected output value of the unknown function at a given input point. These predictions are used to select the next query point in the optimisation process.

Model Architecture

The optimisation strategy uses Bayesian optimisation with a Gaussian Process surrogate model. The Gaussian Process approximates the unknown function based on previously observed query results.

An acquisition function (such as Expected Improvement) is used to balance:

Exploration of uncertain regions

Exploitation of promising regions

The next query is chosen by maximising this acquisition function.

Performance

Performance is evaluated based on the highest output values discovered during the optimisation process. Improvements were tracked week-by-week as additional query points were evaluated.

Visual inspection of optimisation progress and comparison of outputs across rounds were used to assess strategy effectiveness.

Because the true functions were hidden, performance was measured relative to previous results rather than against a known ground truth.

Limitations

The dataset is extremely small, limiting the accuracy of surrogate models.

The true functions are unknown, making evaluation indirect.

The optimisation process may become trapped in local optima.

Results depend heavily on early sampling decisions.

Trade-offs

Several trade-offs were involved in the optimisation process:

Exploration vs exploitation
Exploring new areas may discover better solutions but risks wasting limited queries.

Model complexity vs robustness
More complex surrogate models may capture patterns better but risk overfitting due to the small dataset.

Local refinement vs global search
Focusing on known promising regions improves stability but may miss better solutions elsewhere.