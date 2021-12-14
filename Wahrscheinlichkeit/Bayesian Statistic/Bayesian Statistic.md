[17. Bayesian Statistics - YouTube](https://www.youtube.com/watch?v=bFZ-0FH5hfs)

In the beginning of the process, we come up with a potential probability (parameter) for which we believe that it describes something approximately.
This probability is the 'belief'.

An example for such an belief is a probability p = 0.5 that a randomly drawn person is female.

In the second stage of the process, we draw further samples and update our belief.

In order to model the belief at first place, we need to find a suitable distribution between 0 and 1 on the x axis ([[Verteilung & Wahrscheinlichkeitsfunktion]]). This is called the prior distribution.

After we determined a certain parameter p, we can draw a new sample and get new data.

Then we update our belief from the beginning using the [[Bayes Theorem]]. A is the probability of the parameter. B is the distribution of our new data given the parameter.

The updated distribution which models p will be of the same type as in the beginning (beta -> beta, gaussian -> gaussian).
