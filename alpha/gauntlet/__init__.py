"""Layer 4: the gauntlet.

Seven tests a candidate has to survive, with no threshold written by hand.
``folds`` builds purged time splits; ``gauntlet`` runs the tests and reports
every statistic, judging only once thresholds have been measured on both
sides: the false-acceptance rate with nothing planted, and the joint
false-rejection rate with signals planted at a known IC.
"""
