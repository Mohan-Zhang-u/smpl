BeerFMTEnv
===========

Beer is one of the oldest and most widely consumed alcoholic drinks in the world, and we have a lot of ways to produce them. This class provides a typical (and simple enough) simulation of the industry-level beer fermentation process. The only input that the simulation takes is the reaction temperature. The end goal in this simulation is to reach the stop condition (finish production) with a certain time limit, the quicker the better.

To better assist any control algorithms, we also provided a canonical production process under this simulation. Please consult the 'profile_industrial' in the BeerFMT section `here <https://github.com/smpl-env/smpl/blob/main/examples.ipynb>`_ for more details.

BeerFMTEnv module
-----------------

Following the discription above, we provide APIs as below:

.. automodule:: smpl.envs.beerfmtenv
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:
