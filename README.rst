.. _SMPL: https://github.com/smpl-env/smpl

.. _Openai Gym: https://gym.openai.com/

.. _d4rl: https://github.com/rail-berkeley/d4rl.git

.. _PenSimPy: https://github.com/smpl-env/PenSimPy.git

.. _fastodeint: https://github.com/smpl-env/fastodeint.git

The Simulated Industrial Manufacturing and Process Control Learning Environments (`SMPL`_) supplements several process control environments to the `Openai Gym`_ family, which quenches the pain of performing Deep Reinforcement Learning algorithms on them. Furthermore, we provided `d4rl`_-like wrappers for accompanied datasets, making Offline RL on those environments even smoother.

For the paper `SMPL: Simulated Manufacturing and Process Control Learning Environments <https://openreview.net/forum?id=TscdNx8udf5>`_, you can cite with 

.. code-block::
    @inproceedings{
    zhang2022smpl,
    title={{SMPL}: Simulated Industrial Manufacturing and Process Control Learning Environments},
    author={Mohan Zhang and Xiaozhou Wang and Benjamin Decardi-Nelson and Song Bo and An Zhang and Jinfeng Liu and Sile Tao and Jiayi Cheng and Xiaohong Liu and Dengdeng Yu and Matthew Poon and Animesh Garg},
    booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2022},
    url={https://openreview.net/forum?id=TscdNx8udf5}
    }

Documentation
-------------

The documentation is available here! `https://smpl-env.github.io/smpl-document/index.html <https://smpl-env.github.io/smpl-document/index.html>`_

Install
-------
.. code-block::

    $ git git@github.com:smpl-env/smpl.git
    $ cd smpl
    $ pip install .

.. note::
    You will need to build the `PenSimPy`_ environment with `SMPL`_ separately. Namely, build and install `fastodeint`_ following `this instruction <https://github.com/smpl-env/fastodeint/blob/master/README.md>`_, then install `PenSimPy`_.

    For Linux users, you can just install `fastodeint`_ and `PenSimPy`_ by executing the following commands:

    .. code-block::

        $ sudo apt-get install libomp-dev
        $ sudo apt-get install libboost-all-dev
        $ git clone --recursive git@github.com:smpl-env/fastodeint.git
        $ cd fastodeint
        $ pip install .
        $ cd ..
        $ git clone --recursive git@github.com:smpl-env/PenSimPy.git
        $ cd PenSimPy
        $ pip install .

    If you also want to use the pre-built MPC and EMPC controllers, you would need to install mpctools by CasADi. For Linux users, you can execute the following commands:

    .. code-block::

        $ git clone --recursive git@github.com:smpl-env/mpc-tools-casadi.git
        $ cd mpc-tools-casadi
        $ python mpctoolssetup.py install --user


Example Usage
-------------

See the `jupyter notebook <https://github.com/smpl-env/smpl/blob/main/examples.ipynb>`_ for example use cases.
