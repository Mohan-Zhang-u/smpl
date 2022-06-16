======
MAbEnv
======

.. role:: raw-latex(raw)
   :format: latex
..

Introduction to Modeling and Operation of a Continuous Monoclonal Antibody (mAb) Manufacturing Process
------------------------------------------------------------------------------------------------------

Drugs based on monoclonal antibodies (mAbs) play an indispensable role
in biopharmaceutical industry in aspects of therapeutic and market
potentials. In therapy and diagnosis applications, mAbs are widely used
for the treatment of autoimmune diseases, and cancer, etc. According to
a recent publication, mAbs also show promising results in the treatment
of COVID-19. Until September 22, 2020,
94 therapeutic mAbs have been approved by U.S. Food & Drug
Administration (FDA) and the
number of mAbs approved within 2010-2020 is three times more than those
approved before 2010. In terms
of its market value, it is expected to reach a value of $198.2 billion
in 2023. Integrated continuous manufacturing
of mAbs represents the state-of-the-art in mAb manufacturing and has
attracted a lot of attention, because of the steady-state operations,
high volumetric productivity, and reduced equipment size and capital
cost, etc. However, there is no
existing mathematical model of the integrated manufacturing process and
there is no optimal control algorithm of the entire integrated process.
This project fills the knowledge gaps by first developing a mathematical
model of the integrated continuous manufacturing process of mAbs.

Process description
-------------------

The mAb production process consists of the
upstream and the downstream processes. In the upstream process, mAb is
produced in a bioreactor which provides a conducive environment mAb
growth. The downstream process on the other hand recovers the mAb from
the upstream process for storage. In the upstream process for mAb
production, fresh media is fed into the bioreactor where a conducive
environment is provided for the growth of mAb. A cooling jacket in which
a coolant flows is used to control the temperature of the reaction
mixture. The contents exiting the bioreactor is passed through a
microfiltration unit which recovers part of the fresh media in the
stream. The recovered fresh media is recycled back into the bioreactor
while the stream with high amount of mAb is sent to the downstream
process for further processing. A schematic diagram of upstream process
is shown in Figure :numref:`fig:upstream`.

   .. figure:: imgs/upstream_process.png
      :name: fig:upstream
      :align: center
      :width: 90.0%

      A schematic diagram of the upstream process for mAb production


The objective of the downstream process for mAb production is to purify
the stream with high concentration of mAb from the upstream and obtain
the desired product. It is composed of a
set of fractionating columns, for separating mAb from impurities, and
holdup loops, for virus inactivation (VI) and pH conditioning. The
schematic diagram of downstream process is shown in :numref:`fig:downstream`.

   .. figure:: imgs/downstream_process.png
      :name: fig:downstream
      :align: center
      :width: 90.0%

      A schematic diagram of the downstream process for mAb production


Prebuilt Upstream Controllers
-----------------------------

There are two provided implementation of advanced
process control (APC) techniques on the operation of the upstream continuous mAb
production process,  
Model Predictive Control (MAbUpstreamMPC) and Economic Model Predictive Control
(MAbUpstreamEMPC). Here we provide a brief description of both.

Simulation settings
-------------------

After conducting extensive open-loop tests, the control and prediction
horizons :math:`N` for both controllers was fixed at 100. This implies
that at a sampling time of 1 hour, the controllers plan 100 hours into
the future. The weights on the deviation of the states and input from
the setpoint were identify matrices.

Upstream Simulation Results
---------------------------

The state and input trajectories of the system under the operation of
both MPC and EMPC is shown in Figures :numref:`fig:Figure_1` and
:numref:`fig:Figure_12`. It can be seen that MPC and EMPC uses
different strategies to control the process. As an example, it can be
seen in Figure :numref:`fig:Figure_9` that EMPC initially heats up the
system before gradually reducing it whereas MPC goes to the setpoint and
stays there. Again, EMPC tries to reduce the flow of the recycle stream
while MPC increases it as can be seen in Figure
:numref:`fig:Figure_11`. In both controllers though, the recycle
stream flow rate was kept low. Although the setpoint for the MPC was
determined under the same economic cost function used in the EMPC, it
can be seen that the EMPC does not go to that optimal steady state. This
could be due to the horizon being short for EMPC. Another possibility
could be due to numerical errors since the cost function was not scaled
in the EMPC. The cases where MPC was unable to go to the setpoint could
be due to numerical errors as a result of the large values of the states
and inputs. Further analysis may be required to confirm these
assertions.



   .. figure:: imgs/Figure_1.png
      :align: center
      :name: fig:Figure_1
      :width: 90.0%

      Trajectories of concentration of viable cells in the bioreactor
      and separator under the two control algorithms



   .. figure:: imgs/Figure_2.png
      :align: center
      :name: fig:Figure_2
      :width: 90.0%

      Trajectories of total viable cells in the bioreactor and separator
      under the two control algorithms



   .. figure:: imgs/Figure_3.png
      :align: center
      :name: fig:Figure_3
      :width: 90.0%

      Trajectories of glucose concentration in the bioreactor and
      separator under the two control algorithms



   .. figure:: imgs/Figure_4.png
      :align: center
      :name: fig:Figure_4
      :width: 90.0%

      Trajectories of glutamine concentration in the bioreactor and
      separator under the two control algorithms



   .. figure:: imgs/Figure_5.png
      :align: center
      :name: fig:Figure_5
      :width: 90.0%

      Trajectories of lactate concentration in the bioreactor and
      separator under the two control algorithms



   .. figure:: imgs/Figure_6.png
      :align: center
      :name: fig:Figure_6
      :width: 90.0%

      Trajectories of ammonia concentration in the bioreactor and
      separator under the two control algorithms



   .. figure:: imgs/Figure_7.png
      :align: center
      :name: fig:Figure_7
      :width: 90.0%

      Trajectories of mAb concentration in the bioreactor and separator
      under the two control algorithms



   .. figure:: imgs/Figure_8.png
      :align: center
      :name: fig:Figure_8.png
      :width: 90.0%

      Trajectories of reaction mixture volume in the bioreactor and
      separator under the two control algorithms



   .. figure:: imgs/Figure_9.png
      :align: center
      :name: fig:Figure_9
      :width: 90.0%

      Trajectories of the bioreactor temperature and the coolant
      temperature under the two control algorithms



   .. figure:: imgs/Figure_10.png
      :align: center
      :name: fig:Figure_10
      :width: 90.0%

      Trajectories of flow in and out of the bioreactor under the two
      control algorithms



   .. figure:: imgs/Figure_11.png
      :align: center
      :name: fig:Figure_11
      :width: 90.0%

      Trajectories of the recycle flow rate and the flow rate out of the
      upstream process under the two control algorithms



   .. figure:: imgs/Figure_12.png
      :align: center
      :name: fig:Figure_12
      :width: 90.0%

      Trajectories of glucose in fresh media under the two control
      algorithms


Model parameters
================

.. _upstream-1:

Upstream
--------

.. container::
   :name: tb:upstream_parameters

   .. table:: Parameters for the upstream process model
      :align: center

      +----------------------+------------------------------------+-----------------------------+
      | Parameter            | Unit                               | Value                       |
      +======================+====================================+=============================+
      | :math:`K_{d,amm}`    | :math:`mM`                         | :math:`1.76`                |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`K_{d,gln}`    | :math:`hr^{-1}`                    | :math:`0.0096`              |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`K_{glc}`      | :math:`mM`                         | :math:`0.75`                |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`K_{gln}`      | :math:`mM`                         | :math:`0.038`               |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`KI_{amm}`     | :math:`mM`                         | :math:`28.48`               |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`KI_{lac}`     | :math:`mM`                         | :math:`171.76`              |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`m_{glc}`      | :math:`mmol/(cell \cdot hr)`       | :math:`4.9 \times 10^{-14}` |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`Q_{mAb}^{max}`| :math:`mg/(cell\cdot hr)`          | :math:`6.59 \times 10^{-10}`|
      +----------------------+------------------------------------+-----------------------------+
      | :math:`Y_{amm,gln}`  | :math:`mmol/mmol`                  | :math:`0.45`                |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`Y_{lac,glc}`  | :math:`mmol/mmol`                  | :math:`2.0`                 |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`Y_{X,glc}`    | :math:`cell/mmol`                  | :math:`2.6 \times 10^8`     |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`Y_{X,gln}`    | :math:`cell/mmol`                  | :math:`8.0 \times 10^8`     |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`\alpha_1`     | :math:`(mM \cdot L)/(cell \cdot h)`| :math:`3.4 \times 10^{-13}` |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`\alpha_2`     | :math:`mM`                         | 4.0                         |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`-\Delta H`    | :math:`J/mol`                      | :math:`5.0 \times 10^5`     |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`rho`          | :math:`g/L`                        | :math:`1560.0`              |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`c_p`          | :math:`J/(g ^\circ C)`             | :math:`1.244`               |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`U`            | :math:`J/(h ^\circ C)`             | :math:`4 \times 10^2`       |
      +----------------------+------------------------------------+-----------------------------+
      | :math:`T_{in}`       | :math:`^\circ C`                   | :math:`37.0`                |
      +----------------------+------------------------------------+-----------------------------+

.. _downstream-1:

Downstream
----------

The parameters of downstream model are obtained from the work of
Gomis-Fons et al :raw-latex:`\cite{gomis2020model}` and several
parameters are modified because the process is upscaled from lab scale
to industrial scale. They are summarized in
Table `6.2 <#tb:para_down>`__.

.. container::
   :name: tb:para_down

   .. table:: Parameters of digital twin of downstream
      :align: center

      +---------+-----------------------+---------------------+--------------------------------+
      | Step    | Parameter             | Unit                | Value                          |
      +=========+=======================+=====================+================================+
      | Capture | :math:`q_{max,1}`     | :math:`mg/mL`       | :math:`36.45`                  |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`k_{1}`         | :math:`mL/(mg~min)` | :math:`0.704`                  |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`q_{max,2}`     | :math:`mg/mL`       | :math:`77.85`                  |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`k_{2}`         | :math:`mL/(mg~min)` | :math:`2.1\cdot10^{-2}`        |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`K`             | :math:`mL/mg`       | :math:`15.3`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`D_{eff}`       | :math:`cm^{2}/min`  | :math:`7.6\cdot10^{-5}`        |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`D_{ax}`        | :math:`cm^{2}/min`  | :math:`5.5\cdot10^{-1}v`       |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`k_{f}`         | :math:`cm/min`      | :math:`6.7\cdot10^{-2}v^{0.58}`|
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`r_{p}`         | :math:`cm`          | :math:`4.25\cdot10^{-3}`       |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`L`             | :math:`cm`          | :math:`20`                     |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`V`             | :math:`mL`          | :math:`10^5`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`\epsilon_c`    | :math:`-`           | :math:`0.31`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`\epsilon_p`    | :math:`-`           | :math:`0.94`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`q_{max,elu}`   | :math:`mg/mL`       | :math:`114.3`                  |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`k_{elu}`       | :math:`min^{-1}`    | :math:`0.64`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`H_{0,elu}`     | :math:`M^{\beta}`   | :math:`2.2\cdot10^{-2}`        |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`\beta_{elu}`   | :math:`-`           | :math:`0.2`                    |
      +---------+-----------------------+---------------------+--------------------------------+
      | Loop    | :math:`D_{ax}`        | :math:`cm^{2}/min`  | :math:`2.9\cdot10^{2}v`        |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`L`             | :math:`cm`          | :math:`600`                    |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`V`             | :math:`mL`          | :math:`5\cdot10^5`             |
      +---------+-----------------------+---------------------+--------------------------------+
      | CEX     | :math:`q_{max}`       | :math:`mg/mL`       | :math:`150.2`                  |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`k`             | :math:`min^{-1}`    | :math:`0.99`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`H_{0}`         | :math:`M^{\beta}`   | :math:`6.9\cdot10^{-4}`        |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`\beta`         | :math:`-`           | :math:`8.5`                    |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`D_{app}`       | :math:`cm^{2}/min`  | :math:`1.1\cdot10^{-1}v`       |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`L`             | :math:`cm`          | :math:`10`                     |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`V`             | :math:`mL`          | :math:`5\cdot10^{4}`           |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`\epsilon_{c}`  | :math:`-`           | :math:`0.34`                   |
      +---------+-----------------------+---------------------+--------------------------------+
      | AEX     | :math:`D_{app}`       | :math:`cm^{2}/min`  | :math:`1.6\cdot10^{-1}v`       |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`k`             | :math:`min^{-1}`    | :math:`0`                      |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`L`             | :math:`cm`          | 10                             |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`V`             | :math:`mL`          | :math:`5\cdot10^{4}`           |
      +---------+-----------------------+---------------------+--------------------------------+
      |         | :math:`\epsilon_{c}`  | :math:`-`           | :math:`0.34`                   |
      +---------+-----------------------+---------------------+--------------------------------+

mAbEnv module
----------------

Following the discription above, we provide APIs as below:

.. automodule:: smpl.envs.mabenv
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
