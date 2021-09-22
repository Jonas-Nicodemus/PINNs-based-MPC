<!-- PROJECT SHIELDS -->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Physics-Informed Neural Networks-based Model Predictive Control for Multi-link Manipulators
Prospective contribution to the <a href="https://www.mathmod.at/">MATHMOD 2022 Vienna</a> conference.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <!--<li><a href="#citing">Citing</a></li>-->
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

We discuss nonlinear model predictive control (NMPC) for multi-body dynamics via physics-informed machine learning methods.
Physics-informed neural networks (PINNs) are a promising tool for approximating (partial) differential equations.
PINNs are not suited for control tasks in their original form since they are not designed to handle variable control actions or variable initial conditions.
Thus, we present the idea of enhancing PINNs by adding control actions and initial conditions as additional network inputs. This enables the controller design based on a PINN as an approximation of the underlying system dynamics.
Finally, we present our results using our PINN-based MPC to solve a tracking problem for a complex mechanical system,  a multi-link manipulator.

<!-- For more information, please refer to the following: doi -->

<!--### Citing
If you use this project for academic work, please consider citing our
[publication](doi):

    J. Nicodemus, J. Kneifl, J. Fehr, B. Unger
    Physics-Informed Neural Networks-based Model Predictive Control for Multi-link Manipulators

-->
### Built With

* [TensorFlow](https://www.tensorflow.org/)
* [Python](https://www.python.org/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

A python environment is required, we recommend using a virtual environment.

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:Jonas-Nicodemus/PINNs-based-MPC.git
   ```
2. Go into the directory
   ```sh
   cd PINNs-based-MPC
   ```
3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

There are two executable scripts located in `src`.
* `train_pinn.py` can be executed to learn weights and overwrite the already existing ones,
   which can be found under `resources/weights`.
* `main.py`, then evaluates the PINN first in self-loop prediction mode and subsequently in
  closed-loop mode connected to the real system (emulated by RK45) for the given reference trajectory.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Jonas Nicodemus - jonas.nicodemus@simtech.uni-stuttgart.de

Benjamin Unger - benjamin.unger@simtech.uni-stuttgart.de

Jonas Kneifl - jonas.kneifl@itm.uni-stuttgart.de

Jörg Fehr - joerg.fehr@itm.uni-stuttgart.de

Project Link: [https://github.com/Jonas-Nicodemus/PINNs-based-MPC](https://github.com/Jonas-Nicodemus/PINNs-based-MPC)

[license-shield]: https://img.shields.io/github/license/Jonas-Nicodemus/PINNs-based-MPC.svg?style=for-the-badge
[license-url]: https://github.com/Jonas-Nicodemus/PINNs-based-MPC/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jonas-nicodemus-a34931209/