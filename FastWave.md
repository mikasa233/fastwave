# Introduction #
FastWave is an opensource code software used for seismic wave modeling in complex media with staggered-grid finite difference method on Graphics Processing Units. The speedup tested on a GTX295 GPU with an grid size of 1024x1024 exceeds 100x, which greatly reduces the time used to compute seismic wavefield in inversion and then accelerates the whole inversion process to at least 100 times faster.

# Algorithm #
  * **Equation**: first-order velocity-stress equation is used in modeling;

  * **Method**: high-order staggered-grid finite-difference is implemented on GPUs using shared memory strategy;

  * **Boundary**: split perfectly matched boundary is used for absorbing outgoing wave.


