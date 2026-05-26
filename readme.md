# Variational Diffusion Models

Pytorch implemenetation of Variational Diffusion Models  <!-- add citation here -->
This implementation tries to follow faithfully the original article implementation, and some derivatives.

The model trains on images, but it can work with any type of data.

The implementation includes:

- Likelihood estimation during the sampling process using Hutchinson trace estimator to speed up the trace computation
- Linear, Cosine and Multivariate Learned noising schedule from MuLAN <!-- Add citation here --> 
- Random Fourier Features in addition to the normal input.

## Files
  
    ├── denoiser.py             # denoiser module
    ├── learned_schedule.py     # learnable multivariate noise schedule from MuLAN
    ├── p_ode.py                # ODE sampler class with likelihood estimation
    ├── readme.md
    ├── requirements.txt
    └── vdm.py                  # Pytorch Lightning Variational Diffusion Model implementation
