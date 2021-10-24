# local_selfinhibition_network

Simulation of a standard neural field model consisting of interconnected excitatory and inhibitory firing rate units [Wilson1972, Ermentrout1998]. Each excitatory unit is connected to nearby (excitatory and inhibitory) units via a distance dependent Gaussian connectivity profile. Each inhibitory unit is connected via a Gaussian connectivity profile to nearby excitatory units and only locally coupled to itself.


Code is written in Python 3; in Python 3.8.  <br/>
Required libraries: numpy, scipy, matplotlib


### Organization of the  project

The project has the following structure:

    local_selfinhibition_network
      |- README.md
      |- tools
         |- __init__.py
         |- functions.py
         |- get_EI_np.py
         |- parameter_settings.py
         |- plot_functions.py
         |- Runge_Kutta_Fehlberg.py
         |- save_activity.py
         
      |- run_figures.py
         |- data/
            |- ...
            
            
### Project Data

- [ ] save results from simulation for easier reproduction of figures


### Licensing

Licensed under MIT license. 

### Getting cited

- [ ] submit manuscript to biorxiv


### Scripts

[Python Script]() reproduces figures from manuscript.
