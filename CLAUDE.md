                                                                                                                    
  ## Running Python scripts                                                                                         

  **Never** run scripts with bare `python` or `python3`. Always use:

  micromamba run -n campeones python -m src.campeones_analysis.<module.path>

  For example, to run `src/campeones_analysis/decoding/run_decoding.py`:

  micromamba run -n campeones python -m src.campeones_analysis.decoding.run_decoding

  This applies to every script execution in this project, including one-off runs, tests, and pipeline steps.        

  Una vez guardado, Claude lo cargará automáticamente en esta y todas las sesiones futuras.
