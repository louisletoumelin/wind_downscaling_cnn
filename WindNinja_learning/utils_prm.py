def select_cfg_file(prm):
    if prm["solver"] == "momentum":
        prm["cfg_file"] = prm["working_directory"] + "Models/WindNinja/" + "cli_momentumSolver_diurnal.cfg"
    elif prm["solver"] == "mass":
        prm["cfg_file"] = prm["working_directory"] + "Models/WindNinja/" + "mass_conserving.cfg"
    return prm
