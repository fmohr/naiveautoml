if __name__ == '__main__':

    from py_experimenter.experimenter import PyExperimenter
    from experiment_execution import run_ex

    experimenter = PyExperimenter(experiment_configuration_file_path='/home/lawrence/Documents/seminario/naiveautoml/experiments/configuration.conf', name='example')
    experimenter.fill_table_from_config()
    experimenter.execute(run_ex)
    print(experimenter.get_table())