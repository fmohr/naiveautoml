if __name__ == '__main__':

    from py_experimenter.experimenter import PyExperimenter
    from experiment_execution import run_ex
    import os

    experimenter = PyExperimenter(experiment_configuration_file_path=os.path.split(__file__)[0]+'/configuration.conf', name='example')
    experimenter.fill_table_from_config()
    experimenter.execute(run_ex)
    print(experimenter.get_table())