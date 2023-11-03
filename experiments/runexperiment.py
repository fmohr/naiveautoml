if __name__ == '__main__':

    from py_experimenter.experimenter import PyExperimenter
    from experiment_execution import run_ex
    import os
    
    import logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # configure lccv logger (by default set to WARN, change it to DEBUG if tests fail)
    lccv_logger = logging.getLogger("lccv")
    lccv_logger.handlers.clear()
    lccv_logger.setLevel(logging.WARN)
    lccv_logger.addHandler(ch)
    elm_logger = logging.getLogger("elm")
    elm_logger.handlers.clear()
    elm_logger.setLevel(logging.WARN)
    elm_logger.addHandler(ch)

    experimenter = PyExperimenter(experiment_configuration_file_path=os.path.split(__file__)[0]+'/configuration.conf', name='example')
    experimenter.fill_table_from_config()
    experimenter.execute(run_ex)
    print(experimenter.get_table())