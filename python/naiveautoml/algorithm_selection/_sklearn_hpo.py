import json
from ConfigSpace.read_and_write import json as config_json
from ConfigSpace import ConfigurationSpace


class HPOHelper:

    def __init__(self, search_space):

        self.config_spaces = {}

        self.step_names = []
        for step in search_space:
            step_name = step["name"]
            self.step_names.append(step_name)
            self.config_spaces[step_name] = {}

            for comp in step["components"]:
                config_space_as_string = json.dumps(comp["params"])
                self.config_spaces[step_name][comp["class"]] = config_json.read(config_space_as_string)

    def get_config_space_for_selected_algorithms(self, selected_algorithms):
        """

        :param selected_algorithms: dictionary where keys are slot names and vals are algorithm names
        :return:
        """
        cs = ConfigurationSpace()
        for step, selection in selected_algorithms.items():
            cs.add_configuration_space(
                prefix=step,
                configuration_space=self.config_spaces[step][selection]
            )
        return cs

    def get_hps_by_step(self, hp_config):

        entry = {}
        for step_name in self.step_names:
            relevant_keys = [k for k in hp_config.keys() if k.startswith(f"{step_name}:")]
            if relevant_keys:
                entry[step_name] = {k[len(step_name) + 1:]: hp_config[k] for k in relevant_keys}
            else:
                entry[step_name] = None
        return entry
