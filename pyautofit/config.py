import json
import logging
import copy

class AutoFitConfig:
    """Constructs AutoFitConfig
    """
    def __init__(
            self,
            name='autofit_config',
            version='template',
            classifier_config=None,
    ):
        self.name = name
        self.version = version
        self.classifier_config = classifier_config

    @classmethod
    def from_dict(cfg, config_dict):
        """Create configuration object from python dictionary"""
        config = AutoFitConfig()
        for (key, value) in config_dict.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json(cfg, json_path):
        """Create configuration object from json file"""
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
                return cfg.from_dict(config)
        except Exception as e:
            logging.error(e)
            return False


    def to_dict(self):
        """Return a configuration dictionary"""
        return copy.deepcopy(self.__dict__)

    def to_json(self, json_path):
        """Save current configuration to json file"""
        try:
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f)
        except Exception as e:
            logging.error(e)
            return False


"""Test Script"""
"""
if __name__=='__main__':
    config = AutoFitConfig(name='t', version='t', classifier_config=dict())
    print(config.to_dict())
    config = AutoFitConfig.from_json('configs/autofit_config.json')
    print(config.to_dict())
"""