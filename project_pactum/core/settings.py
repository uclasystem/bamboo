import importlib
import logging
import os

import project_pactum

logger = logging.getLogger(__name__)

class Settings:

	def __init__(self):
		settings_path = os.path.join(project_pactum.BASE_DIR, 'settings.py')
		spec = importlib.util.spec_from_file_location('settings', settings_path)
		instance_settings = importlib.util.module_from_spec(spec)
		try:
			spec.loader.exec_module(instance_settings)
		except:
			logger.warn('settings.py is missing')
			return
		for setting in dir(instance_settings):
			if not setting.isupper():
				continue
			value = getattr(instance_settings, setting)
			setattr(self, setting, value)
