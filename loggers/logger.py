import os
import logging
import datetime
from default_configs.default_config import DefaultConfig

class LOG:
    __instance = None

    def __init__(self):
        logging_level = logging.DEBUG if DefaultConfig.get_config('logging_debug') else logging.INFO
        self.log = None
        self.setup_logger(level=logging_level)
        LOG.__instance = self


    def setup_logger(self, name=__file__, level=logging.INFO):
        logger = logging.getLogger(name)
        if getattr(logger, '_init_done__', None):
            logger.setLevel(level)
            self.log = logger
            return

        logger._init_done__ = True
        logger.propagate = False
        logger.setLevel(level)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s:%(levelname)s::[%(module)s:%(lineno)d]::%(message)s")
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)

        now = datetime.datetime.now()
        date_format = '%Y_%m_%d_%H_%M_%S'
        format_date = now.strftime(date_format)

        save_dir = './loggers'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_handler = logging.FileHandler(f'{save_dir}/log-{format_date}.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        del logger.handlers[:]
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        self.log = logger
        return

    def log_info(self, msg):
        self.log.info(msg)

    def log_debug(self, msg):
        self.log.debug(msg)

    def get_logger(self):
        return self.log

    @staticmethod
    def get_instance():
        if LOG.__instance is None:
            LOG()
            return LOG.__instance
        return LOG.__instance

    @staticmethod
    def info(msg):
        LOG.get_instance().log_info(msg)

    @staticmethod
    def debug(msg):
        LOG.get_instance().log_debug(msg)
