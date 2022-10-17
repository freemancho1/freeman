import os


class LogLevelManager:
    # state:
    #  -1: None(= 0)
    #   0: All
    #   1: Info
    #   2: Warining
    #   3: Error
    
    before_state = "-1"
    current_state = "-1"
    log_levels = [0, 1, 2, 3]
    env_key = "TF_CPP_MIN_LOG_LEVEL"
    
    @classmethod
    def get(cls):
        return None if cls.current_state == "-1" else cls.current_state
            
    @classmethod
    def set(cls, log_level):
        cls.before_state = os.getenv(cls.env_key, "-1")
        if log_level in cls.log_levels:
            cls.current_state = str(log_level)
            os.environ[cls.env_key] = cls.current_state
        else:
            cls.current_state = "-1"
            
    @classmethod
    def reset(cls):
        if cls.before_state == "-1":
            try: 
                os.environ.pop(cls.env_key)
            except:
                pass
        else:
            os.environ[cls.env_key] = cls.before_state
        cls.current_state = cls.before_state

