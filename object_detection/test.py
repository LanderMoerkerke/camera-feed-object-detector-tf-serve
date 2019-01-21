import configparser


_CONFIG_FILE = "../config.ini"

config = configparser.ConfigParser()
config.read(_CONFIG_FILE)

tst = config["Tensorflow"]["save_detection"]

print(bool(tst))
