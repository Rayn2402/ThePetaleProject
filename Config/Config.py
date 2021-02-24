"""
Authors : Mehdi Mitiche

File that contain constant data used by our models

"""

#The Supported Metrics to optimize
METRICS = ["MSE", "ACCURACY"]

#The Values of the hyper parameters to tune
HYPER_PARAMS = {
    "lr":{"min":0.0001,"max":0.1},
    "optimizer_name":{"values": ["Adam","RMSprop","SGD"]},
    "batch_size":{"min":8, "max":128, "step":8 },
    "n_layers":{"min":0,"max":1},
    "n_units":{"min":1,"max":100},
    "dropout":{"min":0,"max":1}, 
}