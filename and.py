"""author : kkc
    email : kkchaudhary11@gmail.com
"""
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s %(message)s]"
logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"running_logs.log"),level=logging.INFO, format=logging_str)

def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(AND)

    logging.info(f"this is the actual data frame {df}")

    X,y = prepare_data(df)
    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename="and.model")
    save_plot(df, "and.png", model)

if __name__=='__main__':
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }

ETA = 0.3 # 0 and 1
EPOCHS = 10
try:
    logging.info(">>>>> Staring traing here")
    main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
    logging.info("<<<<< Training done successfully")
except Exception as e:
    logging.exception(e)    