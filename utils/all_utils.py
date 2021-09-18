import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.colors import ListedColormap
import logging

plt.style.use("fivethirtyeight")

def prepare_data(df):
  """it is used to seperate the dependent and independent features

  Args:
     df (pd:DataFrame) : it is pandas dataFrame

  Returns:
      tuple: it returns the tuple of dependent and independent variables
  """
  logging.info("Preparing the data of dependeing and independent model")
  X = df.drop("y", axis=1)
 
  y = df["y"]

  return X,y

def save_model(model, filename):
  """this will save the model created

  Args:
      model (python object): trained model to 
      filename (str): path to save modelp
  """
  logging.info("saving the model")
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True)
  filePath = os.path.join(model_dir,filename)
  joblib.dump(model, filePath)
  logging.info(f"saved the model at : {filePath}")

def save_plot(df, filename, model):
  def _create_base_plot(df):
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf()
    figure.set_size_inches(10,8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02):
    colors = ("red","blue","lightgreen","grey","cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1, x1.max()+1
    x2_min, x2_max = x2.min() -1, x2.max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    logging.info(xx1)
    logging.info(xx1.ravel())
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()
    

  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) 
  plotPath = os.path.join(plot_dir, filename) 
  plt.savefig(plotPath)
  logging.info(f"saving the plot {plotPath}")