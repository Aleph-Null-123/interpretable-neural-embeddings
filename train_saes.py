import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
from sklearn.model_selection import train_test_split
import os
import tempfile
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

