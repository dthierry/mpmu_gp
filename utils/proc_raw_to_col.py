#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

def main():
    file = sys.argv.pop()

    d = pd.read_csv(file)
    x0 = d.iloc[:, 1].to_numpy()
    x1 = x0.reshape((24, -1))
    dn = pd.DataFrame(x1)
    dn.to_csv("reshaped.csv")

if __name__ == "__main__":
    main()

