# import argparse
import glob
import logging
import sys
import os
import random
import shutil



def separate_train_val(dirPath, testDirPath):
    files = os.listdir(dirPath)
    for index, f in enumerate(files):
        if index == 1000:
            break
        print(shutil.move('./' + dirPath + '/' +  f,testDirPath))

if __name__ == '__main__':
    separate_train_val(sys.argv[1],sys.argv[2])
