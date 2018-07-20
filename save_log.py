# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:56:50 2018

@author: Administrator
"""

import sys  
  
class Logger(object):  
    def __init__(self, fileN="Default.log"):  
        self.terminal = sys.stdout  
        self.log = open(fileN, "a")  
  
    def write(self, message):  
        self.terminal.write(message)  
        self.log.write(message)  
  
    def flush(self):  
        pass  
  
