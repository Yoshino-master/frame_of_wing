'''
Created on 2019年2月2日

@author: jinglingzhiyu
'''
import os
import pandas as pd




class decoder_module():
    def __init__(self,generator):
        self.generator = generator
        self.protocol  = None
    def __call__(self,csv_path,protocol=None):
        if protocol is not None:
            self.protocol = protocol
        return self.generator(csv_path,self.protocol)
    def set_protocol(self,protocol):
        self.protocol = protocol
    def set_generator(self,generator):
        self.generator = generator

class simple_decoder():
    def __init__(self, generator):
        self.generator = generator
        




