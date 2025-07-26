import torch
import numpy as np
from collections import deque
from util import *


def get_HR(sig, fs=30):
    
    sig = sig.detach().cpu().numpy()
    # sig = butter_bandpass(sig, lowcut=0.6, highcut=4, fs=fs)
    
    HR, _ = estimate_average_heartrate(sig, fs)
    return HR



class RangeBasedMemoryBank:
    def __init__(self, range_width=5, max_length=3):
        self.range_width = range_width
        self.max_length = max_length
        self.banks = {}
        self.item_to_index = {"x": 0, "psd": 1, "ppg": 2, "HR": 3}
        

    def _get_range_key(self, key):
        return (key // self.range_width) * self.range_width


    def add(self, key, value):
        assert len(value) == len(self.item_to_index.keys()), f"{len(value)=} != {len(self.item_to_index.keys())=}"
        range_key = self._get_range_key(key)
        if range_key not in self.banks:
            self.banks[range_key] = deque(maxlen=self.max_length)
        self.banks[range_key].append(value)
        
        
    def add_by_sig(self, x, sig):
        HR = get_HR(sig)
        self.add(HR, (x, None, sig, HR))
        

    def get(self, key):
        range_key = self._get_range_key(key)
        return list(self.banks.get(range_key, []))
    
    
    def get_all(self):
        all_values = []
        for deque in self.banks.values():
            all_values.extend(deque)
        return all_values
    
    
    def get_all_item(self, item="x"):
        
        assert item in self.item_to_index.keys(), f"{item=} not in {self.item_to_index.keys()}"
        index = self.item_to_index[item]
        
        all_x = []
        for deque in self.banks.values():
            all_x.extend([item[index] for item in deque])
        return all_x
    
    def get_all_item_and_HR(self, item="x"):
        
        assert item in self.item_to_index.keys(), f"{item=} not in {self.item_to_index.keys()}"
        index = self.item_to_index[item]
        
        all_HR = []
        all_x = []
        for key, deque in self.banks.items():
            all_HR.extend([item[3] for item in deque])
            all_x.extend([item[index] for item in deque])
            
        return all_HR, all_x

    
    
    def get_all_item_by_HR(self, HR, item="x"):
        value = self.get(HR)
        
        assert item in self.item_to_index.keys(), f"{item=} not in {self.item_to_index.keys()}"
        index = self.item_to_index[item]
        
        all_x = [item[index] for item in value]
        
        return all_x
    
    
    def get_all_item_by_sig(self, sig, item="x"):
        HR = get_HR(sig)
        return self.get_all_item_by_HR(HR, item)
    
    
    def print_elem_nums(self):
        for k, v in self.banks.items():
            print(f"{k=}, {len(v)=}")
    
    
if __name__ == '__main__':
    
    # Example usage
    memory_bank = RangeBasedMemoryBank(range_width=200, max_length=3)

    # Adding some values
    memory_bank.add(70, ('x1', 'y1', 'z1', 70))
    memory_bank.add(90, ('x2', 'y2', 'z2', 90))
    memory_bank.add(80, ('x3', 'y3', 'z3', 80))
    memory_bank.add(100, ('x4', 'y4', 'z4', 100))

    # Retrieving all values
    print(memory_bank.get_all_item())
    print(memory_bank.get_all_item_by_HR(70))