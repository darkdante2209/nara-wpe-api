# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 15:15:18 2020

@author: Admin
"""
import pathlib
import os
from AudioProcessing import AudioProcessing
project_root = pathlib.Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
))

filesFromDir = os.listdir(str(project_root / 'my_data'/'clean_data' ))  
for file_name in filesFromDir:
    sound1 = AudioProcessing(str(project_root / 'my_data' / 'clean_data' / file_name))
    sound1.set_reverb(0.2,0.5)
    output_name='reverb_'+file_name
    sound1.save_to_file(str(project_root / 'my_data' / '0.2 - 0.5' / 'reverb_data' / output_name))

