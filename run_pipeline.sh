#!/bin/bash

cd HI_Data_Processing
echo running HI data processing
uv run main.py

cd ../STRUDL
echo extracting STRUDL fronts
uv run extract_fronts_operational.py

cd ../STRUDL_Tracking_Analysis
echo running pre-fitting pipeline
uv run run_pre_fitting_pipeline.py

cd ../ELEvo
echo running ELEvo
uv run ELEvo.py

cd ../STRUDL_Tracking_Analysis
echo fixing STRUDL output
uv run fix_strudl_output.py

cd ../ELEvoHI
echo running ELEvoHI
uv run run_elevohi.py

cd ..