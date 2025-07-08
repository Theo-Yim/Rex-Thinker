#!/bin/bash

export PYTHONPATH="/workspace/Rex-Thinker/GroundingDINO:$PYTHONPATH"

echo 'export PYTHONPATH="/workspace/Rex-Thinker:/workspace/Rex-Thinker/GroundingDINO:$PYTHONPATH"' >> ~/.bashrc

pip install -r requirements_jh.txt
pip install --no-deps supervision contourpy cycler defusedxml fonttools kiwisolver matplotlib pyparsing addict yapf pycocotools