#!/bin/bash
longjob -c  'nice -n python main_experiments_trainvol.py --data cora --edge-norm dsm --epochs 2'
longjob -c  'nice -n python main_experiments_trainvol.py --data citeseer --edge-norm dsm --epochs 2'
longjob -c  'nice -n python main_experiments_trainvol.py --data pubmed --edge-norm dsm --epochs 2'
longjob -c  'nice -n python main_og.py --data cora --edge-norm dsm --epochs 2 --epochs 2'
longjob -c  'nice -n python main_og.py --data citeseer --edge-norm dsm --epochs 2 --epochs 2'
longjob -c  'nice -n python main_og.py --data pubmed --edge-norm dsm --epochs 2 --epochs 2'