#!/bin/bash  
echo "$1"
git add . 
git commit -m "$1" 
git push https://alexjungaalto@github.com/alexjungaalto/FederatedLearning/ main
cd ~/flcoursematerial

