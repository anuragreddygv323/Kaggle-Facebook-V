
# pass in NEIGH environment variable

for x in `seq 1 8`;
do
    python -u grid_knn_demo.py --bayes 2>&1 > knn_params/`printf %02d ${NEIGH}`_`uuid -v4`.log &
    sleep 10
done;
