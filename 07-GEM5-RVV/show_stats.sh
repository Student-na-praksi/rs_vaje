FILE=./vector_bench/stats.txt

echo $FILE

metrics_cpu=("numCycles" "l1dcaches.overallMisses::total")
metrics_vec=("numVecAluAccesses"  "numVecRegReads"  "numVecRegWrites" "numVecInsts" "vecLookups" )

echo "CPU stats"
for metric in "${metrics_cpu[@]}"; do
    echo "-------------------------------"
    grep -ri "$metric" $FILE
done

echo " "
echo "Vector stats"
for metric in "${metrics_vec[@]}"; do
    echo "-------------------------------"
    grep -ri "$metric" $FILE
done
echo " "


