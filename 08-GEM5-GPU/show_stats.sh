FILE=./CU_stats_opt/stats.txt

echo $FILE

metrics_cpu=( "waveLevelParallelism::samples"  "vALUInsts" "vpc" "globalReads" "globalWrites" "ldsBankAccesses" "coalsrLineAddresses::total")

echo "GPU stats"
for metric in "${metrics_cpu[@]}"; do
    echo "-------------------------------"
    grep -ri "$metric" $FILE
done




