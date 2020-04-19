for i in `seq 1 3`;
do
    ./neural.py -k 0 -u 60,30 -l 2.0 -r 0.1 -n 1000 -a 0.002 -q 12 -o 1
    ./neural.py -k 0 -u 50,20 -l 0.5 -r 0.1 -n 1000 -a 0.01 -q 12 -o 2
    ./neural.py -k 0 -u 50,20 -l 0.01 -r 0.04 -n 1000 -a 0.01 -q 12 -o 3
done