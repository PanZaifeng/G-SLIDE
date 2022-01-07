#!/usr/bin/env bash
make &> /dev/null

res=$(./runme ./amazon.json)
#echo -e "$res"
#echo

echo "Epoch-wise results summary:"
echo -e "$res" | grep -E 'Epoch|elaps|corr'
echo

echo "Average time for each part in a batch computation"
iter_num=$(echo -e "$res" | grep Iteration | wc -l)
part_list=(FW0 FW1 BP1 BP0 UD0 UD1 LSH_QY LSH_RC)
part_bar_dat=''
for part in FW0 FW1 BP1 BP0 UD0 UD1 LSH_QY LSH_RC
do
  avg=$(echo -e "$res" | awk '$0 ~ "'$part'" {sum+=$2} END {print sum/"'$iter_num'"}')
  part_bar_dat+="${part/_/\\\\\\\\\\_} $avg\n"
  echo "[$part] ${avg}ms"
done

outdir=results
mkdir -p $outdir

part_bar_fig=$outdir/part_bar.png
part_bar_plt_cmd=$(echo $(cat << EOFMarker
  set term png;
  set out '$part_bar_fig';
  set title 'Time for each part';
  set ylabel 'time [ms]';
  set yrange [0:*];
  set boxwidth 0.5;
  set style fill solid;
  unset key;
  plot '<cat' using 0:2:xtic(1) with boxes;
  set out;
EOFMarker
))

echo -e "$part_bar_dat" | gnuplot -p -e "$part_bar_plt_cmd"

time_dat="$(echo -e "$res" | grep 'elaps' | awk '{print $4 / 1000}')"
acc_dat="$(echo -e "$res" | grep 'corr' | awk '{print $7 * 100}')"

time_list=($(echo $time_dat))
acc_list=($(echo $acc_dat))
time_acc_dat=$(for i in $(seq 0 $(expr ${#time_list[*]} - 1)); do
  echo ${time_list[$i]} ${acc_list[$i]}
done)

time_acc_fig=$outdir/time_acc.png
time_acc_plt_cmd=$(echo $(cat << EOFMarker
  set term png;                                      
  set out '$time_acc_fig';
  set title 'Time-wise accuracy of G-SLIDE';
  set xlabel 'time [s]';
  set ylabel 'accuracy [%]';
  plot '<cat' title 'G-SLIDE' with linespoints;
  set out;
EOFMarker
))

echo -e "$time_acc_dat" | gnuplot -p -e "$time_acc_plt_cmd"

time_epoch_fig=$outdir/time_epoch.png
time_epoch_plt_cmd=$(echo $(cat << EOFMarker
  set term png;
  set out '$time_epoch_fig';
  set logscale y 10;
  set title 'Time consumed over epochs of G-SLIDE';
  set xlabel 'epoch';
  set ylabel 'time [s]';
  plot '<cat' using (\$0+1):1 title 'G-SLIDE' with linespoints;
  set out;
EOFMarker
))

echo -e "$time_dat" | gnuplot -p -e "$time_epoch_plt_cmd"

acc_epoch_fig=$outdir/acc_epoch.png
acc_epoch_plt_cmd=$(echo $(cat << EOFMarker
  set term png;
  set out '$acc_epoch_fig';
  set title 'Accuracy over epochs of G-SLIDE';
  set xlabel 'epoch';
  set ylabel 'accuracy [%]';
  plot '<cat' using (\$0+1):1 title 'G-SLIDE' with linespoints;
  set out;
EOFMarker
))

echo -e "$acc_dat" | gnuplot -p -e "$acc_epoch_plt_cmd"

