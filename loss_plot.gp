set logscale y
set xlabel "Iterations"
set ylabel "Loss"
set terminal svg
set output "loss.svg"
plot "loss.tsv" using 1:2 with lines title "Loss vs Iteration"
