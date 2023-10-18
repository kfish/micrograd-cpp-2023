set logscale y
set xlabel "Iterations"
set ylabel "Loss"
set terminal svg
set output "loss.svg"
set object 1 rect from screen 0,0 to screen 1,1 behind fillcolor rgb "white" fillstyle solid 1.0
plot "loss.tsv" using 1:2 with lines title "Loss vs Iteration"
