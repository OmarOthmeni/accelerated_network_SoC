# Vitis HLS: C simulation, synthesis, co-simulation (optional) and IP export.
#   cd hls && vitis_hls -f run_hls.tcl
# Target: PYNQ-Z1 / PYNQ-Z2 (xc7z020clg400-1) at 100 MHz.

open_project -reset cnn_prj
set_top cnn_top
add_files cnn.cpp -cflags "-I."
add_files -tb tb.cpp -cflags "-I. -DDATA_DIR=\\\"[pwd]/data\\\""

open_solution -reset sol1 -flow_target vivado
set_part xc7z020clg400-1
create_clock -period 10 -name default

csim_design
csynth_design
# cosim_design -trace_level none     ;# slow (RTL simulation of the full test set); uncomment to run
export_design -format ip_catalog -description "INT8 CNN accelerator (MSL rover images)" -vendor omar -version 1.0

exit
