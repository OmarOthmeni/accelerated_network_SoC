# Vivado: Zynq PS + CNN accelerator overlay for PYNQ, then bitstream.
#   vivado -mode batch -source vivado/build_overlay.tcl
# Run hls/run_hls.tcl first (it exports the IP to hls/cnn_prj/sol1/impl/ip).
# Needs the PYNQ board files installed in Vivado. For PYNQ-Z1 use:
#   -tclargs www.digilentinc.com:pynq-z1:part0:1.0

set board_part [expr {$argc > 0 ? [lindex $argv 0] : "tul.com.tw:pynq-z2:part0:1.0"}]
set root [file normalize [file dirname [info script]]/..]
set out  $root/vivado/out

create_project -force cnn_overlay $out/prj -part xc7z020clg400-1
set_property board_part $board_part [current_project]
set_property ip_repo_paths $root/hls/cnn_prj/sol1/impl/ip [current_project]
update_ip_catalog

create_bd_design cnn_overlay
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1"} [get_bd_cells ps7]
# 100 MHz fabric clock + two high-performance ports for the accelerator's AXI masters
set_property -dict [list CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
                         CONFIG.PCW_USE_S_AXI_HP0 {1} CONFIG.PCW_USE_S_AXI_HP1 {1}] [get_bd_cells ps7]

create_bd_cell -type ip -vlnv omar:hls:cnn_top:1.0 cnn_top

# control registers on GP0, image reads on HP0, logit writes on HP1
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Master "/ps7/M_AXI_GP0" Clk "Auto"} [get_bd_intf_pins cnn_top/s_axi_control]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Slave "/ps7/S_AXI_HP0" Clk "Auto"} [get_bd_intf_pins cnn_top/m_axi_gmem0]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config {Slave "/ps7/S_AXI_HP1" Clk "Auto"} [get_bd_intf_pins cnn_top/m_axi_gmem1]

validate_bd_design
save_bd_design
make_wrapper -files [get_files $out/prj/cnn_overlay.srcs/sources_1/bd/cnn_overlay/cnn_overlay.bd] -top
add_files -norecurse $out/prj/cnn_overlay.gen/sources_1/bd/cnn_overlay/hdl/cnn_overlay_wrapper.v

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

# PYNQ needs the .bit and the .hwh with the same base name
file mkdir $root/pynq/overlay
file copy -force $out/prj/cnn_overlay.runs/impl_1/cnn_overlay_wrapper.bit $root/pynq/overlay/cnn.bit
file copy -force $out/prj/cnn_overlay.gen/sources_1/bd/cnn_overlay/hw_handoff/cnn_overlay.hwh $root/pynq/overlay/cnn.hwh

# utilization and timing reports for the README
open_run impl_1
report_utilization -file $root/vivado/reports/utilization.txt
report_timing_summary -file $root/vivado/reports/timing.txt
report_power -file $root/vivado/reports/power.txt
puts "Overlay written to pynq/overlay/ (cnn.bit + cnn.hwh)"
