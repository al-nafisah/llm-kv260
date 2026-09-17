# GEMV_HLS_MODE=csynth|export vitis_hls -f run_hls.tcl
# Synthesises and exports both gemv2 kernels: gemv_fetch_ip.zip, gemv_mac_ip.zip.
# The C testbench does not run here (csim cannot link on this PC); build it
# with the system g++ as CLAUDE.md shows. GEMV_CLK_NS defaults to 4.
set mode "export"
if {[info exists ::env(GEMV_HLS_MODE)]} { set mode $::env(GEMV_HLS_MODE) }
set clk_ns 4
if {[info exists ::env(GEMV_CLK_NS)]} { set clk_ns $::env(GEMV_CLK_NS) }
set here [file dirname [file normalize [info script]]]
foreach top {gemv_fetch gemv_mac} {
    puts "run_hls.tcl: $top at $clk_ns ns"
    open_project -reset ${top}_prj
    set_top $top
    add_files $here/$top.cpp -cflags "-I$here -I$here/../int4_core"
    add_files -tb $here/tb_gemv2.cpp -cflags "-I$here -I$here/../int4_core"
    open_solution -reset "kv260" -flow_target vivado
    set_part xck26-sfvc784-2LV-c
    create_clock -period $clk_ns -name default
    set_clock_uncertainty 0.4
    config_interface -m_axi_addr64=1 -m_axi_latency=64
    config_compile -pipeline_loops 0
    csynth_design
    if {$mode == "export"} { export_design -format ip_catalog -output $here/${top}_ip.zip }
    close_project
}
exit
