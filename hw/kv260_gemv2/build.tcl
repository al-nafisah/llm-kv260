# Vivado 2022.2 block design for gemv2 (hls/gemv2/) on the KV260.
#
#   cd hw/kv260_gemv2 && vivado -mode batch -source build.tcl
#
#   gemv_fetch  m_axi_gmem0..3 -> S_AXI_HP0..HP3_FPD, 128-bit, one each
#               s0a..s3b       -> eight AXI4-Stream Data FIFOs, 16384 x 128 bit
#                                 in UltraRAM (8 URAM each, all 64 of the device)
#   gemv_mac    s0a..s3b       <- the FIFOs
#               m_axi_gmemx, gmemy0a..y3b -> S_AXI_HPC0_FPD through one 9:1 SmartConnect
#   control     M_AXI_HPM0_FPD: fetch at 0xA000_0000, mac at 0xA001_0000
#   pl_clk0     GEMV_MHZ (default 250); pl.dtsi must agree.
set here [file dirname [file normalize [info script]]]
set mhz 250
if {[info exists ::env(GEMV_MHZ)]} { set mhz $::env(GEMV_MHZ) }
set variant ""
if {[info exists ::env(GEMV_VARIANT)]} { set variant "_$::env(GEMV_VARIANT)" }
puts "build.tcl: gemv2, pl_clk0 = $mhz MHz, variant = '$variant'"
set ip_dir $here/ip_repo$variant
file mkdir $ip_dir
foreach ip {gemv_fetch gemv_mac} {
    if {![file exists $ip_dir/$ip/component.xml]} {
        exec unzip -q -o $here/../../hls/gemv2/${ip}_ip.zip -d $ip_dir/$ip
    }
}
set prj $here/kv260_gemv2$variant

create_project -force kv260_gemv2 $prj -part xck26-sfvc784-2LV-c
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]
set_property board_connections {som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:1.3} [current_project]
set_property ip_repo_paths $ip_dir [current_project]
update_ip_catalog

create_bd_design "design_1"

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $mhz \
    CONFIG.PSU__USE__IRQ0 {1} \
] $ps

set fetch [create_bd_cell -type ip -vlnv xilinx.com:hls:gemv_fetch:1.0 gemv_fetch_0]
set mac   [create_bd_cell -type ip -vlnv xilinx.com:hls:gemv_mac:1.0 gemv_mac_0]

# Control: PS master -> both kernels' AXI-Lite slaves (the automation grows
# the one SmartConnect it creates on HPM0 for the second slave).
foreach cell {gemv_fetch_0 gemv_mac_0} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
        Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave "/$cell/s_axi_control" \
        ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
        [get_bd_intf_pins $cell/s_axi_control]
}

# Weights: one fetch master per HP port, no sharing.
foreach {m s} {gmem0 S_AXI_HP0_FPD gmem1 S_AXI_HP1_FPD gmem2 S_AXI_HP2_FPD gmem3 S_AXI_HP3_FPD} {
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
        Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master "/gemv_fetch_0/m_axi_$m" Slave "/zynq_ultra_ps_e_0/$s" \
        ddr_seg {Auto} intc_ip {Auto} master_apm {0}] \
        [get_bd_intf_pins zynq_ultra_ps_e_0/$s]
}

set rst [get_bd_cells -filter {VLNV =~ "xilinx.com:ip:proc_sys_reset:*"}]
set clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
set rstn [get_bd_pins $rst/peripheral_aresetn]

# Activations in and results out: nine light masters, one 9:1 SmartConnect, HPC0.
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc_hpc0]
set_property -dict [list CONFIG.NUM_SI {9} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {1}] $smc
set i 0
foreach m {gmemx gmemy0a gmemy0b gmemy1a gmemy1b gmemy2a gmemy2b gmemy3a gmemy3b} {
    connect_bd_intf_net [get_bd_intf_pins gemv_mac_0/m_axi_$m] [get_bd_intf_pins axi_smc_hpc0/S0${i}_AXI]
    incr i
}
connect_bd_intf_net [get_bd_intf_pins axi_smc_hpc0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
connect_bd_net [get_bd_pins axi_smc_hpc0/aclk] $clk
connect_bd_net [get_bd_pins axi_smc_hpc0/aresetn] $rstn
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk] $clk

# The eight prefetch FIFOs, fetch -> mac, in UltraRAM.
foreach s {s0a s0b s1a s1b s2a s2b s3a s3b} {
    set f [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 fifo_$s]
    set_property -dict [list CONFIG.FIFO_DEPTH {16384} CONFIG.FIFO_MEMORY_TYPE {ultra} \
        CONFIG.TDATA_NUM_BYTES {16} CONFIG.HAS_TKEEP {0} CONFIG.HAS_TLAST {0} CONFIG.HAS_TSTRB {0}] $f
    connect_bd_intf_net [get_bd_intf_pins gemv_fetch_0/$s] [get_bd_intf_pins fifo_$s/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins fifo_$s/M_AXIS] [get_bd_intf_pins gemv_mac_0/$s]
    connect_bd_net [get_bd_pins fifo_$s/s_axis_aclk] $clk
    connect_bd_net [get_bd_pins fifo_$s/s_axis_aresetn] $rstn
}

# Interrupts: fetch on pl_ps_irq0[0], mac on [1].
set cat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 irq_concat]
set_property CONFIG.NUM_PORTS {2} $cat
connect_bd_net [get_bd_pins gemv_fetch_0/interrupt] [get_bd_pins irq_concat/In0]
connect_bd_net [get_bd_pins gemv_mac_0/interrupt] [get_bd_pins irq_concat/In1]
connect_bd_net [get_bd_pins irq_concat/dout] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]

assign_bd_address
set_property offset 0xA0000000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_gemv_fetch_0_Reg}]
set_property range 64K [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_gemv_fetch_0_Reg}]
set_property offset 0xA0010000 [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_gemv_mac_0_Reg}]
set_property range 64K [get_bd_addr_segs {zynq_ultra_ps_e_0/Data/SEG_gemv_mac_0_Reg}]

validate_bd_design
save_bd_design

make_wrapper -files [get_files design_1.bd] -top
add_files -norecurse $prj/kv260_gemv2.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 3
wait_on_run synth_1
foreach r [get_runs -filter {IS_SYNTHESIS && NAME != "synth_1"}] {
    wait_on_run $r
    if {[get_property PROGRESS $r] != "100%"} { error "OOC run $r did not finish" }
}
launch_runs impl_1 -to_step write_bitstream -jobs 3
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "impl_1 did not finish" }

set out $here/out$variant
file mkdir $out
file copy -force $prj/kv260_gemv2.runs/impl_1/design_1_wrapper.bit $out/gemv2.bit
file copy -force $prj/kv260_gemv2.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh $out/gemv2.hwh
open_run impl_1
report_timing_summary -file $out/timing_summary.rpt
report_utilization -file $out/utilization.rpt
puts "BITSTREAM: $out/gemv2.bit"
