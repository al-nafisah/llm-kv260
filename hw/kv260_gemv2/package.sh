#!/usr/bin/env bash
# Wrap gemv2's bitstream for xmutil: hw/kv260_gemv2/out/gemv2/ with
# gemv2.bit.bin, gemv2.dtbo (or pl.dtsi when there is no dtc), shell.json.
# GEMV_VARIANT=<tag> packages out_<tag>/ instead, so an experiment never
# overwrites the shipped engine. GEMV_MHZ sets pl_clk0 in the overlay.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
variant=${GEMV_VARIANT:+_$GEMV_VARIANT}
mhz=${GEMV_MHZ:-250}
outroot=$here/out$variant
out=$outroot/gemv2
mkdir -p "$out"
sed "s/assigned-clock-rates = <250000000>/assigned-clock-rates = <${mhz}000000>/" "$here/pl.dtsi" > "$outroot/pl.dtsi"
source /opt/Xilinx/Vivado/2022.2/settings64.sh >/dev/null
cat > "$outroot/gemv2.bif" <<BIF
all:
{
    [destination_device = pl] $outroot/gemv2.bit
}
BIF
(cd "$outroot" && bootgen -image gemv2.bif -arch zynqmp -process_bitstream bin -w)
mv "$outroot/gemv2.bit.bin" "$out/gemv2.bit.bin"
if command -v dtc >/dev/null; then dtc -@ -I dts -O dtb -o "$out/gemv2.dtbo" "$outroot/pl.dtsi"; else echo "no dtc here: compile pl.dtsi on the board (dtc -@ -I dts -O dtb -o gemv2.dtbo pl.dtsi)"; cp "$outroot/pl.dtsi" "$out/"; fi
printf '{\n    "shell_type" : "XRT_FLAT",\n    "num_slots": "1"\n}\n' > "$out/shell.json"
ls -l "$out"
