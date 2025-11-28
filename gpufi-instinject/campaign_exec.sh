#!/bin/bash

# ---------------------------------------------- START ONE-TIME PARAMETERS ----------------------------------------------
# needed by gpgpu-sim for real register usage on PTXPlus mode
export PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
CONFIG_FILE=./gpgpusim.config
TMP_DIR=./logs
CACHE_LOGS_DIR=./cache_logs
TMP_FILE=tmp.out
# persistent list of invalid parameter combinations to skip
INVALID_COMBOS_FILE=./invalid_param_combos.txt
RUNS=4272
COMPONENT_SET="0"
BATCH=$(( $(grep -c ^processor /proc/cpuinfo) - 1 )) # -1 core for computer not to hang
DELETE_LOGS=0 # if 1 then all logs will be deleted at the end of the script
INJECT_BIT_FLIP_COUNT=1

# Optional: specify PTX virtual register name(s) to inject (overrides index-based selection)
# Examples: %f1, %r36, %rd7, %p2; multiple names can be colon-delimited like "%f1:%r36".

# Default register name; overridden per-injection if register_used.txt exists
REGISTER_NAME="%r90"

# ---------------------------------------------- END ONE-TIME PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER GPGPU CARD PARAMETERS ----------------------------------------------
# L1 cache size per SIMT core (30 SIMT cores on RTX 2060, 30 clusters with 1 core each)
L1D_SIZE_BITS=524345  # nsets=1, line_size=128 bytes + 57 bits, assoc=512 (64 KB per core)
L1C_SIZE_BITS=524345 # nsets=128, line_size=64 bytes + 57 bits, assoc=8 (64 KB per core)
L1T_SIZE_BITS=1048633 # nsets=4, line_size=128 bytes + 57 bits, assoc=256 (128 KB per core)
# L2 cache total size from all sub partitions (RTX 2060: 12 memory controllers × 2 sub partitions = 24 total)
L2_SIZE_BITS=24576057 # (nsets=64, line_size=128 bytes + 57 bits, assoc=16) x 24 sub partitions = 3 MB total
# ---------------------------------------------- END PER GPGPU CARD PARAMETERS ------------------------------------------------

# ---------------------------------------------- START PER KERNEL/APPLICATION PARAMETERS (+profile=1) ----------------------------------------------
CUDA_UUT="./MatrixMultiplication wA 16 hA 16 wB 16 hB 16"
# total cycles for all kernels
CYCLES=3723
# Get the exact cycles, max registers and SIMT cores used for each kernel with profile=1 
# fix cycles.txt with kernel execution cycles
# (e.g. seq 1 10 >> cycles.txt, or multiple seq commands if a kernel has multiple executions)
# use the following command from profiling execution for easier creation of cycles.txt file
# e.g. grep "_Z12lud_diagonalPfii" cycles.in | awk  '{ system("seq " $12 " " $18 ">> cycles.txt")}'
CYCLES_FILE=./cycles.txt
# File with aggregated danger region intervals copied from simulator logs
CYCLE_REGION_FILE=./cycle_region.txt
# 1: choose a cycle from the register's danger region intervals; 0: avoid them
INJECT_WITHIN_DANGER_REGION=1
MAX_REGISTERS_USED=42
SHADER_USED="0"
SUCCESS_MSG='Fault Injection Test Success!'
FAILED_MSG='Fault Injection Test Failed!'
TIMEOUT_VAL=20s
DATATYPE_SIZE=32
# lmem and smem values are taken from gpgpu-sim ptx output per kernel
# e.g. GPGPU-Sim PTX: Kernel '_Z9vectorAddPKdS0_Pdi' : regs=8, lmem=0, smem=0, cmem=380
# if 0 put a random value > 0
LMEM_SIZE_BITS=1
SMEM_SIZE_BITS=32768
# ---------------------------------------------- END PER KERNEL/APPLICATION PARAMETERS (+profile=1) ------------------------------------------------

FAULT_INJECTION_OCCURRED="Fault injection"
CYCLES_MSG="gpu_tot_sim_cycle ="

masked=0
performance=0
SDC=0
crashes=0

# ---------------------------------------------- START PER INJECTION CAMPAIGN PARAMETERS (profile=0) ----------------------------------------------
# 0: perform injection campaign, 1: get cycles of each kernel, 2: get mean value of active threads, during all cycles in CYCLES_FILE, per SM,
# 3: single fault-free execution
profile=0

# 1: per warp bit flip, 0: per thread bit flip
per_warp=0
# in which kernels to inject the fault. e.g. 0: for all running kernels, 1: for kernel 1, 1:2 for kernel 1 & 2 
kernel_n=0
# in how many blocks (smems) to inject the bit flip
blocks=1

build_combo_key_from_vars() {
    # Build a canonical key string from currently selected variables
    # Keep ordering stable for matching with analysis_fault.py
    # Exclude reg_bits/local_bits/shared_bits/l1*_shader,l1*_bits/l2_bits from the filter key
    echo -n "comp=${components_to_flip};per_warp=${per_warp};kernel=${kernel_n};"
    echo -n "thread=${thread_rand};warp=${warp_rand};block=${block_rand};cycle=${total_cycle_rand};"
    echo -n "reg_name=${REGISTER_NAME};reg_rand_n=${register_rand_n}"
}

select_cycle_for_register() {
    local reg_name="$1"
    local fallback_cycle="$2"
    if [[ -z "${reg_name}" || -z "${fallback_cycle}" ]]; then
        echo "${fallback_cycle}"
        return
    fi
    if [[ ! -f "${CYCLE_REGION_FILE}" ]]; then
        echo "${fallback_cycle}"
        return
    fi
    local line
    line=$(grep -F "reg=${reg_name} " "${CYCLE_REGION_FILE}" | head -n 1)
    if [[ -z "${line}" ]]; then
        echo "${fallback_cycle}"
        return
    fi
    local intervals="${line#*cycles=}"
    if [[ -z "${intervals}" ]]; then
        echo "${fallback_cycle}"
        return
    fi

    if [[ "${INJECT_WITHIN_DANGER_REGION}" -eq 1 ]]; then
        local chosen_cycle
        chosen_cycle=$(python3 - "${intervals}" "${fallback_cycle}" <<'PY'
import random
import sys

intervals = sys.argv[1].strip()
fallback = sys.argv[2]

def parse_ranges(spec):
    ranges = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
        else:
            a = b = part
        try:
            start = int(a)
            end = int(b)
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges

ranges = parse_ranges(intervals)
total = sum((end - start + 1) for start, end in ranges if end >= start)
if total <= 0 or not ranges:
    print(fallback)
    sys.exit(0)

pick = random.randint(0, total - 1)
for start, end in ranges:
    span = end - start + 1
    if span <= 0:
        continue
    if pick < span:
        print(start + pick)
        break
    pick -= span
PY
)
        if [[ -n "${chosen_cycle}" ]]; then
            echo "${chosen_cycle}"
            return
        fi
    else
        local chosen_cycle
        chosen_cycle=$(python3 - "${intervals}" "${CYCLES_FILE}" "${fallback_cycle}" <<'PY'
import random
import sys

intervals = sys.argv[1].strip()
cycles_path = sys.argv[2]
fallback = sys.argv[3]

def parse_ranges(spec):
    ranges = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
        else:
            a = b = part
        try:
            start = int(a)
            end = int(b)
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges

def in_range(val, ranges):
    for start, end in ranges:
        if start <= val <= end:
            return True
    return False

ranges = parse_ranges(intervals)
try:
    with open(cycles_path, 'r') as fh:
        candidates = [int(line.strip()) for line in fh if line.strip()]
except OSError:
    candidates = []

safe = [c for c in candidates if not in_range(c, ranges)]

if safe:
    print(random.choice(safe))
else:
    print(fallback)
PY
)
        if [[ -n "${chosen_cycle}" ]]; then
            echo "${chosen_cycle}"
            return
        fi
    fi

    echo "${fallback_cycle}"
}

initialize_config() {
    # 0:RF, 1:local_mem, 2:shared_mem, 3:L1D_cache, 4:L1C_cache, 5:L1T_cache, 6:L2_cache (e.g. components_to_flip=0:1 for both RF and local_mem)
    # random component to flip from COMPONENT_SET
    while true; do
        components_to_flip=$(shuf -e ${COMPONENT_SET} -n 1)
        # random number for choosing a random thread after thread_rand % #threads operation in gpgpu-sim
        thread_rand=$(shuf -i 0-6000 -n 1)
        # random number for choosing a random warp after warp_rand % #warp operation in gpgpu-sim
        warp_rand=$(shuf -i 0-6000 -n 1)
        # random cycle for fault injection
        total_cycle_rand="$(shuf ${CYCLES_FILE} -n 1)"
        if [[ "$profile" -eq 3 ]]; then
            total_cycle_rand=-1
        fi
        # Randomize REGISTER_NAME per injection if register list is available
        if [[ -f "register_used.txt" && -s "register_used.txt" ]]; then
            REGISTER_NAME=$(shuf -n 1 register_used.txt | tr -d '\r')
        fi
        if [[ "$profile" -ne 3 && "${total_cycle_rand}" != "-1" ]]; then
            total_cycle_rand=$(select_cycle_for_register "${REGISTER_NAME}" "${total_cycle_rand}")
        fi
        # in which registers to inject the bit flip
        # register_rand_n="$(shuf -i 1-${MAX_REGISTERS_USED} -n 1)"; register_rand_n="${register_rand_n//$'\n'/:}"
        register_rand_n=1
        # example: if -i 1-32 -n 2 then the two commands below will create a value with 2 random numbers, between [1,32] like 3:21. Meaning it will flip 3 and 21 bits.
        reg_bitflip_rand_n=$(shuf -i 1-${DATATYPE_SIZE} -n ${INJECT_BIT_FLIP_COUNT} | paste -sd:)
        # same format like reg_bitflip_rand_n but for local memory bit flips
        local_mem_bitflip_rand_n=$(shuf -i 1-${LMEM_SIZE_BITS} -n ${INJECT_BIT_FLIP_COUNT} | paste -sd:)
        # random number for choosing a random block after block_rand % #smems operation in gpgpu-sim
        block_rand=$(shuf -i 0-6000 -n 1)
        # same format like reg_bitflip_rand_n but for shared memory bit flips
        shared_mem_bitflip_rand_n=$(shuf -i 1-${SMEM_SIZE_BITS} -n ${INJECT_BIT_FLIP_COUNT} | paste -sd:)
        # randomly select one or more shaders for L1 data cache fault injections 
        l1d_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"; l1d_shader_rand_n="${l1d_shader_rand_n//$'\n'/:}"
        # same format like reg_bitflip_rand_n but for L1 data cache bit flips
        l1d_cache_bitflip_rand_n=$(shuf -i 1-${L1D_SIZE_BITS} -n 1 | paste -sd:)
        # randomly select one or more shaders for L1 constant cache fault injections 
        l1c_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"; l1c_shader_rand_n="${l1c_shader_rand_n//$'\n'/:}"
        # same format like reg_bitflip_rand_n but for L1 constant cache bit flips
        l1c_cache_bitflip_rand_n=$(shuf -i 1-${L1C_SIZE_BITS} -n 1 | paste -sd:)
        # randomly select one or more shaders for L1 texture cache fault injections 
        l1t_shader_rand_n="$(shuf -e ${SHADER_USED} -n 1)"; l1t_shader_rand_n="${l1t_shader_rand_n//$'\n'/:}"
        # same format like reg_bitflip_rand_n but for L1 texture cache bit flips
        l1t_cache_bitflip_rand_n=$(shuf -i 1-${L1T_SIZE_BITS} -n 1 | paste -sd:)
        # same format like reg_bitflip_rand_n but for L2 cache bit flips
        l2_cache_bitflip_rand_n=$(shuf -i 1-${L2_SIZE_BITS} -n 1 | paste -sd:)

        # Build combo key and check against invalid list
        combo_key=$(build_combo_key_from_vars)
        if [[ -f "${INVALID_COMBOS_FILE}" ]] && grep -Fxq "${combo_key}" "${INVALID_COMBOS_FILE}"; then
            # invalid combo previously observed; re-sample
            continue
        fi
        break
    done
# ---------------------------------------------- END PER INJECTION CAMPAIGN PARAMETERS (profile=0) ------------------------------------------------

    sed -i -e "s/^-components_to_flip.*$/-components_to_flip ${components_to_flip}/" ${CONFIG_FILE}
    sed -i -e "s/^-profile.*$/-profile ${profile}/" ${CONFIG_FILE}
    sed -i -e "s/^-last_cycle.*$/-last_cycle ${CYCLES}/" ${CONFIG_FILE}
    sed -i -e "s/^-thread_rand.*$/-thread_rand ${thread_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-warp_rand.*$/-warp_rand ${warp_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-total_cycle_rand.*$/-total_cycle_rand ${total_cycle_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-register_rand_n.*$/-register_rand_n ${register_rand_n}/" ${CONFIG_FILE}
    # If a specific register name is provided, override index-based selection; otherwise reset to empty
    if [[ -n "${REGISTER_NAME}" ]]; then
        # Escape chars that sed replacement cares about
        RN_ESC=${REGISTER_NAME//&/\\&}
        sed -i -e "s|^-register_name[[:space:]].*$|-register_name ${RN_ESC}|" "${CONFIG_FILE}"
    else
        sed -i -e 's|^-register_name[[:space:]].*$|-register_name ""|' "${CONFIG_FILE}"
    fi
    sed -i -e "s/^-reg_bitflip_rand_n.*$/-reg_bitflip_rand_n ${reg_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-per_warp.*$/-per_warp ${per_warp}/" ${CONFIG_FILE}
    sed -i -e "s/^-kernel_n.*$/-kernel_n ${kernel_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-local_mem_bitflip_rand_n.*$/-local_mem_bitflip_rand_n ${local_mem_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-block_rand.*$/-block_rand ${block_rand}/" ${CONFIG_FILE}
    sed -i -e "s/^-block_n.*$/-block_n ${blocks}/" ${CONFIG_FILE}
    sed -i -e "s/^-shared_mem_bitflip_rand_n.*$/-shared_mem_bitflip_rand_n ${shared_mem_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-shader_rand_n.*$/-shader_rand_n ${shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l1d_shader_rand_n.*$/-l1d_shader_rand_n ${l1d_shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l1d_cache_bitflip_rand_n.*$/-l1d_cache_bitflip_rand_n ${l1d_cache_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l1c_shader_rand_n.*$/-l1c_shader_rand_n ${l1c_shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l1c_cache_bitflip_rand_n.*$/-l1c_cache_bitflip_rand_n ${l1c_cache_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l1t_shader_rand_n.*$/-l1t_shader_rand_n ${l1t_shader_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l1t_cache_bitflip_rand_n.*$/-l1t_cache_bitflip_rand_n ${l1t_cache_bitflip_rand_n}/" ${CONFIG_FILE}
    sed -i -e "s/^-l2_cache_bitflip_rand_n.*$/-l2_cache_bitflip_rand_n ${l2_cache_bitflip_rand_n}/" ${CONFIG_FILE}
}

gather_results() {
    for file in ${TMP_DIR}${1}/${TMP_FILE}*; do
        # Derive index for matching saved config
        idx=${file##*${TMP_FILE}}
        cfg_path="${TMP_DIR}${1}/${CONFIG_FILE}${idx}"
        if [[ -f "${cfg_path}" ]]; then
            # Extract parameters from saved config to emit a canonical key line
            # Helper to read value after a flag from the config
            get_val() { grep -E "^$1\b" "${cfg_path}" | awk '{print $2}' | tail -n1; }
            components_to_flip_cfg=$(get_val "-components_to_flip")
            thread_rand_cfg=$(get_val "-thread_rand")
            warp_rand_cfg=$(get_val "-warp_rand")
            total_cycle_rand_cfg=$(get_val "-total_cycle_rand")
            register_rand_n_cfg=$(get_val "-register_rand_n")
            register_name_cfg=$(get_val "-register_name")
            reg_bitflip_rand_n_cfg=$(get_val "-reg_bitflip_rand_n")
            local_mem_bitflip_rand_n_cfg=$(get_val "-local_mem_bitflip_rand_n")
            block_rand_cfg=$(get_val "-block_rand")
            shared_mem_bitflip_rand_n_cfg=$(get_val "-shared_mem_bitflip_rand_n")
            l1d_shader_rand_n_cfg=$(get_val "-l1d_shader_rand_n")
            l1d_cache_bitflip_rand_n_cfg=$(get_val "-l1d_cache_bitflip_rand_n")
            l1c_shader_rand_n_cfg=$(get_val "-l1c_shader_rand_n")
            l1c_cache_bitflip_rand_n_cfg=$(get_val "-l1c_cache_bitflip_rand_n")
            l1t_shader_rand_n_cfg=$(get_val "-l1t_shader_rand_n")
            l1t_cache_bitflip_rand_n_cfg=$(get_val "-l1t_cache_bitflip_rand_n")
            l2_cache_bitflip_rand_n_cfg=$(get_val "-l2_cache_bitflip_rand_n")
            kernel_n_cfg=$(get_val "-kernel_n")

            combo_line="comp=${components_to_flip_cfg};per_warp=${per_warp};kernel=${kernel_n_cfg};"
            combo_line+="thread=${thread_rand_cfg};warp=${warp_rand_cfg};block=${block_rand_cfg};cycle=${total_cycle_rand_cfg};"
            combo_line+="reg_name=${register_name_cfg};reg_rand_n=${register_rand_n_cfg};reg_bits=${reg_bitflip_rand_n_cfg};"
            combo_line+="local_bits=${local_mem_bitflip_rand_n_cfg};shared_bits=${shared_mem_bitflip_rand_n_cfg};"
            combo_line+="l1d_shader=${l1d_shader_rand_n_cfg};l1d_bits=${l1d_cache_bitflip_rand_n_cfg};"
            combo_line+="l1c_shader=${l1c_shader_rand_n_cfg};l1c_bits=${l1c_cache_bitflip_rand_n_cfg};"
            combo_line+="l1t_shader=${l1t_shader_rand_n_cfg};l1t_bits=${l1t_cache_bitflip_rand_n_cfg};"
            combo_line+="l2_bits=${l2_cache_bitflip_rand_n_cfg}"

            echo "[INJ_PARAMS] [Run ${1}] ${TMP_FILE}${idx} ${combo_line}"
        fi
        grep -a -iq "${SUCCESS_MSG}" "$file"; success_msg_grep=$(echo $?)
	grep -a -i "${CYCLES_MSG}" "$file" | tail -1 | grep -a -q "${CYCLES}"; cycles_grep=$(echo $?)
        grep -a -iq "${FAILED_MSG}" "$file"; failed_msg_grep=$(echo $?)
        if grep -a -qE "FI_WRITER|FI_READER" "$file"; then
            grep -a -hE "FI_WRITER|FI_READER" "$file" | while IFS= read -r line; do
            echo "[Run ${1}] Effects from ${file}: $line"
            done
        fi
        result=${success_msg_grep}${cycles_grep}${failed_msg_grep}
        
        # Get file name for display
        filename=$(basename "$file")
        
        case $result in
        "001")
            let RUNS--
            let masked++
            echo "[Run ${1}] ${filename}: Masked (no performance impact)" ;;
        "011")
            let RUNS--
            let masked++ 
            let performance++
            echo "[Run ${1}] ${filename}: Masked (with performance impact)" ;;
        "100" | "110")
            let RUNS--
            let SDC++
            echo "[Run ${1}] ${filename}: SDC" ;;
        *)
            grep -a -iq "${FAULT_INJECTION_OCCURRED}" "$file"
            if [ $? -eq 0 ]; then
                let RUNS--
                let crashes++
                echo "[Run ${1}] ${filename}: DUE (Crash)"
            else
                echo "[Run ${1}] ${filename}: Unclassified (${result})"
            fi ;;
        esac
    done
}

parallel_execution() {
    batch=$1
    mkdir ${TMP_DIR}${2} > /dev/null 2>&1
    for i in $( seq 1 $batch ); do
        initialize_config
        # unique id for each run (e.g. r1b2: 1st run, 2nd execution on batch)
        sed -i -e "s/^-run_uid.*$/-run_uid r${2}b${i}/" ${CONFIG_FILE}
        cp ${CONFIG_FILE} ${TMP_DIR}${2}/${CONFIG_FILE}${i} # save state
        timeout ${TIMEOUT_VAL} $CUDA_UUT > ${TMP_DIR}${2}/${TMP_FILE}${i} 2>&1 &
    done
    wait
    gather_results $2
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt > /dev/null 2>&1
        rm -r ${TMP_DIR}${2} > /dev/null 2>&1 # comment out to debug output
    fi
    if [[ "$profile" -ne 1 ]]; then
        # clean intermediate logs anyway if profile != 1
        rm _ptx* _cuobjdump_* _app_cuda* *.ptx f_tempfile_ptx gpgpu_inst_stats.txt > /dev/null 2>&1
    fi
}

main() {
    # Normalize existing invalid combos to reduced keys (idempotent)
    if [[ -f "${INVALID_COMBOS_FILE}" ]]; then
        tmp_reduced=$(mktemp)
        awk -F';' '
        function getv(kv, k) { return (k in kv)?kv[k]:"" }
        {
            delete kv
            for (i=1; i<=NF; ++i) {
                split($i, a, "=")
                k=a[1]; sub(/^\s+|\s+$/,"",k)
                v=a[2]; sub(/^\s+|\s+$/,"",v)
                kv[k]=v
            }
            key = "comp=" getv(kv,"comp") ";per_warp=" getv(kv,"per_warp") ";kernel=" getv(kv,"kernel") ";"
            key = key "thread=" getv(kv,"thread") ";warp=" getv(kv,"warp") ";block=" getv(kv,"block") ";cycle=" getv(kv,"cycle") ";"
            key = key "reg_name=" getv(kv,"reg_name") ";reg_rand_n=" getv(kv,"reg_rand_n")
            if (!(key in seen)) { print key; seen[key]=1 }
        }' "${INVALID_COMBOS_FILE}" > "$tmp_reduced" 2>/dev/null || true
        if [[ -s "$tmp_reduced" ]]; then
            mv "$tmp_reduced" "${INVALID_COMBOS_FILE}"
        else
            rm -f "$tmp_reduced"
        fi
    fi
    # Remove all directories whose names start with 'logs'
    find . -type d -name "logs*" -exec rm -rf {} + 2>/dev/null || true
    
    if [[ "$profile" -eq 1 ]] || [[ "$profile" -eq 2 ]] || [[ "$profile" -eq 3 ]]; then
        RUNS=1
    fi
    # MAX_RETRIES to avoid flooding the system storage with logs infinitely if the user
    # has wrong configuration and only Unclassified errors are returned
    MAX_RETRIES=3
    LOOP=1
    mkdir ${CACHE_LOGS_DIR} > /dev/null 2>&1
    while [[ $RUNS -gt 0 ]] && [[ $MAX_RETRIES -gt 0 ]]
    do
        echo "runs left ${RUNS}" # DEBUG
        let MAX_RETRIES--
        LOOP_START=${LOOP}
        unset LAST_BATCH
        if [ "$BATCH" -gt "$RUNS" ]; then
            BATCH=${RUNS}
            LOOP_END=$(($LOOP_START))
        else
            BATCH_RUNS=$(($RUNS/$BATCH))
            if (( $RUNS % $BATCH )); then
                LAST_BATCH=$(($RUNS-$BATCH_RUNS*$BATCH))
            fi
            LOOP_END=$(($LOOP_START+$BATCH_RUNS-1))
        fi

        for i in $( seq $LOOP_START $LOOP_END ); do
            parallel_execution $BATCH $i
            let LOOP++
        done

        if [[ ! -z ${LAST_BATCH+x} ]]; then
            parallel_execution $LAST_BATCH $LOOP
            let LOOP++
        fi
    done

    if [[ $MAX_RETRIES -eq 0 ]]; then
        echo "Probably \"${CUDA_UUT}\" was not able to run! Please make sure the execution with GPGPU-Sim works!"
    else
        echo "Masked: ${masked} (performance = ${performance})"
        echo "SDCs: ${SDC}"
        echo "DUEs: ${crashes}"
    fi
    if [[ "$DELETE_LOGS" -eq 1 ]]; then
        rm -r ${CACHE_LOGS_DIR} > /dev/null 2>&1 # comment out to debug cache logs
    fi
}

main "$@"
exit 0
