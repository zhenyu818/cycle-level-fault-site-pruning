# Cycle-Level Fault Injection Pruning 🚀

This project builds on **[gpuFI-4](https://github.com/caldi-uoa/gpuFI-4.git)** to prune fault injection points at the cycle level. It first discovers **danger regions** (cycles where registers are vulnerable), then restricts injections to those regions, and finally rescales results so they represent full-region fault injection.

## Repository Layout
- `gpufi-instinject/`: end-to-end fault injection runner; produces logs and pruned `test_result` outputs.
- `accel/`: post-processing to convert pruned results back to full-region estimates.

## Quickstart
1) Pull the container image  
`docker pull accelsim/ubuntu-18.04_cuda-11:latest`

2) Launch the container and mount this repo  
```
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  accelsim/ubuntu-18.04_cuda-11:latest
```

3) Discover vulnerable cycles (danger regions)  
`cd gpufi-instinject`  
On the first run, set `RUN_PER_EPOCH=1` in `inst_fault_inject_exp.sh`, then run:  
`bash inst_fault_inject_exp.sh`  
Check the generated `logs` for lines starting with `[danger region]`, which list vulnerable cycles for each register.

4) Save danger regions for pruning  
Copy all `[danger region]` lines into `gpufi-instinject/cycle_region.txt` 📌

5) Run fault injection only in danger regions  
`bash inst_fault_inject_exp.sh` again. Results land in `gpufi-instinject/test_result/` as CSVs.

6) Rescale results to full-region equivalents  
Copy the same `[danger region]` content into `accel/danger.log`. Place the CSV from `test_result` inside `accel/`, then run from `accel/`:  
`python calc_p.py <csv_filename> <T> <R>`  
- `T`: total cycle count  
- `R`: total register count  
The script prints the converted probability metrics.

## Notes
- Requires Docker with CUDA support and Python 3 inside the container.
- Adjust other parameters in `gpufi-instinject/inst_fault_inject_exp.sh` (e.g., target app, components, GPU arch) as needed.
- `accel.py` only sums registers present in both `danger.log` and the CSV; make sure the danger region file matches the CSV’s registers.
