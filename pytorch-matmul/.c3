project: pytorch-matmul-example
script: run.sh          # a bash script with your commands
hardware: l40           # l40, a100 or h100
time: "00:10:00"        # maximum runtime

docker:
  image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime   # public image with the dependencies
  requires_accelerator: cuda

output:
  - ./results           # folders collected as results
