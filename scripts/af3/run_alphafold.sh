#!/bin/bash -l

#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 45G
#SBATCH --gres gpu:1
#SBATCH --partition l40s
#SBATCH --time 70:00:00
#SBATCH --account lpdi
#SBATCH --job-name alphafold3

json_path=""
input_dir=""
output_dir=""
run_data_pipeline="true"

# Parse input flags
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -i|--input)
      if [ -f "$2" ]; then
        json_path="$2"
        json_path=$(realpath $json_path)
      elif [ -d "$2" ]; then
        input_dir="$2"
        input_dir=$(realpath $input_dir)
      else
        echo "Error: Invalid input path '$2'" >&2
        exit 1
      fi
      shift # past argument
      shift # past value
      ;;
    -o|--output)
      output_dir="$2"
      output_dir=$(realpath $output_dir)
      shift # past argument
      shift # past value
      ;;
    --no-msa)
      run_data_pipeline="false"
      shift # past argument
      ;;
    *)
      echo "Error: Unknown option '$1'" >&2
      exit 1
      ;;
  esac
done

# Check that either json_path or input_dir is set
if [ -z "$json_path" ] && [ -z "$input_dir" ]; then
  echo "Error: Either -i/--input <file|directory> or --no-msa must be provided" >&2
  exit 1
fi

# Check that output_dir is set
if [ -z "$output_dir" ]; then
  echo "Error: -o/--output <directory> must be provided" >&2
  exit 1
fi

OPTIONS=""
if [ ! -z "$json_path" ]; then
  OPTIONS="$OPTIONS --json_path=$json_path"
fi
if [ ! -z "$input_dir" ]; then
  OPTIONS="$OPTIONS --input_dir=$input_dir"
fi
OPTIONS="$OPTIONS --output_dir=$output_dir"
OPTIONS="$OPTIONS --run_data_pipeline=$run_data_pipeline"

echo "Running AlphaFold with the following options:"
echo $OPTIONS

# setup env
conda activate /work/lpdi/users/dobbelst/tools/af3-env

XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS XLA_PYTHON_CLIENT_PREALLOCATE XLA_CLIENT_MEM_FRACTION

cd /work/lpdi/users/dobbelst/tools/alphafold3
python run_alphafold.py \
	--model_dir=/work/lpdi/users/dobbelst/tools/alphafold3 \
	--db_dir=/work/lpdi/databases/alphafold3_dbs --num_recycles 100 \
	$OPTIONS
