#!/bin/bash

cpus=1
datapath='/usr/users/cidbn1/neurodyn'
dataset="AlzheimerMice_Hayashi"


SUBMIT_FILE="./sbatch_submit.sh"
ON_CLUSTER=false

mice=$(find $datapath/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
# echo "Found mice in dataset $dataset: $mice"
# read -p 'Which mouse should be processed? ' mouse

for mouse in $mice
do
  # mkdir -p $HOME/data/$mouse
  echo "Processing mouse $mouse"
  mkdir -p /scratch/users/$USER/data/$dataset/$mouse

  ## getting all sessions of $mouse to loop through
  session_names=$(find $datapath/$dataset/$mouse/Session* -maxdepth 0 -type d -exec basename {} \;)

  # s=1
  for session_name in $session_names
  do
    echo "Running $session_name..."
    if $ON_CLUSTER; then
      cat > $SUBMIT_FILE <<- EOF
#!/bin/bash
#SBATCH -A all
#SBATCH -p medium
#SBATCH -c $cpus
#SBATCH -t 00:05:00
#SBATCH --mem=1000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 ~/placefields/data_pipeline/preprocessing/align_session.py $datapath $dataset $mouse $session_name
EOF
    sbatch $SUBMIT_FILE
    rm $SUBMIT_FILE
  else
    python3 -W ignore ~/placefields/data_pipeline/preprocessing/align_session.py $datapath $dataset $mouse $session_name > /dev/null
  fi

  done
done
