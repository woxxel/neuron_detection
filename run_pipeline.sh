#!/bin/bash

cpus=8
datapath='/usr/users/cidbn1/neurodyn'
dataset="AlzheimerMice_Hayashi"
# dataset="Shank2Mice_Hayashi"

SUBMIT_FILE="./sbatch_submit.sh"

mice=$(find $datapath/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
# echo "Found mice in dataset $dataset: $mice"
# read -p 'Which mouse should be processed? ' mouse

for mouse in $mice
do
  # mkdir -p $HOME/data/$mouse
  mkdir -p /scratch/users/$USER/data/$dataset/$mouse

  ## getting all sessions of $mouse to loop through
  session_names=$(find $datapath/$dataset/$mouse/Session* -maxdepth 0 -type d -exec basename {} \;)

  # s=1
  for session_name in $session_names
  do
    if test -f /scratch/users/$USER/data/$dataset/$mouse/$session_name/OnACID_results.hdf5; then
      # echo "$session_name already processed - skipping"
      continue
    fi

    session_path=$datapath/$dataset/$mouse/$session_name

    if test -d $session_path/images; then
      echo "Processing $session_path"

      ## writing sbatch submission commands to bash-file
      cat > $SUBMIT_FILE <<- EOF
#!/bin/bash
#SBATCH -A all
#SBATCH -p medium
#SBATCH -c $cpus
#SBATCH -t 02:00:00
#SBATCH --mem=20000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=1

python3 ~/placefields/data_pipeline/process_session.py $dataset $mouse $session_path $cpus
EOF

      sbatch $SUBMIT_FILE
      rm $SUBMIT_FILE
    fi

    ## only process first 5 sessions (for now)
    # if [[ $s -eq 5 ]]; then
    #   break
    # fi
    # ((s++))

  done
done
