#!/bin/bash

cpus=8
datapath_in='/usr/users/cidbn1/neurodyn'
datapath_out='/usr/users/cidbn1/placefields'
dataset="AlzheimerMice_Hayashi"
# dataset="Shank2Mice_Hayashi"

SUBMIT_FILE="./sbatch_submit.sh"

mice=$(find $datapath_in/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
echo "Found mice in dataset $dataset: $mice"
read -p 'Which mouse should be processed? ' mouse

# for mouse in $mice
# do
  mkdir -p $datapath_out/$dataset/$mouse

  ## getting all sessions of $mouse to loop through
  session_names=$(find $datapath_in/$dataset/$mouse/Session* -maxdepth 0 -type d -exec basename {} \;)

  s=1
  for session_name in $session_names
  do

    if ! $(test -d $datapath_out/$dataset/$mouse/$session_name); then
      mkdir -p $datapath_out/$dataset/$mouse/$session_name; 
    fi

    # if test -f $datapath_out/$dataset/$mouse/$session_name/OnACID_results.hdf5; then
    #   echo "$session_name already processed - skipping"
    #   continue
    # fi

    session_path=$datapath_in/$dataset/$mouse/$session_name

    if test -d $session_path/images; then
      echo "Processing mouse $mouse, $session_name"

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

python3 ./process_session.py $datapath_in $datapath_out $dataset $mouse $session_name $cpus
EOF

      sbatch $SUBMIT_FILE
      rm $SUBMIT_FILE
    fi

    ## only process first 5 sessions (for now)
    if [[ $s -eq 3 ]]; then
      break
    fi
    ((s++))

  done
# done
