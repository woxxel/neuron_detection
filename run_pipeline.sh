#!/bin/bash

dataset="AlzheimerMice_Hayashi"
#mouse="555wt"
mice=$(find /usr/users/cidbn1/neurodyn/$dataset/* -maxdepth 0 -type d -exec basename {} \;)
echo "Found mice in dataset $dataset: $mice"
read -p 'Which mouse should be processed? ' mouse

mkdir -p $HOME/data/$mouse

## getting all sessions of $mouse to loop through
session_paths=$(find /usr/users/cidbn1/neurodyn/$dataset/$mouse/Session* -maxdepth 0 -type d)

s=1
for session_path in $session_paths
do
  echo $session_path

  ## writing sbatch submission commands to bash-file
  FILE="./sbatch_submit.sh"
  cat > $FILE <<- EOF
#!/bin/bash
#SBATCH -p medium
#SBATCH -t 08:00:00
#SBATCH --mem=20000

module use /usr/users/cidbn_sw/sw/modules
module load cidbn_caiman-1.9.10_py-3.9
source activate caiman-1.9.10_py-3.9

python3 ~/placefields/programs/process_session.py $mouse $session_path
EOF

  sbatch $FILE
  rm $FILE

  ## only process first 5 sessions (for now)
  # if [[ $s -eq 5 ]]; then
  #   break
  # fi
  ((s++))

done
