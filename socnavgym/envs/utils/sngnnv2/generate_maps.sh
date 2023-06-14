#!/bin/bash

#if [ ! -d "./unlabelled_data" ]
#then
#  curl -f ljmanso.com/files/SocNav2_unlabelled_data.tgz --output unlabelled.tgz || {
#    echo "Download failed"
#    exit 1
#    }
#  mkdir "unlabelled_data"
#  tar zxvf unlabelled.tgz -C ./unlabelled_data/
#  rm unlabelled.tgz
#  find unlabelled_data/SocNav2_unlabelled_data/ -name '*.*' -exec mv {} unlabelled_data/ \;
#  cd unlabelled_data/ && rm -rf SocNav2_unlabelled_data/ && cd ../
#fi

if [ ! -d "./unlabelled_data_robot" ]
then
  mkdir -p unlabelled_data_robot/
  python3 add_robot_pose_to_dataset.py unlabelled_data unlabelled_data_robot
fi

TEMPFILE=/tmp/$$.tmp
echo 0 > $TEMPFILE
FILES="unlabelled_data_robot/*.json"
mkdir -p images_dataset/

for f in $FILES
do
  if [ ! -f "images_dataset/${f##*/}" ]; then
    echo "Processing $f"
    COUNTER=$[$(cat $TEMPFILE) + 1]
    echo $COUNTER > $TEMPFILE

    python3 showcase_static.py "example_model" "$f" 150 &
    cp "$f" images_dataset/

    if [ $COUNTER -eq 2 ]
    then
      echo 0 > $TEMPFILE
      wait
    fi
  fi

done
unlink $TEMPFILE

