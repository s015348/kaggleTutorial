#!/bin/bash

REPO="/kaggleTutorial"
GIT_URL="https://github.com/s015348$REPO"
KAGGLE_CNAME="facial-keypoints-detection"
CNAME="facial"
KAGGLE_USERNAME="your username of kaggle"
KAGGLE_PWD="your password of kaggle"
TRAIN_FILENAME="training.zip"
TEST_FILENAME="test.zip"
TABLE_FILENAME="IdLookupTable.csv"
TESTFILE="$REPO/$CNAME/gpu_test.py"
DATAFILE="$REPO/$CNAME/training.csv"

clone_code() {
	cd /
	git clone $GIT_URL
	cd $REPO
	git status
	python "$TESTFILE"	
}

update_code() {
	cd $REPO
	git status
	git pull $GIT_URL
	git status
}

download_data() {
	cd "$REPO/$CNAME"
	read -p "Input your kaggle username: " KAGGLE_USERNAME
	read -s -p "                  password: " KAGGLE_PWD
	echo "\n\nStart downloading with user ${KAGGLE_USERNAME}"
	kg download -u $KAGGLE_USERNAME -p $KAGGLE_PWD -c $KAGGLE_CNAME -f $TRAIN_FILENAME
	kg download -u $KAGGLE_USERNAME -p $KAGGLE_PWD -c $KAGGLE_CNAME -f $TEST_FILENAME
	kg download -u $KAGGLE_USERNAME -p $KAGGLE_PWD -c $KAGGLE_CNAME -f $TABLE_FILENAME
	unzip $TRAIN_FILENAME
	unzip $TEST_FILENAME
}

# Initializaing or updating codes from git repo
if [ ! -d "$REPO" ]; then
	echo "${REPO} does not exist, cloning code..."
	clone_code
else
	echo "${REPO} exists, updating it..."	
	update_code
fi

# Initializaing data from kaggle website
if [ ! -f "$DATAFILE" ]; then
	echo "${DATAFILE} does not exist, downloading..."
	download_data
else
	echo "${DATAFILE} exist"
fi

echo "\n   Type cd $REPO/$CNAME && python example1.py to run your codes\n"
