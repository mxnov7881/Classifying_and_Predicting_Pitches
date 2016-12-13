for D in ./datasets/*
do
    currdir=$(basename $D)
    pref="./datasets/"
    echo "------ DATA SET: ${currdir} ---------------"
    echo "Binary SVM"
    python ./src/classify.py --mode train --algorithm binary_svm --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train 
	python ./src/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".test  --predictions-file "$pref$currdir/$currdir".predictions
	python compute_accuracy.py "$pref$currdir/$currdir".test "$pref$currdir/$currdir".predictions
    echo "kNN"
    python ./src/classify.py --mode train --algorithm binary_knn --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".train
	python ./src/classify.py --mode test --model-file "$pref$currdir/$currdir".model --data "$pref$currdir/$currdir".test  --predictions-file "$pref$currdir/$currdir".predictions
	python compute_accuracy.py "$pref$currdir/$currdir".test "$pref$currdir/$currdir".predictions
done