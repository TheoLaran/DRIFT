#echo 'installing tensorflow'
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp34-cp34m-linux_x86_64.whl
#pip3 install --upgrade --user $TF_BINARY_URL
END=5
touch em/evalMetrics_ml_1m
touch rv/relevanceVector_ml_1m
for method in "SAROS" "Main"
do
  cd java/src
  echo 'making relevance vector'
  javac -cp ../binaries/commons-lang3-3.5.jar  preProcess/ConvertIntoRelVecGeneralized.java preProcess/InputOutput.java
  java -cp . preProcess.ConvertIntoRelVecGeneralized ../../res/gt_"$method" ../../res/pr_"$method" ../../rv/relevanceVector_ml_1m 10
  cd -
  echo 'compute offline metrics'
  python3 compOfflineEvalMetrics.py ml_1m
  cp em/evalMetrics_ml_1m res/PANDOR/"$method"
done

python3.7 utils/res_to_latex.py

