# Real-time Object Detection using Tensorflow Android
* Video: https://youtu.be/0oBequpSGXM 

* Step 1: Install Android Studio https://developer.android.com/studio/...
* Step 2: Install python if not installed (Anaconda: https://conda.io/docs/user-guide/inst...)
* Step 3: Install Tensorflow if not in your python package
 git clone https://github.com/tensorflow/tensorflow
* Step 4: Open Android Studio
  select Open an existing Android Studio project
Select the tensorflow/examples/android directory from wherever you cloned the TensorFlow Github repo. Click OK.
  If it asks you to do a Gradle Sync, click OK.
  Install various platforms and tools If it asks.
* Step 5: Open the build.gradle file 
  (you can go to 1:Project in the side panel and find it under the Gradle Scripts zippy under Android).
  Look for the nativeBuildSystem variable and set it to none if it isn't already:
  // set to 'bazel', 'cmake', 'makefile', 'none'
  def nativeBuildSystem = 'none'
* Step 6: Click the Run button (the green arrow) or use Run - Run 'android' from the top menu.
  If it asks you to use Instant Run, click Proceed Without Instant Run.
