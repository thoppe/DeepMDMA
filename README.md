# Deep MDMA

Deep MDMA is a combination of [Lucid](https://github.com/tensorflow/lucid) and [librosa](https://librosa.github.io/librosa/).
The goal is to visualize the movement between layers of a neural network set to music.
While Lucid makes beautiful images, independently training and interpolating between them leaves the animation disjointed.
This is because the locally minimized regions of the CPPN are far from each other for each sample .
The trick was to reuse the initial coordinates from the previous model to train the next.
This provides continuity to train one image into another.

[![Secret Crates 🎧💊🎧 Deep MDMA 🎧💊🎧](https://img.youtube.com/vi/qPi5UPAlwl8/0.jpg)](https://www.youtube.com/watch?v=qPi5UPAlwl8)

To use, start with python 3, install tensorflow, and all the requirements

   pip install -r requirements.txt




