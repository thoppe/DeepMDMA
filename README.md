# Deep MDMA
----------------------------------------------------------------

Deep MDMA is a combination of [Lucid](https://github.com/tensorflow/lucid) and [librosa](https://librosa.github.io/librosa/). The goal is to visualize the movement between layers of a neural network set to music.

[![Secret Crates 🎧💊🎧 Deep MDMA 🎧💊🎧](https://img.youtube.com/vi/qPi5UPAlwl8/0.jpg)](https://www.youtube.com/watch?v=qPi5UPAlwl8)

While Lucid make beautiful images, independently training multiple images and interpolating leaves the animation disjointed.
This is because the regions of the CPPN for each sample are far from each other.
The trick was to reuse the initial coordinates from the previous model to train the next.
This provides continuity to train one image into another.


