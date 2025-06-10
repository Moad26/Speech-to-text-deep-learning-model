# Speech to text Deep Learning model

Uses LIBRISPEECH librery to train on, it turns flac audio to preprocessed spectrogram files, then a cnn model transform them feature maps.
And then a Bidrectionnal Lstm model takes those feature maps and transform them to text with the help of a ctc decoder.
