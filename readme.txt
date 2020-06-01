***********************************************************************************************************************************************************
1) To run all the python program and python notebook, following packages are required as listed:
	pydub, librosa, pandas, numpy, sklearn, itertools, matplotlib, keras
***********************************************************************************************************************************************************
2) Description of the program files:
	a) config.py -- Contains parameters for processing of audio files
	b) ConvertToWav.py -- Converts .au file to .wav file
	c) CreateDataset.py -- Creates dataset by doing feature extraction on .wav file
	d) FeatureExtraction.py -- Invokes program to do feature Extraction
	e) music_genre_cnn_2d_5_genre.py -- Converts .au to .wav, generates dataset and runs CNN model on 5 genres.
	f) music_genre_cnn_2d_10_genre.py -- Converts .au to .wav, generates dataset and runs CNN model on 10 genres.
	g) music_genre_without_pca.ipynb -- Reads dataset extracted by FeatureExtraction.py and uses Machine Learning techniques without feature elimination
	h) music_genre_pca.ipynb -- Reads dataset extracted by FeatureExtraction.py and uses Machine Learning techniques with feature elimination
***********************************************************************************************************************************************************