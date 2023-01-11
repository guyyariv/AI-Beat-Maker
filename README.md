# AI-Beat-Maker

Description:  
Creating music has become easier with remix generators and separate drum line creators,
but there is still no solution for automatically creating the desired combination.
We use a combination of machine learning and DSP methods in order to automatically generate
a remix out of a given sample, and sync the result with a generated drumbeat created by
using a trained neural network.
Thus, we have created a beat-making program to give our users the ability to create a brand-new Hip Hop
beat in the simplest touch of a button, without the need for advanced and complicated music studio applications.


## Running Instructions

1. Clone this repo
2. Activate virtual environment
3. Install requirements
```angular2html
pip install -r requirements.txt
```
4. Add the sample you wish to remix into the 'samples' directory
5. Run main.py  
Parameters required:  
--track -> samples title with no extension (must be a wav file)  
Optional parameters:  
--drums -> True if you wish to generate drums for the remix else False (default=True)  
--bass -> True if you wish to generate bass for the remix else False (default=False)  
--k -> Number of clusters you wish to split your sample (default=2)   

For example:
```angular2html
python main.py --track=track_name --drums=True -- bass=False --k=4
```

If you would like to retrain the LSTM Drum-Generator, run:
```angular2html
python drums_generator/lstm.py
```
You can then find the new model weights in the 'drums_generator/weights' directory.  
All that's left is to update the predict.py file to load your new weights.
