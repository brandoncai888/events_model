### Total Variation Distance (TVD)
Calculate TVD results: <code>TVD_results.txt</code> by simply adding any new files to <code>TVD_filenames.txt</code> and running <code>TVD_calculator.py</code>

### Noise
##### Generation
To just generate poisson noise, 
<code> python -m noise.generate_poisson --rate 1.0 --duration 20 --width 346 --height 260 --folder .</code>

##### Analysis Pipeline
Run noise generation/evaluation pipeline example command: <code>python noise_gen_pipeline.py --rates 0.1,0.2,0.4,1.0,2.0,4.0,10.0 --durations 400,200,100,40,20,10,4 --width 346 --height 260 --paired --folder data_equal_counts --video 1.0</code>

In the example each $rate \cdot duration = 40$ so that the number of events per pixel is $\mathrm{Pois}(40)$ long enough exposure time to assess the accuracy of the model against ideal distributions without the effects of random variation from samples being too small. These generated noise datas all achieve TVD $\approx 1\%$. 

We can also use the same duration and vary the rate: <code>python noise_gen_pipeline.py --rates 0.1,0.2,0.4,1.0,2.0,4.0,10.0 --durations 20 --width 346 --height 260 --folder data_equal</code> Note that this noise data has varying TVD due to lower rates having very few points per pixel: 

|frequency (20s duration)|0.1  |0.2  |0.4 |1.0 |2.0 |4.0 |10.0|
|---------|-----|-----|----|----|----|----|----|
|TVD   |16.9%|10.7%|5.2%|2.0%|0.9%|0.5%|0.2%|


### Object
#### Raindrops dataset 
##### Format Conversion
.aedat to .csv conversion using <code>python object/aedat_to_csv.py --aedat 30Hz,45Hz --folder object</code>

##### Visualize in video
visualize using <code>python ./visualize.py --filename object/45Hz.csv --duration 5 --fps 100</code>

##### Grab a snippet
<code>python ./visualize.py --filename object/45Hz.csv --duration .05 --fps 10000 --start 2.67 --slowdown 10</code>