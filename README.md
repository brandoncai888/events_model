
Run noise generation/evaluation pipeline example command. Here each $rate \cdot duration = 40$ so that the number of events per pixel is $\mathrm{Pois}(40)$ long enough exposure time to assess the accuracy of the model against ideal distributions without the effects of random variation from samples being too small. 

<code>python noise_gen_pipeline.py --rates 0.1,0.2,0.4,1.0,2.0,4.0,10.0 --durations 400,200,100,40,20,10,4 --width 346 --height 260 --paired --folder data --video 1.0</code>

To just generate poisson noise, 

<code> python -m noise.generate_poisson --rate 1.0 --duration 20 --width 346 --height 260 --folder .</code>

Update <code>TVD_results.txt</code> by simply adding any new files to <code>TVD_filenames.txt</code> and running <code>TVD_calculator.py</code>