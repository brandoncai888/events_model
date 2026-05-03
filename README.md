
Run noise generation pipeline example command. Here each rate * duration = 40 so that the number of events per pixel is Pois(40) long enough exposure time to assess the accuracy of the model against ideal distributions without the effects of random variation from samples being too small. 

<code>python noise_gen_pipeline.py --rates 0.1,0.2,0.4,1.0,2.0,4.0,10.0 --durations 400,200,100,40,20,10,4 --width 346 --height 260 --paired --folder data --video 1.0</code>