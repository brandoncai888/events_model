### Total Variation Distance (TVD)
Calculate TVD results: <code>TVD_results.txt</code> by simply adding any new files to <code>TVD_filenames.txt</code> and running <code>TVD_calculator.py</code>

For managed noise graph CSVs, you can also build the filenames automatically: <code>python TVD_calculator.py --rates 0.1,0.2,0.4 --durations 20 --data_root data</code>

### Managed file layout
All scripts now use <code>file_manager.py</code> for data paths:

<code>data/(noise or object)/(frequency or name)/(optional time-slice)/(events or iets or pictures or videos or tracks)/(file)</code>

Examples:

<code>data/noise/1.0Hz/events/poisson_noise_1.0Hz_20.0s.csv</code>

<code>data/noise/1.0Hz/iets/poisson_noise_1.0Hz_20.0s_iet.pkl</code>

<code>data/object/45/2.67_2.71/pictures/45_2.67_2.71_ON_iet_hist.png</code>

You can ask the manager for a path directly:

<code>python file_manager.py --source noise --artifact events --rate 1.0 --duration 20.0</code>

The old <code>--folder</code> argument is still accepted as an alias for <code>--data_root</code>.

### Analysis
Build an inter-event-time grid from a managed event CSV:

<code>python iets.py --source object --dataset 45 --slice 2.67_2.71 --polarity ON --data_root data</code>

Generate IET histogram graphs from the saved grid:

<code>python graphs.py --source object --dataset 45 --slice 2.67_2.71 --polarity ON --min_iet 0.00001 --max_iet 1 --data_root data</code>

Track event center-of-mass motion in 1 ms windows:

<code>python center_of_mass.py --source object --dataset 45 --slice 2.67_2.71 --polarity ON --window 0.001 --data_root data</code>

This writes COM snapshot and velocity CSVs under the managed <code>tracks</code> folder. Velocity is reported as <code>vx_pixels_per_s</code>, <code>vy_pixels_per_s</code>, and <code>speed_pixels_per_s</code>.

Create a video of average event-count heatmaps in 8x8 pixel bins over small time windows:

<code>python temporal_counts.py --source object --dataset 45 --slice 2.67_2.71 --polarity ON --window 0.001 --size 8 --fps 30 --data_root data --no_show</code>

Use COM velocity to predict neighboring-pixel event timing and plot residuals:

<code>python neighbor_time_prediction.py --source object --dataset 45 --slice 2.67_2.71 --polarity ON --window 0.001 --axis x --data_root data</code>

By default, positive <code>vx</code> predicts the pixel to the right and negative <code>vx</code> predicts the pixel to the left, with <code>predicted_dt = 1 / abs(vx)</code>. The actual event is the neighbor-pixel event nearest to <code>event_t + predicted_dt</code> in time. The residual plot includes negative residuals down to <code>-0.001s</code>, shows the average projected speed in the title, and marks both zero residual and mean residual as vertical lines; use <code>--min_residual</code> and <code>--max_residual</code> to change the range. Use <code>--dx 1 --dy 0</code> to force a right-neighbor prediction.

### Noise
##### Generation
To just generate poisson noise, 
<code> python -m noise.generate_poisson --rate 1.0 --duration 20 --width 346 --height 260 --data_root data</code>

##### Analysis Pipeline
Run noise generation/evaluation pipeline example command: <code>python noise_gen_pipeline.py --rates 0.1,0.2,0.4,1.0,2.0,4.0,10.0 --durations 400,200,100,40,20,10,4 --width 346 --height 260 --paired --data_root data --video 1.0</code>

In the example each $rate \cdot duration = 40$ so that the number of events per pixel is $\mathrm{Pois}(40)$ long enough exposure time to assess the accuracy of the model against ideal distributions without the effects of random variation from samples being too small. These generated noise datas all achieve TVD $\approx 1\%$. 

We can also use the same duration and vary the rate: <code>python noise_gen_pipeline.py --rates 0.1,0.2,0.4,1.0,2.0,4.0,10.0 --durations 20 --width 346 --height 260 --data_root data</code> Note that this noise data has varying TVD due to lower rates having very few points per pixel: 

|frequency (20s duration)|0.1  |0.2  |0.4 |1.0 |2.0 |4.0 |10.0|
|---------|-----|-----|----|----|----|----|----|
|TVD   |16.9%|10.7%|5.2%|2.0%|0.9%|0.5%|0.2%|


### Object
#### Raindrops dataset 
##### Format Conversion
.aedat to .csv conversion using <code>python object/aedat_to_csv.py --aedat 30,45 --data_root data</code>

The converter looks first in the managed events folder, then falls back to legacy locations such as <code>object/45Hz.aedat</code>.

##### Grab a time slice and polarities
<code>python object/isolate.py --source object --dataset 45 --mintime 2.67 --maxtime 2.71</code>

##### Visualize in video
visualize using <code>python ./visualize.py --source object --dataset 45 --duration 5 --fps 100</code>

##### Grab a snippet
<code>python ./visualize.py --source object --dataset 45 --duration .02 --fps 2500 --start .99 --slowdown 25</code>
