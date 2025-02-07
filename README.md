# Laptime Sim Python

A simple and fast laptime simulator for determining the laptime when racing optimal speed.

It uses an approximate implementation of the algoritm described in this paper.
http://jameshakewill.com/Lap_Time_Simulation.pdf

I'm no math wizard. Optimization of laptimes is done by bruteforce trying different lines and using the fastest line found. By trying thousands of different options, we can come close (enough) to the optimum raceline. Some would call this AI...


## Installation
Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create python virtual environment and install dependencies:

There is a streamlit app to visualize the results.
```bash
streamlit run src/streamlit_apps/Welcome.py
```

Also a docker compose file to run the streamlit app in docker is provided.
```
docker compose up --build -d
```
This will run the app on [http://localhost:8501](http://localhost:8501)

## Usage
The main.py will run an optimization of all cars on all tracks.

## Examples of usage

![streamlit_trackview](/resources/images/streamlit_trackview.png)

![streamlit_car_properties](/resources/images/streamlit_car_properties.png)

