# Laptime Sim Python
Laptime sim

Clone the repository to your current directory

```console
git clone https://github.com/Jan-Jaap/laptime-sim .

```


Use [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) to create python3.12 virtual environment and install dependencies:
```console 
poetry env use "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
poetry install

```
Open a shell and launch streamlit webapp to start the WebApp:
```console
poetry shell
streamlit run src\streamlit_apps\Welcome.py

```

Use the webapp running on [localhost](http://localhost:8501)

![streamlit_trackview](/images/streamlit_trackview.png)

![streamlit_car_properties](/images/streamlit_car_properties.png)

![streamlit_laptime_optimizer](/images/streamlit_laptime_optimizer.png)
