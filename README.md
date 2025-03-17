# :earth_americas: GDP dashboard template

A simple Streamlit app showing the GDP of different countries in the world.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gdp-dashboard-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

TODOS
1. Instead of reference range compared to OP, use reference range as an actual reference range. Like the 930 setup.
2. Check the mid previous candle condition, it's probably buggy
3. Check the MFE and MAE calculations, probably buggy
4. Add avg time to reach target or stop
5. Add a condition not to add new setups if there is active current one.