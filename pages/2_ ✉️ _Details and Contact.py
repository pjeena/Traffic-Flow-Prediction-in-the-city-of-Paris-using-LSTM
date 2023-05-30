import streamlit as st

#st.write("## Code")

st.write(
    "See more in the public repo [GitHub"
    " repository](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM)"
)

st.code(
    f"""
This app forecasts the live traffic for the next 3 hours in the famous streets of Paris. Additionally, it also provides 
statistics for the historial traffic data such as distribution of traffic among 2 wheelers and 4 wheelers, most busiest 
junction etc.'
""",
    "python",
)
