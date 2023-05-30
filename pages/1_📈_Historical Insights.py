import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import glob    
import os
import pickle

st.set_page_config(layout="wide")
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 1.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)


## junctions in the city of paris
junction_labels = sorted(['[Paris] Rivoli x Nicolas Flamel',
                    '[Paris] Amsterdam x Clichy',
                    '[Paris] Quai de Valmy',
                    '[Paris] Quai de Jemmapes',
                    '[Paris] CF892 Rivoli x Bourdonnais',
                    '[Paris] CF318 Rivoli x Lobau',
                    '[Paris] CF1 Rivoli x Sébastopol',
#                    '[Paris] CF4 Poissonnière x Montmartre EST',
                    '[Paris] CF4 Poissonnière x Montmartre OUEST'])

#dir_path = '../data/vehicles_EDA/*.parquet'
#file_names = sorted(glob.glob(dir_path))


with st.sidebar:
    statistics_type = st.radio(
        'Statistics :',
        ("By each junction", "Between different junctions")
    )

    st.markdown(
    """<style>
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


if statistics_type=='By each junction':

            option_1 = st.selectbox('Junction Name',junction_labels)
            path_data_junction = os.path.join( 'data', 'vehicles_EDA', option_1 + '.parquet' )
            df = pd.read_parquet(path_data_junction)


            df["Hour"] = df["t"].dt.hour
            df["Day of the Month"] = df["t"].dt.day
            df["Day Name"] = df["t"].dt.day_name()
            df['Total'] = df['2_wheelers'] + df['4_wheelers']
            df = df.reset_index(drop=True)
            df_hour = df.groupby("Hour")[["2_wheelers", "4_wheelers", 'Total']].mean().reset_index()
            df_hour["2_wheelers"] = df_hour["2_wheelers"].astype(int)
            df_hour["4_wheelers"] = df_hour["4_wheelers"].astype(int)
            df_hour["Total"] = df_hour["Total"].astype(int)


            df_day = (
                df.groupby("Day of the Month")[["2_wheelers", "4_wheelers", "Total"]].mean().reset_index()
            )
            df_day["2_wheelers"] = df_day["2_wheelers"].astype(int)
            df_day["4_wheelers"] = df_day["4_wheelers"].astype(int)
            df_day["Total"] = df_day["Total"].astype(int)

            df_day_name = df.groupby("Day Name")[["2_wheelers", "4_wheelers","Total"]].mean().reset_index()
            df_day_name["2_wheelers"] = df_day_name["2_wheelers"].astype(int)
            df_day_name["4_wheelers"] = df_day_name["4_wheelers"].astype(int)
            df_day_name["Total"] = df_day_name["Total"].astype(int)
            cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_day_name = df_day_name.groupby(['Day Name']).sum().reindex(cats).reset_index()


            df_types = [df_hour, df_day, df_day_name]
            title_names = ['Average Traffic by hour', 'Average Traffic by day of month', 'Average Traffic by day name']
            for i, type in enumerate(["Hour", "Day of the Month", "Day Name"]):
                fig = px.line(
                    df_types[i],
                    x=type,
                    y=["2_wheelers", "4_wheelers","Total"],
                    labels={"variable": "type of vehicle", "value": "Vehicles"},
                    #              hover_data={"date": "|%B %d, %Y"},
                    title=title_names[i],
                    markers=True,
                )
                fig.update_layout(xaxis=dict(tickmode="linear", tick0=0.0, dtick=1.0))
                fig.update_traces(marker_size=10)
                fig.update_xaxes(title_font=dict(size=16, family="Courier", color="black"))
                fig.update_yaxes(title_font=dict(size=16, family="Courier", color="black"))
                # fig.update_layout(xaxis = dict(tickfont = dict(size=13)))
                # fig.update_layout(yaxis = dict(tickfont = dict(size=13)))

                st.plotly_chart(fig, theme="streamlit", use_container_width=True)



#            pickle.dump(df_types, open('data/vehicles_EDA/model.pkl', 'wb'))



dir_path = 'data/vehicles_EDA/*.parquet'
file_names = sorted(glob.glob(dir_path))

for file in file_names:


            df = pd.read_parquet(file)


            df["Hour"] = df["t"].dt.hour
            df["Day of the Month"] = df["t"].dt.day
            df["Day Name"] = df["t"].dt.day_name()
            df['Total'] = df['2_wheelers'] + df['4_wheelers']
            df = df.reset_index(drop=True)
            df_hour = df.groupby("Hour")[["2_wheelers", "4_wheelers", 'Total']].mean().reset_index()
            df_hour["2_wheelers"] = df_hour["2_wheelers"].astype(int)
            df_hour["4_wheelers"] = df_hour["4_wheelers"].astype(int)
            df_hour["Total"] = df_hour["Total"].astype(int)


            df_day = (
                df.groupby("Day of the Month")[["2_wheelers", "4_wheelers", "Total"]].mean().reset_index()
            )
            df_day["2_wheelers"] = df_day["2_wheelers"].astype(int)
            df_day["4_wheelers"] = df_day["4_wheelers"].astype(int)
            df_day["Total"] = df_day["Total"].astype(int)

            df_day_name = df.groupby("Day Name")[["2_wheelers", "4_wheelers","Total"]].mean().reset_index()
            df_day_name["2_wheelers"] = df_day_name["2_wheelers"].astype(int)
            df_day_name["4_wheelers"] = df_day_name["4_wheelers"].astype(int)
            df_day_name["Total"] = df_day_name["Total"].astype(int)
            cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            df_day_name = df_day_name.groupby(['Day Name']).sum().reindex(cats).reset_index()


            df_types = [df_hour, df_day, df_day_name]


            pickle.dump(df_types, open(  file.split('.')[-2] + '.pkl', 'wb'))






dir_path = 'data/vehicles_EDA/*.pkl'
file_names = sorted(glob.glob(dir_path))

#file = file_names[0]
df_hour_junc = []
df_day_junc = []
df_dayname_junc = []

for file in file_names:
        df = pickle.load(open(file, 'rb'))
        df_hour_junc.append(   df[0].add_suffix(  '_' + file.split('/')[-1].split('.')[0])   )
        df_day_junc.append(   df[1].add_suffix(  '_' + file.split('/')[-1].split('.')[0])   )
        df_dayname_junc.append(   df[2].add_suffix(  '_' + file.split('/')[-1].split('.')[0])   )


df_hour_junc =  pd.concat(df_hour_junc,axis=1)
df_day_junc =  pd.concat(df_day_junc,axis=1)
df_dayname_junc =  pd.concat(df_dayname_junc,axis=1)


if statistics_type=='Between different junctions':
            
            option_2 = st.selectbox('Type',['2_wheelers', '4_wheelers','Total'])



            df_types = [df_hour_junc, df_day_junc, df_dayname_junc]
            title_names = ['Average Traffic by hour', 'Average Traffic by day of month', 'Average Traffic by day name']
            for i, type in enumerate(["Hour_[Paris] Amsterdam x Clichy", "Day of the Month_[Paris] Amsterdam x Clichy", "Day Name_[Paris] Amsterdam x Clichy"]):
                            columns_to_plot =[type] + list(df_types[i].filter(regex='^{}'.format(option_2)).columns)
                            df_to_plot = df_types[i][columns_to_plot]
                            df_to_plot.columns = df_to_plot.columns.map(lambda x: x.removeprefix(option_2 + '_')) 

                            fig = px.line(
                                df_to_plot,
                                x=type,
                                y=df_to_plot.columns[1:],
                                labels={"variable": "Junction", "value": "Vehicles",
                                        df_to_plot.columns[0] : df_to_plot.columns[0].split('_')[0] },
                                #              hover_data={"date": "|%B %d, %Y"},
                                title=title_names[i],
                                markers=True,
                            )
                            fig.update_layout(xaxis=dict(tickmode="linear", tick0=0.0, dtick=1.0))
                            fig.update_traces(marker_size=10)
                            fig.update_xaxes(title_font=dict(size=16, family="Courier", color="black"))
                            fig.update_yaxes(title_font=dict(size=16, family="Courier", color="black"))
                            # fig.update_layout(xaxis = dict(tickfont = dict(size=13)))
                            # fig.update_layout(yaxis = dict(tickfont = dict(size=13)))

                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)