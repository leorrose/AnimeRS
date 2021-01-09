import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
import zipfile
import io
import shutil
from typing import Dict, List, Tuple
from collections import defaultdict
from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise.dump import load
from surprise.prediction_algorithms.predictions import Prediction
from surprise.trainset import Trainset


script_dir: str = os.path.dirname(os.path.realpath(__file__))
data_path: str = f"{script_dir}/data"

def download_zip(url: str, extract_path:str = None) -> None:
  """
  Method to download a zip from url and extract to given directory.

  Args:
    url (str): the zip url
    extract_path (str): the path to extract if none will extract in the 
    running directory. defaults to None
  """
  # get zip
  response: requests.Response = requests.get(url)
  # test status code is 200
  response.raise_for_status()
  # read response content (zip) into zipfile object
  zip: zipfile.ZipFile = zipfile.ZipFile(io.BytesIO(response.content))
  # extract zip to given path
  zip.extractall(extract_path)

def download_data() -> None:
  """
  Method to download partial data set from our github repository
  """
  # delete old folder
  if not os.path.exists(f'{data_path}/rating.csv') or not os.path.exists(f'{data_path}/anime.csv'):
    if os.path.exists(f'{data_path}'):
        shutil.rmtree(f'{data_path}')

    # github repository base url
    base_url: str = "https://github.com/leorrose/anime/raw/main"
    # partial image data set url
    anime_data_url: str = f"{base_url}/anime_data.zip?raw=true"
    # download zip
    download_zip(anime_data_url, data_path)


@st.cache
def load_data() -> pd.DataFrame:
    """
    Method to load data into data frame

    Returns:
        (pd.Dataframe): data frame of users
    """
    # read and combine to one pdf
    rating_df: pd.DataFrame = pd.read_csv(f"{data_path}/rating.csv")
    anime_df: pd.DataFrame = pd.read_csv(f"{data_path}/anime.csv")
    data_df: pd.DataFrame  =  pd.merge(rating_df, anime_df, on="anime_id")
    
    # remove rows with missing values and rows with user rating of -1
    data_df: pd.DataFrame = data_df.dropna()
    data_df: pd.DataFrame = data_df[data_df.rating_x != -1]

    # remove all users that gave more than 250 reviews and anime's that have less than 250 reviews
    min_anime_ratings:int = 250
    filter_anime: pd.Series = data_df['anime_id'].value_counts() > min_anime_ratings
    filter_anime: pd.DataFrame = filter_anime[filter_anime].index.tolist()

    min_user_ratings:int = 250
    filter_users: pd.Series = data_df['user_id'].value_counts() > min_user_ratings
    filter_users: pd.DataFrame = filter_users[filter_users].index.tolist()
    data_df: pd.DataFrame  = data_df[
        ( data_df['anime_id'].isin(filter_anime)) & (data_df['user_id'].isin(filter_users))]

    # take a sample of 100,000
    data_df: pd.DataFrame  = data_df.sample(n=100000, random_state=42)

    return data_df

@st.cache
def get_top_n(predictions: List[Prediction], n: int=10
                              ) -> Dict[int, List[Tuple[int, float]]]:
    """
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(List[Prediction]): The list of predictions, as 
        returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. 
        Default is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
      # add to users list the item id and its estimated rating
      top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
      # sort users item prediction by estimated rating
      user_ratings.sort(key=lambda x: x[1], reverse=True)
      # slice only top n
      top_n[uid] = user_ratings[:n]
    
    return top_n


download_data()
data_df: pd.DataFrame = load_data()

# main view
st.title("Anime Recommendation System")

# side bar
user_ids: List[int] = st.sidebar.multiselect("Select User Id", sorted(data_df.user_id.unique()))
number_of_recommendations:int = int(st.sidebar.number_input("Number of Recommendation", 
                                min_value=1, max_value=30, value =10))
render_recommendations: bool = st.sidebar.button("Get Recommendation")

if render_recommendations:
    for user_id in user_ids:
        # get all anime's
        anime_ids: pd.Series = data_df.anime_id.unique()

        # create a user data frame to predict ratings
        user_to_predict = {'user_id': [user_id] * len(anime_ids),
                        'anime_id': anime_ids,
                        'rating_x': [0] * len(anime_ids)}
        predict_data_df: pd.DataFrame = pd.DataFrame(user_to_predict)

        # create data object from dataframe
        reader: Reader = Reader(rating_scale=(1, 10))
        predict_data: Dataset = Dataset.load_from_df(predict_data_df, reader)

        # read trained model
        algo: SVD = load(f'{script_dir}/svd.pickle')[1]

        # predict ratings
        testset: Trainset = predict_data.build_full_trainset().build_testset()
        predictions: List[Prediction] = algo.test(testset)

        # get top 10
        top_n: Dict[int, List[Tuple[int, float]]] = get_top_n(predictions, 
                                                            n=number_of_recommendations)

        for uid, user_ratings in top_n.items():
            st.markdown(f"## ***User id: {uid}***")
            st.subheader(f"Recomandations (Users anime are not included):")
            st.dataframe(pd.DataFrame([data_df.loc[data_df.anime_id == iid, 
                                        'name'].iloc[0] for (iid, _) in user_ratings]))

        st.subheader(f"Users Anime's:")
        st.dataframe(data_df[data_df.user_id==user_id].sort_values(by='rating_x', 
                        ascending=False)["name"].reset_index(drop=True))
