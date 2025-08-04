import re
import os
import ast
import sys
import json
import time
import runpy
import chardet
import mimetypes
import traceback
import importlib
import numpy as np
import pandas as pd
from typing import ClassVar

from tornado.process import task_id

from alpha.team import Team
from alpha.roles import Role
from alpha.logs import logger
from alpha.schema import Message
from alpha.actions import Action, UserRequirement

from src.llm_utils import SeLLM
from src.llm_utils import load_config
from src.team_config import data_analysis_Input_Analyst

from src.Tools import EDA_Tools
from src.Tools.date_utils import check_date_column
from src.Tools.plot_utils import boxplot,heatmap_plot,violin_plot,density_plot
from src.oss_utils import download_to_file,oss_upload_by_path,get_image_url

from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings

from dotenv import load_dotenv
from utils import read_data_file

load_dotenv()
server_base = os.getenv('server_base')
base_path = os.getenv('base_path')

handler = {"sink": sys.stdout, "level": "ERROR"}
logger.configure(handlers=[handler])
init_file_names=None
df_data_list = None


async def start(
        idea: str = "",
        investment: float = 0,
        n_round: int = 1,
        add_human: bool = True,
):
    team = Team()
    team.hire(
        [
            data_analysis_Input_Analyst(),
        ]
    )

    team.run_project(idea)
    await team.run(n_round=n_round)


async def main():
    while True:
        userInput = input("\n\n老板，您好：")
        if userInput == "结束" or userInput == "exit":
            break
        else:
            await start(userInput)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
