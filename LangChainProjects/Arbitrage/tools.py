from langchain.tools import StructuredTool
import requests
def usebballapi(st):
    """st is a string, for example "bets", which will call the bets, use apibasketball to see how to call each thing"""
    url = "https://api-basketball.p.rapidapi.com/"
    url = url + st
    headers = {
        "X-RapidAPI-Key": "30dc89b030msh4c5443fc00c447ap1a6a04jsn8112d78a4d7e",
        "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)
    return(response.json())
    
bballtool = StructuredTool.from_function(usebballapi)