from cv_json import json_to_query
from vectorstore import query_vector_store

def chatbot_response(user_input, json_file):
    query = json_to_query(json_file)
    results = query_vector_store(query)
    return f"User Input: {user_input}, JSON Query Result: {results}"