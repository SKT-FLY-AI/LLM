import json

def json_to_query(file):
    with open(file) as json_file:
        data = json.load(json_file)
        print(data)
        ID = data['IMG-id']
        color_code = data['color-code']
        bristol = data['bristol']
        blood = data['blood']

        query = f"                                      "
        
        response = """model="gpt-4o"
            response_format={"type": "json_object"},
            messages=[
                {"role":"system",
                "content": "You are a helpful assistant designed to ouput JSON."},
                {"role": "user", "content": """f"{query}""""},
            ]"""

        return query, response