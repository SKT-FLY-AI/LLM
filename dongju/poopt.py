from Input_text_generator import json_to_query

def query_to_gpt():
    query,response = json_to_query('/mnt/sdb/tmp/poopsee/langchain/via_gpt/dummy.json')
    print(query)
    print(f"response json string :: {response}")
    
if __name__ == "__main__":
    query_to_gpt()