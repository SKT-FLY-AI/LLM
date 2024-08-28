import json


'''
* 병 있을 경우 *
이 강아지는 {disease}을 앓고 있는데 
나한테  우리아이 건강 상태를 파악하기 위해서, 이 문서를 기반으로 질문을 한 가지만 해줘
질문할 내용이 없으면, 만약 건강에 이상이 있는거 같으면, 병원을 가라고 해주고 
이상이 없는거 같으면 결론을 내려주면 좋을거 같아
'''
'''
* 병 없을 경우 *
나한테 우리아이 건강 상태를 파악하기 위해서, 이 문서를 기반으로 질문을 한 가지만 해줘
질문할 내용이 없으면, 만약 건강에 이상이 있는거 같으면, 병원을 가라고 해주고 
이상이 없는거 같으면 결론을 내려주면 좋을거 같아
'''

def json_to_query(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        #print(data)

        ID = data['IMG-id']
        fecal = data['fecal']   # 강아지 똥 차트 : fecal score chart dog
        main_color_code = data['main-color-code']
        sub_color_code = data['sub-color-code']

        query = f"                                      "
        
        response = """
            model="gpt-4o"
            response_format={"type": "json_object"},
            messages=[
                {"role":"system",
                "content": "You are a helpful assistant designed to ouput JSON."},
                {"role": "user", "content": """f"{query}""""},
            ]
            """
        
        return query, response