# google AI API Key : AIzaSyByS04-dE_qqitVHzJBDHa84nbj8R69oe4



from dotenv import load_dotenv
import os

import keras
import keras_nlp
# .env 파일을 로드하여 환경 변수를 설정합니다.
load_dotenv()


# print("dotenv 로드 완료")

# 환경 변수에 설정된 키 확인
# print("환경 변수에 설정된 키:")
# for key in os.environ:
#     print(key)

os.environ["KERAS_BACKEND"] = "tensorflow" # torch or tensor could be work

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

