import json


class Retrieval:
    def __init__(self, data_path: str = "./data/newjeans.json"):
        with open(data_path, "r") as file:
            data: dict[str, str] = json.load(file)
            self.data = data

    def retrieve(self, query: str) -> str | None:
        # ####
        # 이 주석을 지우고 다음 코드를 완성해보자.
        # query 는 유저의 발화문이다. self.data 에는 json 값이 key-value 형태로 저장되어 있다.
        # 만약 query 가 json 의 key 값을 키워드로 포함하고 있으면 "{key}: {value}" 를 str 으로 리턴하자.
        # 아니라면 None 을 리턴하자.
        #
        # 예시 1:
        # query: who is minji? ("minji" is in json key)
        # @ return "minji: The leader of NewJeans..."
        # 
        #
        # 예시 2:
        # query: this is an empty query (None of the json key matches)
        # @ return None
        # ###
        for key in self.data:
            if key.lower() in query.lower():
                return f"{key}: {self.data[key]}"
        return None


    def print_data(self) -> None:
        print(self.data)
