import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from typing import List, Dict
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Validate required environment variables
required_env_vars = [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

class MultiIndexHealthcareBot:
    def __init__(self):
        self.model = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "800"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Initialize all indexes
        self.indexes = {
            'food': pc.Index('food-unite'),
            'health_qa': pc.Index('health-qa'),
            'wellness': pc.Index('wellness-chat'),
            'guidelines': pc.Index('healthcare-guidelines')
        }

        # Configure weights for different indexes
        self.index_weights = {
            'food': 1.0,
            'health_qa': 1.0,
            'wellness': 1.0,
            'guidelines': 1.0
        }

        # 일반적인 의료 정보 제공을 위한 프롬프트
        self.general_prompt = """
        당신은 친절하고 전문적인 의료 정보 제공 챗봇입니다.
        답변 구조와 형식:
        1. 답변의 구성
        - 첫 문단은 질문에 대한 정의나 개요를 설명합니다
        - 중간 문단은 구체적인 설명과 예시를 제공합니다
        - 마지막 문단은 실용적인 조언이나 주의사항을 담습니다
        
        2. 문단 작성
        - 각 문단은 2-3개의 문장으로 구성합니다
        - 문단과 문단 사이에 빈 줄을 넣어 구분합니다
        - 문장은 자연스럽게 이어지도록 합니다
        
        3. 설명 방식
        - 일반인이 이해하기 쉬운 쉬운 말로 설명합니다
        - 전문 용어는 괄호 안에 쉬운 설명을 함께 씁니다
        - 의학적 수치나 단위는 이해하기 쉽게 풀어서 설명합니다
        - markdown 강조표시는 사용하지 않습니다
        """

        # 감정/심리 상담을 위한 프롬프트
        self.counseling_prompt = """
        당신은 따뜻하고 친근한 상담사입니다. 
        다음과 같은 방식으로 상담해주세요:

        1. 답변 구조
        - 각 문단은 2-3개의 연관된 문장으로 구성하세요
        - 문단과 문단 사이에는 반드시 빈 줄을 넣어 구분하세요
        - 첫 문단은 공감과 이해를 표현하세요
        - 중간 문단들은 구체적인 제안을 담으세요
        - 마지막 문단은 희망적인 메시지로 마무리하세요

        2. 답변 스타일
        - 친근하고 자연스러운 말투를 사용하세요
        - markdown 강조표시는 사용하지 마세요
        - 각 제안은 '첫째', '둘째' 등으로 구분하세요
        - 전문가 상담이 필요한 경우 자연스럽게 마지막 부분에서 언급하세요

        3. 내용 구성
        - 내담자의 감정이 자연스럽다는 것을 알려주세요
        - 2-3가지 실천 가능한 구체적인 제안을 해주세요
        - 조언은 구체적이되 부담스럽지 않게 해주세요
        - 전체 답변은 4-5개의 짧은 문단으로 구성하세요
        """

    def get_embeddings(self, text: str) -> List[float]:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding

    def classify_query(self, query: str) -> Dict[str, float]:
        query_lower = query.lower()
        weights = self.index_weights.copy()
        
        # 감정/심리 상담 관련 키워드
        self.is_counseling_query = any(word in query_lower for word in [
            '우울', '불안', '분노', '스트레스', '화나', '짜증', '걱정',
            '불면', '수면', '괴로', '힘들', '고민', '무기력', '외로',
            '불행', '슬픔', '우울증', '감정', '기분'
        ])
        
        if self.is_counseling_query:
            weights['wellness-chat'] = 1.5
            weights['health-qa'] = 1.0
            
        # 질병 가이드라인 관련 키워드
        if any(word in query_lower for word in [
            '당뇨', '고혈압', '이상지질혈증', 'copd', '천식', '콩팥병', 
            '우울증', '심방세동', '골다공증', '영양소', '기준'
        ]):
            weights['healthcare-guidelines'] = 1.5
            weights['health-qa'] = 1.2
        
        # 일반 건강 Q&A 관련 키워드
        if any(word in query_lower for word in [
            '증상', '치료', '병원', '약', '검사', '아프', '통증',
            '부작용', '질환', '예방'
        ]):
            weights['health-qa'] = 1.5
            weights['healthcare-guidelines'] = 1.2
        
        # 식품영양 관련 키워드
        if any(word in query_lower for word in [
            '음식', '식단', '영양', '먹', '식사', '다이어트', '요리',
            '칼로리', '단백질', '탄수화물', '지방'
        ]):
            weights['food-unite'] = 1.5
        
        return weights

    def search_all_indexes(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.get_embeddings(query)
        weights = self.classify_query(query)
        
        all_results = []
        for index_name, index in self.indexes.items():
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                for match in results.matches:
                    match.score *= weights[index_name]
                    if not hasattr(match, 'metadata'):
                        match.metadata = {}
                    match.metadata['source'] = index_name
                    all_results.append(match)
                
            except Exception as e:
                print(f"Error querying {index_name}: {str(e)}")
                continue

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k * 2]

    def format_context(self, relevant_docs, min_score: float = 0.5) -> str:
        context = "\n관련 정보:\n"
        for doc in relevant_docs:
            if hasattr(doc, 'metadata') and doc.score >= min_score:
                text = doc.metadata.get('text', 'No content available')
                context += f"{text}\n"
        return context

    def generate_response(self, query: str) -> str:
        try:
            relevant_docs = self.search_all_indexes(query)
            context = self.format_context(relevant_docs)
            
            if hasattr(self, 'is_counseling_query') and self.is_counseling_query:
                system_prompt = self.counseling_prompt
                user_prompt = f"Context: {context}\n\nQuestion: {query}\n\n사용자의 감정에 공감하면서 구체적인 도움이 되는 답변을 제공해주세요."
            else:
                system_prompt = self.general_prompt
                user_prompt = f"Context: {context}\n\nQuestion: {query}\n\n위 질문에 대해 제공된 컨텍스트를 바탕으로 자연스럽고 이해하기 쉽게 설명해주세요."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

# 챗봇 관련 라우트
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({"error": "No question provided"}), 400
        
        chatbot = MultiIndexHealthcareBot()
        response = chatbot.generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 영양소 계산기 관련 라우트
@app.route('/nutrition')
def nutrition():
    return render_template('nutrition.html')

@app.route('/api/search-foods', methods=['POST'])
def search_foods():
    try:
        query = request.json.get('query')
        print(f"Received search query: {query}")
        
        if not query:
            return jsonify({"error": "No search query provided"}), 400

        # OpenAI 임베딩 생성
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding = embedding_response.data[0].embedding  # 여기서 embedding 생성

        # Pinecone 검색
        food_index = pc.Index('food-unite')
        results = food_index.query(
            vector=embedding,  # 생성된 embedding 사용
            top_k=10,
            include_metadata=True
        )

        # 결과 로깅
        print(f"Found {len(results.matches)} results")
        for match in results.matches:
            print(f"ID: {match.id}")
            print(f"Metadata: {match.metadata}")
            print("---")

        # 결과 형식화
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "name": match.metadata.get('name_ko', '검색결과 없음'),
                "score": match.score
            })

        return jsonify(formatted_results)
    except Exception as e:
        print(f"Error in search_foods: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/food-details', methods=['POST'])
def get_food_details():
    try:
        food_id = request.json.get('food_id')
        print(f"요청된 음식 ID: {food_id}")  # 로깅
        
        food_index = pc.Index('food-unite')
        vector_data = food_index.fetch([food_id])
        
        print(f"가져온 데이터: {vector_data}")  # 로깅
        
        if food_id not in vector_data.vectors:
            return jsonify({"error": "음식을 찾을 수 없습니다"}), 404

        metadata = vector_data.vectors[food_id].metadata
        
        # 영양소 데이터 구성
        nutrient_data = {
            "열량": {
                "value": float(metadata.get('nutrient_energy', 0)),
                "unit": metadata.get('nutrient_energy_unit', 'kcal')
            },
            "단백질": {
                "value": float(metadata.get('nutrient_protein', 0)),
                "unit": metadata.get('nutrient_protein_unit', 'g')
            },
            "지방": {
                "value": float(metadata.get('nutrient_fat', 0)),
                "unit": metadata.get('nutrient_fat_unit', 'g')
            },
            "탄수화물": {
                "value": float(metadata.get('nutrient_carbohydrate', 0)),
                "unit": metadata.get('nutrient_carbohydrate_unit', 'g')
            }
        }
        
        return jsonify({"nutrients": nutrient_data})
    except Exception as e:
        print(f"에러 발생: {str(e)}")  # 로깅
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)