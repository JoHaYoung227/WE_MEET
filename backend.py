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
CORS(app)  # For allowing cross-origin requests

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
        self.max_tokens = int(os.getenv("MAX_TOKENS", "500"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Initialize all indexes
        self.indexes = {
            'food': pc.Index('food-unite'),
            'health_qa': pc.Index('health-qa'),
            'wellness': pc.Index('wellness-chat'),
            'guidelines': pc.Index('healthcare-guidelines')
        }

        # Configure weights for different indexes based on query type
        self.index_weights = {
            'food': 1.0,
            'health_qa': 1.0,
            'wellness': 1.0,
            'guidelines': 1.0
        }

        self.system_prompt = """
        당신은 의료 정보를 제공하는 헬스케어 어시스턴트입니다.

        다음 데이터베이스의 정보를 기반으로 답변합니다:
        1. 의료 가이드라인: 주요 질병에 대한 의학적 지침
        2. 건강 Q&A: 실제 의료 상담 사례
        3. 정신건강 상담: 심리/정서 관련 전문가 상담
        4. 식품영양 정보: 영양 및 식단 관련 데이터

        응답 지침:
        1. 데이터베이스의 정보와 일반적인 의학 지식을 자연스럽게 통합하여 응답하세요.
        2. 의학적 정보를 제시할 때는 근거와 출처를 함께 언급하세요.
        3. 전문적인 판단이 필요한 경우, 의료진과의 상담을 추천하되 대화의 맥락을 자연스럽게 유지하세요.
        4. 모든 답변은 공감적이고 이해하기 쉬운 한국어로 작성하세요.
        5. 일반적인 의학 지식을 설명할 때는 [GENERAL MEDICAL KNOWLEDGE] 표시를 사용하세요.
        """

    def get_embeddings(self, text: str) -> List[float]:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding

    def classify_query(self, query: str) -> Dict[str, float]:
        query_lower = query.lower()
        weights = self.index_weights.copy()
        
        # 질병 가이드라인 관련 키워드
        if any(word in query_lower for word in [
            '당뇨', '고혈압', '이상지질혈증', 'copd', '천식', '콩팥병', 
            '우울증', '심방세동', '골다공증', '영양소', '기준'
        ]):
            weights['healthcare-guidelines'] = 1.5
            weights['health-qa'] = 1.2
        
        # 정신건강 상담 관련 키워드
        if any(word in query_lower for word in [
            '우울', '불안', '분노', '스트레스', '화나', '짜증', '걱정',
            '불면', '수면', '괴로', '힘들', '고민'
        ]):
            weights['wellness-chat'] = 1.5
            weights['health-qa'] = 1.0
        
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
        """Search across all indexes and combine results"""
        query_embedding = self.get_embeddings(query)
        weights = self.classify_query(query)
        
        print(f"\n=== 검색 시작 ===")
        print(f"Query: {query}")
        print(f"Weights: {weights}")
        
        all_results = []
        for index_name, index in self.indexes.items():
            try:
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                print(f"\n--- {index_name} 검색 결과 ---")
                if len(results.matches) == 0:
                    print("결과 없음")
                
                for match in results.matches:
                    print(f"\nScore: {match.score}")
                    if hasattr(match, 'metadata'):
                        print(f"메타데이터 키: {match.metadata.keys()}")
                        print(f"전체 메타데이터: {match.metadata}")
                    else:
                        print("메타데이터 없음")
                    
                # Add source and apply weight to scores
                for match in results.matches:
                    match.score *= weights[index_name]
                    if not hasattr(match, 'metadata'):
                        match.metadata = {}
                    match.metadata['source'] = index_name
                    all_results.append(match)
                
            except Exception as e:
                print(f"Error querying {index_name}: {str(e)}")
                continue
        
        print("\n=== 최종 결과 ===")
        print(f"총 {len(all_results)}개의 결과 찾음")
        
        # Sort by adjusted scores and take top results
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k * 2]

    def format_context(self, relevant_docs, min_score: float = 0.5) -> str:
        """Format retrieved documents into context string"""
        context = "\nRelevant information:\n"
        for i, doc in enumerate(relevant_docs, 1):
            if hasattr(doc, 'metadata') and doc.score >= min_score:
                source = doc.metadata.get('source', 'unknown')
                text = doc.metadata.get('text', 'No content available')
                context += f"{i}. [{source}] (Score: {doc.score:.3f}) {text}\n"
        return context

    def generate_response(self, query: str) -> str:
        try:
            relevant_docs = self.search_all_indexes(query)
            context = self.format_context(relevant_docs)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {query}\n\nProvide a comprehensive answer using the provided context. For each point you make, indicate which source it comes from. If you need to add any general medical knowledge not found in the context, clearly mark it as [GENERAL MEDICAL KNOWLEDGE]."}
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

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
