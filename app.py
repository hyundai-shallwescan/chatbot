from flask import Flask, request, jsonify
import psycopg2
import os
import re
import openai 

from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()
app = Flask(__name__)

# 데이터베이스 연결
DB_CONFIG = {
    "host": os.getenv('DB_HOST'),
    "port": os.getenv('DB_PORT'),
    "dbname": os.getenv('DB_NAME'),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD')
}

def get_db_connection():
    """PostgreSQL 데이터베이스에 연결"""
    return psycopg2.connect(**DB_CONFIG)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/match', methods=['POST'])
def match_products():
    try:
        user_query = request.args.get('query')

        match_threshold = 0.3
        match_count = 2

        # 사용자 메세지 임베딩
        embedded_query = openai.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        ).data[0].embedding

        connection = get_db_connection()
        cursor = connection.cursor()

        # match_products 함수 호출
        query = """
        SELECT id, title, price, thumbnail_image, description, similarity 
        FROM match_products(%s::VECTOR(1536), %s, %s);
        """
        cursor.execute(query, (embedded_query, match_threshold, match_count))

        results = cursor.fetchall()

        products = []
        for result in results:
            product = {
                "id": result[0],
                "title": result[1],
                "price": result[2],
                "thumbnailImage": result[3],
                "description": result[4],
                "similarity": result[5]
            }
            products.append(product)

        contents = []
        for result in results:
            contents.append(result[4])  # description 추출

        text = ' '.join(contents)
        new_text = re.sub(r"\s+|\n", " ", text)

        # 프롬프트 생성
        user_prompt = f"Document:\n{new_text}\n\nQ: {user_query}\nA:"

        # GPT API 호출
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question truthfully using document"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            top_p=1.0,
            max_tokens=8000
        )

        # gpt 응답 내용 
        gpt_response = completion.choices[0].message.content

        return jsonify({
            "message": gpt_response,
            "product": products
        }), 200

    except psycopg2.Error as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == '__main__':
    app.run(debug=True)