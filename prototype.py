import requests
import json
import tiktoken
import fasttext
import fasttext.util
import numpy as np
import re
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

class Assistant():
    API_URL = "http://localhost:1234/v1/chat/completions"
    HEADERS = {"Content-Type": "application/json"}
    MODEL_NAME = "gemma-3-1b-it"
    MAX_CONTEXT_TOKENS = 16000

    def __init__(self):
        self.faq_entries = self.load_faq('faq.json')
        self.faq_questions = [entry['question'] for entry in self.faq_entries]

        # Загружаем FastText модель
        fasttext.util.download_model('ru', if_exists='ignore')
        self.ft_model = fasttext.load_model('cc.ru.300.bin')

        # Строим эмбеддинги вопросов из FAQ
        self.question_embeddings = np.array([self.ft_model.get_sentence_vector(q) for q in self.faq_questions])

    def load_faq(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean_input(self, text: str) -> str:
        # Удаление ESC-последовательностей и "битых" символов
        text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        return text.strip()

    def count_tokens(self, text: str) -> int:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def find_relevant_entries(self, user_question: str, top_n=10, threshold=0.55):
        user_embedding = self.ft_model.get_sentence_vector(user_question).reshape(1, -1)
        sim_scores = cosine_similarity(user_embedding, self.question_embeddings)[0]

        top_indices = [i for i, score in enumerate(sim_scores) if score > threshold]
        sorted_top = sorted(top_indices, key=lambda i: sim_scores[i], reverse=True)

        return [self.faq_entries[i] for i in sorted_top[:top_n]]

    def build_context(self, entries: List[Dict]) -> str:
        context = []
        token_count = 0
        for entry in entries:
            entry_text = f"Вопрос: {entry['question']}\nОтвет: {entry['answer']}"
            entry_tokens = self.count_tokens(entry_text)
            if token_count + entry_tokens > self.MAX_CONTEXT_TOKENS:
                break
            context.append(entry_text)
            token_count += entry_tokens
        return "\n".join(context)

    def ask_with_faq(self, question: str) -> str:
        relevant = self.find_relevant_entries(question)
        context = self.build_context(relevant)

        messages = [
            {"role": "system", "content": f"Отвечай используя только:\n{context}"},
            {"role": "user", "content": question}
        ]

        try:
            resp = requests.post(self.API_URL, headers=self.HEADERS, json={
                "model": self.MODEL_NAME,
                "messages": messages,
                "temperature": 0.25,
                "max_tokens": 12000
            }, timeout=10)
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Ошибка: {str(e)}"

    def get_answer(self, question: str):
        try:
            question = self.clean_input(question)
            return self.ask_with_faq(question)
        except Exception as e:
            print(f"Критическая ошибка: {str(e)}")
            return "Не удалось обработать запрос"


if __name__ == "__main__":
    assist = Assistant()
    while True:
        q = input("\nYou: ")
        ans = assist.get_answer(q)
        print(f"\nAss: {ans}")
