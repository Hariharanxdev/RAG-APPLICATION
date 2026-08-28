from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL
from prompts import create_prompt


class Chatbot:
    """Generate answers using Groq and retrieved context."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured in the .env file."
            )

        self.client = Groq(api_key=GROQ_API_KEY)

        # Store conversation history
        self.conversation_history = []

    def generate_answer(self, question, retrieved_chunks):
        """Generate an answer using retrieved context and conversation history."""

        if not retrieved_chunks:
            answer = (
                "I couldn't find this information "
                "in the provided documents."
            )

            self.conversation_history.append({
                "question": question,
                "answer": answer
            })

            return answer

        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"Source: {chunk['source']}\n"
                f"Page: {chunk['page']}\n"
                f"Chunk: {chunk['chunk_id']}\n"
                f"Content: {chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        # Build conversation context
        previous_conversation = ""

        for item in self.conversation_history:
            previous_conversation += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n\n"
            )

        prompt = create_prompt(
            context=context,
            question=question
        )

        if previous_conversation:
            prompt = f"""
Previous Conversation:
{previous_conversation}

{prompt}
"""

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG assistant. "
                        "Answer using the provided document context. "
                        "Use previous conversation only to understand "
                        "references and follow-up questions. "
                        "Do not invent information."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        # Save conversation
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })

        return answer

    def clear_history(self):
        """Clear the current conversation history."""

        self.conversation_history = []