from transformers import DPRContextEncoder, DPRQuestionEncoder, T5ForConditionalGeneration, T5Tokenizer
import faiss
import torch
import json

# Load the dataset
with open('dataset.json') as f:
    dataset = json.load(f)

# Load DPR models and tokenizer
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
generator = T5ForConditionalGeneration.from_pretrained('t5-small')

# Tokenize and encode the context passages
contexts = [item["context"] for item in dataset]
context_embeddings = context_encoder(tokenizer(contexts, return_tensors='pt', padding=True, truncation=True)['input_ids']).pooler_output

# Indexing with FAISS
index = faiss.IndexFlatL2(context_embeddings.shape[1])
index.add(context_embeddings.detach().cpu().numpy())

# Function to get response
def get_response(question):
    question_embedding = question_encoder(tokenizer(question, return_tensors='pt')['input_ids']).pooler_output.detach().cpu().numpy()
    _, I = index.search(question_embedding, k=5)
    retrieved_contexts = [contexts[i] for i in I[0]]

    # Concatenate retrieved contexts and generate answer
    input_text = " ".join(retrieved_contexts) + " question: " + question
    input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    outputs = generator.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Test the function
if __name__ == "__main__":
    question = "What information is provided in the policy booklet?"
    print(get_response(question))
