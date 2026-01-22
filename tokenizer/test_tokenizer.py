from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

text = "hello there hello world"
encoded = tokenizer.encode(text)

print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
