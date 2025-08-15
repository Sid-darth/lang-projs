""" try out tiktoken """
import tiktoken

input_text = input("enter something: ")

# set encoding
enc = tiktoken.get_encoding("r50k_base")

# encode text
tokens = enc.encode(input_text)
print(f"Input: {input_text}, Len of tokens: {len(tokens)}; Tokens: {tokens}")

# decode tokens to retrieve text
text = enc.decode(tokens)
print(f"Decoded text : s{text}")