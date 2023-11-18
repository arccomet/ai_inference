import pysbd
import nltk
# nltk.download('punkt')

seg = pysbd.Segmenter(language="en", clean=True)

text = "Hello. YES. Please. Thank you. very cool cool cool cool cool cool, and cool"

result = seg.segment(text)

print(result)


def break_into_chunks(text, word_count):
    sentences = nltk.sent_tokenize(text)
    print(sentences)
    chunks = []
    current_chunk = []

    word_count_so_far = 0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        print(sentence)
        print(len(current_chunk))
        if word_count_so_far + len(words) <= word_count or len(current_chunk) == 0:
            current_chunk.append(sentence)
            word_count_so_far += len(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk.clear()
            word_count_so_far = 0

            # Put into next chunk
            current_chunk.append(sentence)

    # Add the last chunk if it's not empty
    if len(current_chunk) > 0:
        chunks.append(' '.join(current_chunk))

    print(len(''.join(chunks)))

    return chunks


def old_break_into_chunks(text, word_count):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []

    word_count_so_far = 0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        if word_count_so_far + len(words) <= word_count:
            current_chunk.append(sentence)
            word_count_so_far += len(words)
        else:
            if len(current_chunk) > 0:  # Ensures we have a complete sentence
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                word_count_so_far = len(words)

    # Add the last chunk if it's not empty
    if len(current_chunk) > 0:
        chunks.append(' '.join(current_chunk))

    print(len(''.join(chunks)))

    return chunks


print(">>", break_into_chunks(text, 10))