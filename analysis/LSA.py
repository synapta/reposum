from sklearn.feature_extraction.text import CountVectorizer

def feed_data(input_data):
    for string in string_list:
        yield string[0]

cv = CountVectorizer(stop_words="english", analyzer="word")




tr = cv.fit_transform(read_strings(abstracts))
print(tr)
