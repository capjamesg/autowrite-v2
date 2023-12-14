import re

import nltk
import pygtrie
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
# from spellchecker import SpellChecker

from pysurprisal import Surprisal

SUFFIX_PREFIX_CHARS = (",", ".", "!", "?")
PRIORITY_BOOST = 2

# if exists, open indices.pickle
import pickle
import os

if os.path.exists("indices.pickle"):
    with open("indices.pickle", "rb") as f:
        indices = pickle.load(f)
else:
    indices = {}

def get_words(url):
    print(url)
    words = ""

    if url.endswith(".md") or url.endswith(".txt"):
        with open(url, "r") as f:
            post = f.read()

        words += post
    else:
        post = (
            BeautifulSoup(requests.get(url).text, "html.parser")
            .get_text()
            .replace("\n", " ")
        )

        try:
            post_text = (
                BeautifulSoup(post, "html.parser")
                .get_text()
                .replace("\n", " ")
                #
            )
            words += post_text
        except Exception as e:
            print(e)
            pass

    # rstrp all words
    words = " ".join([re.sub(r"[^a-zA-Z0-9]+", " ", word) for word in words.split(" ")])

    # remove words < 3 chars and > 20 chars
    words = " ".join(
        [word for word in words.split(" ") if len(word) > 3 and len(word) < 20]
    )

    bigrams = nltk.bigrams(words.split(" "))

    # lowercase all words
    bigrams = [(word[0], word[1]) for word in bigrams]

    trigrams = nltk.trigrams(words.split(" "))
    trigrams = [(word[0], word[1], word[2]) for word in trigrams]

    quadgrams = nltk.ngrams(words.split(" "), 4)
    quadgrams = [(word[0], word[1], word[2], word[3]) for word in quadgrams]

    return words, bigrams, trigrams, quadgrams


def build_surprisal_index(words):
    surprisals = Surprisal(words)

    surprisals.calculate_surprisals()

    # print(surprisals.surprisals)

    vocab = surprisals.surprisals.keys()

    return surprisals, vocab


def autocomplete(query, trie):
    N = 10

    # distances = {}

    # for word in vocab:
    #     word_distance = distance(word, query)
    #     distances[word] = word_distance

    # print(sorted(distances.items(), key=lambda x: x[1])[0:N])

    results = trie.keys(prefix=query)

    # order by surprisals
    results = sorted(results, key=lambda x: trie[x])

    return results[0:N]

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

tuned_words = {}

@app.route("/surprisal")
def surprisal():
    query = request.args.get("query")
    previous_word = request.args.get("previous_word")
    rules = request.args.get("rules") or ""
    urls = request.args.get("urls").split(",") or []
    all_words_written_in_document = request.args.get("all_words") or ""
    
    # remove last
    all_words_written_in_document = " ".join(all_words_written_in_document.split(" ")[0:-1])

    # rules is X-Y,X-Y, etc.
    rules = rules.split(",")
    rules = [rule.split("-") for rule in rules]

    for rule in rules:
        if query == rule[0]:
            return jsonify({"previous_word": rule[1], "next_word_predictions": ["[OVERWRITE_PREVIOUS_WORD]"]})

    all_surprisals = {}
    vocabulary = set()
    all_bigrams = []
    all_quadgrams = []

    urls = [url for url in urls if url != ""]

    if len(urls) == 0:
        # get surprisals for all posts in /Users/james/blog/james-coffee-blog/_posts/*.md
        import glob

        urls = glob.glob("/Users/james/blog/james-coffee-blog/_posts/*.md")# + ["/Users/james/src/pysurprisal/nytimes_news_articles.txt"]

        # get 20 urls
        # urls = urls[0:20]

    all_words = ""
    all_bigrams = []
    ny_words = {}

    if "all" not in indices:
        for u in urls:
            words, bigrams, trigrams, _ = get_words(u)

            all_words += words
            all_bigrams += bigrams

            # if u.endswith("nytimes_news_articles.txt"):
            #     for word in words.split(" "):
            #         ny_words[word] = True

    urls = ["all"]

    for url in urls:
        if url not in indices:
            print("Building index for", url)
            
            surprisals, vocab = build_surprisal_index(all_words)

            print("Calculating priority for", url)

            # if url.endswith("?priority"):
            surprisals.surprisals = {
                k: v * PRIORITY_BOOST for k, v in surprisals.surprisals.items() if k not in ny_words
            }

            print("Calculating arrangement for", url)

            # remove words w/ less than three letters
            vocab = [word for word in vocab if len(word) > 3]
            surprisals.surprisals = {
                k: v for k, v in surprisals.surprisals.items() if len(k) > 3
            }

            all_surprisals.update(surprisals.surprisals)
            vocabulary.update(vocab)

            trie = pygtrie.CharTrie()

            # for bigram in all_bigrams:
            #     bigram = " ".join(bigram)
            #     trie[bigram] = surprisals.surprisals.get(
            #         bigram[0], 0
            #     ) - surprisals.surprisals.get(bigram[1], 0)

            for word in vocab:
                trie[word] = surprisals.surprisals.get(word, 0)

            indices[url] = {
                "bigrams": all_bigrams,
                "trigrams": trigrams,
                "surprisals": surprisals,
                "vocab": vocab,
                "trie": trie,
            }

            # pickle indices
            import pickle

            with open("indices.pickle", "wb") as f:
                pickle.dump(indices, f, pickle.HIGHEST_PROTOCOL)
        else:
            bigrams = indices[url]["bigrams"]
            trigrams = indices[url]["trigrams"]
            surprisals = indices[url]["surprisals"]
            vocab = indices[url]["vocab"]

            all_surprisals.update(surprisals.surprisals)
            vocabulary.update(vocab)
            all_bigrams += bigrams
            trie = indices[url]["trie"]

    # tune surprisals with all_words_written_in_document
    for word in all_words_written_in_document.split(" "):
        if word not in tuned_words:
            print("Tuning", word)
            all_surprisals[word] = all_surprisals.get(word, 1) / 3

            tuned_words[word] = True

            trie[word] = 0
            
    trie = autocomplete(query, trie)

    # try to correct
    # if previous_word not in surprisals.surprisals:
    #     spell = SpellChecker()

    #     # add all surprisal words to spellcheck that appear more than 3 times (to prevent typos from being populated as correct words)
    #     word_counts = surprisals.counts

    #     words_with_counts = [word for word in word_counts if word_counts[word] > 1]

    #     spell.word_frequency.load_words(words_with_counts)

    #     corrected_word = spell.correction(previous_word)

    #     if corrected_word in surprisals.surprisals:
    #         previous_word = corrected_word

    #     # if upper first char has lower surprisal, use that
    #     if surprisals.surprisals.get(
    #         previous_word.capitalize(), 100
    #     ) < surprisals.surprisals.get(previous_word, 100):
    #         previous_word = previous_word.capitalize()
    #     else:
    #         previous_word = corrected_word


    # return nothing if previous_word < 3 chars
    # if len(previous_word) < 3:
    #     return jsonify({"previous_word": previous_word, "next_word_predictions": []})

    # if previous word is in surprisals, use that
    if previous_word and previous_word.lower() in all_surprisals:
        previous_word = previous_word
        trie = [previous_word]

    return jsonify(
        {
            "next_word_predictions": trie,
            "previous_word": previous_word,
        }
    )


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
