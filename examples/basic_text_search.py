from vectordb import VectorDB, DummyEmbeddings

def main():
    dim = 16
    embed = DummyEmbeddings(dim)
    db = VectorDB(dim, embeddings=embed)

    texts = [
        "I love machine learning",
        "Pasta is delicious",
        "Neural networks learn patterns",
        "Football is my hobby",
        "Building a vector database"
    ]

    for t in texts:
        db.add_text(t, {"source": "demo"})

    res = db.search_text("deep learning", k=3)
    for r in res:
        print(r)

if __name__ == "__main__":
    main()
