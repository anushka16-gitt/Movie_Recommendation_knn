import os
from recommender import read_data, clean_data, make_matrix, suggest_movies
from sklearn.neighbors import NearestNeighbors

def main():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    movies, ratings = read_data(folder)
    print(f"Movies: {len(movies)}, Ratings: {len(ratings)}")

    ratings = clean_data(ratings)
    print(f"Ratings after filtering: {len(ratings)}")

    matrix, title_list = make_matrix(ratings, movies)

    knn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn_model.fit(matrix.values)

    while True:
        name = input("Enter a movie title (or 'quit' to exit): ").strip()

        if name.lower() in ("quit", "exit", "q"):
            print("Closing")
            break

        if not name:
            continue

        found, suggestions = suggest_movies(name, matrix, knn_model, title_list)

        if found is None:
            print(f"No match found for '{name}'. Try again.\n")
        else:
            print(f"\nRecommendations for: {found}")
            for i, (title, score) in enumerate(suggestions, 1):
                print(f"{i}.{title} (similarity: {score:.2%})")
            print()


if __name__ == "__main__":
    main()
