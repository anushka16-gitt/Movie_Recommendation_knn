# Movie Recommendation System

A movie recommendation system built using K-Nearest Neighbors (KNN) algorithm and collaborative filtering. It suggests similar movies based on user rating patterns from the MovieLens dataset.

## How It Works

1. Loads movie and rating data from the MovieLens dataset
2. Filters out users and movies with too few ratings to reduce noise
3. Builds a movie-user rating matrix (rows = movies, columns = users)
4. Fits a KNN model using cosine similarity on the matrix
5. Takes a movie name as input and returns the most similar movies

## Dataset

Uses the [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) dataset:
- 100,836 ratings from 610 users across 9,742 movies

Download the dataset, extract it, and place the CSV files in the `data/` folder:

```
Movie_Recommendation/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── recommender.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- pandas
- numpy
- scikit-learn
- fuzzywuzzy
- python-Levenshtein

## Usage

```bash
python main.py
```

### Example

```
Movies: 9742, Ratings: 100836
Ratings after filtering: 74673
Enter a movie title (or 'quit' to exit): Toy Story

Recommendations for: Toy Story (1995)
1.Jurassic Park (1993) (similarity: 65.87%)
2.Forrest Gump (1994) (similarity: 63.94%)
3.Toy Story 2 (1999) (similarity: 62.37%)
4.Star Wars: Episode IV - A New Hope (1977) (similarity: 61.62%)
5.Shrek (2001) (similarity: 61.39%)
```

## Project Structure

| File | Description |
|------|-------------|
| `recommender.py` | Contains functions for loading data, filtering, building the matrix, and generating recommendations |
| `main.py` | Entry point that runs the interactive recommendation loop |
| `requirements.txt` | Python dependencies |

## Algorithm

- **Collaborative Filtering**: Recommends movies based on how users rated them, not based on movie content
- **KNN with Cosine Similarity**: Finds movies with the most similar rating vectors across all users
- **Fuzzy Matching**: Allows users to type approximate movie titles using Levenshtein distance

## Built With

- Python
- scikit-learn (KNN)
- pandas (data handling)
- fuzzywuzzy (string matching)
# Movie_Recommendation_knn
