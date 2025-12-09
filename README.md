# Book Recommendation Ranking Solution

**Final Score: 0.889 NDCG@20**

## Approach

### Model
- **CatBoost Ranker** with PairLogitPairwise loss
- Ensemble of 5 models with different random seeds
- **Rank-based averaging** (more robust than score averaging)

### Negative Sampling Strategy
- **150 negatives per user** from top-800 popular books
- Key insight: negatives must have rich features (popularity, interactions, etc.)
- Random/unpopular negatives performed poorly (0.59 on public)

### Features (16 total)

| Feature | Description |
|---------|-------------|
| times_read | Number of times book was read |
| total_interactions | Total user-book interactions |
| read_rate | Ratio of reads to interactions |
| avg_user_rating | Average rating given to book |
| unique_users | Number of unique users who interacted |
| user_books_read | How many books user has read |
| user_read_rate | User's read ratio |
| user_avg_rating | User's average rating |
| publication_year | Book publication year |
| avg_rating | Book's global average rating |
| book_age | Years since publication |
| age | User age |
| gender | User gender |
| genre_match | Overlap between user's preferred genres and book genres |
| popularity | log(1 + total_interactions) |
| rating_compatibility | 1 - |book_rating - user_avg_rating| / 5 |

## Key Insights

1. **Negative sampling source matters more than quantity**
   - Popular books (top-800) as negatives: 0.88+
   - Random books as negatives: 0.59 (disaster)
   - Candidates distribution as negatives: 0.59

2. **Lower validation != always better public score**
   - Too low validation means model doesn't learn
   - Optimal: validation ~0.85-0.87

3. **Candidate books distribution**
   - Only 4.6% from top-800
   - 34% cold books (no training data)
   - 42.7% rank 5000+

4. **Rank-based ensemble > Score-based ensemble**
   - More robust to score scale differences
   - Gave +0.2% improvement

## Files

- `solution.ipynb` - Final solution notebook
- `data/` - Dataset directory
  - `train.csv` - Training interactions
  - `books.csv` - Book metadata
  - `users.csv` - User metadata
  - `book_genres.csv` - Book-genre mappings
  - `candidates.csv` - Candidate books for each test user
  - `targets.csv` - Test user IDs

## How to Run

```bash
# Install dependencies
pip install pandas numpy catboost scikit-learn

# Run notebook
jupyter notebook solution.ipynb
```

## Results Progression

| Experiment | Val NDCG | Public NDCG |
|------------|----------|-------------|
| Baseline (15 neg) | 0.93 | 0.86 |
| 100 neg | 0.87 | 0.88 |
| 150 neg + ens3 | 0.85 | 0.887 |
| 150 neg + rank ens5 | 0.84 | **0.889** |
