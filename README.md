# CineSage Insights - A Predictive and Visualization Tool for Filmmakers
## CSE 6242 A Team 20

CineSage Insights is a project designed to redefine the movie production planning process by integrating data analysis and visualizations. This tool is specifically tailored for filmmakers, offering a novel approach to making data-driven decisions. It factors in various aspects such as plot premise, budget, star cast, release time-frame, and targeted geographies, enabling a comprehensive prediction and visualization of a movie's potential success and revenue.

### Live Demo
https://cinesageinsights.azurewebsites.net/

### Project Structure
The repository is organized as follows:

- `.github/workflows/main.yml`: Contains the workflow configuration for GitHub actions.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `Cluster Analysis/`: This directory includes notebooks and models for cluster analysis.
    - `ClusterUserGenre.ipynb`: Jupyter notebook for user-genre cluster analysis.
    - `gmm_model.joblib`: Saved Gaussian Mixture Model.
    - `kmeans_model.joblib`: Saved K-Means Model.
- `dataset/`: Contains various datasets used in the project.
    - `cleaned_dataset/`: Cleaned datasets for training and testing.
    - `created/`: Datasets created during the project.
    - `dataset_download_links.json`: JSON file with links to download datasets.
    - `downloaded/`: Initially downloaded datasets.
    - `download_large_csv.py`: Script to download large CSV files.
    - `README.md`: Information about the datasets.
- `Distributed CSVs/`: Contains distributed CSV files for box office collection and missing IDs.
- `Flask/`: Flask application for the web-based interface of the tool.
    - `app.py`: Main Flask application file.
    - `best_model_overall_revenue.pth`: Saved best model for revenue prediction.
    - `genre_prediction.py`: Script for genre prediction.
    - `lib/`: Libraries used in the Flask application.
    - `static/`: Static files like JavaScript and images for the Flask app.
    - `templates/`: HTML templates for the web interface.
    - `utils.py`: Utility functions for the Flask app.
- `model_experiments/`: Contains notebooks and data for various model experiments.
- `report/`: Includes report files and plots.
- `scripts/`: Scripts used for data scraping and processing.
- `README.md`: General information and overview of the repository.

### Getting Started
To get started with CineSage Insights, clone the repository and explore the various directories to understand the structure and components of the project.

### Contribution
This project is a collaborative effort of Team 20 in CSE 6242. Contributions are welcome to enhance the tool's capabilities and accuracy.

---

CineSage Insights is not just a tool; it's a pathway for filmmakers to embrace the digital transformation in movie production, ensuring data-driven success in the competitive world of cinema.
