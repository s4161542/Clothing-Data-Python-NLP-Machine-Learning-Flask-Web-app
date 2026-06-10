from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import re
import gensim.downloader as api
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def docvecs(embeddings, docs):
    # making an  empty matrix where each row will contain one document vector.
    vecs = np.zeros((len(docs), embeddings.vector_size))

    #Iterating through each tokenized review.
    for i, doc in enumerate(docs):

        # Keep only the words that  is present in  the GloVe model.
        valid_words = [word for word in doc if word in embeddings.key_to_index]

        # If no words are found in the model convert the vector to zero
        if len(valid_words) == 0:
            continue

        # Getting the vector for  valid word.
        word_vectors = np.vstack([embeddings[word] for word in valid_words])

        # Add all word vectors together to create one document vector.
        doc_vector = np.sum(word_vectors, axis=0)

        # Storing the document vector in the matrix.
        vecs[i, :] = doc_vector

    # Returning all document vectors.
    return vecs


def simple_stem(word):
    # Converting the word to lowercase and removing whitespaces.
    word = word.lower().strip()
    #Hard coded the most common suffixes to singular form 
    # Convert plural words ending with "ies" into singular form.
    if word.endswith("ies"):
        word = word[:-3] + "y"

    # Dealing with words ending with "sses".
    elif word.endswith("sses"):
        word = word[:-2]

    # Removing simple plural "s" from longer words.
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]

    # Return the cleaned word.
    return word


def tokenize(text):
    # Convert any input into lowercase text.
    text = str(text).lower()

    # tokenisation using regex 
    tokens = re.findall(r"[a-zA-Z]+", text)

    # Apply simple stemming to each token.
    tokens = [simple_stem(token) for token in tokens]

    # Return the final list of tokens.
    return tokens



# Flask app.
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "shopping_secret_key")

# Parsing the  CSV file.
clothes_data = pd.read_csv(BASE_DIR / "assignment3_II.csv")

# Filling missing values of text columns .
text_columns = [
    "Title",
    "Review Text",
    "Division Name",
    "Department Name",
    "Class Name",
    "Clothes Title",
    "Clothes Description"
]

for column in text_columns:
    clothes_data[column] = clothes_data[column].fillna("")


# Fill missing values of number columns .
number_columns = [
    "Rating",
    "Recommended IND",
    "Positive Feedback Count"
]

for column in number_columns:
    clothes_data[column] = clothes_data[column].fillna(0)


# Converting Clothing ID to integer.
clothes_data["Clothing ID"] = clothes_data["Clothing ID"].astype(int)


# Create one row per clothing item.
items_data = (
    clothes_data
    .sort_values("Clothing ID")
    .groupby("Clothing ID", as_index=False)
    .first()
)


# Get unique department names for the website nav bar.
departments = sorted(clothes_data["Department Name"].dropna().unique())


# This DataFrame stores new user reviews.
user_reviews = pd.DataFrame(
    columns=[
        "Review ID",
        "Clothing ID",
        "Age",
        "Title",
        "Review Text",
        "Rating",
        "Recommended IND"
    ]
)

# Loading the saved Logistic Regression model.
with open(BASE_DIR / "logistic_regression_model.pkl", "rb") as file:
    model = pickle.load(file)


glove_model = None


def get_glove_model():
    global glove_model

    if glove_model is None:
        glove_model = api.load("glove-wiki-gigaword-100")

    return glove_model


def predict_recommendation(review_title, review_description):
    # Combining title and description into one review text.
    review_text = review_title + " " + review_description

    # Converting the review text into tokens.
    tokenized_review = tokenize(review_text)

    # Converting the tokenized review into a GloVe document vector.
    review_vector = docvecs(get_glove_model(), [tokenized_review])

    # Predicting whether the review recommends the item or not.
    prediction = model.predict(review_vector)

    # Return 0 or 1 as an integer.
    return int(prediction[0])

def get_item(clothing_id):
    # Converting clothing_id into an integer so it matches the CSV column type.
    clothing_id = int(clothing_id)

    # Find the item with the matching Clothing ID.
    item = items_data[items_data["Clothing ID"] == clothing_id]

    # If no item is found, return None.
    if len(item) == 0:
        return None

    # Converting the item row into a dictionary for  use in templates.
    return item.iloc[0].to_dict()


def get_reviews_for_item(clothing_id):
    global user_reviews

    # Converting clothing_id into an integer.
    clothing_id = int(clothing_id)

    # Select original reviews from the CSV file for this item.
    original_reviews = clothes_data[clothes_data["Clothing ID"] == clothing_id][["Age","Title","Review Text","Rating","Recommended IND","Positive Feedback Count"]].copy()

    # Adding a empty Review ID for original CSV reviews.
    original_reviews["Review ID"] = ""

    # Naming these reviews as original reviews.
    original_reviews["Source"] = "Original"

    # Selecting reviews submitted by the user for this item.
    new_reviews = user_reviews[
        user_reviews["Clothing ID"] == clothing_id
    ].copy()

    # Adding missing columns to user reviews so they match the original reviews.
    if len(new_reviews) > 0:
        new_reviews["Positive Feedback Count"] = 0
        new_reviews["Source"] = "User"

        # Put user reviews first, then original reviews below.
        all_reviews = pd.concat(
            [new_reviews, original_reviews],
            ignore_index=True
        )

        # Returnning reviews as a list of dictionaries for the HTML template.
        return all_reviews.to_dict("records")

    # If there are no user reviews, return only original reviews.
    return original_reviews.to_dict("records")


def get_item_stats(clothing_id):
    # Getting all reviews for this item.
    reviews = get_reviews_for_item(clothing_id)

    # Returning default values if the item has no reviews.
    if len(reviews) == 0:
        return {
            "average_rating": 0,
            "recommend_percent": 0,
            "review_count": 0
        }

    # Adding all rating values together.
    total_rating = sum(float(review["Rating"]) for review in reviews)

    # Adding all recommendation labels together.
    total_recommend = sum(int(review["Recommended IND"]) for review in reviews)

    # Calculating average rating.
    average_rating = round(total_rating / len(reviews), 2)

    # Calculating the percentage of reviewers who recommended the item.
    recommend_percent = round((total_recommend / len(reviews)) * 100, 1)

    # Return the statistics as a dictionary.
    return {
        "average_rating": average_rating,
        "recommend_percent": recommend_percent,
        "review_count": len(reviews)
    }


def search_items(search_string):
    # Changing the search input into  tokens.
    search_tokens = tokenize(search_string)

    # Returning no results if the search box is empty.
    if len(search_tokens) == 0:
        return []

    # Store matched items here.
    search_results = []

    # Iterating through every clothing item.
    for index, row in items_data.iterrows():

        # Combining important item information into one searchable text.
        item_text = (str(row["Clothes Title"]) + " " +str(row["Clothes Description"]) + " " +str(row["Department Name"]) + " " +str(row["Class Name"]))
        # Tokenizing the item text.
        item_tokens = tokenize(item_text)
        # Check if any search word appears in the item text.
        for token in search_tokens:
            if token in item_tokens:

                # Add the matching item to the results.
                search_results.append(row.to_dict())

                # Stop checking this item after the first match.
                break

    # Return all matched items.
    return search_results



# Routes

@app.route("/")
def index():
    # Showing the first 12 items on the home page.
    featured_items = items_data.head(12).to_dict("records")

    # Render the home page template.
    return render_template(
        "home.html",
        departments=departments,
        featured_items=featured_items
    )


@app.route("/category/<department_name>")
def category(department_name):
    # Find all items that belong to the selected department.
    category_items = items_data[
        items_data["Department Name"].str.lower() == department_name.lower()
    ]

    # Renderring the category page.
    return render_template("category.html",departments=departments, department_name=department_name,items=category_items.to_dict("records"),item_count=len(category_items))


@app.route("/item/<int:clothing_id>")
def item_detail(clothing_id):
    # Get the selected item.
    item = get_item(clothing_id)

    # Showing not found page if the item does not exist.
    if item is None:
        return render_template("not_found.html",departments=departments)

    # Getting all reviews for this item.
    reviews = get_reviews_for_item(clothing_id)

    # Getting rating and recommendation statistics for this item.
    stats = get_item_stats(clothing_id)

    # Rendering the item detail page.
    return render_template("item.html",departments=departments,item=item,reviews=reviews,stats=stats )


@app.route("/search", methods=["POST"])
def search():
    # Getting the search keyword from the form.
    search_string = request.form.get("searchword", "").strip()

    # Search for matching items.
    search_results = search_items(search_string)

    # Count the number of matching results.
    num_results = len(search_results)

    # Render the search results page.
    return render_template("search.html",departments=departments,search_string=search_string, num_results=num_results, article_search=search_results)


@app.route("/add_review/<int:clothing_id>", methods=["GET", "POST"])
def add_review(clothing_id):
    global user_reviews

    # Getting the item that the user is reviewing.
    item = get_item(clothing_id)

    # Showing not found page if the item does not exist.
    if item is None:
        return render_template("not_found.html", departments=departments)

    # Empty string for prediction before the user clicks the prediction button.
    prediction = ""

    # if form submitted run this code.
    if request.method == "POST":

        # Reading form values typed by the user.
        age = request.form.get("age", "")
        title = request.form.get("title", "")
        description = request.form.get("description", "")
        rating = request.form.get("rating", "")

        # Checking which button was clicked.
        button_value = request.form.get("button", "")

        # If the user clicks Predict Recommendation then run the model.
        if button_value == "Predict Recommendation":
            prediction = predict_recommendation(title, description)

            # Showing the form again with the prediction result.
            return render_template("add_review.html", departments=departments, item=item,prediction=prediction,age=age,title=title,description=description,rating=rating)

        # If the user clicks Save Review then store the review.
        elif button_value == "Save Review":

            #  the final recommendation chosen by the user.
            recommendation = request.form.get("recommendation", "")

            # Creating a short unique review ID.
            review_id = "T" + str(len(user_reviews) + 1)

            # Storing the new review information in a dictionary.
            new_review = {
                "Review ID": review_id,
                "Clothing ID": clothing_id,
                "Age": age,
                "Title": title,
                "Review Text": description,
                "Rating": rating,
                "Recommended IND": recommendation
            }

            # Adding the new review to the user_reviews DataFrame.
            user_reviews = pd.concat(
                [user_reviews, pd.DataFrame([new_review])],
                ignore_index=True
            )

            # Go back to the item page so the review appears under old reviews.
            return redirect(url_for("item_detail", clothing_id=clothing_id))

    # Showing the empty review form for GET requests.
    return render_template("add_review.html", departments=departments,item=item, prediction=prediction, age="", title="",description="",rating="")


@app.route("/review/<review_id>")
def review_detail(review_id):
    # Finding the saved user review using Review ID.
    review = user_reviews[user_reviews["Review ID"] == review_id]

    # Showing not found page if the review does not exist.
    if len(review) == 0:
        return render_template(
            "not_found.html",
            departments=departments
        )

    # Converting the review row into a dictionary.
    review = review.iloc[0].to_dict()

    # Getting the clothing item connected to this review.
    clothing_id = int(review["Clothing ID"])
    item = get_item(clothing_id)

    # Render the saved review page.
    return render_template("review.html",departments=departments,review=review,item=item )

@app.route("/about")
def about():
    # Render the about page.
    return render_template("about.html",departments=departments)





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


#References:
#1. Flask documentation: https://flask.palletsprojects.com/
#2. Pandas documentation: https://pandas.pydata.org/docs/
#3. Gensim documentation: https://radimrehurek.com/gensim
#4. Scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
#5. ww3schools HTML tutorial: https://www.w3schools.com/html/
#6. w3schools CSS tutorial: https://www.w3schools.com/css/
#7. w3schools JavaScript tutorial: https://www.w3schools.com/js/
#8 . Stack Overflow: https://stackoverflow.com/
#9. w3schools http tutorial: https://www.w3schools.com/tags/ref_httpmethods.asp
