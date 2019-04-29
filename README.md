# Predicting Users' Votes

The attached TSV file contains information about 146,646 pictures posted at a major Russian railfanning website. (No need to click!) The columns, separated by TABs, are explained in the table:

1. etitle - picture category
2. region - region where the picture has been taken
3. takenon - the date when the picture was taken, represented as YYYY-MM-DD; if the day or months are not known, 00 is used instead; some values in this column may be missing
4. votedon - the date when the picture was posted to the site; some values in this column may be missing
5. author_id - the ID of the author, represented as a positive integer number
6. votes - the number of (up)votes for the picture
7. viewed - the number of times the pictures were viewed
8. n_comments - the number of comments to the picture.

Some values in the last three columns may be negative. Treat the negative numbers as NAs.

You are to write a script that will predict the number of upvotes based on any other information from the table, using any predictive model. You will possibly need:

1. Split the table into the training and testing parts.
2. Convert some or all date-based columns to Pandas DateTime.
3. Convert some or all categorical columns to dummies.
4. Select important features.
5. Choose a predictive model.
6. Fit the model and assess the fit quality.
7. Cross-validate the model.
8. Repeat steps 4-7, if necessary.

Your dataset has 70% of rows of the complete dataset. I have the remaining 30%. I will evaluate the quality of your model by applying it to my part of the dataset. You will earn a passing grade if the score of the model is at least 51% (better than tossing a coin).

After you train the model, pickle it as "model.p" and upload together with the complete Python code. You should also provide a report that contains the following items:

- The name of the chosen model.
- Pie charts of distributions of pictures by category and region.
- Histograms of upvotes, views, and comments.
- Line graphs of the average number of pictures, upvotes, views, and comments by year; put all four graphs in one chart, but use the log Y-axis and provide a legend.
- A histogram of the difference between the true and predicted number of upvotes.
- The scores of your model on the training and testing parts.

Hint: Newer pictures may have fewer views/comments/upvotes, because of the shorter lifespan

Hint: Newer pictures may have more views/comments/upvotes, because of the progressively increasing traffic

Hint: Pictures posted by "older" authors may have more views/comments/upvotes, because of the author's reputation; can you estimate an author's lifespan on the site from the data that you have?

Hint: The log of the number of upvotes may be easier to estimate than the number itself


