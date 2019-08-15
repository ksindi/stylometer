import csv

import tweepy

config = {
    k: os.getenv(k, default)
    for k, default in [
        ("APP_ENV", "local"),
        ("LOG_LEVEL", "INFO"),
        # Twitter API credentials
        ("TWITTER_CONSUMER_KEY", ""),
        ("TWITTER_CONSUMER_SECRET_KEY", ""),
        ("TWITTER_ACCESS_TOKEN", ""),
        ("TWITTER_ACCESS_TOKEN_SECRET", ""),
    ]
}


def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # Authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(config["CONSUMER_KEY"], config["CONSUMER_SECRET"])
    auth.set_access_token(config["ACCESS_KEY"], config["ACCESS_SECRET"])
    api = tweepy.API(auth)

    # Initialize a list to hold all the tweepy Tweets
    alltweets = []

    # Make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # Save most recent tweets
    alltweets.extend(new_tweets)

    # Save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # Keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        # All subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(
            screen_name=screen_name, count=200, max_id=oldest
        )

        # Save most recent tweets
        alltweets.extend(new_tweets)

        # Update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    # Transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [
        [tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")]
        for tweet in alltweets
    ]

    # Write the csv
    with open("%s_tweets.csv" % screen_name, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(outtweets)

    pass


if __name__ == "__main__":
    # Pass in the username of the account you want to download
    get_all_tweets("J_tsar")
