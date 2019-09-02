import argparse
import csv
import os

import tqdm
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


def remove_if_exists(filename: str):
    try:
        os.remove(filename)
    except OSError:
        pass


def get_all_tweets(screen_name: str, filename: str):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # Authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(
        config["TWITTER_CONSUMER_KEY"], config["TWITTER_CONSUMER_SECRET_KEY"]
    )
    auth.set_access_token(
        config["TWITTER_ACCESS_TOKEN"], config["TWITTER_ACCESS_TOKEN_SECRET"]
    )

    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Initialize a list to hold all the tweepy Tweets
    alltweets = []

    # Create progress bar. 3240 represets max allowed tweets to scrape per user.
    pbar = tqdm.tqdm(total=3240, desc=screen_name, leave=True)

    # Make initial request for most recent tweets (200 is the maximum allowed count)
    # https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline
    # https://github.com/tweepy/tweepy/blob/bc2deca369f89e43905aa147a4eee48ff522b028/tweepy/api.py#L123
    new_tweets = api.user_timeline(
        screen_name=screen_name, count=200, include_rts=False, exclude_replies=True
    )

    pbar.update(len(new_tweets))

    # Save most recent tweets
    alltweets.extend(new_tweets)

    # Save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # Keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        # All subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(
            screen_name=screen_name,
            count=200,
            max_id=oldest,
            include_rts=False,
            exclude_replies=True,
        )

        pbar.update(len(new_tweets))

        # Save most recent tweets
        alltweets.extend(new_tweets)

        # Update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

    # Transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [
        [tweet.id_str, tweet.created_at, tweet.text, screen_name] for tweet in alltweets
    ]

    # Write the csv
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text", "screen_name"])
        writer.writerows(outtweets)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: allo comma separated list as option
    parser.add_argument(
        "--usernames", help="File of line separated usernames", required=True
    )
    parser.add_argument("--out_dir", help="Directory to dump tweets", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate file if already exists",
        required=False,
    )
    args = parser.parse_args()
    print("Args: ", args)

    with open(args.usernames) as f:
        for line in tqdm.tqdm(f.readlines(), leave=True):
            screen_name = line.strip()
            fpath = os.path.join(args.out_dir, f"{screen_name}_tweets.csv")
            if not os.path.isfile(fpath) or args.force:
                remove_if_exists(fpath)
                get_all_tweets(screen_name, fpath)
