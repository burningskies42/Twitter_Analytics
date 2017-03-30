import pandas as pd
from io import StringIO

def tweet2df(new_str):
    # new_str = new_str[2:-2]
    
    return df

new_str ='''
{"created_at":"Wed Mar 29 14:29:07 +0000 2017","id":847093266109517824,"id_str":"847093266109517824","text":"Get $10.00 off the Evecase 2L Hydration Daypack w\/code 768C6DBC today! https:\/\/t.co\/38fj1GcXKe #hiking https:\/\/t.co\/VnSGGptzkz #giveaway","source":"\u003ca href=\"http:\/\/www.amazon.com\" rel=\"nofollow\"\u003eAmazon\u003c\/a\u003e","truncated":false,"in_reply_to_status_id":null,"in_reply_to_status_id_str":null,"in_reply_to_user_id":null,"in_reply_to_user_id_str":null,"in_reply_to_screen_name":null,"user":{"id":830098798642860032,"id_str":"830098798642860032","name":"paul","screen_name":"Ijij6393Paul","location":null,"url":null,"description":null,"protected":false,"verified":false,"followers_count":4,"friends_count":465,"listed_count":0,"favourites_count":0,"statuses_count":849,"created_at":"Fri Feb 10 16:59:10 +0000 2017","utc_offset":null,"time_zone":null,"geo_enabled":false,"lang":"zh-cn","contributors_enabled":false,"is_translator":false,"profile_background_color":"F5F8FA","profile_background_image_url":"","profile_background_image_url_https":"","profile_background_tile":false,"profile_link_color":"1DA1F2","profile_sidebar_border_color":"C0DEED","profile_sidebar_fill_color":"DDEEF6","profile_text_color":"333333","profile_use_background_image":true,"profile_image_url":"http:\/\/pbs.twimg.com\/profile_images\/842438029046345728\/jh0tezK9_normal.jpg","profile_image_url_https":"https:\/\/pbs.twimg.com\/profile_images\/842438029046345728\/jh0tezK9_normal.jpg","default_profile":true,"default_profile_image":false,"following":null,"follow_request_sent":null,"notifications":null},"geo":null,"coordinates":null,"place":null,"contributors":null,"is_quote_status":false,"retweet_count":0,"favorite_count":0,"entities":{"hashtags":[{"text":"hiking","indices":[95,102]},{"text":"giveaway","indices":[127,136]}],"urls":[{"url":"https:\/\/t.co\/38fj1GcXKe","expanded_url":"http:\/\/amzn.to\/2nbqiQ9","display_url":"amzn.to\/2nbqiQ9","indices":[71,94]},{"url":"https:\/\/t.co\/VnSGGptzkz","expanded_url":"https:\/\/giveaway.amazon.com\/p\/05543ddbd876b464\/?ref_=tsm_4_tw_p_tw","display_url":"giveaway.amazon.com\/p\/05543ddbd876\u2026","indices":[103,126]}],"user_mentions":[],"symbols":[]},"favorited":false,"retweeted":false,"possibly_sensitive":false,"filter_level":"low","lang":"en","timestamp_ms":"1490797747529"}
'''


new_str = tweet2df(new_str)

for s in new_str:

    print(s)

# print(new_str)