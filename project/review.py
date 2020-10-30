from . import views

def CommentFetch (url):
    from selenium import webdriver
    import time
    import re
    import csv
    import string
    import time
    import sys
    import os
    module_dir = os.path.dirname(__file__)
    print(module_dir)

    t_end = time.time() + 60 * .1



    driver = webdriver.Chrome('C:\webdrivers\chromedriver')
    yt_link = url
    print(
        "-------------------------------------------------------------------------------------------------------------------")
    driver.get(yt_link)
    driver.maximize_window()
    time.sleep(5)
    title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
    print("Video Title: " + title)
    print(
        "-------------------------------------------------------------------------------------------------------------------")

    comment_section = driver.find_element_by_xpath('//*[@id="comments"]')
    driver.execute_script("arguments[0].scrollIntoView();", comment_section)
    time.sleep(7)

    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    #while True:
    while time.time() < t_end:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

        # Wait to load page
        time.sleep(2)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)

    name_elems = driver.find_elements_by_xpath('//*[@id="author-text"]')
    comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
    num_of_names = len(name_elems)
    file_path = os.path.join(module_dir, 'static/output.csv')
    f = open(file_path, "w", encoding='utf8')
    for i in range(num_of_names):
        username = name_elems[i].text  # .replace(",", "|")
        # username = emoji_pattern.sub(r'', username)
        # username = str(username).replace("\n", "---")
        comment = comment_elems[i].text  # .replace(",", "|")
        # comment = emoji_pattern.sub(r'', comment)
        # comment = str(comment).replace("\n", "---")

        # if isEnglish(comment) == False:
        #   comment = "NOT ENGLISH"

        f.write(comment + "\n")  # comment.translate({ord(i):None for i in '' if i not in string.printable})

    f.close()
    driver.close()