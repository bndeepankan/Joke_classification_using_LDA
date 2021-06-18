import requests
import json


def verbAtlas(word_list):
    """
    :param word_list: list of words annotated as Noun, verb, etc.
    :return: None
    """

    for tags, word in word_list.items():
        for val in word:
            response = requests.get("http://verbatlas.org/api/verbatlas/predicate?lemma=%s" % val)
            if response.status_code == 200:
                content = json.loads(response.text)
                for id in content.keys():
                    print(val, content[id]['va_frame_id'], content[id]['va_frame_name'])
            else:
                print(val, [])
