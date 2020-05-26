import json
import urllib.request

JSON_FILE_PATH = 'https://nextcloud.readerbench.com/index.php/s/bsKbYDrCSYFKNze/download'

class JaccSimilarity:
    def __init__(self, topics, level):

        self.similarity = {}
        self.level = level
        self.topics = topics
        self.parse()

    def parse(self):
        # content = urllib.request.urlopen(JSON_FILE_PATH).read().decode("utf-8")

        with open('rb_api/mass_customization/jacc_' + self.level + '.json') as f:
            data = json.load(f)

        for topic in self.topics:
            self.similarity[topic] = data[topic]


