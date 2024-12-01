from util.knuCrawler import KnuCrawler
from util.dataPreprocessor import DataPreprocessor
from util.embeddingTexts import EmbeddingTexts


class WebEmbeddingPipeline:
    def __init__(self, index, model):
        self.index = index
        self.model = model

        self.preprosessor = DataPreprocessor()
        self.crawler = KnuCrawler(self.preprosessor)
        self.embedding = EmbeddingTexts(self.index, self.model, self.preprosessor)

    def run(self):
        self.crawler.run()
        self.embedding.run()
        self.preprosessor.saveResults()
