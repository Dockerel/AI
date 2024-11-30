import numpy as np


class EmbeddingTexts:
    def __init__(self, index, model, preprosessor):
        self.index = index
        self.model = model

        self.preprosessor = preprosessor
        self.texts = self.preprosessor.texts
        self.image_url = self.preprosessor.image_url
        self.titles = self.preprosessor.titles
        self.doc_urls = self.preprosessor.doc_urls
        self.doc_dates = self.preprosessor.doc_dates

    # -1, 0, 1 범위로 벡터 정규화 함수
    def normalize_vector(self, vectors):
        # 벡터를 numpy array로 변환
        vectors = np.array(vectors)

        # 각 벡터의 최소값과 최대값을 구함
        min_vals = vectors.min(axis=1, keepdims=True)
        max_vals = vectors.max(axis=1, keepdims=True)

        # Min-Max 정규화 (0, 1 범위로 정규화한 후 -1, 0, 1 범위로 변환)
        normalized_vectors = (vectors - min_vals) / (
            max_vals - min_vals
        )  # 0 to 1 범위로 변환
        normalized_vectors = 2 * normalized_vectors - 1  # -1 to 1 범위로 변환

        # 0을 포함하는 -1, 0, 1 범위로 조정
        normalized_vectors = np.round(normalized_vectors)

        return normalized_vectors

    def batch_vectors(self, vectors, batch_size):
        for i in range(0, len(vectors), batch_size):
            yield vectors[i : i + batch_size]

    def run(self):
        ids = [str(x) for x in range(0, len(self.texts))]
        embedded_datas = self.model.encode(self.texts).tolist()

        # embedded_datas에 대해 벡터 정규화
        normalized_embedded_datas = self.normalize_vector(embedded_datas).tolist()

        meta_datas = [
            {
                "title": self.titles[i],
                "text": self.texts[i],
                "url": self.doc_urls[i],
                "date": self.doc_dates[i],
            }
            for i in range(len(self.texts))
        ]
        records = list(zip(ids, normalized_embedded_datas, meta_datas))

        # 배치 업로드
        batch_size = 100  # 한 번에 업로드할 벡터 수 (요청 크기에 따라 조정)
        for batch in self.batch_vectors(records, batch_size):
            self.index.upsert(vectors=batch)
