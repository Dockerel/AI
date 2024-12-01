import json
from util.characterTextSplitter import CharacterTextSplitter


class DataPreprocessor:
    def __init__(self):
        self.text_splitter = CharacterTextSplitter()
        self.texts = []
        self.image_url = []
        self.titles = []
        self.doc_urls = []
        self.doc_dates = []

    def processAnnouncement(self, document_data):
        for title, doc, image, date, url in document_data:
            if not doc.strip():  # doc가 비어있는 경우
                split_texts = self.text_splitter.split_text(doc)
                self.texts.extend(split_texts)
                self.titles.extend(
                    [title] * len(split_texts)
                )  # 제목을 분리된 텍스트와 동일한 길이로 추가
                self.doc_urls.extend([url] * len(split_texts))
                self.doc_dates.extend(
                    [date] * len(split_texts)
                )  # 분리된 각 텍스트에 동일한 날짜 적용

                # 이미지 URL도 저장
                if image:  # 이미지 URL이 비어 있지 않은 경우
                    self.image_url.extend(
                        [image] * len(split_texts)
                    )  # 동일한 길이로 이미지 URL 추가
                else:  # 이미지 URL이 비어 있는 경우
                    self.image_url.extend(
                        ["No content"] * len(split_texts)
                    )  # "No content" 추가
            else:
                self.texts.append("No content")
                self.titles.append(title)
                self.doc_urls.append(url)
                self.doc_dates.append(date)
                self.image_url.append(image if image else "No content")

    def processProfessors(self, prof_data):
        professor_texts = []
        professor_image_urls = []
        professor_titles = []
        professor_doc_urls = []
        professor_doc_dates = []

        # prof_data는 extract_professor_info_from_urls 함수의 반환값
        for title, doc, image, date, url in prof_data:
            if (
                isinstance(doc, str) and doc.strip()
            ):  # 교수 정보가 문자열로 있고 비어있지 않을 때
                split_texts = self.text_splitter.split_text(doc)
                professor_texts.extend(split_texts)
                professor_titles.extend(
                    [title] * len(split_texts)
                )  # 교수 이름을 분리된 텍스트와 동일한 길이로 추가
                professor_doc_urls.extend([url] * len(split_texts))
                professor_doc_dates.extend(
                    [date] * len(split_texts)
                )  # 분리된 각 텍스트에 동일한 날짜 적용

                # 이미지 URL도 저장
                if image:  # 이미지 URL이 비어 있지 않은 경우
                    professor_image_urls.extend(
                        [image] * len(split_texts)
                    )  # 동일한 길이로 이미지 URL 추가
                else:
                    professor_image_urls.extend(
                        ["No content"] * len(split_texts)
                    )  # "No content" 추가
            else:
                professor_texts.append("No content")
                professor_titles.append(title)
                professor_doc_urls.append(url)
                professor_doc_dates.append(date)
                professor_image_urls.append(
                    image if image else "No content"
                )  # 이미지 URL 추가

        # 교수 정보 데이터를 기존 데이터와 합치기
        self.texts.extend(professor_texts)
        self.image_url.extend(professor_image_urls)
        self.titles.extend(professor_titles)
        self.doc_urls.extend(professor_doc_urls)
        self.doc_dates.extend(professor_doc_dates)

    def processCompanyInfos(self, company_data):
        for title, doc, image, date, url in company_data:
            if (
                isinstance(doc, str) and doc.strip()
            ):  # doc가 문자열인지 확인하고 비어있지 않은지 확인
                split_texts = self.text_splitter.split_text(doc)
                self.texts.extend(split_texts)
                self.titles.extend(
                    [title] * len(split_texts)
                )  # 제목을 분리된 텍스트와 동일한 길이로 추가
                self.doc_urls.extend([url] * len(split_texts))
                self.doc_dates.extend(
                    [date] * len(split_texts)
                )  # 분리된 각 텍스트에 동일한 날짜 적용

                # 이미지 URL도 저장
                if image:  # 이미지 URL이 비어 있지 않은 경우
                    self.image_url.extend(
                        [image] * len(split_texts)
                    )  # 동일한 길이로 이미지 URL 추가
                else:  # 이미지 URL이 비어 있는 경우
                    self.image_url.extend(
                        ["No content"] * len(split_texts)
                    )  # "No content" 추가
            else:
                self.texts.append("No content")
                self.titles.append(title)
                self.doc_urls.append(url)
                self.doc_dates.append(date)
                self.image_url.append(
                    image if image else "No content"
                )  # 이미지 URL 추가

    def processSeminarInfos(self, seminar_data):
        for title, doc, image, date, url in seminar_data:
            if (
                isinstance(doc, str) and doc.strip()
            ):  # doc가 문자열인지 확인하고 비어있지 않은지 확인
                split_texts = self.text_splitter.split_text(doc)
                self.texts.extend(split_texts)
                self.titles.extend(
                    [title] * len(split_texts)
                )  # 제목을 분리된 텍스트와 동일한 길이로 추가
                self.doc_urls.extend([url] * len(split_texts))
                self.doc_dates.extend(
                    [date] * len(split_texts)
                )  # 분리된 각 텍스트에 동일한 날짜 적용

                # 이미지 URL도 저장
                if image:  # 이미지 URL이 비어 있지 않은 경우
                    self.image_url.extend(
                        [image] * len(split_texts)
                    )  # 동일한 길이로 이미지 URL 추가
                else:  # 이미지 URL이 비어 있는 경우
                    self.image_url.extend(
                        ["No content"] * len(split_texts)
                    )  # "No content" 추가
            else:
                self.texts.append("No content")
                self.titles.append(title)
                self.doc_urls.append(url)
                self.doc_dates.append(date)
                self.image_url.append(
                    image if image else "No content"
                )  # 이미지 URL 추가

    def saveResults(self):
        with open("data/texts.json", "w") as file:
            json.dump(self.texts, file)
        with open("data/image_url.json", "w") as file:
            json.dump(self.image_url, file)
        with open("data/titles.json", "w") as file:
            json.dump(self.titles, file)
        with open("data/doc_urls.json", "w") as file:
            json.dump(self.doc_urls, file)
        with open("data/doc_dates.json", "w") as file:
            json.dump(self.doc_dates, file)
