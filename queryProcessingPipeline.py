import json, re, pytz, time
import numpy as np
from datetime import datetime
from konlpy.tag import Okt
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher


class QueryProcessingPipeline:
    def __init__(self, index, model):
        with open("data/texts.json", "r") as file:
            self.texts = json.load(file)
        with open("data/image_url.json", "r") as file:
            self.image_url = json.load(file)
        with open("data/titles.json", "r") as file:
            self.titles = json.load(file)
        with open("data/doc_urls.json", "r") as file:
            self.doc_urls = json.load(file)
        with open("data/doc_dates.json", "r") as file:
            self.doc_dates = json.load(file)

        self.index = index
        self.model = model

        self.bm25_titles = None

    def transformed_query(self, content):
        # 중복된 단어를 제거한 명사를 담을 리스트
        query_nouns = []

        # 1. 숫자와 특정 단어가 결합된 패턴 추출 (예: '2024학년도', '1월' 등)
        pattern = r"\d+(?:학년도|년|학년|월|일|학기|시|분|초|기|개|차)?"
        number_matches = re.findall(pattern, content)
        query_nouns += number_matches
        # 추출된 단어를 content에서 제거
        for match in number_matches:
            content = content.replace(match, "")

        # 2. 영어와 한글이 붙어 있는 패턴 추출 (예: 'SW전공' 등)
        eng_kor_pattern = r"\b[a-zA-Z]+[가-힣]*\b"
        eng_kor_matches = re.findall(eng_kor_pattern, content)
        # 영어 부분을 대문자로 변환

        eng_kor_matches_upper = [match.upper() for match in eng_kor_matches]
        # 변환된 단어를 query_nouns에 추가
        query_nouns += eng_kor_matches_upper

        # content에서 원래 추출된 단어를 정확히 제거
        for match in eng_kor_matches:
            content = re.sub(rf"\b{re.escape(match)}\b", "", content)

        # 3. 영어 단어 단독으로 추출
        english_words = re.findall(r"\b[a-zA-Z]+\b", content)
        query_nouns += english_words
        # 추출된 단어를 content에서 제거
        for match in english_words:
            content = content.replace(match, "")
        if "컴학" in content:
            query_nouns.append("컴퓨터학부")
        if "차" in content:
            query_nouns.append("차")
        if "국가 장학금" in content:
            query_nouns.append("국가장학금")
            content = content.replace("국가 장학금", "")
        if "종프" in content:
            query_nouns.append("종합설계프로젝트")
        if "종합설계프로젝트" in content:
            query_nouns.append("종합설계프로젝트")
        if "대회" in content:
            query_nouns.append("경진대회")
            content = content.replace("대회", "")
        if "튜터" in content:
            query_nouns.append("TUTOR")
            content = content.replace("튜터", "")  # '튜터' 제거
        if "탑싯" in content:
            query_nouns.append("TOPCIT")
            content = content.replace("탑싯", "")
        if "시험" in content:
            query_nouns.append("시험")
        if "하계" in content:
            query_nouns.append("여름")
            query_nouns.append("하계")
        if "동계" in content:
            query_nouns.append("겨울")
            query_nouns.append("동계")
        if "겨울" in content:
            query_nouns.append("겨울")
            query_nouns.append("동계")
        if "여름" in content:
            query_nouns.append("여름")
            query_nouns.append("하계")
        if "성인지" in content:
            query_nouns.append("성인지")
        if "첨성인" in content:
            query_nouns.append("첨성인")
        if "글솦" in content:
            query_nouns.append("글솝")
        if "수꾸" in content:
            query_nouns.append("수강꾸러미")
        if "장학금" in content:
            query_nouns.append("장학생")
            query_nouns.append("장학")
        if "장학생" in content:
            query_nouns.append("장학금")
            query_nouns.append("장학")
        if "대해" in content:
            content = content.replace("대해", "")
        if "에이빅" in content:
            query_nouns.append("에이빅")
            query_nouns.append("ABEEK")
            content = content.replace("에이빅", "")
        if "채용" in content and any(
            keyword in content for keyword in ["모집", "공고"]
        ):
            if "모집" in content:
                content = content.replace("모집", "")
            if "공고" in content:
                content = content.replace("공고", "")
        # 비슷한 의미 모두 추가 (세미나)
        related_keywords = ["세미나", "특강", "강연"]
        if any(keyword in content for keyword in related_keywords):
            for keyword in related_keywords:
                query_nouns.append(keyword)
        # "공지", "사항", "공지사항"을 query_nouns에서 '공지사항'이라고 고정하고 나머지 부분 삭제
        keywords = ["공지", "사항", "공지사항"]
        if any(keyword in content for keyword in keywords):
            # 키워드 제거
            for keyword in keywords:
                content = content.replace(keyword, "")
                query_nouns.append("공지사항")  # 'query_noun'에 추가
        # 5. Okt 형태소 분석기를 이용한 추가 명사 추출
        okt = Okt()
        additional_nouns = [noun for noun in okt.nouns(content) if len(noun) > 1]
        query_nouns += additional_nouns
        if "인도" not in query_nouns and "인턴십" in query_nouns:
            query_nouns.append("베트남")
            query_nouns.append("다낭")

        # 6. "수강" 단어와 관련된 키워드 결합 추가
        if "수강" in content:
            related_keywords = ["변경", "신청", "정정", "취소", "꾸러미"]
            for keyword in related_keywords:
                if keyword in content:
                    # '수강'과 결합하여 새로운 키워드 추가
                    combined_keyword = "수강" + keyword
                    query_nouns.append(combined_keyword)
                    if "수강" in query_nouns:
                        query_nouns.remove("수강")
                    for keyword in related_keywords:
                        if keyword in query_nouns:
                            query_nouns.remove(keyword)
        # 최종 명사 리스트에서 중복된 단어 제거
        query_nouns = list(set(query_nouns))
        return query_nouns

    def initialize_bm25_titles(self):
        # BM25 유사도 계산
        tokenized_titles = [
            self.transformed_query(title) for title in self.titles
        ]  # 제목마다 명사만 추출하여 토큰화

        # 기존과 동일한 파라미터를 사용하고 있는지 확인
        self.bm25_titles = BM25Okapi(
            tokenized_titles, k1=1.5, b=0.75
        )  # 기존 파라미터 확인

    # 날짜를 파싱하는 함수
    def parse_date_change_korea_time(self, date_str):
        clean_date_str = date_str.replace("작성일", "").strip()
        naive_date = datetime.strptime(clean_date_str, "%y-%m-%d %H:%M")
        # 한국 시간대 추가
        korea_timezone = pytz.timezone("Asia/Seoul")
        return korea_timezone.localize(naive_date)

    def calculate_weight_by_days_difference(self, post_date, current_date, query_nouns):

        # 날짜 차이 계산 (일 단위)
        days_diff = (current_date - post_date).days

        # 기준 날짜 (24-01-01 00:00) 설정
        baseline_date_str = "24-01-01 00:00"
        baseline_date = self.parse_date_change_korea_time(baseline_date_str)

        # 작성일이 기준 날짜 이전이면 가중치를 1.35로 고정
        if post_date <= baseline_date:
            return 1.4

        # '최근', '최신' 등의 키워드가 있는 경우, 최근 가중치를 추가
        add_recent_weight = (
            1.0
            if any(
                keyword in query_nouns for keyword in ["최근", "최신", "지금", "현재"]
            )
            else 0
        )

        # **5일 단위 구분**: 최근 문서에 대한 세밀한 가중치 부여
        if days_diff <= 10:
            return 1.30 + add_recent_weight
        elif days_diff <= 20:
            return 1.27 + add_recent_weight
        elif days_diff <= 30:
            return 1.25 + add_recent_weight
        elif days_diff <= 60:
            return 1.20 + add_recent_weight
        elif days_diff <= 90:
            return 1.16
        # **월 단위 구분**: 2개월 이후는 월 단위로 단순화
        month_diff = (days_diff - 90) // 30
        month_weight_map = {
            0: 1.10,  # 2.5~3.5개월
            1: 1.08 - add_recent_weight / 2,  # 3.5~4.5개월
            2: 1.05 - add_recent_weight / 2,  # 4.5~5.5개월
            3: 1.02 - add_recent_weight / 2,  # 5.5~6.5개월
        }

        # 기본 가중치 반환 (6개월 이후)
        return month_weight_map.get(month_diff, 1 - add_recent_weight / 5)

    def get_korean_time(self):
        return datetime.now(pytz.timezone("Asia/Seoul"))

    # 유사도를 조정하는 함수
    def adjust_date_similarity(self, similarity, date_str, query_nouns):
        # 현재 한국 시간
        current_time = self.get_korean_time()
        # 작성일 파싱
        post_date = self.parse_date_change_korea_time(date_str)
        # 가중치 계산
        weight = self.calculate_weight_by_days_difference(
            post_date, current_time, query_nouns
        )
        # 조정된 유사도 반환
        return similarity * weight

    def find_url(url, doc_url):
        for i, urls in enumerate(doc_url):
            if urls.startswith(url):  # indexs와 시작이 일치하는지 확인
                return i

    # 사용자 질문에서 추출한 명사와 각 문서 제목에 대한 유사도를 조정하는 함수
    def adjust_similarity_scores(self, query_noun, title, similarities):

        for idx, titl in enumerate(title):
            # 제목에 포함된 query_noun 요소의 개수를 센다
            # 제목에 포함된 query_noun 요소의 개수를 센다
            matching_noun = [noun for noun in query_noun if noun in titl]

            for noun in matching_noun:
                similarities[idx] += len(noun) * 0.25
                if re.search(r"\d", noun):  # 숫자가 포함된 단어 확인
                    if noun in title:  # 본문에도 숫자 포함 단어가 있는 경우 추가 조정
                        similarities[idx] += len(noun) * 0.22
                    else:
                        similarities[idx] += len(noun) * 0.19
            # print(title,similarities[idx])
            # query_noun에 "대학원"이 없고 제목에 "대학원"이 포함된 경우 유사도를 0.1 감소
            keywords = ["대학원", "대학원생"]

            # 조건 1: 둘 다 키워드 포함
            if any(keyword in query_noun for keyword in keywords) and any(
                keyword in titl for keyword in keywords
            ):
                similarities[idx] += 2.0
            # 조건 2: query_noun에 없고, title에만 키워드가 포함된 경우
            if not any(keyword in query_noun for keyword in keywords) and any(
                keyword in titl for keyword in keywords
            ):
                similarities[idx] -= 2.0
            if (
                "현장" and "실습" and "현장실습"
            ) not in query_noun and "대체" in query_noun:
                similarities[idx] -= 1
            if "외국인" not in query_noun and "외국인" in title:
                similarities[idx] -= 2.0
            if self.texts[idx] == "No content":
                similarities[idx] *= 1.65  # 본문이 "No content"인 경우 유사도를 높임
            if "마일리지" in query_noun and "마일리지" in title:
                similarities[idx] += 1
            if (
                "신입생" in query_noun
                and "수강신청" in query_noun
                and "일괄수강신청" in title
            ):
                similarities[idx] += 1
        return similarities

    def last_filter_keyword(self, DOCS, query_noun, user_question):
        # 필터링에 사용할 키워드 리스트
        Final_best = DOCS
        # 키워드가 포함된 경우 유사도를 조정하고, 유사도 기준으로 내림차순 정렬
        for idx, doc in enumerate(DOCS):
            score, title, date, text, url, image = doc

            if (
                any(keyword in query_noun for keyword in ["인턴", "인턴십"])
                and "다낭" in title
                and "베트남" in title
            ):
                score += 2.0
            if "여름" in query_noun and "겨울" in title:
                score -= 1.0
            if "겨울" in query_noun and "여름" in title:
                score -= 1.0
            if "변경" in query_noun and "변경" in title:
                score += 1.0
            if "1학기" in query_noun and "1학기" in title:
                score += 1.0
            if "2학기" in query_noun and "2학기" in title:
                score += 1.0
            if "1학기" in query_noun and "2학기" in title:
                score -= 1.0
            if "2학기" in query_noun and "1학기" in title:
                score -= 1.0
            if any(
                keyword in title for keyword in ["종프", "종합설계프로젝트"]
            ) and any(keyword in user_question for keyword in ["종프", "종합설계프로젝트"]):
                score += 0.7
            if any(
                keyword in title
                for keyword in [
                    "심컴",
                    "심화컴퓨터전공",
                    "심화 컴퓨터공학",
                    "심화컴퓨터공학",
                ]
            ):
                if any(
                    keyword in user_question for keyword in ["심컴", "심화컴퓨터전공"]
                ):
                    score += 0.6
                else:
                    score -= 0.8
            elif any(
                keyword in title
                for keyword in [
                    "글로벌소프트웨어전공",
                    "글로벌SW전공",
                    "글로벌소프트웨어융합전공",
                    "글솝",
                    "글솦",
                ]
            ):
                if any(
                    keyword in user_question
                    for keyword in [
                        "글로벌소프트웨어융합전공",
                        "글로벌소프트웨어전공",
                        "글로벌SW전공",
                        "글솝",
                        "글솦",
                    ]
                ):
                    score += 0.65
                else:
                    if not "SDG" in query_noun:
                        score -= 0.8
            elif any(keyword in title for keyword in ["인컴", "인공지능컴퓨팅"]):
                if any(
                    keyword in user_question for keyword in ["인컴", "인공지능컴퓨팅"]
                ):
                    score += 0.8
                else:
                    if not "SDG" in query_noun:
                        score -= 0.8
            if not any(
                keyword in user_question
                for keyword in ["벤처스타트업아카데미", "스타트업"]
            ) and any(keyword in title for keyword in ["벤처스타트업아카데미", "스타트업"]):
                score -= 0.9
            if any(
                keyword in text for keyword in ["계약학과", "대학원", "타대학원"]
            ) and not any(keyword in query_noun for keyword in ["계약학과", "대학원", "타대학원"]):
                score -= 0.4  # 유사도 점수를 0.1 낮추기
            keywords = ["대학원", "대학원생"]

            # 조건 1: 둘 다 키워드 포함
            if any(keyword in query_noun for keyword in keywords) and any(
                keyword in title for keyword in keywords
            ):
                score += 2.0
            # 조건 2: query_noun에 없고, title에만 키워드가 포함된 경우
            elif not any(keyword in query_noun for keyword in keywords) and any(
                keyword in title for keyword in keywords
            ):
                score -= 2.0
            if any(keyword in query_noun for keyword in ["대학원", "대학원생"]) and any(
                keyword in title for keyword in ["대학원", "대학원생"]
            ):
                score += 2.5
            if (
                any(keyword in query_noun for keyword in ["담당", "업무", "일"])
                or any(
                    keyword in query_noun
                    for keyword in ["직원", "교수", "선생", "선생님"]
                )
            ) and date == "작성일24-01-01 00:00":
                ### 종프 팀과제 담당 교수 누구야와 같은 질문인데 엉뚱하게 파인콘에서 직원이 유사도 높게 측정된 경우를 방지하기 위함.
                if any(keys in query_noun for keys in ["교수"]):
                    check = 0
                    compare_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor"  ## 직원에 해당하는 URL임.
                    if compare_url == url:
                        check = 1
                    if check == 0:
                        score += 0.5
                    else:
                        score -= 0.9  ###직원이니까 유사도 나가라..
                else:
                    score += 1.5

            if not any(keys in query_noun for keys in ["교수"]) and any(
                keys in title for keys in ["담당교수", "교수"]
            ):
                score -= 0.7

            match = re.search(r"(?<![\[\(])\b수강\w*\b(?![\]\)])", title)

            if match:
                full_keyword = match.group(0)
                # query_nouns에 포함 여부 확인
                if full_keyword not in query_noun:
                    score -= 0.85
                else:
                    score += 0.85
            # 조정된 유사도 점수를 사용하여 다시 리스트에 저장
            Final_best[idx] = (score, title, date, text, url, image)
            # print(Final_best[idx])
        return Final_best

    def cluster_documents_by_similarity(self, docs, threshold=0.89):
        clusters = []

        for doc in docs:
            title = doc[1]
            added_to_cluster = False
            # 기존 클러스터와 비교
            for cluster in clusters:
                # 첫 번째 문서의 제목과 현재 문서의 제목을 비교해 유사도를 계산
                cluster_title = cluster[0][1]
                similarity = SequenceMatcher(None, cluster_title, title).ratio()
                # 유사도가 threshold 이상이면 해당 클러스터에 추가
                if similarity >= threshold:
                    # print(f"{doc[0]} {cluster[0][0]}  {title} {cluster_title}")
                    cluster_date = self.parse_date_change_korea_time(cluster[0][2])
                    doc_in_date = self.parse_date_change_korea_time(doc[2])
                    compare_date = abs(cluster_date - doc_in_date).days
                    ## 두 비교 문서 유사도 차이 0.5 미만이면서 두 문서 날짜 차이가 60일 미만인 경우 추가한다.
                    if cluster_title == title or (
                        -doc[0] + cluster[0][0] < 0.5
                        and cluster[0][3] != doc[2]
                        and compare_date < 60
                    ):
                        cluster.append(doc)
                    added_to_cluster = True
                    break

            # 유사한 클러스터가 없으면 새로운 클러스터 생성
            if not added_to_cluster:
                clusters.append([doc])

        return clusters

    def organize_documents_v2(self, sorted_cluster):
        # 첫 번째 문서를 기준으로 초기 설정
        top_doc = sorted_cluster[0]
        top_title = top_doc[1]

        # new_sorted_cluster 초기화 및 첫 번째 문서와 동일한 제목을 가진 문서들을 모두 추가
        new_sorted_cluster = []
        # titles에서 top_title과 같은 제목을 가진 모든 문서를 new_sorted_cluster에 추가
        count = 0
        for i, title in enumerate(self.titles):
            if title == top_title:
                new_similar = top_doc[0]
                count += 1
                new_doc = (
                    top_doc[0],
                    self.titles[i],
                    self.doc_dates[i],
                    self.texts[i],
                    self.doc_urls[i],
                    self.image_url[i],
                )
                new_sorted_cluster.append(new_doc)
        for i in range(count - 1):
            fix_similar = list(new_sorted_cluster[i])
            fix_similar[0] = fix_similar[0] + 0.3 * count
            new_sorted_cluster[i] = tuple(fix_similar)
        # sorted_cluster에서 new_sorted_cluster에 없는 제목만 추가
        for doc in sorted_cluster:
            doc_title = doc[1]
            # 이미 new_sorted_cluster에 추가된 제목은 제외
            if doc_title != top_title:
                new_sorted_cluster.append(doc)

        return new_sorted_cluster, count

    def best_docs(self, user_question):
        # 사용자 질문
        start = time.time()
        query_noun = self.transformed_query(user_question)
        query_time = time.time() - start
        print(f"명사화 시간{query_time}")
        if len(query_noun) == 0:
            return None, None
        # print(f"=================\n\n question: {user_question} 추출된 명사: {query_noun}")

        #######  최근 공지사항, 채용, 세미나, 행사, 특강의 단순한 정보를 요구하는 경우를 필터링 하기 위한 매커니즘 ########
        remove_noticement = [
            "제일",
            "가장",
            "공고",
            "공지사항",
            "필독",
            "첨부파일",
            "수업",
            "컴퓨터학부",
            "컴학",
            "상위",
            "정보",
            "관련",
            "세미나",
            "행사",
            "특강",
            "강연",
            "공지사항",
            "채용",
            "공고",
            "최근",
            "최신",
            "지금",
            "현재",
        ]
        query_nouns = [noun for noun in query_noun if noun not in remove_noticement]
        return_docs = []
        key = None
        numbers = 5  ## 기본으로 5개 문서 반환할 것.
        check_num = 0
        for noun in query_nouns:
            if "개" in noun:
                # 숫자 추출
                num = re.findall(r"\d+", noun)
                if num:
                    numbers = int(num[0])
                    check_num = 1
        if (
            any(
                keyword in query_noun
                for keyword in [
                    "세미나",
                    "행사",
                    "특강",
                    "강연",
                    "공지사항",
                    "채용",
                    "공고",
                ]
            )
            and any(
                keyword in query_noun for keyword in ["최근", "최신", "지금", "현재"]
            )
            and len(query_nouns) < 1
            or check_num == 1
        ):
            # print(query_nouns)
            if "공지사항" in query_noun:
                key = ["공지사항"]
                i = 0
                return_docs.append(
                    (
                        self.titles[i],
                        self.doc_dates[i],
                        self.texts[i],
                        self.doc_urls[i],
                        self.image_url[i],
                    )
                )
                i += 1
                start = i
                while i < numbers:
                    if self.titles[start] != self.titles[start - 1]:
                        i += 1
                    return_docs.append(
                        (
                            self.titles[start],
                            self.doc_dates[start],
                            self.texts[start],
                            self.doc_urls[start],
                            self.image_url[start],
                        )
                    )
                    start += 1

            elif "채용" in query_noun:
                key = ["채용"]
                url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b&wr_id="
                ind = self.find_url(url, self.doc_urls)
                i = ind
                return_docs.append(
                    (
                        self.titles[i],
                        self.doc_dates[i],
                        self.texts[i],
                        self.doc_urls[i],
                        self.image_url[i],
                    )
                )
                i += 1
                start = i
                while i < ind + numbers:
                    if self.titles[start] != self.titles[start - 1]:
                        i += 1
                    return_docs.append(
                        (
                            self.titles[start],
                            self.doc_dates[start],
                            self.texts[start],
                            self.doc_urls[start],
                            self.image_url[start],
                        )
                    )
                    start += 1
            else:
                url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4&wr_id="
                ind = self.find_url(url, self.doc_urls)
                other_key = ["세미나", "행사", "특강", "강연"]
                key = [keyword for keyword in other_key if keyword in user_question]
                i = ind
                return_docs.append(
                    (
                        self.titles[i],
                        self.doc_dates[i],
                        self.texts[i],
                        self.doc_urls[i],
                        self.image_url[i],
                    )
                )
                i += 1
                start = i
                while i < ind + numbers:
                    if self.titles[start] != self.titles[start - 1]:
                        i += 1
                    return_docs.append(
                        (
                            self.titles[start],
                            self.doc_dates[start],
                            self.texts[start],
                            self.doc_urls[start],
                            self.image_url[start],
                        )
                    )
                    start += 1

        # for idx, (titl, dat, tex, ur, image_ur) in enumerate(return_docs):
        #     print(f"순위 {idx+1}: 제목: {titl},본문 {len(tex)} 날짜: {dat}, URL: {ur}")
        #     print("-" * 50)
        if len(return_docs) > 0:
            return return_docs, key

        #######################################그 외의 경우는 일반적으로 처리함 #####################################################################################

        remove_noticement = [
            "제일",
            "가장",
            "공고",
            "공지사항",
            "필독",
            "첨부파일",
            "수업",
            "컴퓨터학부",
            "컴학",
            "상위",
            "관련",
        ]
        query_noun = [noun for noun in query_noun if noun not in remove_noticement]
        start = time.time()
        title_question_similarities = self.bm25_titles.get_scores(
            query_noun
        )  # 제목과 사용자 질문 간의 유사도
        title_question_similarities /= 24

        adjusted_similarities = self.adjust_similarity_scores(
            query_noun, self.titles, title_question_similarities
        )
        # 유사도 기준 상위 15개 문서 선택
        top_20_titles_idx = np.argsort(title_question_similarities)[-25:][::-1]

        # 결과 출력
        # print("최종 정렬된 BM25 문서:")
        # for idx in top_20_titles_idx:  # top_20_titles_idx에서 각 인덱스를 가져옴
        #     print(f"  제목: {titles[idx]} 유사도: {title_question_similarities[idx]} 날짜: {doc_dates[idx]}")
        #     print("-" * 50)

        Bm25_best_docs = [
            (
                self.titles[i],
                self.doc_dates[i],
                self.texts[i],
                self.doc_urls[i],
                self.image_url[i],
            )
            for i in top_20_titles_idx
        ]
        bm25_time = time.time() - start
        print(f"BM25 생성 시간 : {bm25_time}")
        ####################################################################################################
        start = time.time()
        # 1. Dense Retrieval - Text 임베딩 기반 20개 문서 추출
        embedded_datas = self.model.encode(user_question).tolist()  # 사용자 질문 임베딩

        # Pinecone에서 텍스트에 대한 가장 유사한 벡터 20개 추출
        pinecone_results_text = self.index.query(
            vector=embedded_datas, top_k=60, include_values=False, include_metadata=True
        )
        pinecone_similarities_text = [
            res["score"] for res in pinecone_results_text["matches"]
        ]
        pinecone_docs_text = [
            (
                res["metadata"].get("title", ""),
                res["metadata"].get("date", ""),
                res["metadata"].get("text", ""),
                res["metadata"].get("url", ""),
            )
            for res in pinecone_results_text["matches"]
        ]

        dense_time = time.time() - start
        print(f"파인콘 추출 시간 {dense_time}")

        start = time.time()
        #####파인콘으로 구한  문서 추출 방식 결합하기.
        combine_dense_docs = []

        # 1. 본문 기반 문서를 combine_dense_docs에 먼저 추가
        for idx, text_doc in enumerate(pinecone_docs_text):
            text_similarity = pinecone_similarities_text[idx] * 3.3
            text_similarity = self.adjust_date_similarity(
                text_similarity, text_doc[1], query_noun
            )
            matching_noun = [noun for noun in query_noun if noun in text_doc[2]]

            # # 본문에 포함된 명사 수 기반으로 유사도 조정
            for noun in matching_noun:
                text_similarity += len(noun) * 0.22
                if re.search(r"\d", noun):  # 숫자가 포함된 단어 확인
                    if (
                        noun in text_doc[2]
                    ):  # 본문에도 숫자 포함 단어가 있는 경우 추가 조정
                        text_similarity += len(noun) * 0.24
                    else:
                        text_similarity += len(noun) * 0.20
            combine_dense_docs.append(
                (text_similarity, text_doc)
            )  # (유사도, (제목, 날짜, 본문, URL))

        ####query_noun에 포함된 키워드로 유사도를 보정
        # 유사도 기준으로 내림차순 정렬
        combine_dense_docs.sort(key=lambda x: x[0], reverse=True)
        dense1_time = time.time() - start
        print(f"파인콘 뽑은 문서 유사도 조정 시간 {dense1_time}")
        # ## 결과 출력
        # print("\n통합된 파인콘문서 유사도:")
        # for score, doc in combine_dense_docs:
        #     title, date, text, url = doc
        #     print(f"제목: {title} 유사도: {score} 날짜: {date}")
        #     print('---------------------------------')

        #################################################3#################################################3
        #####################################################################################################3
        start = time.time()
        # Step 1: combine_dense_docs에 제목, 본문, 날짜, URL을 미리 저장

        # combine_dense_doc는 (유사도, 제목, 본문 내용, 날짜, URL) 형식으로 데이터를 저장합니다.
        combine_dense_doc = []

        # combine_dense_docs의 내부 구조에 맞게 두 단계로 분해
        for score, (title, date, text, url) in combine_dense_docs:
            combine_dense_doc.append((score, title, text, date, url, "No Content"))

        combine_dense_doc = self.last_filter_keyword(
            combine_dense_doc, query_noun, user_question
        )

        # Step 2: combine_dense_docs와 BM25 결과 합치기
        final_best_docs = []

        # combine_dense_docs와 BM25 결과를 합쳐서 처리
        for score, title, text, date, url, img in combine_dense_doc:
            matched = False
            for bm25_doc in Bm25_best_docs:
                if bm25_doc[0] == title:  # 제목이 일치하면 유사도를 합산
                    combined_similarity = (
                        score + adjusted_similarities[self.titles.index(bm25_doc[0])]
                    )
                    # combined_similarity=adjust_date_similarity(combined_similarity,bm25_doc[1])
                    final_best_docs.append(
                        (
                            combined_similarity,
                            bm25_doc[0],
                            bm25_doc[1],
                            bm25_doc[2],
                            bm25_doc[3],
                            bm25_doc[4],
                        )
                    )
                    matched = True
                    break
            if not matched:

                # 제목이 일치하지 않으면 combine_dense_docs에서만 유사도 사용
                final_best_docs.append((score, title, date, text, url, "No content"))

        # 제목이 일치하지 않는 BM25 문서도 추가
        for bm25_doc in Bm25_best_docs:
            matched = False
            for score, title, text, date, url, img in combine_dense_doc:
                if (
                    bm25_doc[0] == title and bm25_doc[1] == text
                ):  # 제목이 일치하면 matched = True로 처리됨
                    matched = True
                    break
            if not matched:
                # 제목이 일치하지 않으면 BM25 문서만 final_best_docs에 추가
                if bm25_doc[0] in self.titles:
                    combined_similarity = adjusted_similarities[
                        self.titles.index(bm25_doc[0])
                    ]  # BM25 유사도 가져오기
                    combined_similarity = self.adjust_date_similarity(
                        combined_similarity, bm25_doc[1], query_noun
                    )
                    final_best_docs.append(
                        (
                            combined_similarity,
                            bm25_doc[0],
                            bm25_doc[1],
                            bm25_doc[2],
                            bm25_doc[3],
                            bm25_doc[4],
                        )
                    )
        final_best_docs.sort(key=lambda x: x[0], reverse=True)
        final_best_docs = final_best_docs[:20]

        combine_time = time.time() - start
        print(f"Bm+ pinecone 결합 시간 {combine_time}")

        # print("\n\n\n\n필터링 전 최종문서 (유사도 큰 순):")
        # for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(final_best_docs):
        #     print(f"순위 {idx+1}: 제목: {titl}, 유사도: {scor},본문 {len(tex)} 날짜: {dat}, URL: {ur}")
        #     print("-" * 50)

        start = time.time()

        final_best_docs = self.last_filter_keyword(
            final_best_docs, query_noun, user_question
        )
        final_best_docs.sort(key=lambda x: x[0], reverse=True)
        filter_time = time.time() - start
        print(f"키워드 필터 이후 시간 {filter_time}")
        print("\n\n\n\n중간필터 최종문서 (유사도 큰 순):")
        for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(
            final_best_docs[:10]
        ):
            print(
                f"순위 {idx+1}: 제목: {titl}, 유사도: {scor},본문 {len(tex)} 날짜: {dat}, URL: {ur}"
            )
            print("-" * 50)

        start = time.time()
        # Step 1: Cluster documents by similarity
        clusters = self.cluster_documents_by_similarity(final_best_docs)
        cluster_time = time.time() - start
        print(f"군집화 시간{cluster_time}")

        # Step 2: Compare cluster[0] cluster[1] top similarity and check condition
        top_0_cluster_similar = clusters[0][0][0]
        top_1_cluster_similar = clusters[1][0][0]

        keywords = ["최근", "최신", "현재", "지금"]

        start = time.time()

        if (
            top_0_cluster_similar - top_1_cluster_similar <= 0.3
        ):  ## 질문이 모호했다는 의미일 수 있음.. (예를 들면 수강신청 언제야? 인데 구체적으로 1학기인지, 2학기인지, 겨울, 여름인지 모르게..)
            # 날짜를 비교해 더 최근 날짜를 가진 클러스터 선택
            # 조금더 세밀하게 들어가자면?
            # print("세밀하게..")
            if (
                any(keyword in word for word in query_noun for keyword in keywords)
                or top_0_cluster_similar - clusters[len(clusters) - 1][0][0] <= 0.3
            ):
                # print("최근이거나 뽑은 문서들이 유사도 0.3이내")
                if top_0_cluster_similar - clusters[len(clusters) - 1][0][0] <= 0.3:
                    # print("최근이면서 뽑은 문서들이 유사도 0.3이내 real")
                    sorted_cluster = sorted(
                        clusters, key=lambda doc: doc[0][2], reverse=True
                    )
                    sorted_cluster = sorted_cluster[0]
                else:
                    # print("최근이면서 뽑은 문서들이 유사도 0.3이상")
                    if top_0_cluster_similar - top_1_cluster_similar <= 0.3:
                        # print("최근이면서 뽑은 두문서의 유사도 0.3이하이라서 두 문서로 줄임")
                        date1 = self.parse_date_change_korea_time(clusters[0][0][2])
                        date2 = self.parse_date_change_korea_time(clusters[1][0][2])
                        result_date = (date1 - date2).days
                        if result_date < 0:
                            result_docs = clusters[1]
                        else:
                            result_docs = clusters[0]
                        sorted_cluster = sorted(
                            result_docs, key=lambda doc: doc[2], reverse=True
                        )

                    else:
                        sorted_cluster = sorted(
                            clusters, key=lambda doc: doc[0][0], reverse=True
                        )
                        sorted_cluster = sorted_cluster[0]
            else:
                # print("두 클러스터로 판단해보자..")
                if top_0_cluster_similar - top_1_cluster_similar <= 0.1:
                    # print("진짜 차이가 없는듯..?")
                    date1 = self.parse_date_change_korea_time(clusters[0][0][2])
                    date2 = self.parse_date_change_korea_time(clusters[1][0][2])
                    result_date = (date1 - date2).days
                    if result_date < 0:
                        # print("두번째 클러스터가 더 크네..?")
                        result_docs = clusters[1]
                    else:
                        # print("첫번째 클러스터가 더 크네..?")
                        result_docs = clusters[0]
                    sorted_cluster = sorted(
                        result_docs, key=lambda doc: doc[2], reverse=True
                    )
                else:
                    # print("에이 그래도 유사도 차이가 있긴하네..")
                    result_docs = clusters[0]
                    sorted_cluster = sorted(
                        result_docs, key=lambda doc: doc[0], reverse=True
                    )
        else:  # 질문이 모호하지 않을 가능성 업업
            number_pattern = r"\d"
            period_word = ["여름", "겨울"]
            if (
                any(keyword in word for word in query_noun for keyword in keywords)
                or not any(re.search(number_pattern, word) for word in query_noun)
                or not any(key in word for word in query_noun for key in period_word)
            ):
                # print("최근 최신이라는 말 드가거나 2가지 모호한 판단 기준")
                if any(re.search(number_pattern, word) for word in query_noun) or any(
                    key in word for word in query_noun for key in period_word
                ):
                    # print("최신인줄 알았지만 유사도순..")
                    result_docs = clusters[0]
                    num = 0
                    for doc in result_docs:
                        if re.search(r"\d+차", doc[1]):
                            num += 1
                    if num > 1:
                        sorted_cluster = sorted(
                            result_docs, key=lambda doc: doc[2], reverse=True
                        )
                    else:
                        sorted_cluster = sorted(
                            result_docs, key=lambda doc: doc[0], reverse=True
                        )
                else:
                    # print("너는 그냥 최신순이 맞는거여..")
                    result_docs = clusters[0]
                    sorted_cluster = sorted(
                        result_docs, key=lambda doc: doc[2], reverse=True
                    )
            else:
                # print("진짜 유사도순대로")
                result_docs = clusters[0]
                sorted_cluster = sorted(
                    clusters[0], key=lambda doc: doc[0], reverse=True
                )

        final1_time = time.time() - start
        print(f"군집화 문서 비교해서 상위문서1개 뽑는 시간{final1_time}")

        # 예시 사용
        start = time.time()
        final_cluster, count = self.organize_documents_v2(sorted_cluster)
        result_time = time.time() - start
        print(f"동일한 문서 제목인거 추가 리스트에 시간{result_time}")
        print("\n\n\n\n최종 상위 문서 (유사도 및 날짜 기준 정렬):")
        for idx, (scor, titl, dat, tex, ur, image_ur) in enumerate(final_cluster):
            print(
                f"순위 {idx+1}: 제목: {titl}, 유사도: {scor}, 날짜: {dat}, URL: {ur} 내용: {len(tex)}   이미지{len(image_ur)}"
            )
            print("-" * 50)
        print("\n\n\n")
        # return final_cluster[:count], query_noun

    def run(self, user_question):
        self.initialize_bm25_titles()
        self.best_docs(user_question)