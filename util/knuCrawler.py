import requests, re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


class KnuCrawler:

    def __init__(self, preprosessor):

        self.preprocessor = preprosessor

        self.base_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_1"

    def get_latest_wr_id_공지사항(self):
        url = self.base_url
        response = requests.get(url)
        if response.status_code == 200:
            match = re.search(r"wr_id=(\d+)", response.text)
            if match:
                return int(match.group(1))
        return None

    def make_crawling_urls_공지사항(self):
        urls = []
        target_pages = [27510, 27047, 27614, 27246, 25900, 27553, 25896]

        now_number = self.get_latest_wr_id_공지사항()
        for number in range(now_number, 27726, -1):
            urls.append(f"{self.base_url}&wr_id=" + str(number))

        for page in target_pages:
            urls.append(f"{self.base_url}&wr_id={page}")

        return urls

    def extract_text_and_date_from_url(self, urls):
        all_data = []

        def fetch_text_and_date(url):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")

                # 제목 추출
                title_element = soup.find("span", class_="bo_v_tit")
                title = (
                    title_element.get_text(strip=True)
                    if title_element
                    else "Unknown Title"
                )

                # 본문 텍스트와 이미지 URL을 분리하여 저장
                text_content = "Unknown Content"  # 텍스트 초기화
                image_content = []  # 이미지 URL을 담는 리스트 초기화

                # 본문 내용 추출
                paragraphs = soup.find("div", id="bo_v_con")
                if paragraphs:
                    # paragraphs 내부에서 'p', 'div', 'li' 태그 텍스트 추출
                    text_content = "\n".join(
                        [
                            element.get_text(strip=True)
                            for element in paragraphs.find_all(["p", "div", "li"])
                        ]
                    )
                    # print(text_content)
                    if text_content.strip() == "":
                        text_content = ""
                    # 이미지 URL 추출
                    for img in paragraphs.find_all("img"):
                        img_src = img.get("src")
                        if img_src:
                            image_content.append(img_src)

                # 날짜 추출
                date_element = soup.select_one("strong.if_date")  # 수정된 선택자
                date = (
                    date_element.get_text(strip=True)
                    if date_element
                    else "Unknown Date"
                )

                # 제목이 Unknown Title이 아닐 때만 데이터 추가
                if title != "Unknown Title":
                    return (
                        title,
                        text_content,
                        image_content,
                        date,
                        url,
                    )  # 문서 제목, 본문 텍스트, 이미지 URL 리스트, 날짜, URL 반환
                else:
                    return (
                        None,
                        None,
                        None,
                        None,
                        None,
                    )  # 제목이 Unknown일 경우 None 반환
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return None, None, None, None, url

        with ThreadPoolExecutor() as executor:
            results = executor.map(fetch_text_and_date, urls)

        # 유효한 데이터만 추가
        all_data = [
            (title, text_content, image_content, date, url)
            for title, text_content, image_content, date, url in results
            if title is not None
        ]
        return all_data

    def crawlAnnouncements(self):
        urls = self.make_crawling_urls_공지사항()
        all_data = self.extract_text_and_date_from_url(
            urls
        )  # 크롤링 및 전처리 된 데이터
        return all_data

    # 정교수
    def extract_professor_info_from_정교수_url(self, url):
        all_data = []

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find("div", id="dr").find_all("li")

            for professor in professor_elements:
                # 이미지 URL 추출
                image_element = professor.find("div", class_="dr_img").find("img")
                image_content = (
                    image_element["src"] if image_element else "Unknown Image URL"
                )

                # 이름 추출
                name_element = professor.find("div", class_="dr_txt").find("h3")
                title = (
                    name_element.get_text(strip=True)
                    if name_element
                    else "Unknown Name"
                )

                # 연락처와 이메일 추출 후 하나의 텍스트로 결합
                contact_info = professor.find("div", class_="dr_txt").find_all("dd")
                contact_number = (
                    contact_info[0].get_text(strip=True)
                    if len(contact_info) > 0
                    else "Unknown Contact Number"
                )
                email = (
                    contact_info[1].get_text(strip=True)
                    if len(contact_info) > 1
                    else "Unknown Email"
                )
                text_content = f"{title}, {contact_number}, {email}"

                # 날짜와 URL 설정
                date = "작성일24-01-01 00:00"

                prof_url_element = professor.find("a")
                prof_url = (
                    prof_url_element["href"] if prof_url_element else "Unknown URL"
                )

                # 각 교수의 정보를 all_data에 추가
                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

        return all_data

    # 초빙교수
    def extract_professor_info_from_초빙교수_url(self, url):
        all_data = []

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find("div", id="Student").find_all("li")

            for professor in professor_elements:
                # 이미지 URL 추출
                image_element = professor.find("div", class_="img").find("img")
                image_content = (
                    image_element["src"] if image_element else "Unknown Image URL"
                )

                # 이름 추출
                name_element = professor.find("div", class_="cnt").find(
                    "div", class_="name"
                )
                title = (
                    name_element.get_text(strip=True)
                    if name_element
                    else "Unknown Name"
                )

                # 연락처와 이메일 추출
                contact_place = (
                    professor.find("div", class_="dep").get_text(strip=True)
                    if professor.find("div", class_="dep")
                    else "Unknown Contact Place"
                )
                email_element = (
                    professor.find("dl", class_="email").find("dd").find("a")
                )
                email = (
                    email_element.get_text(strip=True)
                    if email_element
                    else "Unknown Email"
                )

                # 텍스트 내용 조합
                text_content = (
                    f"성함(이름):{title}, 연구실(장소):{contact_place}, 이메일:{email}"
                )

                # 날짜와 URL 설정
                date = "작성일24-01-01 00:00"
                prof_url = url

                # 각 교수의 정보를 all_data에 추가
                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

        return all_data

    # 직원
    def extract_professor_info_from_직원_url(self, url):
        all_data = []

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # 교수 정보가 담긴 요소들 선택
            professor_elements = soup.find("div", id="Student").find_all("li")

            for professor in professor_elements:
                # 이미지 URL 추출
                image_element = professor.find("div", class_="img").find("img")
                image_content = (
                    image_element["src"] if image_element else "Unknown Image URL"
                )

                # 이름 추출
                name_element = professor.find("div", class_="cnt").find("h1")
                title = (
                    name_element.get_text(strip=True)
                    if name_element
                    else "Unknown Name"
                )

                # 연락처 추출
                contact_number_element = professor.find("span", class_="period")
                contact_number = (
                    contact_number_element.get_text(strip=True)
                    if contact_number_element
                    else "Unknown Contact Number"
                )

                # 연구실 위치 추출
                contact_info = professor.find_all("dl", class_="dep")
                contact_place = (
                    contact_info[0].find("dd").get_text(strip=True)
                    if len(contact_info) > 0
                    else "Unknown Contact Place"
                )

                # 이메일 추출
                email = (
                    contact_info[1].find("dd").find("a").get_text(strip=True)
                    if len(contact_info) > 1
                    else "Unknown Email"
                )

                # 담당 업무 추출
                role = (
                    contact_info[2].find("dd").get_text(strip=True)
                    if len(contact_info) > 2
                    else "Unknown Role"
                )

                # 텍스트 내용 조합
                text_content = f"성함(이름):{title}, 연락처(전화번호):{contact_number}, 사무실(장소):{contact_place}, 이메일:{email}, 담당업무:{role}"

                # 날짜와 URL 설정
                date = "작성일24-01-01 00:00"
                prof_url = url

                # 각 교수의 정보를 all_data에 추가
                all_data.append((title, text_content, image_content, date, prof_url))

        except Exception as e:
            print(f"Error processing {url}: {e}")

        return all_data

    def crawlProfessors(self):
        prof_data = []

        정교수_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_1&lang=kor"
        초빙교수_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_2&lang=kor"
        직원_url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub2_5&lang=kor"

        prof_data += self.extract_professor_info_from_정교수_url(정교수_url)
        prof_data += self.extract_professor_info_from_초빙교수_url(초빙교수_url)
        prof_data += self.extract_professor_info_from_직원_url(직원_url)

        return prof_data

    def get_latest_wr_id_취업정보(self):
        url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b"
        response = requests.get(url)
        if response.status_code == 200:
            # re.findall을 사용하여 모든 wr_id 값을 찾아 리스트로 반환
            match = re.findall(r"wr_id=(\d+)", response.text)
            if match:
                # wr_ids 리스트에 있는 모든 wr_id 값을 출력
                max_wr_id = max(int(wr_id) for wr_id in match)
                return max_wr_id
        return None

    def make_crawling_urls_취업정보(self):
        now_company_number = self.get_latest_wr_id_취업정보()

        company_urls = []
        for number in range(now_company_number, 1149, -1):
            company_urls.append(
                "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_3_b&wr_id="
                + str(number)
            )
        return company_urls

    def extract_company_from_url(self, urls):
        all_data = []

        def fetch_text_and_date(url):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")

                # 제목 추출
                title_element = soup.find("span", class_="bo_v_tit")
                title = (
                    title_element.get_text(strip=True)
                    if title_element
                    else "Unknown Title"
                )

                # 본문 텍스트와 이미지 URL을 분리하여 저장
                text_content = "Unknown Content"  # 텍스트 초기화
                image_content = []  # 이미지 URL을 담는 리스트 초기화

                # 본문 내용 추출
                paragraphs = soup.find("div", id="bo_v_con")
                if paragraphs:
                    # paragraphs 내부에서 'p', 'div', 'li' 태그 텍스트 추출
                    text_content = "\n".join(
                        [
                            element.get_text(strip=True)
                            for element in paragraphs.find_all(["p", "div", "li"])
                        ]
                    )
                    # print(text_content)
                    if text_content.strip() == "":
                        text_content = ""
                    # 이미지 URL 추출
                    for img in paragraphs.find_all("img"):
                        img_src = img.get("src")
                        if img_src:
                            image_content.append(img_src)

                # 날짜 추출
                date_element = soup.select_one("strong.if_date")  # 수정된 선택자
                date = (
                    date_element.get_text(strip=True)
                    if date_element
                    else "Unknown Date"
                )

                # 제목이 Unknown Title이 아닐 때만 데이터 추가
                if title != "Unknown Title":
                    return (
                        title,
                        text_content,
                        image_content,
                        date,
                        url,
                    )  # 문서 제목, 본문 텍스트, 이미지 URL 리스트, 날짜, URL 반환
                else:
                    return (
                        None,
                        None,
                        None,
                        None,
                        None,
                    )  # 제목이 Unknown일 경우 None 반환
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return None, None, None, None, url

        with ThreadPoolExecutor() as executor:
            results = executor.map(fetch_text_and_date, urls)

        # 유효한 데이터만 추가
        all_data = [
            (title, text_content, image_content, date, url)
            for title, text_content, image_content, date, url in results
            if title is not None
        ]
        return all_data

    def crawlCompanyInfos(self):
        urls = self.make_crawling_urls_취업정보()
        company_data = self.extract_company_from_url(urls)
        return company_data

    def get_latest_wr_id_세미나(self):
        url = "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4"
        response = requests.get(url)
        if response.status_code == 200:
            # re.findall을 사용하여 모든 wr_id 값을 찾아 리스트로 반환
            match = re.findall(r"wr_id=(\d+)", response.text)
            if match:
                # wr_ids 리스트에 있는 모든 wr_id 값을 출력
                max_wr_id = max(int(wr_id) for wr_id in match)
                return max_wr_id
        return None

    def making_crawling_urls_세미나(self):
        now_seminar_number = self.get_latest_wr_id_세미나()
        seminar_urls = []
        for number in range(now_seminar_number, 246, -1):
            seminar_urls.append(
                "https://cse.knu.ac.kr/bbs/board.php?bo_table=sub5_4&wr_id="
                + str(number)
            )
        return seminar_urls

    def extract_seminar_from_url(self, urls):
        all_data = []

        def fetch_text_and_date(url):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, "html.parser")

                # 제목 추출
                title_element = soup.find("span", class_="bo_v_tit")
                title = (
                    title_element.get_text(strip=True)
                    if title_element
                    else "Unknown Title"
                )

                # 본문 텍스트와 이미지 URL을 분리하여 저장
                text_content = "Unknown Content"  # 텍스트 초기화
                image_content = []  # 이미지 URL을 담는 리스트 초기화

                # 본문 내용 추출
                paragraphs = soup.find("div", id="bo_v_con")
                if paragraphs:
                    # paragraphs 내부에서 'p', 'div', 'li' 태그 텍스트 추출
                    text_content = "\n".join(
                        [
                            element.get_text(strip=True)
                            for element in paragraphs.find_all(["p", "div", "li"])
                        ]
                    )
                    # print(text_content)
                    if text_content.strip() == "":
                        text_content = ""
                    # 이미지 URL 추출
                    for img in paragraphs.find_all("img"):
                        img_src = img.get("src")
                        if img_src:
                            image_content.append(img_src)

                # 날짜 추출
                date_element = soup.select_one("strong.if_date")  # 수정된 선택자
                date = (
                    date_element.get_text(strip=True)
                    if date_element
                    else "Unknown Date"
                )

                # 제목이 Unknown Title이 아닐 때만 데이터 추가
                if title != "Unknown Title":
                    return (
                        title,
                        text_content,
                        image_content,
                        date,
                        url,
                    )  # 문서 제목, 본문 텍스트, 이미지 URL 리스트, 날짜, URL 반환
                else:
                    return (
                        None,
                        None,
                        None,
                        None,
                        None,
                    )  # 제목이 Unknown일 경우 None 반환
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return None, None, None, None, url

        with ThreadPoolExecutor() as executor:
            results = executor.map(fetch_text_and_date, urls)

        # 유효한 데이터만 추가
        all_data = [
            (title, text_content, image_content, date, url)
            for title, text_content, image_content, date, url in results
            if title is not None
        ]
        return all_data

    def crawlSeminarInfos(self):
        urls = self.making_crawling_urls_세미나()
        seminar_data = self.extract_seminar_from_url(urls)
        return seminar_data

    def run(self):
        # 공지사항
        announcement_data = self.crawlAnnouncements()
        self.preprocessor.processAnnouncement(announcement_data)

        # 정교수, 초빙교수, 직원
        professor_data = self.crawlProfessors()
        self.preprocessor.processProfessors(professor_data)

        # 취업정보
        company_data = self.crawlCompanyInfos()
        self.preprocessor.processCompanyInfos(company_data)

        # 세미나
        seminar_data = self.crawlSeminarInfos()
        self.preprocessor.processSeminarInfos(seminar_data)
