import re
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Union, Tuple

def _parse_timestamp(timestamp):
    
    # Loại bỏ phần ICT khỏi chuỗi thời gian
    dt_without_tz = " ".join(timestamp.split()[:-1])
    # Chuyển đổi chuỗi thành datetime
    dt = pd.to_datetime(dt_without_tz)
    # Áp dụng lại múi giờ ICT
    return dt.tz_localize('Asia/Bangkok')

def custom_sort(item: str, priority_list: List[str] = None, loop_list: List[str] = None) -> Tuple:
    if priority_list is None:
        priority_list = []
    if loop_list is None:
        loop_list = []
    
    def split_key(s: str) -> Union[Tuple[str, float, float, float, float, int], Tuple[None]]:
        pattern = r"([A-Za-z]+)(\d+)(?:_(\d+))?(?:_(\d+))?(?:RANK(\d+))?(?:LOOP([A-Za-z0-9]+))?"
        match = re.match(pattern, s)
        
        if match:
            key = match.group(1)
            numlevel1 = float(match.group(2))
            numlevel2 = float(match.group(3)) if match.group(3) else 0
            numlevel3 = float(match.group(4)) if match.group(4) else 0
            numrank = float(match.group(5)) if match.group(5) else 0
            if loop_list and match.group(6):
                numloop = loop_list.index(match.group(6)) if match.group(6) in loop_list else 0
            else:
                numloop = 0
            return (key, numlevel1, numlevel2, numlevel3, numrank, numloop)
        else:
            # Trả về giá trị mặc định để tránh lỗi khi giải nén
            return (None, 0, 0, 0, 0, 0)

    # Sắp xếp theo thứ tự ưu tiên trong priority_list nếu có
    for i, pattern in enumerate(priority_list):
        if item.startswith(pattern):
            # Tách chuỗi thành tuple để so sánh
            split = split_key(item)
            if split[0] is not None:  # Chỉ trả về nếu regex khớp
                return (i, *split)
    
    # Trả về vị trí cuối nếu không có ưu tiên
    return (len(priority_list), item)

# def custom_sort(item, priority_list=[]):
#     def split_key(item):
#         # Sử dụng regex để tách chuỗi thành prefix, số chính (int hoặc float), và phần phụ
#         match = re.match(r"([A-Za-z]+)([\d.]+)?(?:[._](\d+))?", item)
#         if match:
#             prefix = match.group(1)
#             # Sử dụng float để xử lý số thập phân, nếu không có thì trả về -1
#             num1 = float(match.group(2)) if match.group(2) else -1
#             num2 = int(match.group(3)) if match.group(3) else -1
#             return (prefix, num1, num2)
#         return (item, -1, -1)

#     # Hàm sort_key tùy chỉnh
#         # Kiểm tra từng ký tự đại diện và trả về thứ tự ưu tiên của nó
#     for i, pattern in enumerate(priority_list):
#         if item.startswith(pattern):
#             # Tách chuỗi thành tuple để so sánh
#             split = split_key(item)
#             return (i, *split)
#     return (len(priority_list), item)

def parse_html(text: str, max_length=200):
    def truncate_text(text: str, max_length):
        if len(text) <= max_length:
            return text
        
        truncate_sign = ['?', ',', '-', '_', ':', '(', '.']

        # Tách chuỗi thành các câu dựa vào dấu chấm "."
        for i in truncate_sign:
            if i in text:
                sentences = text.split(i)
                break

        # Ghép lại các câu nhưng loại bỏ câu cuối cùng nếu chuỗi vượt quá giới hạn
        truncated_text = ''
        
        for sentence in sentences[:-1]:  # Duyệt qua các câu trừ câu cuối
            # Thêm câu hiện tại vào chuỗi đã cắt và một dấu chấm nếu cần
            if len(truncated_text) + len(sentence) + 1 <= max_length:
                truncated_text += sentence + '.'
            else:
                break
        if truncated_text == '':
            truncated_text = text
        return truncated_text.strip()  # Xoá khoảng trắng dư thừa

    # Chỉ chuyển thành chuỗi một lần, nếu cần thiết
    if not isinstance(text, str):
        text = str(text)
    
    # Kiểm tra nhanh xem nội dung có phải là HTML
    if bool(BeautifulSoup(text, "html.parser").find()): 
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
    
    # Sử dụng replace thay cho re.sub cho các ký tự cụ thể
    for char in ['"', "'", "’"]:
        text = text.replace(char, '')

    # Tối ưu hóa regex chỉ dùng một lần re.sub
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) > max_length:
        truncate = truncate_text(text, max_length=max_length)
        print(f'*Text: {text}')
        print(f'-> Truncate to: {truncate}')
        return truncate
    else:
        return text