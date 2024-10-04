import re
from bs4 import BeautifulSoup

def custom_sort(item, priority_list=[]):
    # Bỏ qua ký tự '$' nếu có
    # item = item.lstrip('$').upper()  # Bỏ ký tự '$' và chuyển thành chữ hoa

    # # Tìm các phần chữ cái và số trong item
    # parts = re.findall(r'([A-Z]+)(\d+\.?\d*)?', item)
    # key = []

    # # Phần đầu tiên trong parts phải là chữ cái
    # letter_part = parts[0][0]
    
    # # Kiểm tra xem phần chữ cái có nằm trong danh sách ưu tiên không
    # if letter_part in priority_list:
    #     key.append(priority_list.index(letter_part))
    # else:
    #     key.append(len(priority_list))  # Đặt ở cuối nếu không có trong danh sách ưu tiên
    
    # # Thêm số để sắp xếp, bao gồm cả số thập phân
    # for letter, number in parts:
    #     if number:
    #         key.extend([int(n) for n in number.split('.') if n])
    # return key

    def split_key(item):
        # Sử dụng regex để tách chuỗi thành prefix, số chính (int hoặc float), và phần phụ
        match = re.match(r"([A-Za-z]+)([\d.]+)?(?:[._](\d+))?", item)
        if match:
            prefix = match.group(1)
            # Sử dụng float để xử lý số thập phân, nếu không có thì trả về -1
            num1 = float(match.group(2)) if match.group(2) else -1
            num2 = int(match.group(3)) if match.group(3) else -1
            return (prefix, num1, num2)
        return (item, -1, -1)

    # Hàm sort_key tùy chỉnh
        # Kiểm tra từng ký tự đại diện và trả về thứ tự ưu tiên của nó
    for i, pattern in enumerate(priority_list):
        if item.startswith(pattern):
            # Tách chuỗi thành tuple để so sánh
            split = split_key(item)
            return (i, *split)
    return (len(priority_list), item)

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