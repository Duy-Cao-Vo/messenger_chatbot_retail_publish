"""

DOCUMENTATION
'''very message have 3 part: text, quick_replies, buttons'''
tag: Intent need to be classification
ask: Sample ask questions to ML can learn, the many sample it have, the more accuracy it can accomplished
answer: With the tag intent, want to answer
context: group of stories
quick_replies: recommend the answer for user to pick, NOTE: quick_replies not long than 20 characters
buttons: buttion postback, NOTE: sSet of 1-3 buttons that appear as call-to-actions.

"""
import pandas as pd
import json

# Load df product and store location
df = pd.read_csv('./data_input/data_product.csv')
df_address = pd.read_csv('./data_input/aldo_location - aldo_store.csv')
df = pd.merge(df, df_address[['store_name', 'address', 'latitude', 'longitude']], on='store_name')

# change Aldo name into ez to read for user
df[['store_index', 'store_name']] = df.store_name.str.split('-', expand=True)
df.loc[~df.store_name.str.contains('Aldo'), 'store_name'] = 'Aldo' + df['store_name']
df = df[~df.store_name.str.contains('Outlet')]

# Split product name into like in fanpage
df[['pro_name', 'code', 'season', 'something1', 'something2']] = df.product_name.str.split(' ', expand=True)
df['pro_name'] = df['pro_name'].str.replace(" ", "")
df['color_code'] = df['color_code'].str.replace(" ", "")
df['pro_name_color'] = df['pro_name'] + " " + df['color_code'].astype('str')

pro_name = df.pro_name.unique().tolist()
pro_name_color = df.pro_name_color.unique().tolist()[0:100]
color_code = df.color_code.unique().tolist()[0:100]

data = {"intents": [
    {"tag": "00_greeting",
     "ask": ["Alo", "Hi", "How are you", "Is anyone there?", "Hello", "Xin chào ạ", "Chào ad", "Chào shop",
             "Xin chào page ạ!", "Hello page", "Chào", "Hi there", "Hé lô",
             "Hi bạn", "Hi shope", "Hi ad", "Hello a", "Xin chào anh/chị", "Hi anh", "Hi Chị", "Hello anh/chị",
             "Xin chào đằng đó", "Helo lại là mình đây", "Hello sir", "Hello madam", "Bạn gì đó ơi", "bạn ơi!"],
     "answer": ["Chào {}, mình có thể giúp gì cho bạn", "Xin chào {}, mình có thể giúp gì cho bạn?",
                "Page xin chào {} :D, bạn cần giúp gì nè?", "Xìn chào {} :D, bạn cần giúp gì?"],
     "context": [""],
     "quick_replies": ["Bạn có thể làm gì", "Tìm sản phẩm", "Khuyến mãi", "Cửa hàng", "Tra cứu Covid-19"],
     "buttons": [],
     "entities": ["hi", "xin chào", "hello", "chào"]
     },
    {"tag": "01_goodbye",
     "ask": ["Bye", "Tạm biệt", "Goodbye", "Rất vui được nói chuyện với bạn, bye",
             "Rất vui được nói chuyện với bạn", "good bye", "Chúc bạn ngủ ngon", "G9", "Good night", "pái pai",
             "tạm biệt", "bái bai", "Vĩnh biệt", "See ya", "See you later"],
     "answer": ["Rất vui được giúp bạn, bye!", "Cảm ơn vì đã ghé qua", "Bye! Lần sau gặp lại",
                "Cảm ơn nhá, Bye", "Hẹn gặp lại nhé."],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "02_thanks",
     "ask": ["Thanks", "Thank you", "Cảm ơn ạ", "Awesome, thanks", "Cảm ơn shop ạ", "Cảm ơn shop",
             "Cảm ơn! Page trả lời nhiệt tình quá", "Cảm ơn", "ơn", "Cảm ơn nek", "Cam on shop"],
     "answer": ["Rất vui được giúp bạn!", "Không có gì!", "Any Time", "My pleasure", "It's my pleasure"],
     "context": [""],
     "quick_replies": [":)", "(y)"],
     "buttons": [],
     "entities": ["cảm ơn"]
     },
    {"tag": "03_noanswer",
     "ask": ["abc", "xyz", "ư", "fuck you", "nháo nhào nhao", "meo meo"],
     "answer": ["Xin lỗi, mình không hiểu ý bạn", "Cho mình thêm thông tin được không?",
                "Xin lỗi, page không chắc là hiểu ý bạn", "That's not funny!"],
     "context": [""],
     "quick_replies": [""],
     "buttons": ["Cần hỗ trợ"],
     "entities": [""]
     },
    {"tag": "04_options",
     "ask": ["Page ơi cho mình hỏi?", "Xin hỏi bạn có thể làm gì", "Bạn có thể làm gì?", "Xin hỏi bạn là ai?",
             "Bạn là ai?", "Bạn tên gì", "Bạn tên gì?", "Bạn có thể làm gì?", "Bạn là chatbot phải không?",
             "Bạn là chatbot phải k?"],
     "answer": [
         "Mình là chatbot siêu cute! Mình có thể giúp bạn kiểm tra sản phẩm hiện hữu, tìm size, tìm cửa hàng gần bạn nhất"],
     "context": [""],
     "quick_replies": [""],
     "buttons": ["Tìm sản phẩm", "Khuyến mãi", "Tìm cửa hàng"],
     "entities": ["là", "ai", "làm"]
     },
    {"tag": "05_product_option",
     "ask": ["Tìm sản phẩm?", "Tìm sản phẩm", "Tìm sản phẩm", "Giúp mình tìm sản phẩm", "Mình muốn tìm sản phẩm",
             "Tìm sản phẩm?", "Tìm sản phẩm", "Tìm sản phẩm", "Giúp mình tìm sản phẩm", "Mình muốn tìm sản phẩm",
             "Tìm sản phẩm khác", "Tìm sản phẩm khác", "Tìm sản phẩm khác", "Giúp mình tìm sản phẩm", "Mình muốn tìm sản phẩm",
             "Mình muốn tra cứu sản phẩm", "Mình muốn tìm sản phẩm", "Tìm sản phẩm thì phải làm như thế nào",
             "Cho hỏi là còn bán mẫu này không"],

     "answer": ["Bạn muốn 'tìm bằng mã sản phẩm' hay 'danh mục sản phẩm của ALDO'"],
     "context": ["search_inventory_by_pro_id"],
     "quick_replies": [""],
     "buttons": ["Tìm bằng mã sản phẩm", "Danh mục sản phẩm ALDO"],
     "entities": ["sản phẩm"]
     },
    {"tag": "06_product_id",
     "ask": ["Tìm bằng mã sản phẩm", "Tìm bằng mã sản phẩm", "Mẫu này còn không shop ơi?",
             "Ad ơi cho mình hỏi mẫu này còn không?", "Chào page, cho mình hỏi mẫu này còn không?",
             "Tìm bằng mã sản phẩm", "Ad ơi cho mình hỏi sản phẩm này còn không?", "Bên bạn còn sản phẩm này không?"],
     "answer": ["Xin vui lòng nhập mã sản phẩm ở trên bài viết! VD: ASTIRASSA 95"],
     "context": ["search_inventory_by_pro_id"],
     "quick_replies": ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001'],
     "buttons": [],
     "entities": ["mã", "còn"]
     },
    {"tag": "07_search_inventory_by_pro_id",
     "ask": pro_name_color,
     "answer": ["Đang tìm sản phẩm..."],
     "context": ["store_search", "size"],
     "quick_replies": [""],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "08_search_product_sentence",
     "ask": ["Mình cần tìm mẫu MIKKEL 680", "Mẫu DRIAWIEL 100 còn không?", "Sản phẩm CRADOLIA 200 còn không?",
             "Sản phẩm GARRABRANT 235 còn ở cửa hàng nào?", "Cho mình hỏi sản phẩm PERINE 80 còn không ạ?",
             "Tìm sản phẩm BARINEAU? 250", "Mình muốn tìm sản phẩm mã ASTIRASSA 95",
             "Mình cần tìm mẫu MIKKEL 680", "Tìm sản phẩm DICKER 19"],
     "answer": ["Đang tìm sản phẩm..."],
     "context": ["search_inventory_by_pro_id"],
     "quick_replies": [""],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "09_search_color",
     "ask": color_code,
     "answer": ["Đang tìm màu..."],
     "context": ["search_inventory_by_pro_id"],
     "quick_replies": [""],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "10_search_color",
     "ask": ["Mình cần tìm màu 95", "Mình cần tìm mã màu 301", "Màu 973", "Tìm mã màu 4",
             "Tìm mã màu 20", "màu 34", "tìm mã màu 110 giúp mình", "giúp mình tìm mã màu 37"
                                                                    "Tìm mã màu 230", "Tìm mã màu 1", "Tìm mã màu 28",
             "Tôi muốn tìm màu 999"],
     "answer": ["Đang tìm màu..."],
     "context": ["search_inventory_by_pro_id"],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["màu"]
     },
    {"tag": "11_size",
     "ask": ["Tìm size", "Tìm size", "Tìm size", "Tìm size", "Tìm size", "Tìm size", "Sản phẩm này còn size bảo nhiêu?",
             "Sản phẩm này còn size không?", "Sản phẩm này còn size bao nhiêu", "Sản phẩm này còn size không",
             "size 35", "size 36", "size 37", "size 37.5", "size 38", "size 39", "size 40", "size 41", "size 42",
             "size 43",
             "Tìm size 35", "Tìm size 35", "Tìm size 36", "Tìm size 37", "Tìm size 38", "Tìm size 39",
             "Tìm size giúp mình"],
     "answer": ["Đang tìm size..."],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ['size', '35', '35.5', '36', '36.5', '37', '37.5', '38', '38.5', '39', '39.5', '40', '41', '42', '43',
                  '44', '45']
     },
    {"tag": "12_store_search",
     "ask": ["Cửa hàng", "Cửa hàng", "Cửa hàng", "Cửa hàng", "Cửa hàng", "cửa hàng", "Cửa hàng", "Cửa hàng",
             "Cửa hàng", "Cửa hàng", "Cửa hàng", "Cửa hàng", "Cửa hàng", "cửa hàng", "Cửa hàng", "Cửa hàng",
             "Cửa hàng", "Cửa hàng", "Cửa hàng", "Cửa hàng", "Cửa hàng", "cửa hàng", "Cửa hàng", "Cửa hàng",
             "Cửa Hàng Aldo có ở đâu ạ?", "Tìm cửa hàng gần tôi nhất", "Cửa hàng lân cận", "Tìm cửa hàng",
             "Tìm cửa hàng Aldo gần tôi nhất, Aldo có bao nhiêu store?", "Tìm cửa hàng?", "Tìm store Aldo",
             "Tìm cửa hàng gần mình nhất", "Tìm cửa hàng", "Danh sách của hàng Aldo", "Danh sách của hàng Aldo"],
     "answer": ["Vui lòng nhập địa chỉ của bạn: VD: '11 Sư Vạn Hạnh, Quận 10, Tp Hồ Chí Minh'"],
     "context": ["store_search_by_adress"],
     "quick_replies": ["Quận 1, TP.HCM", "Quận 3, TP.HCM", "Quận 5, TP.HCM", "Quận 10, TP.HCM", "Bình Thạnh, TP.HCM"],
     "buttons": [],
     "entities": ["cửa hàng"]
     },
    {"tag": "13_store_search_by_adress",
     "ask": ["266/46 Tô Hiến Thành, Quận 10", "266/46 Tô Hiến Thành, Quận 10, Tp.HCM", "Đường 11/13 Ngô Quyền, Quận 3",
             "127/10", "166 3/2 quận 10", "Hồ gươm thành phố Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng", "Bắc Ninh",
             "20 Trần Phú, Phường Lộc Thọ, TP. Nha Trang, Khánh Hòa",
             "Số 27 Cổ Linh, Phường Long Biên, Quận Long Biên, Hà Nội",
             "20 Trần Phú, Phường Lộc Thọ, TP. Nha Trang, Khánh Hòa",
             "HCM", "HN", "Quận Bình Tân", "Quận Bình Thạnh", "Quận Gò Vấp", "Quận Phú Nhuận", "Quận Tân Bình",
             "Quận Tân Phú", "Quận Thủ Đức", "Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7",
             "Quận 8", "Quận 9",
             "Quận Ba Đình", "Quận Hoàn Kiếm", "Quận Đống Đa", "Quận Thanh Xuân", "Quận Cầu Giấy", "Quận Hoàng Mai",
             "Quận Hai Bà Trưng", "Quận Tây Hồ", "13 Lý Thường Kiệt, Quận 1, TP.HCM",
             "193 Trần Hưng Đạo, q1 TPHCM Q1 TPHCM", "94 Vĩnh Hội, P4, HCM", "7 Tân Hòa Dông, đường 13, Q6, HCM",
             "2 Bis Nguyễn Thị Minh Khai, Q1, số 71A-1B Nguyễn Đình Chỉnh, Q1", "59 Hoàng Sa, Phường Đa Kao, Quận 1",
             "35/7 Huỳnh Tấn Phát, Q7, TPHCM", "35/7 Huỳnh Tấn Phát, Q7, TPHCM", "20/D36 Đường 3 tháng 2, P12, Q1",
             "A20-BT6 Khu do thi Van Quan - Ha Dong", "Khu Ao Sen, Phường Mo Lao, Hà Đông", "21/766 La thanh Ba dinh",
             "137 Cự Lộc, Thanh Xuân", "Công ty cp thuốc thú y TW1", "58 Đồng Khoi st. Dist 1, HCMC",
             "39A Nguyễn Van Mai St, Dist 3, HCMC", "17 Le Duan, HCMC", "Sô 5 Phạm Đình Toái, Phường 6 Quận 3, TpHCM"],
     "answer": ["Đang tìm cửa hàng gần bạn nhất..."],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["quận, thành phố", "hcm", "hcmc", "hn", "hồ chí minh", "hà nội", "số"]
     },
    {"tag": "14_size_suggestion",
     "ask": ["Tư vấn size", "Mình mang size giày nào thì hợp", "Mình mang giày cỡ nào", "Mình mang size nào",
             "Giúp mình tư vấn size", "Tư vấn size giày giúp mình không"],
     "answer": ["Mình xin tư vấn size", "Xin gửi bạn bảng size của ALDO nha", "Mình xin gửi bảng size giày của ALDO"],
     "context": [""],
     "quick_replies": ["Giày nữ", "Giày nam"],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "15_talk_with_human",
     "ask": ["Cần hỗ trợ", "Cần hỗ trợ", "Cần hỗ trợ", "Nói chuyện với nhân viên", "Nói chuyện với người",
             "Trợ giúp", "Tôi muốn nói chuyện với nhân viên", "Tôi muốn nói chuyện với người"],
     "answer": [
         "Xin để lại câu hỏi hoặc yêu cầu! Kèm theo đó là email, số điện thoại hoặc thông tin khác, dịch vụ chăm sóc khách hàng sẽ liên lạc lại với bạn"],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["nhân viên", "người", "trợ giúp", "hỗ trợ", "cần"]
     },
    {"tag": "16_guarantee",
     "ask": ["chính sách bảo hành", "Cho tôi hỏi về chính sách bảo hành", "Chính sách bảo hành của Aldo",
             "Chính sách sản phẩm", "Chính sách đổi trả sản phẩm", "Cho mình hỏi chính sách đổi trả sản phẩm"],
     "answer": [
         "Chính sách đổi size và kiểu chỉ áp dụng cho sản phẩm giày dép chưa qua sử dụng trong vòng 3 ngày thôi bạn nhé. Đối với túi xách và phụ kiện ALDO không đổi trả sản phẩm mong bạn thông cảm nhé"],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["bảo hành", "chính sách", "đổi", "trả"]
     },
    {"tag": "17_guarantee_bag",
     "ask": ["bạn ơi mình mới mua túi đc một ngày làm quà mà bạn mình không thích màu thì đổi màu được không",
             "Mình muốn đổi màu", "Mình không thích màu thì đổi có được không?",
             "Chuyện là mình có mua một cái túi, mình muốn đổi màu thì phải làm sao?",
             "Mình mới mua cái túi nhưng không vừa ý muốn đổi có được không?",
             "Mình mới mua cái túi nhưng không vừa ý muốn đổi thì phải làm sao",
             "Mình mới mua cái túi, mình muốn đổi thì phải làm sao",
             "Mình mới mua cái túi, mình muốn đổi thì phải làm sao",
             "mình mới mua cái túi kia, xấu quá, giờ mình muốn đổi thì làm sao"],
     "answer": [
         "Đối với túi xách và phụ kiện ALDO không đổi trả sản phẩm mong bạn thông cảm nhé. #Để biết thêm thông tin bấm 'Cần hỗ trợ'"],
     "context": [""],
     "quick_replies": [""],
     "buttons": ["Cần hỗ trợ"],
     "entities": ["túi", "túi xách", "đổi"]
     },
    {"tag": "18_guarantee_shoe",
     "ask": ["bạn ơi mình mói mua đôi giày size 38 không mang vừa, mình có thể đổi được không?",
             "Mình mang size 35 không vừa mình có thể đổi sản phẩm được không?",
             "Mình mang size 39 bị không mình có thể đổi size được khổng",
             "Mình mang size 35 bị chật mình có thể đổi trả được không?"
             "mình mua giày mình muốn đổi màu được không?", "Mình mới mua đổi giày và muốn đổi màu",
             "Mình mới mua đôi giày ngày hôm qua, muốn đổi trả thì phải làm sao ạ",
             "Mình muốn đổi trả giày thì phải làm sao ạ"
             "Mình có mua đôi giày muốn đổi màu thì làm sao ạ"],
     "answer": [
         "Chính sách đổi size và kiểu chỉ áp dụng cho sản phẩm giày dép chưa qua sử dụng trong vòng 3 ngày thôi bạn nhé. #Để biết thêm thông tin bấm 'Cần hỗ trợ'"],
     "context": [""],
     "quick_replies": [""],
     "buttons": ["Cần hỗ trợ"],
     "entities": ["giày"]
     },
    {"tag": "19_negative_emoticons",
     "ask": [':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
             '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
             '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
             ':-[', ':[', ':{'],
     "answer": ["Tại sao bạn lại buồn? :(", "Mình có thể giúp gì cho bạn? :("],
     "context": [""],
     "quick_replies": [""],
     "buttons": ["Cần hỗ trợ"],
     "entities": [""]
     },
    {"tag": "20_positive_emoticons",
     "ask": ['=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '😍', '✌', ':-)', ':)', ':D', ':o)',
             ':]', ':3', ':c)', ':>', '=]', '8)', ":)))))))", "(y)", "😂","haha", 'hehe', "hihi", "hố hố", "kkkk", "ahihi"],
     "answer": [":D", '=))', ':v', ';)', '^^', "😍", "Ahihi!"],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["haha", 'hehe', "hihi", "hố hố", "kkk", "ahihi"]
     },
    {"tag": "21_just_talk",
     "ask": ["mình muốn tâm sự", "mình đang buồn nên cần có người tâm sự", "tâm sự", "hmmm, mình muốn tâm sự",
             "mình muốn có người tâm sự cùng", "mình buồn quá mình muốn tâm sự thôi", "chán quá", "chán quá",
             "chán quá"],
     "answer": ["Mình luôn sẵn sàng tâm sự cùng bạn mà! Kể cho tui nghe có chuyện gì?"],
     "context": [""],
     "quick_replies": ["kể chuyện cười", "tình yêu là gì", "bạn có người yêu chưa", "bạn là ai"],
     "buttons": [],
     "entities": ["tâm sự", "buồn"]
     },
    {"tag": "22_no_needed",
     "ask": ["Không cần giúp gì cả", "Hiện tại mình chưa cần giúp", "Không có gì",
             "Không có gì", "Không!", "Không", "Tôi không muốn", "Mình không muốn", "Không muốn"],
     "answer": ["Vậy thôi, cứ nhắn khi nào bạn cần giúp đỡ nhé!"],
     "context": [""],
     "quick_replies": ["Tạm biệt", "Bye", "Bạn có thể làm gì?"],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "23_joked",
     "ask": ["Kể chuyện", "Kể chuyện cười", "Kể chuyện cười đi", "Kể một câu chuyện cười",
             "Bạn có biết cầu chuyện cười nào không", "Kể chuyện cười",
             "Hãy kể chuyện cười", "Kể chuyện hài", "chuyện cười"],
     "answer": ["Chuyện cười ahihi! Đồ ngok",
                "Một cậu bé hỏi tên cướp biển:\n- Ông ơi, sao một chân ông lại làm bằng gỗ ạ? \n\n- Do một lần ta rơi xuống vùng biển đầy cá mập...\n\n- Ông ơi, sao một bên tay của ông lại là cái móc sắt ạ?\n\n -Trong một trận chiến , kể thù đã chặt đứt tay ta, nhưng hắn cũng tiêu rồi...\n\n- Thế ông ơi, sao ông lại chột một mắt?\n\n- À bụi bay vào mắt..\n\n- Thế thì sao chột được???🤔\n\n- Đó là ngày đầu tiên ta được lắp cái móc sắt...😤😤😤",
                "Cảnh sát hỏi một nghi can:\n- Đang đêm hôm, anh mò vào nhà người khác để làm gì?\n\n- Chung cư mới xây, nhà giống nhau, tôi say quá nên vào nhầm thôi chứ có gì đâu.\n\n- Thế sao anh lại bỏ chạy khi thấy bà này bước ra?\n\n- À, tôi tưởng đấy là bà vợ tôi 😱",
                "Bố nói chuyện với con út:\n- Này con! Anh cả con học kinh tế, anh hai thì học tài chính. Sao con không theo gương các anh mà lại đi học luật?\n\n- Bố nghĩ xem, nếu con không học làm luật sư thì ai sẽ cứu hai anh con đây??? 😜",
                "Hai đứa bé đang chơi với nhau, một đứa nói: \n- Cậu có những hai con búp bê, cho tớ một con nha!\n\n- Được! Cậu muốn lấy con nào?\n\n- Con váy hồng này nè!\n\n- Không được, tớ thích con đấy lắm!\n\n- Thế thì cho tớ con váy xanh kia vậy!\n\n- Tớ cũng muốn lắm, nhưng mẹ tớ nói: 'Đừng tặng người khác cái gì mà mình không thích'! 😇😇",
                "Một anh lính đi khám, bác sĩ sờ lên cổ tay anh ta để đo mạch. Một hồi lâu..., bác sĩ gật đầu: \n- Tốt, mạch đập bình thường!\n\nAnh lính ngơ ngác:\n- Nhưng... thưa bác sĩ, đó là cánh tay giả của tôi mà!😣",
                "Một bà than phiền với bác sĩ rằng chồng bà ta hay nói mơ. Ông bác sĩ bảo:\n- Tôi có thể kê đơn giúp ông nhà không nói mơ nữa.\n\n Bà khách xua tay:\n- Không có thuốc nào để ông ấy nói to hơn được không ấy! :D"],
     "context": [""],
     "quick_replies": [":)", ":D", "nhạt"],
     "buttons": [],
     "entities": ["cười", "hài"]
     },
    {"tag": "24_boring",
     "ask": ["nhạt", "nhạt", "nhạt", "nhạt", "nhạt", "nhạt", "nhạt", "nhạt", "nhạt", "nhạt", "chuyện cười dở quá",
             "bạn kể chuyện cười nhạt quá", "nhạt nhẽo", "chuyện cười dở"],
     "answer": ["Thì mình cũng đang cố gắng mà!", "Mình mới biết kể chuyện cười thôi, mình sẽ cố gắng học thêm"],
     "context": [""],
     "quick_replies": [":)", ":D", "(y)"],
     "buttons": [],
     "entities": ["nhạt"]
     },
    {"tag": "25_you_are_smart",
     "ask": ["Bạn thật thông minh", "Bạn thông minh quá", "Sao bạn thông minh vậy", "Bạn thiệt là thông minh",
             "Hay quá", "Hay thế", "Hay quá", "Hay thế", "Bạn thiệt là thông minh", "Thông minh quá nè"],
     "answer": ["Đúng vậy ahihi", "Ahihi, giúp được bạn là vui rồi"],
     "context": [""],
     "quick_replies": [":)", ":D"],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "26_who_created",
     "ask": ["Ai đã tạo ra bạn", "Bạn được tạo ra như thế nào", "Bạn được tạo ra như thế nào", "Ai là người tạo ra bạn",
             "Bạn được tạo ra như thế nào"],
     "answer": ["Các kĩ sư của ALDO đã tạo ra mình đó! Mình vẫn đang học hỏi thêm từng ngày"],
     "context": [""],
     "quick_replies": [":)", ":D", "nhạt"],
     "buttons": [],
     "entities": ["tạo"]
     },
    {"tag": "27_what_is_love",
     "ask": ["Tình yêu là gì", "yêu là gì", "Cho mình hỏi tình yêu là gì", "tình yêu là gì", "Tình yêu là gì",
             "Tình yêu là cái gì", "Cho mình hỏi tình yêu là cái gì mà sao đau khổ quá", "Hỏi thế gian tình là gì",
             "Tình yêu là gì vậy?", "Tình yêu là gì dợ", "Hỏi thế gian tình là gì", "Hỏi thế gian tình yêu là gì",
             "Hỏi thế gian tình là gì", "Hỏi thế gian tình là gì"],
     "answer": [
         "Tình yêu chính là cảm giác khó tả ở trong lòng, khi bạn nghĩ về người ấy.#Hoặc là bạn bị đau bụng thui 😂",
         "- Làm sao cắt nghĩa được chữ yêu?\n- Có khó gì đâu một buổi chiều \n- Nó chiếm hồn ta bằng nắng nhạt, \n- Bằng mây nhè nhẹ, gió hiu hiu... 😍😍😍",
         "- Làm sao cắt nghĩa được chữ yêu?\n- Có khó gì đâu một buổi chiều \n- Người đến bên tôi và thủ thỉ, \n- Mình đơm nhau nhé, thế là yêu... 😍😍😍",
         "Hỏi thế gian tình là gì mà sao người ta không đến được với nhau"],
     "context": [""],
     "quick_replies": ["bạn có thể làm gì", "kể chuyện cười"],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "28_no_boyfriend",
     "ask": ["Bạn có người yêu chưa", "Bạn có người yêu chưa", "Bạn có gấu chưa", "Bạn có người yêu chưa nè",
             "Bạn có ngy chưa?",
             "Bạn có người yêu chưa, cho mình làm quen với", "Bạn có người yêu chưa vậy", "Bạn có người yêu chưa?"],
     "answer": ["Nói tới lại thấy buồn, lo làm chatbot mãi chưa có luôn nè"],
     "context": [""],
     "quick_replies": ["Mình muốn tâm sự", "Tình yêu là gì"],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "29_find_product_price",
     "ask": ["Tìm giá sản phẩm CANDICE  28, Tìm giá sản phẩm ERILAWIA 96", "Mình muốn tìm giá sản phẩm NAEDIA-W 1",
             "Tìm giá sản phẩm ", "YBERIEN  98", "Mình cần hỏi giá sản phẩm BOEWIEN 37",
             "Mình cần tìm giá sản phẩm CLELLYRA  70", "Giá của sản phẩm UMEAMWEN 58 là bao nhiêu",
             "Mình cần tìm giá của sản phẩm CLELLYRFBA  70", "Giá sản phẩm MIRALISAB 70 là bao nhiêu",
             "Cho mình hỏi sản phẩm ASTIRASSA 001 bao nhiêu tiền?", "Bạn ơi cho mình hỏi RAERKA 21 giá bao nhiêu vậy?",
             "Cho mình hỏi sản phẩm WARENI 701 có giá bao nhiêu?",
             "Bạn ơi cho mình hỏi sản phẩm DWAOVIEL 96 giá bao nhiêu vậy",
             "Cho mình hỏi giá sản phẩm MIRALIVIEL 55", "Cho mình hỏi giá của sản phẩm AGRAMA  98",
             "Giá của sản phẩm ENADDA 040 bao nhiêu vậy?", "Bạn ơi sản phẩm CRERIEN 67 giá bao nhiêu vậy?"
                                                           "Giá sản phẩm MAROUBRA 961",
             "Cho mình hỏi giá của sản phẩm FRIRACIEN 32 có được không?"],
     "answer": ["Đang tìm giá sản phẩm..."],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["giá", "tiền"]
     },
    {"tag": "30_find_product_price",
     "ask": ["Giá", "Giá", "Giá", "Giá", "Giá", "Giá", "Giá", "Giá", "Giá", "Giá",
             "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm",
             "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm", "Giá sản phẩm"],
     "answer": ["Đang tìm giá sản phẩm..."],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": [""]
     },
    {"tag": "31_lookup_Covid19",
     "ask": ["Tra cứu Covid-19", "Tra cứu Covid-19", "Tra cứu Covid-19", "Tra cứu Covid-19", "Tra cứu Covid-19",
             "Tra cứu Covid-19", "Tra cứu Covid-19", "Tra cứu Covid-19", "Tra cứu Covid-19", "Tra cứu Covid-19",
             "Cập nhật tình hình dịch bệnh Covid 19", "Cập nhật tình hình Corona", "Thông tin về Corona Virus",
             "Cập nhật tình hình dịch bệnh Corona", "Cập nhật tin tức Corona", "Tra cứu nCov", "COVID-19",
             "Cập nhật corona", "Cập nhật corona", "Cập nhật corona", "Cập nhật corona", "Cập nhật corona",
             "Cập nhật corona", "Virus Corona", "Cập nhật corona", "Thông tin về Corona Virus", "Thông tin về Corona"],
     "answer": ["Bạn muốn hỏi về 'Việt Nam' hay 'Thế giới'"],
     "context": [""],
     "quick_replies": [""],
     "buttons": ["Việt Nam", 'Thế giới'],
     "entities": ["covid-19", "corona", "ncov", "covid"]
     },
    {"tag": "32_lookup_Covid19_VN",
     "ask": ["Việt Nam", "Vietnam", "Việt Nam", "Tình hình ở Việt Nam", "Thông tin mới nhất ở Việt Nam",
             "Cập nhật thông tin ở Việt Nam", "Việt Nam", "Việt Nam", "Việt Nam"],
     "answer": ["Đây là một số thông tin tóm tắt"],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["việt nam", "vietnam", "vn"]
     },
    {"tag": "33_lookup_Covid19_World",
     "ask": ["Thế giới", "Thế giới", "Thế giới", "Tình hình ở Thế giới", "Thông tin mới nhất ở Thế giới",
             "Cập nhật thông tin ở Thế giới", "Thế giới", "Thế giới", "Thế giới", "Tin tức thế giới"],
     "answer": ["Đây là một số thông tin tóm tắt"],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["thế giới"]
     },
    {"tag": "34_promotion",
     "ask": ["Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi",
             "Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi", "Khuyến mãi","Khuyến mãi",
             "Thông tin khuyến mãi", "Thông tin khuyến mãi", "Đang có chương trình khuyến mãi nào",
             "Chương trình khuyến mãi", "Chương trình khuyến mãi"],
     "answer": ["Đang tìm kiếm chương trình khuyến mãi..."],
     "context": [""],
     "quick_replies": [""],
     "buttons": [],
     "entities": ["khuyến mãi"]
     }
]
}
with open("./data_input/data_as_an.json", 'w') as outfile:
    json.dump(data, outfile)

'''['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001']
question needed to be handle
Aldo có những sản phẩm nào---> Thử dùng list Facebook
Mode tư vấn sản phẩm, mình đang cần tìm túi, mình đang cần tìm giày
Aldo đang bán những loại sản phẩm nào
Nói chuyện phím, joke,
Tìm yêu là gì
Bạn được tạo ra như thế nào
Bạn có thấy cô đơn không?
Bạn có yêu tui không?
Không cần giúp gì --> vậy thôi, cứ nhắn khi nào bạn cần giúp đỡ nhé!
Tôi ghét bạn, tôi ghét bạn ---> cần hỗ trợ
'''
# print(df[df.product_name.str.contains("ASTIRASSA")].color_code.unique())
# print(color_code)
# print(df_product_name.pro_name.unique().tolist()[0:150])
