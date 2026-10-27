# Mô tả: Xây dựng và chuẩn hóa tệp dữ liệu nhãn chuẩn Ground Truth cho tập ảnh kiểm định QA (data/holdout).
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

# Color map for classes (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),      # battery: red
    1: (0, 255, 0),      # biological: green
    2: (0, 165, 255),    # cardboard: orange
    3: (255, 0, 255),    # clothes: magenta
    4: (255, 255, 0),    # glass: cyan
    5: (192, 192, 192),  # metal: silver
    6: (255, 255, 255),  # paper: white
    7: (255, 0, 0),      # plastic: blue
    8: (0, 255, 255),    # shoes: yellow
    9: (128, 128, 128),  # trash: gray
}

RAW_GT = [
    {
        "filename": "base64-1717231151649602366964.jpeg",
        "objects": [
            {
                "object_id": "img01_obj01",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [180, 695, 825, 1115],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai bia Saigon màu nâu nằm ngang bên trái, thân thủy tinh hổ phách, nhãn trắng và cổ bọc giấy bạc."
            },
            {
                "object_id": "img01_obj02",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [325, 615, 555, 810],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Hũ thủy tinh nhỏ nắp vàng nằm ngay phía trên chai bia Saigon bên trái."
            },
            {
                "object_id": "img01_obj03",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [1355, 370, 1845, 680],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai bia Saigon thứ hai bên phải, thân thủy tinh vỡ đuôi, cổ hướng góc dưới phải có giấy bạc."
            },
            {
                "object_id": "img01_obj04",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [1115, 1010, 1410, 1334],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai thủy tinh trong suốt nắp đỏ ở tiền cảnh sát đáy ảnh."
            },
            {
                "object_id": "img01_obj05",
                "class_name": "trash",
                "class_id": 9,
                "bbox_xyxy": [1190, 810, 1585, 1120],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Khẩu trang y tế màu trắng dùng một lần nằm trên mảnh kính."
            },
            {
                "object_id": "img01_obj06",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 720, 150, 920],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Mảnh vải vụn/giẻ lau màu xám ở góc dưới cùng bên trái."
            },
            {
                "object_id": "img01_obj07",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [0, 195, 2000, 1334],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Bãi phế liệu mảnh kính vỡ màu xanh dương và đen trải rộng toàn bộ mặt đất trung tâm."
            }
        ]
    },
    {
        "filename": "battery-sample.png",
        "objects": [
            {
                "object_id": "img02_obj01",
                "class_name": "battery",
                "class_id": 0,
                "bbox_xyxy": [14, 50, 503, 371],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Pin laptop màu đen dạng thanh dẹt, có cáp nối, tem nhãn thông số pin lithium và biểu tượng tái chế."
            }
        ]
    },
    {
        "filename": "download (2).jpg",
        "objects": [
            {
                "object_id": "img03_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [112, 10, 168, 62],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai nhựa truyền dịch y tế màu trắng đục ở phía trên giữa bãi rác."
            },
            {
                "object_id": "img03_obj02",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [140, 50, 203, 112],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai nhựa truyền dịch y tế trong suốt ở trung tâm bãi rác."
            },
            {
                "object_id": "img03_obj03",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [0, 0, 299, 168],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống rác thải y tế gồm bao bì túi nhựa và chai dịch truyền vứt bừa bãi trên nền đất."
            }
        ]
    },
    {
        "filename": "images (10).jpg",
        "objects": [
            {
                "object_id": "img04_obj01",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 160, 640, 398],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống rác quần áo phế thải thời trang nhanh trải dài tiền cảnh và sườn phải sa mạc Atacama."
            },
            {
                "object_id": "img04_obj02",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 120, 325, 295],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Gò đồi quần áo cũ phế thải bên sườn trái sa mạc Atacama."
            }
        ]
    },
    {
        "filename": "images (11).jpg",
        "objects": [
            {
                "object_id": "img05_obj01",
                "class_name": "metal",
                "class_id": 5,
                "bbox_xyxy": [0, 140, 684, 448],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống phế liệu linh kiện điện tử, khung kim loại máy giặt, vỏ máy vi tính và dây cáp e-waste."
            }
        ]
    },
    {
        "filename": "images (12).jpg",
        "objects": [
            {
                "object_id": "img06_obj01",
                "class_name": "cardboard",
                "class_id": 2,
                "bbox_xyxy": [32, 36, 535, 519],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chồng thùng bìa carton màu nâu kraft bên dưới và bao quanh khối rác."
            },
            {
                "object_id": "img06_obj02",
                "class_name": "paper",
                "class_id": 6,
                "bbox_xyxy": [110, 90, 475, 445],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống giấy báo, giấy văn phòng trắng vụn và vò nhàu nằm phía trên thùng carton."
            }
        ]
    },
    {
        "filename": "images (13).jpg",
        "objects": [
            {
                "object_id": "img07_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [0, 0, 538, 371],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống chai nhựa PET tái chế trong suốt các nhãn hiệu nước giải khát, dầu ăn, hóa chất."
            }
        ]
    },
    {
        "filename": "images (14).jpg",
        "objects": [
            {
                "object_id": "img08_obj01",
                "class_name": "shoes",
                "class_id": 8,
                "bbox_xyxy": [0, 0, 260, 365],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống giày dép cũ phế thải khổng lồ góc bên trái nhà xưởng."
            },
            {
                "object_id": "img08_obj02",
                "class_name": "shoes",
                "class_id": 8,
                "bbox_xyxy": [250, 0, 547, 365],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống giày dép cũ phế thải bên phải và xung quanh công nhân phân loại."
            }
        ]
    },
    {
        "filename": "images (15).jpg",
        "objects": [
            {
                "object_id": "img09_01",
                "class_name": "shoes",
                "class_id": 8,
                "bbox_xyxy": [113, 17, 192, 68],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đôi dép nhựa tổ ong màu vàng nhạt trên bậc thang trên cùng."
            },
            {
                "object_id": "img09_02",
                "class_name": "shoes",
                "class_id": 8,
                "bbox_xyxy": [160, 84, 226, 148],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đôi dép quai ngang sandal màu be trên bậc thang giữa bên phải."
            },
            {
                "object_id": "img09_03",
                "class_name": "shoes",
                "class_id": 8,
                "bbox_xyxy": [98, 92, 190, 160],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đôi giày thể thao sneaker Vans màu đen sọc trắng buộc dây ở bậc giữa."
            },
            {
                "object_id": "img09_04",
                "class_name": "shoes",
                "class_id": 8,
                "bbox_xyxy": [15, 114, 110, 191],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đôi dép xỏ ngón lê màu xám ở bậc thang dưới cùng bên trái."
            }
        ]
    },
    {
        "filename": "images (17).jpg",
        "objects": [
            {
                "object_id": "img10_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [0, 162, 262, 365],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Thùng rác tái chế bằng nhựa trong suốt mờ bên trái có logo recycle."
            },
            {
                "object_id": "img10_obj02",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [250, 178, 535, 365],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Thùng rác tái chế bằng nhựa màu đen bên phải có logo recycle."
            },
            {
                "object_id": "img10_obj03",
                "class_name": "metal",
                "class_id": 5,
                "bbox_xyxy": [0, 185, 115, 252],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Hộp thiếc kim loại hình bông tuyết màu xanh trong thùng bên trái."
            },
            {
                "object_id": "img10_obj04",
                "class_name": "metal",
                "class_id": 5,
                "bbox_xyxy": [195, 218, 258, 250],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Lon nước ngọt kim loại màu đỏ cắm ống hút đen."
            },
            {
                "object_id": "img10_obj05",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [68, 105, 150, 240],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai thủy tinh màu xanh lá cây trong thùng bên trái."
            },
            {
                "object_id": "img10_obj06",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [41, 110, 86, 190],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Cổ chai thủy tinh trong suốt nằm sau chai xanh lá cây."
            },
            {
                "object_id": "img10_obj07",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [250, 18, 330, 205],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai nhựa trong suốt cao có ngấn rãnh ở thùng đen bên trái."
            },
            {
                "object_id": "img10_obj08",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [328, 52, 405, 200],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Bình nhựa màu trắng đục nắp vặn nhọn ở thùng đen."
            },
            {
                "object_id": "img10_obj09",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [400, 45, 470, 200],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai nước khoáng nhựa trong suốt nắp xanh dương ở thùng đen."
            },
            {
                "object_id": "img10_obj10",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [425, 110, 510, 210],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Hộp/chai nhựa trong suốt miệng rộng phía ngoài cùng bên phải thùng đen."
            },
            {
                "object_id": "img10_obj11",
                "class_name": "metal",
                "class_id": 5,
                "bbox_xyxy": [360, 175, 415, 210],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Lon nhôm kim loại bị ép bẹp ở mép thùng đen."
            }
        ]
    },
    {
        "filename": "images (18).jpg",
        "objects": [
            {
                "object_id": "img11_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [26, 12, 384, 495],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Thùng rác công cộng có bánh xe đẩy bằng nhựa màu xanh lá cây in dòng chữ HÃY BỎ RÁC VÀO THÙNG."
            }
        ]
    },
    {
        "filename": "images (19).jpg",
        "objects": [
            {
                "object_id": "img12_obj01",
                "class_name": "biological",
                "class_id": 1,
                "bbox_xyxy": [90, 0, 597, 335],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống rác hữu cơ ủ phân compost gồm vỏ dưa hấu, vỏ trứng, cà chua thối, rau củ quả trên rơm rạ."
            }
        ]
    },
    {
        "filename": "images (3).jpg",
        "objects": [
            {
                "object_id": "img13_obj01",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [49, 18, 120, 165],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai bia thủy tinh màu xanh lá cây bị vỡ đáy dạt vào bờ cát biển."
            }
        ]
    },
    {
        "filename": "images (4).jpg",
        "objects": [
            {
                "object_id": "img14_obj01",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [37, 48, 185, 200],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai thủy tinh trong suốt đựng dầu màu hổ phách có vòi rót kim loại trên nền trắng."
            }
        ]
    },
    {
        "filename": "images (5).jpg",
        "objects": [
            {
                "object_id": "img15_obj01",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [80, 15, 146, 224],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Chai thủy tinh trong suốt có nắp gài kim loại và nút cao su đỏ trên nền trắng."
            }
        ]
    },
    {
        "filename": "images (6).jpg",
        "objects": [
            {
                "object_id": "img16_obj01",
                "class_name": "biological",
                "class_id": 1,
                "bbox_xyxy": [0, 0, 318, 159],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Rác thải hữu cơ sinh hoạt gồm vỏ chuối, vỏ cam, hoa héo và lá cây mục nát."
            }
        ]
    },
    {
        "filename": "images (7).jpg",
        "objects": [
            {
                "object_id": "img17_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [0, 0, 538, 371],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống chai nhựa PET tái chế trong suốt các nhãn hiệu nước giải khát, dầu ăn, hóa chất (file trùng với images (13))."
            }
        ]
    },
    {
        "filename": "images (8).jpg",
        "objects": [
            {
                "object_id": "img18_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [12, 8, 412, 463],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Giỏ/sọt đựng rác văn phòng bằng nhựa đục lỗ màu xanh lam nhạt trên nền trắng."
            }
        ]
    },
    {
        "filename": "images (9).jpg",
        "objects": [
            {
                "object_id": "img19_obj01",
                "class_name": "cardboard",
                "class_id": 2,
                "bbox_xyxy": [0, 0, 515, 388],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Các mảnh bìa carton sóng màu nâu kraft bị xé rách lộ rõ cấu trúc lõi gợn sóng."
            }
        ]
    },
    {
        "filename": "OIP (1).webp",
        "objects": [
            {
                "object_id": "img20_obj01",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 95, 313, 237],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống rác quần áo phế thải thời trang nhanh trải dài tiền cảnh và sườn phải sa mạc Atacama (ảnh thu nhỏ)."
            },
            {
                "object_id": "img20_obj02",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 70, 160, 175],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Gò đồi quần áo cũ phế thải bên sườn trái sa mạc Atacama (ảnh thu nhỏ)."
            }
        ]
    },
    {
        "filename": "OIP.webp",
        "objects": [
            {
                "object_id": "img21_obj01",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 0, 274, 234],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Cận cảnh đống quần áo cũ nhiều màu sắc lẫn lộn (áo phông, quần jean, áo khoác) chất thành đống."
            }
        ]
    },
    {
        "filename": "Phan-Loai-Rac-Tai-Ng-01.jpg",
        "objects": [
            {
                "object_id": "img22_obj01",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [230, 220, 460, 550],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Bao tải/túi ni lông rác lớn bằng nhựa màu xanh dương chứa đầy rác bên lề đường."
            },
            {
                "object_id": "img22_obj02",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [340, 445, 675, 705],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Túi rác nhựa màu đỏ chứa các hộp xốp đựng thức ăn và ly nhựa dùng một lần."
            },
            {
                "object_id": "img22_obj03",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [590, 580, 755, 715],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Ly nhựa trà sữa trong suốt cắm ống hút nhựa trắng nằm trong bọc túi bóng dưới lòng đường."
            },
            {
                "object_id": "img22_obj04",
                "class_name": "cardboard",
                "class_id": 2,
                "bbox_xyxy": [680, 570, 765, 650],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Vỏ hộp/bịch sữa TH true MILK bằng giấy/carton phức hợp nằm cạnh đống rác."
            },
            {
                "object_id": "img22_obj05",
                "class_name": "plastic",
                "class_id": 7,
                "bbox_xyxy": [680, 380, 710, 400],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Nắp chai nhựa màu vàng nằm trên bậc tam cấp bê tông."
            },
            {
                "object_id": "img22_obj06",
                "class_name": "paper",
                "class_id": 6,
                "bbox_xyxy": [285, 590, 375, 655],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Vỏ bao bì bánh kẹo giấy màu vàng vứt rơi trên mặt đường nhựa."
            }
        ]
    },
    {
        "filename": "photo-5-1642556540067542343380.webp",
        "objects": [
            {
                "object_id": "img23_obj01",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [0, 15, 620, 376],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Ngọn núi rác thải dệt may thời trang nhanh khổng lồ với hàng nghìn bộ quần áo cũ ở bãi rác."
            }
        ]
    },
    {
        "filename": "pin_vuong_4f15ab32849e43a8abdaef2bf000463f.webp",
        "objects": [
            {
                "object_id": "img24_obj01",
                "class_name": "battery",
                "class_id": 0,
                "bbox_xyxy": [252, 120, 551, 676],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Pin vuông Panasonic Hi-TOP 9V màu đỏ vỏ kim loại có hai đầu cực snap connector."
            }
        ]
    },
    {
        "filename": "rac-1562828175-1562828212-3852-1562828313.webp",
        "objects": [
            {
                "object_id": "img25_obj01",
                "class_name": "glass",
                "class_id": 4,
                "bbox_xyxy": [0, 95, 532, 299],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Bãi chai bia thủy tinh màu xanh lá cây có dán nhãn giấy trắng bị vứt tràn ngập trên thảm cỏ ven đường."
            }
        ]
    },
    {
        "filename": "rac-thai-huu-co-1.jpg",
        "objects": [
            {
                "object_id": "img26_obj01",
                "class_name": "biological",
                "class_id": 1,
                "bbox_xyxy": [0, 0, 600, 410],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Rác thải hữu cơ từ nhà bếp gồm bắp ngô, ớt chuông đỏ, bánh mì, vỏ cam mốc, cà chua ủng phân hủy."
            }
        ]
    },
    {
        "filename": "rac-thai-quan-ao-4.jpg",
        "objects": [
            {
                "object_id": "img27_obj01",
                "class_name": "clothes",
                "class_id": 3,
                "bbox_xyxy": [100, 175, 590, 350],
                "reviewer": "Agent_D_QA",
                "review_rationale": "Đống quần áo cũ phế thải đổ trộm tại khu đất trống ven bìa rừng."
            }
        ]
    }
]

def main():
    root_dir = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[2] / "data" / "holdout")))
    manifest_p = Path("reports/qa_acceptance/manifest_hinh_anh_rac.json")
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    file_dict = {f["filename"]: f for f in manifest["files"]}

    vis_dir = Path("reports/qa_acceptance/gt_visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    gt_images = []
    total_objects = 0
    all_obj_ids = set()

    for idx, item in enumerate(RAW_GT, 1):
        fn = item["filename"]
        if fn not in file_dict:
            raise ValueError(f"Image {fn} not in manifest!")
        
        info = file_dict[fn]["pil_decode"]["info"]
        w, h = info["width"], info["height"]
        sha256 = file_dict[fn]["sha256"]
        img_path = Path(file_dict[fn]["absolute_path"])

        # Load image for drawing
        with open(img_path, "rb") as f_img:
            arr_bytes = bytearray(f_img.read())
        np_arr = np.asarray(arr_bytes, dtype=np.uint8)
        draw_im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        objs = []
        for obj in item["objects"]:
            obj_id = obj["object_id"]
            if obj_id in all_obj_ids:
                raise ValueError(f"Duplicate object ID: {obj_id}")
            all_obj_ids.add(obj_id)

            cname = obj["class_name"]
            cid = obj["class_id"]
            if cid != CLASS_NAMES.index(cname):
                raise ValueError(f"Class id {cid} mismatch with {cname}")

            bbox = obj["bbox_xyxy"]
            x1, y1, x2, y2 = bbox
            # Bounding box validity checks
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"Invalid bbox {bbox} in {fn} (dims: {w}x{h})")

            objs.append(obj)
            total_objects += 1

            # Draw on debug image
            color = CLASS_COLORS.get(cid, (0, 255, 0))
            cv2.rectangle(draw_im, (x1, y1), (x2, y2), color, 2)
            label_text = f"{cname}:{cid}"
            cv2.putText(draw_im, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save annotated image
        out_vis_path = vis_dir / f"{idx:02d}_{fn}.jpg"
        cv2.imwrite(str(out_vis_path), draw_im)

        gt_images.append({
            "image_index": idx,
            "filename": fn,
            "sha256": sha256,
            "width": w,
            "height": h,
            "object_count": len(objs),
            "objects": objs
        })

    frozen_gt = {
        "metadata": {
            "dataset_name": "Hinh_Anh_Rac_Acceptance_GroundTruth",
            "version": "1.0_FROZEN",
            "frozen_at": "2026-10-09T03:15:00Z",
            "reviewer": "Agent_D_QA",
            "role": "QA Independent Evaluator",
            "classes": CLASS_NAMES,
            "total_images": len(gt_images),
            "total_objects": total_objects
        },
        "images": gt_images
    }

    out_gt_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
    with open(out_gt_path, "w", encoding="utf-8") as f:
        json.dump(frozen_gt, f, indent=2, ensure_ascii=False)

    print(f"Generated Frozen GT: {len(gt_images)} images, {total_objects} instances.")
    print(f"Saved to {out_gt_path}")
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()
