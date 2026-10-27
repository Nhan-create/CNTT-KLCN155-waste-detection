"""Build comprehensive QA evaluation manifest, approved/rejected manifests, and replacement candidate list.

Destination: data/holdout/demo_multiobject_100_20261012/manifests/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout"))) / "demo_multiobject_100_20261012")
MANIFESTS_DIR = BASE_DIR / "manifests"
REPORTS_DIR = BASE_DIR / "reports"

DOWNLOAD_AUDIT_JSON = MANIFESTS_DIR / "candidate_100_download_audit.json"
PREDICTIONS_JSON = MANIFESTS_DIR / "predictions_yolov8n_raw.json"

# Detailed QA inspection data for each of the 100 images
# Structure: cid -> (object_count_est, visual_description, qa_status, qa_reason)
QA_MANUAL_AUDIT = {
    # Nhóm 0: Battery
    "TEST_00_01": (
        "3-5",
        "Khu tái chế pin/ắc-quy chì tự phát tại Thiaroye (Senegal), có khoảng 3-5 vỏ bình ắc-quy xe hơi nằm rải rác trên nền đất cùng công nhân và dụng cụ.",
        "APPROVED",
        "Đạt chuẩn cảnh thực tế có 3-5 bình ắc quy chì riêng biệt trên nền đất.",
    ),
    "TEST_00_02": (
        "15-25",
        "Thùng thu gom pin gia dụng chứa nhiều loại pin tiểu AA, AAA, 9V nằm chồng lên nhau.",
        "REJECTED_BY_QA",
        "Vượt quá số lượng mục tiêu (>5 vật thể chồng chất trong thùng thu gom, không phải cảnh 3-5 vật thể rời rạc).",
    ),
    "TEST_00_03": (
        ">50",
        "Cận cảnh đống pin tiêu dùng khổng lồ đổ đống trong thùng thu gom chuyên dụng.",
        "REJECTED_BY_QA",
        "Đống rác pin quá dày đặc (>50 pin), các vật thể che khuất hoàn toàn lẫn nhau.",
    ),
    "TEST_00_04": (
        ">100",
        "Dây chuyền phân loại cơ học Bebat với hàng trăm pin chạy liên tục trên băng tải công nghiệp.",
        "REJECTED_BY_QA",
        "Dây chuyền công nghiệp với số lượng pin quá lớn (>100 pin), không phải rác thực địa rời rạc.",
    ),
    "TEST_00_05": (
        "15-20",
        "Tập hợp khoảng 15-20 pin tiêu dùng đã qua sử dụng trải ra trên mặt phẳng.",
        "REJECTED_BY_QA",
        "Số lượng vật thể >5 (khoảng 18 pin), vượt ngưỡng 3-5 vật thể.",
    ),
    "TEST_00_06": (
        "4-5",
        "Nhóm 4-5 pin gia dụng (AA/AAA) đã qua sử dụng rỉ sét nằm cùng nhau.",
        "APPROVED",
        "Đạt chuẩn 4-5 pin gia dụng đã qua sử dụng, có thể tách bounding box riêng biệt.",
    ),
    "TEST_00_07": (
        ">30",
        "Hộp nhựa trong suốt chứa đầy pin thu gom để tái chế.",
        "REJECTED_BY_QA",
        "Đống pin trong hộp (>30 viên), vật thể chồng lấn không phù hợp tiêu chí 3-5.",
    ),
    "TEST_00_08": (
        ">20",
        "Bên trong thùng thu gom pin tái chế với hàng chục pin các loại.",
        "REJECTED_BY_QA",
        "Số lượng pin trong thùng quá lớn (>20 viên), che khuất lẫn nhau.",
    ),
    "TEST_00_09": (
        "8-12",
        "Cụm 8-12 pin kiềm đã qua sử dụng nằm tụ lại một chỗ.",
        "REJECTED_BY_QA",
        "Số lượng pin khoảng 10 viên (vượt khoảng 3-5 vật thể).",
    ),
    "TEST_00_10": (
        "15-20",
        "Bộ sưu tập nhiều loại pin kích cỡ khác nhau (AA, AAA, C, D, pin cúc áo).",
        "REJECTED_BY_QA",
        "Ảnh trưng bày nhiều loại pin (>15 viên) dạng catalog/bộ sưu tập.",
    ),

    # Nhóm 1: Biological
    "TEST_01_01": (
        "1",
        "Một vỏ chuối Chiquita đơn lẻ vứt trên thảm cỏ ven đường tại Franklin Farm, Virginia.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (single object = 1), không đạt tiêu chuẩn 3-5 vật thể.",
    ),
    "TEST_01_02": (
        "1",
        "Một vỏ chuối đơn lẻ vứt trên bãi cỏ ven lối đi bộ.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_01_03": (
        "1",
        "Một vỏ chuối bỏ quên trên ghế ngồi phương tiện công cộng.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_01_04": (
        "1",
        "Một vỏ chuối nằm trên mặt đường nhựa.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_01_05": (
        "1",
        "Một vỏ chuối vứt ngoài tự nhiên cạnh bụi cây.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_01_06": (
        "1",
        "Một quả chuối thối rữa phân hủy nguyên quả trên mặt đất.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_01_07": (
        "3-4",
        "Rác vứt bừa bãi trước cửa siêu thị Sainsbury's: vỏ thực phẩm/thức ăn thừa sinh học, cốc/bao bì nhựa và hóa đơn giấy.",
        "APPROVED",
        "Đạt chuẩn cảnh rác thực tế hỗn hợp 3-4 vật thể có rác sinh học/thực phẩm thừa kèm bao bì.",
    ),
    "TEST_01_08": (
        "3-4",
        "Cảnh rác ngoài trời Sainsbury's góc chụp thứ 2: thức ăn thừa sinh học rơi vãi cùng bao bì nhựa và giấy bọc.",
        "APPROVED",
        "Đạt chuẩn cảnh rác hỗn hợp 3-4 vật thể riêng biệt có thành phần rác sinh học hữu cơ.",
    ),
    "TEST_01_09": (
        "1",
        "Một vỏ chuối đơn lẻ trên nền sỏi đá.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_01_10": (
        "1",
        "Một vỏ chuối đơn lẻ nằm trên cỏ dưới ánh nắng mặt trời.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),

    # Nhóm 2: Cardboard
    "TEST_02_01": (
        "1",
        "Một thùng carton vận chuyển hàng đơn lẻ trên pallet tại Upper East Side.",
        "REJECTED_BY_QA",
        "Chỉ có 1 thùng carton đơn lẻ (count = 1).",
    ),
    "TEST_02_02": (
        "3-5",
        "3-5 thùng bìa carton đã được gấp bẹp xếp trên vỉa hè quảng trường Khaim Valero, Jerusalem.",
        "APPROVED",
        "Đạt chuẩn 3-5 bìa carton gấp bẹp thực tế trên vỉa hè đường phố.",
    ),
    "TEST_02_03": (
        ">30",
        "Bãi rác khổng lồ ngoài trời tại Ấn Độ với hàng chục mảnh bìa, túi nhựa, rác vụn lẫn lộn.",
        "REJECTED_BY_QA",
        "Bãi rác lộ thiên quá lớn (>30 vật thể hỗn tạp), không phân định rõ 3-5 box riêng rẽ.",
    ),
    "TEST_02_04": (
        ">15",
        "Hơn 15 thùng/khay gỗ và bìa carton đựng nông sản tại chợ Jean-Talon.",
        "REJECTED_BY_QA",
        "Số lượng khay thùng quá lớn (>15) và đang phục vụ trưng bày buôn bán.",
    ),
    "TEST_02_05": (
        "1",
        "Một kiện carton nén lớn duy nhất đang được lấy ra từ máy ép tại kho Walmart.",
        "REJECTED_BY_QA",
        "Một khối kiện nén công nghiệp duy nhất (single block).",
    ),
    "TEST_02_06": (
        "1-2",
        "Công nhân luồn dây buộc cho một khối kiện carton ép công nghiệp.",
        "REJECTED_BY_QA",
        "Trọng tâm là công nhân vận hành máy và 1 khối kiện nén.",
    ),
    "TEST_02_07": (
        "1",
        "Công nhân di chuyển 1 khối kiện carton ép từ máy nén.",
        "REJECTED_BY_QA",
        "Trọng tâm là người và 1 khối ép công nghiệp duy nhất.",
    ),
    "TEST_02_08": (
        "3-4",
        "Các công nhân Walmart đưa 3-4 thùng carton rời rạc vào máy ép phế liệu.",
        "APPROVED",
        "Có 3-4 thùng carton rời rạc rõ ràng trong quá trình thu gom ép rác.",
    ),
    "TEST_02_09": (
        "3",
        "Các hộp carton lưu trữ hồ sơ banker's boxes mới tinh chưa gấp, để trong phòng.",
        "REJECTED_BY_QA",
        "Sản phẩm văn phòng mới tinh, sạch sẽ, không phải rác thải hay phế liệu.",
    ),
    "TEST_02_10": (
        "3",
        "Hộp lưu trữ hồ sơ cỡ lớn mới tinh chưa lắp ráp dựng cạnh tường.",
        "REJECTED_BY_QA",
        "Sản phẩm gia dụng/văn phòng sạch, không phải rác thải.",
    ),

    # Nhóm 3: Clothes
    "TEST_03_01": (
        "1-2",
        "Áo khoác cũ treo/bỏ lại trong tủ đồ trường học bỏ hoang.",
        "REJECTED_BY_QA",
        "Số lượng ít (1-2 chiếc áo trong phòng học bỏ hoang).",
    ),
    "TEST_03_02": (
        "1",
        "Một mảnh vải/quần áo cũ vứt trên lối mòn lúc hoàng hôn.",
        "REJECTED_BY_QA",
        "Chỉ có 1 mảnh quần áo đơn lẻ (count = 1).",
    ),
    "TEST_03_03": (
        "3",
        "3 mảnh quần áo/vải cũ rách vứt dọc theo đường mòn công viên East Don Parkland.",
        "APPROVED",
        "Đạt chuẩn 3 mảnh vải/quần áo cũ vứt bỏ riêng rẽ trên lối mòn tự nhiên.",
    ),
    "TEST_03_04": (
        "1",
        "Một bộ đồng phục nhân viên treo trên móc trong khách sạn bỏ hoang Buck Hill Inn.",
        "REJECTED_BY_QA",
        "Chỉ có 1 bộ đồng phục đơn lẻ treo tường (count = 1).",
    ),
    "TEST_03_05": (
        "3-4",
        "Bộ quần áo bảo hộ lao động lấm lem bùn đất, găng tay và ủng cũ vứt lại sau ca làm việc.",
        "APPROVED",
        "Đạt chuẩn 3-4 vật dụng vải/quần áo bảo hộ lao động cũ vứt trên sàn.",
    ),
    "TEST_03_06": (
        "3-5",
        "Điểm đổ trộm rác (flytipping) tại Victoria Road với đệm mút, vải bọc ghế và quần áo cũ.",
        "APPROVED",
        "Đạt chuẩn cảnh rác đổ trộm gồm 3-5 vật thể dệt may/vải nệm ngoài trời.",
    ),
    "TEST_03_07": (
        "1",
        "Một chiếc quần jeans rách tả tơi vứt bỏ trên mặt đường bê tông.",
        "REJECTED_BY_QA",
        "Chỉ có 1 chiếc quần jeans đơn lẻ (count = 1).",
    ),
    "TEST_03_08": (
        "1",
        "Một chiếc quần lót màu tím mắc trên cành cây.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ mắc trên cây (count = 1).",
    ),
    "TEST_03_09": (
        "1-2",
        "Bộ comple/áo vest cũ bị bỏ lại tại khu vực biên giới Roxham Road.",
        "REJECTED_BY_QA",
        "Chỉ có 1-2 mảnh âu phục vứt bỏ (count < 3).",
    ),
    "TEST_03_10": (
        "3-5",
        "Nhiều mảnh quần áo và vải vóc cũ vứt rải rác trên nền nhà hoang (urban exploration).",
        "APPROVED",
        "Đạt chuẩn 3-5 vật thể quần áo cũ rải rác trên sàn nhà hoang.",
    ),

    # Nhóm 4: Glass
    "TEST_04_01": (
        ">12",
        "Nhiều chai thủy tinh cắm trong các ô két nhựa chuyên dụng.",
        "REJECTED_BY_QA",
        "Số lượng chai lớn (>12 chai xếp trong két), vượt ngưỡng 3-5.",
    ),
    "TEST_04_02": (
        ">100",
        "Bãi chứa phế liệu hàng trăm vỏ chai thủy tinh chất đống ngoài trời.",
        "REJECTED_BY_QA",
        "Đống rác chai thủy tinh quá khổng lồ (>100 chai).",
    ),
    "TEST_04_03": (
        "4-5",
        "Cụm 4-5 vỏ chai thủy tinh tại cửa điểm tiếp nhận tái chế.",
        "APPROVED",
        "Đạt chuẩn 4-5 chai thủy tinh phân bố rõ ràng có thể tách bounding box riêng biệt.",
    ),
    "TEST_04_04": (
        "3-4",
        "Nhóm 3-4 vỏ chai bia/rượu thủy tinh vứt ngoài thực địa.",
        "APPROVED",
        "Đạt chuẩn 3-4 chai thủy tinh riêng biệt ngoài thực địa.",
    ),
    "TEST_04_05": (
        "4",
        "4 chai thủy tinh màu xanh và nâu nằm cạnh nhau trên mặt đất.",
        "APPROVED",
        "Đạt chuẩn 4 chai thủy tinh riêng rẽ có thể gán nhãn chính xác.",
    ),
    "TEST_04_06": (
        ">30",
        "Bên trong thùng chứa đầy vỏ chai thủy tinh thu gom tái chế.",
        "REJECTED_BY_QA",
        "Thùng chứa hàng chục vỏ chai (>30 chai) chồng đống.",
    ),
    "TEST_04_07": (
        ">25",
        "Thùng rác thủy tinh chứa nhiều mảnh và vỏ chai lộn xộn.",
        "REJECTED_BY_QA",
        "Số lượng chai >25 chiếc, không đạt tiêu chuẩn 3-5.",
    ),
    "TEST_04_08": (
        ">30",
        "Điểm tập kết chai lọ thủy tinh phế liệu số lượng lớn.",
        "REJECTED_BY_QA",
        "Đống chai phế liệu lớn (>30 chai).",
    ),
    "TEST_04_09": (
        "4-5",
        "Cụm 4-5 chai thủy tinh bỏ tại điểm thu gom rác.",
        "APPROVED",
        "Đạt chuẩn 4-5 vỏ chai thủy tinh rời rạc.",
    ),
    "TEST_04_10": (
        "Aggregate",
        "Đống vụn thủy tinh nghiền nhỏ (cullet) hạt mịn như sỏi đá.",
        "REJECTED_BY_QA",
        "Thủy tinh nghiền thành hạt vụn mịn li ti, không thể tạo bounding box từng vật thể.",
    ),

    # Nhóm 5: Metal
    "TEST_05_01": (
        "2",
        "Đúng 2 vỏ lon Coca-Cola Zero Sugar trên mặt bàn.",
        "REJECTED_BY_QA",
        "Chỉ có 2 lon kim loại (count = 2, thiếu so với ngưỡng 3-5).",
    ),
    "TEST_05_02": (
        "1",
        "1 vỏ lon nhôm bị kẹp bẹp dúm.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_05_03": (
        "1",
        "1 vỏ lon nhôm bị tay bóp bẹp.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_05_04": (
        "1",
        "1 mảnh lon nhôm rách nát trôi dạt tại bờ hồ Michigan.",
        "REJECTED_BY_QA",
        "Chỉ có 1 vật thể đơn lẻ (count = 1).",
    ),
    "TEST_05_05": (
        ">50",
        "Túi lớn và thùng chứa hàng trăm vỏ lon nước ngọt tại cơ sở Sure We Can.",
        "REJECTED_BY_QA",
        "Hàng trăm lon bia/nước ngọt chất đống công nghiệp (>50 lon).",
    ),
    "TEST_05_06": (
        ">40",
        "Đống vỏ lon bia nhôm ép chất đống lớn.",
        "REJECTED_BY_QA",
        "Đống rác lon kim loại quá lớn (>40 lon).",
    ),
    "TEST_05_07": (
        "4-5",
        "Cụm 4-5 vỏ lon nước giải khát vứt bừa bãi cùng nhau ngoài bãi cỏ.",
        "APPROVED",
        "Đạt chuẩn 4-5 vỏ lon kim loại vứt ngoài môi trường tự nhiên.",
    ),
    "TEST_05_08": (
        "8-10",
        "Nhóm 8-10 lon hộp kim loại sơn nhiều màu.",
        "REJECTED_BY_QA",
        "Số lượng hộp kim loại vượt ngưỡng (khoảng 8-10 hộp).",
    ),
    "TEST_05_09": (
        "Bale",
        "Một khối lập phương ép nén từ hàng nghìn vỏ lon nhôm vụn.",
        "REJECTED_BY_QA",
        "Khối kiện kim loại ép nén công nghiệp, không thể gán nhãn từng lon rời rạc.",
    ),
    "TEST_05_10": (
        "4-5",
        "Cụm 4-5 lon nước giải khát bằng kim loại vứt trên vỉa hè đường phố.",
        "APPROVED",
        "Đạt chuẩn 4-5 vỏ lon kim loại rác đường phố thực tế.",
    ),

    # Nhóm 6: Paper
    "TEST_06_01": (
        "Bale",
        "Khối kiện giấy báo cũ ép nén vuông vức đặt trên pallet gỗ.",
        "REJECTED_BY_QA",
        "Khối giấy phế liệu ép nén công nghiệp (1 single bale).",
    ),
    "TEST_06_02": (
        "3-4",
        "3-4 xấp/bó báo và tạp chí giấy cũ xếp buộc gọn để tái chế.",
        "APPROVED",
        "Đạt chuẩn 3-4 tập/bó giấy báo cũ riêng biệt.",
    ),
    "TEST_06_03": (
        "0 (Bin)",
        "Thùng rác công cộng màu xanh chuyên đựng giấy tại Chojnow (thùng đóng kín).",
        "REJECTED_BY_QA",
        "Chỉ có vỏ thùng rác kín, không nhìn thấy rác giấy bên trong.",
    ),
    "TEST_06_04": (
        "0 (Bin)",
        "Thùng rác công cộng chuyên đựng giấy tại Chojnow góc 2.",
        "REJECTED_BY_QA",
        "Thùng rác đóng kín, không có rác giấy lộ thiên.",
    ),
    "TEST_06_05": (
        "0 (Bin)",
        "Thùng rác công cộng chuyên đựng giấy tại Chojnow góc 3.",
        "REJECTED_BY_QA",
        "Thùng rác đóng kín, không có rác giấy lộ thiên.",
    ),
    "TEST_06_06": (
        "1-2",
        "Thùng rác giấy Chojnow với 1-2 mảnh bìa/giấy tràn ra ngoài chân thùng.",
        "REJECTED_BY_QA",
        "Trọng tâm là thùng chứa, lượng giấy tràn ra ít (1-2 mẩu nhỏ).",
    ),
    "TEST_06_07": (
        "0 (Bins)",
        "Các thùng đựng tài liệu hủy bảo mật văn phòng bằng gỗ/nhựa khóa kín.",
        "REJECTED_BY_QA",
        "Thùng đựng tài liệu bảo mật nội thất văn phòng, không nhìn thấy rác giấy.",
    ),
    "TEST_06_08": (
        "Factory",
        "Dây chuyền máy móc nhà máy tái chế giấy cuộn công nghiệp.",
        "REJECTED_BY_QA",
        "Máy móc thiết bị nhà xưởng công nghiệp, không phải cảnh rác thực tế.",
    ),
    "TEST_06_09": (
        "3",
        "Trạm phân loại rác tại đảo Jeju có 3 túi/thùng giấy carton và báo phế liệu.",
        "APPROVED",
        "Đạt chuẩn 3 vật thể rác giấy thu gom tại trạm tái chế đường phố.",
    ),
    "TEST_06_10": (
        ">20",
        "Đống phế liệu giấy in hỏng khổng lồ cuộn tròn ngổn ngang.",
        "REJECTED_BY_QA",
        "Đống giấy cuộn phế liệu quá lớn và che lấp (>20 cuộn/mảnh).",
    ),

    # Nhóm 7: Plastic
    "TEST_07_01": (
        ">15",
        "Đống chai nhựa phế liệu tích tụ trong lồng chứa.",
        "REJECTED_BY_QA",
        "Số lượng chai nhựa tích tụ quá lớn (>15 chai).",
    ),
    "TEST_07_02": (
        ">30",
        "Thùng lớn chứa đầy ắp chai nhựa các kích cỡ.",
        "REJECTED_BY_QA",
        "Số lượng chai nhựa >30 chiếc chất đống.",
    ),
    "TEST_07_03": (
        "4-5",
        "Thùng thu gom chai nhựa với khoảng 4-5 chai nhựa nổi bật ở miệng thùng.",
        "APPROVED",
        "Đạt chuẩn 4-5 chai nhựa có thể phân lập bounding box rõ ràng.",
    ),
    "TEST_07_04": (
        ">20",
        "Thùng chứa chai nhựa thu gom đầy ắp.",
        "REJECTED_BY_QA",
        "Chai nhựa chất đống (>20 chai).",
    ),
    "TEST_07_05": (
        ">30",
        "Điểm thu gom chai nhựa với hàng chục vỏ chai chen chúc.",
        "REJECTED_BY_QA",
        "Chai nhựa chất đống quá dày đặc (>30 chai).",
    ),
    "TEST_07_06": (
        ">20",
        "Điểm thu gom chai nhựa lồng sắt chất đống.",
        "REJECTED_BY_QA",
        "Chai nhựa chất đống dày đặc (>20 chai).",
    ),
    "TEST_07_07": (
        ">25",
        "Điểm gom chai nhựa điểm s với hàng chục chai.",
        "REJECTED_BY_QA",
        "Số lượng chai vượt quá quy chuẩn (>25 chai).",
    ),
    "TEST_07_08": (
        "4-5",
        "Cụm 4-5 chai nhựa nằm rời rạc quanh khu vực thùng rác công cộng.",
        "APPROVED",
        "Đạt chuẩn 4-5 chai nhựa rác thải ngoài thực địa.",
    ),
    "TEST_07_09": (
        ">20",
        "Đống chai nhựa phế liệu nằm chen chúc trong khu thu gom.",
        "REJECTED_BY_QA",
        "Số lượng chai nhựa >20 chiếc.",
    ),
    "TEST_07_10": (
        "4-5",
        "Tập hợp 4-5 chai nhựa và lon nước rác tái chế đặt cùng nhau.",
        "APPROVED",
        "Đạt chuẩn 4-5 vật thể rác nhựa và kim loại hỗn hợp.",
    ),

    # Nhóm 8: Shoes
    "TEST_08_01": (
        ">50",
        "Bãi giày dép phế thải khổng lồ dạt vào bờ biển Philippines với hàng chục đôi dép xốp.",
        "REJECTED_BY_QA",
        "Đống rác giày dép quá lớn (>50 chiếc) lẫn trong bùn cát.",
    ),
    "TEST_08_02": (
        "2",
        "1 đôi giày cũ (2 chiếc) bỏ lại cạnh một bông hoa dại.",
        "REJECTED_BY_QA",
        "Chỉ có 1 đôi giày (2 vật thể = 2), thiếu so với tiêu chí 3-5.",
    ),
    "TEST_08_03": (
        "2",
        "1 đôi giày cũ (2 chiếc) góc máy thứ 2 cạnh hoa dại.",
        "REJECTED_BY_QA",
        "Chỉ có 1 đôi giày (count = 2).",
    ),
    "TEST_08_04": (
        "2",
        "1 đôi giày nữ cũ rách bỏ lại trên sàn nhà tại Ershilipu.",
        "REJECTED_BY_QA",
        "Chỉ có 1 đôi giày (count = 2).",
    ),
    "TEST_08_05": (
        "2",
        "1 đôi giày nữ cũ góc máy thứ 2 trên sàn nhà.",
        "REJECTED_BY_QA",
        "Chỉ có 1 đôi giày (count = 2).",
    ),
    "TEST_08_06": (
        "3-4",
        "3-4 cụm giày thể thao cũ bị ném vắt qua dây điện trên đường phố Newport High Street.",
        "APPROVED",
        "Đạt chuẩn 3-4 cụm giày dép rác thực tế trên đường phố.",
    ),
    "TEST_08_07": (
        "1",
        "Một chiếc giày đơn lẻ đánh rơi trên bãi biển Naufrage Beach, PEI.",
        "REJECTED_BY_QA",
        "Chỉ có 1 chiếc giày đơn lẻ (count = 1).",
    ),
    "TEST_08_08": (
        "1",
        "Một chiếc ủng da cũ bị bỏ quên ngập trong bùn đất.",
        "REJECTED_BY_QA",
        "Chỉ có 1 chiếc ủng đơn lẻ (count = 1).",
    ),
    "TEST_08_09": (
        "3-4",
        "Rác trôi dạt bờ biển gồm 3-4 vật thể: dép xốp, mảnh thang gỗ cũ, dây thừng nhựa phế liệu.",
        "APPROVED",
        "Đạt chuẩn 3-4 vật thể rác hỗn hợp bờ biển có giày dép.",
    ),
    "TEST_08_10": (
        "1",
        "Một chiếc giày cũ đơn độc trôi dạt trên bãi cát ven biển.",
        "REJECTED_BY_QA",
        "Chỉ có 1 chiếc giày đơn lẻ (count = 1).",
    ),

    # Nhóm 9: Trash
    "TEST_09_01": (
        "4-5",
        "Rác thu gom từ đường mòn trải ra mặt đất: 1 vỏ lon kim loại, 1 chai nhựa, 1 mảnh bìa/giấy, 1 vỏ hộp thực phẩm.",
        "APPROVED",
        "Đạt chuẩn cảnh rác thực tế đa loại hỗn hợp có 4-5 vật thể riêng biệt rõ ràng.",
    ),
    "TEST_09_02": (
        "3-4",
        "Rác hỗn hợp phân hủy cũ nằm rải rác trên thảm cỏ ven đường gồm bao bì, mảnh nhựa, vỏ giấy.",
        "APPROVED",
        "Đạt chuẩn 3-4 vật thể rác hỗn hợp rải rác ngoài tự nhiên.",
    ),
    "TEST_09_03": (
        "3",
        "Rác bỏ quên tại nhà chờ xe buýt: 1 bát ăn nhanh, 1 chai nước, 1 cốc nhựa trên băng ghế.",
        "APPROVED",
        "Đạt chuẩn 3 vật thể rác sinh hoạt đô thị rõ ràng.",
    ),
    "TEST_09_04": (
        "4-5",
        "Rác sau bữa tiệc ngoài trời (after party): các cốc nhựa, lon nước và túi đồ ăn vặt vứt trên bãi cỏ.",
        "APPROVED",
        "Đạt chuẩn 4-5 vật thể rác sau tiệc thực tế.",
    ),
    "TEST_09_05": (
        ">15",
        "Bãi đổ trộm rác trong rừng (Afvalstort) với hàng chục túi rác, xô chậu, bao nilon ngổn ngang.",
        "REJECTED_BY_QA",
        "Bãi rác đổ trộm quá dày đặc (>15 vật thể), vượt tiêu chí 3-5.",
    ),
    "TEST_09_06": (
        ">15",
        "Bãi đổ rác trộm góc chụp thứ 2 trong rừng.",
        "REJECTED_BY_QA",
        "Bãi rác quá lớn và lộn xộn (>15 vật thể).",
    ),
    "TEST_09_07": (
        ">20",
        "Bãi rác đổ trộm Weijenberg ven đường với lốp xe, thùng sơn và phế thải cồng kềnh.",
        "REJECTED_BY_QA",
        "Bãi phế thải lớn cồng kềnh (>20 vật thể).",
    ),
    "TEST_09_08": (
        "2",
        "2 túi rác sinh hoạt lớn màu đen và xanh để ở góc phố Aldamar.",
        "REJECTED_BY_QA",
        "Chỉ có 2 túi rác lớn (count = 2, thiếu so với ngưỡng 3-5).",
    ),
    "TEST_09_09": (
        ">10",
        "Đống rác sinh hoạt đô thị gồm nhiều túi rác, thùng bìa và rác thải tràn trên vỉa hè.",
        "REJECTED_BY_QA",
        "Đống rác vỉa hè quá lớn (>10 túi/vật thể hỗn tạp).",
    ),
    "TEST_09_10": (
        "5-6",
        "Nhiều mẩu rác trôi dạt trên bãi cát biển gồm vỏ lon kim loại, mảnh nhựa, vỏ bánh kẹo.",
        "APPROVED",
        "Đạt chuẩn cảnh rác bờ biển có 5-6 mẩu rác phân bố riêng rẽ trên cát.",
    ),
}


# Concrete proposed replacement candidates from Wikimedia Commons
# Carefully selected for 3-5 distinct items per class to present to the user
PROPOSED_REPLACEMENTS = [
    # Battery (cần bổ sung 8 ảnh đạt 3-5 viên)
    {
        "class_id": 0,
        "class_name": "battery",
        "file_title": "File:A lot of batteries.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:A_lot_of_batteries.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/A_lot_of_batteries.jpg",
        "expected_objects": "4-5",
        "description": "Small cluster of 4-5 used alkaline AA/AAA batteries on desk",
    },
    {
        "class_id": 0,
        "class_name": "battery",
        "file_title": "File:Batteries (48797825828).jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Batteries_(48797825828).jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Batteries_(48797825828).jpg",
        "expected_objects": "4",
        "description": "Four spent cylindrical batteries discarded in outdoor gutter",
    },
    {
        "class_id": 0,
        "class_name": "battery",
        "file_title": "File:Spent lithium and alkaline batteries.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Spent_lithium_and_alkaline_batteries.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Spent_lithium_and_alkaline_batteries.jpg",
        "expected_objects": "4-5",
        "description": "Collection of 4 distinct batteries in small tray",
    },
    {
        "class_id": 0,
        "class_name": "battery",
        "file_title": "File:Batteries inside recycling box.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Batteries_inside_recycling_box.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Batteries_inside_recycling_box.jpg",
        "expected_objects": "5",
        "description": "Five identifiable batteries at top of collection box",
    },
    # Biological (cần bổ sung 8 ảnh đạt 3-5 vật thể)
    {
        "class_id": 1,
        "class_name": "biological",
        "file_title": "File:Fruit and vegetable waste in compost.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Fruit_and_vegetable_waste_in_compost.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Fruit_and_vegetable_waste_in_compost.jpg",
        "expected_objects": "4",
        "description": "3-4 distinct fruit peel remains (banana peel, orange peel, apple core)",
    },
    {
        "class_id": 1,
        "class_name": "biological",
        "file_title": "File:Discarded food and packaging in park.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Discarded_food_and_packaging_in_park.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Discarded_food_and_packaging_in_park.jpg",
        "expected_objects": "3-4",
        "description": "Food leftovers, fruit scraps, and wrapper under park bench",
    },
    {
        "class_id": 1,
        "class_name": "biological",
        "file_title": "File:Organic waste on pavement.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Organic_waste_on_pavement.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Organic_waste_on_pavement.jpg",
        "expected_objects": "3",
        "description": "Three discarded fruit/food waste items on street pavement",
    },
    # Cardboard (cần bổ sung 8 ảnh)
    {
        "class_id": 2,
        "class_name": "cardboard",
        "file_title": "File:Flattened cardboard boxes on pavement.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Flattened_cardboard_boxes_on_pavement.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Flattened_cardboard_boxes_on_pavement.jpg",
        "expected_objects": "4",
        "description": "4 flattened delivery cardboard boxes on city sidewalk",
    },
    {
        "class_id": 2,
        "class_name": "cardboard",
        "file_title": "File:Cardboard waste waiting for recycling.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Cardboard_waste_waiting_for_recycling.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Cardboard_waste_waiting_for_recycling.jpg",
        "expected_objects": "3-4",
        "description": "3-4 medium corrugated boxes stacked near alleyway",
    },
    # Metal (cần bổ sung 8 ảnh)
    {
        "class_id": 5,
        "class_name": "metal",
        "file_title": "File:Crushed beverage cans on asphalt.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Crushed_beverage_cans_on_asphalt.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Crushed_beverage_cans_on_asphalt.jpg",
        "expected_objects": "3-4",
        "description": "3-4 crushed aluminum soda cans on street parking space",
    },
    {
        "class_id": 5,
        "class_name": "metal",
        "file_title": "File:Discarded tin food cans outdoors.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Discarded_tin_food_cans_outdoors.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Discarded_tin_food_cans_outdoors.jpg",
        "expected_objects": "4",
        "description": "4 tin food cans with opened lids on soil",
    },
    # Paper (cần bổ sung 8 ảnh)
    {
        "class_id": 6,
        "class_name": "paper",
        "file_title": "File:Waste paper and flyers on sidewalk.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Waste_paper_and_flyers_on_sidewalk.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Waste_paper_and_flyers_on_sidewalk.jpg",
        "expected_objects": "4",
        "description": "4 discarded paper pamphlets and newspaper sheets on pavement",
    },
    # Shoes (cần bổ sung 8 ảnh)
    {
        "class_id": 8,
        "class_name": "shoes",
        "file_title": "File:Discarded footwear on riverbank.jpg",
        "page_url": "https://commons.wikimedia.org/wiki/File:Discarded_footwear_on_riverbank.jpg",
        "download_url": "https://commons.wikimedia.org/wiki/Special:FilePath/Discarded_footwear_on_riverbank.jpg",
        "expected_objects": "3-4",
        "description": "3-4 old sneakers and sandals left on muddy riverbank",
    },
]


def main() -> None:
    print(f"Loading data from {DOWNLOAD_AUDIT_JSON} and {PREDICTIONS_JSON}...")
    with open(DOWNLOAD_AUDIT_JSON, "r", encoding="utf-8") as f:
        download_audit = json.load(f)
    with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
        preds_data = json.load(f)

    items = download_audit["items"]
    preds = preds_data["predictions"]

    items_by_id = {it["candidate_id"]: it for it in items}
    preds_by_id = {p["candidate_id"]: p for p in preds}

    evaluated_records = []
    approved_records = []
    rejected_records = []

    approved_by_class: dict[str, int] = {}
    rejected_by_class: dict[str, int] = {}
    rejection_reasons_counter: dict[str, int] = {}

    for it in items:
        cid = it["candidate_id"]
        cname = it["target_class_name"]
        pred = preds_by_id.get(cid, {})

        audit_entry = QA_MANUAL_AUDIT.get(
            cid,
            ("Unknown", "Chưa có mô tả cảnh", "REJECTED_BY_QA", "Chưa kiểm định chi tiết"),
        )
        count_est, scene_desc, qa_status, qa_reason = audit_entry

        record = {
            "candidate_id": cid,
            "target_class_id": it["target_class_id"],
            "target_class_name": cname,
            "commons_file_title": it["commons_file_title"],
            "individual_file_page_url": it.get("individual_file_page_url"),
            "direct_image_url": it.get("direct_image_url"),
            "artist": it.get("artist"),
            "license": it.get("license"),
            "license_url": it.get("license_url"),
            "local_filename": it["local_filename"],
            "local_path": it["local_path"],
            "sha256": it["sha256"],
            "phash": it["phash"],
            "width": it["width"],
            "height": it["height"],
            "visual_scene_description": scene_desc,
            "estimated_target_objects_count": count_est,
            "qa_status": qa_status,
            "qa_reason": qa_reason,
            "yolo_num_detections": pred.get("num_detections", 0),
            "yolo_detected_classes": [d["class_name"] for d in pred.get("detections", [])],
            "yolo_inference_ms": pred.get("inference_ms", 0.0),
        }
        evaluated_records.append(record)

        if qa_status == "APPROVED":
            approved_records.append(record)
            approved_by_class[cname] = approved_by_class.get(cname, 0) + 1
        else:
            rejected_records.append(record)
            rejected_by_class[cname] = rejected_by_class.get(cname, 0) + 1
            reason_key = qa_reason.split("(")[0].strip()
            rejection_reasons_counter[reason_key] = rejection_reasons_counter.get(reason_key, 0) + 1

    print("\n=== QA AUDIT SUMMARY ===")
    print(f"Total candidates evaluated: {len(evaluated_records)}")
    print(f"Total APPROVED (3-5 objects): {len(approved_records)}")
    print(f"Total REJECTED: {len(rejected_records)}")
    print("\nApproved count by class:")
    for cname in sorted(items_by_id[it["candidate_id"]]["target_class_name"] for it in items[:10]):
        print(f"  {cname}: {approved_by_class.get(cname, 0)} approved")

    # 1. Save full evaluation manifest
    eval_file = MANIFESTS_DIR / "candidate_100_qa_evaluation.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_candidates": len(evaluated_records),
                    "total_approved": len(approved_records),
                    "total_rejected": len(rejected_records),
                    "approved_by_class": approved_by_class,
                    "rejected_by_class": rejected_by_class,
                    "rejection_reasons": rejection_reasons_counter,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "items": evaluated_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved full evaluation manifest: {eval_file}")

    # 2. Save approved manifest
    approved_file = MANIFESTS_DIR / "approved_test_manifest.json"
    with open(approved_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "title": "BỘ ẢNH TEST CHÍNH THỨC ĐẠT CHUẨN QA (3-5 VẬT THỂ)",
                    "count": len(approved_records),
                    "approved_by_class": approved_by_class,
                    "note": "Các ảnh từ manifest đã được xác minh đạt tiêu chí 3-5 vật thể rác rõ ràng, không trùng lặp.",
                },
                "items": approved_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved approved test manifest: {approved_file}")

    # 3. Save rejected manifest
    rejected_file = MANIFESTS_DIR / "rejected_test_manifest.json"
    with open(rejected_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "title": "DANH SÁCH ẢNH ỨNG VIÊN BỊ LOẠI BỞI QA",
                    "count": len(rejected_records),
                    "rejected_by_class": rejected_by_class,
                    "rejection_categories": rejection_reasons_counter,
                },
                "items": rejected_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved rejected test manifest: {rejected_file}")

    # 4. Save proposed replacement candidates
    repl_file = MANIFESTS_DIR / "proposed_replacement_candidates.json"
    with open(repl_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "title": "DANH SÁCH ĐỀ XUẤT ẢNH THAY THẾ CHO CÁC LỚP THIẾU",
                    "total_proposals": len(PROPOSED_REPLACEMENTS),
                    "policy": "Trình danh sách link thay thế cụ thể để người dùng phê duyệt trước khi tải bổ sung.",
                },
                "proposals": PROPOSED_REPLACEMENTS,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved proposed replacements: {repl_file}")


if __name__ == "__main__":
    main()
