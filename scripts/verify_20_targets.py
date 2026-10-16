import sys
import json
import time
import urllib3
import requests

sys.stdout.reconfigure(encoding='utf-8')
urllib3.disable_warnings()

HEADERS = {"User-Agent": "WasteDetectionResearchBot/1.0 (academic research; contact@university.edu)"}

def search_one(q):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srnamespace=6&srlimit=5&srsearch={requests.utils.quote(q)}"
    r = requests.get(url, headers=HEADERS, verify=False, timeout=12)
    items = r.json().get("query", {}).get("search", [])
    for it in items:
        title = it["title"]
        # get imageinfo
        info_url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url|size|mime&titles={requests.utils.quote(title)}"
        r_info = requests.get(info_url, headers=HEADERS, verify=False, timeout=12)
        pages = r_info.json().get("query", {}).get("pages", {})
        for pid, p in pages.items():
            ii = p.get("imageinfo", [{}])[0]
            mime = ii.get("mime", "")
            if mime.startswith("image/") and ii.get("width", 0) >= 400 and ii.get("height", 0) >= 300:
                return {
                    "title": title,
                    "url": ii.get("url"),
                    "width": ii.get("width"),
                    "height": ii.get("height"),
                    "mime": mime,
                    "size": ii.get("size")
                }
        time.sleep(0.3)
    return None

TARGETS = [
    ("kho_01_don_loai_pin", "Pin bị oxy hóa / rỉ sét", "Corroded Screwdriver Battery filetype:bitmap"),
    ("kho_02_don_loai_thuy_tinh", "Chai thủy tinh vỡ trong bụi cỏ", "Broken Glass - Keller Island filetype:bitmap"),
    ("kho_03_don_loai_kim_loai", "Vỏ lon bia bẹp dí trong đất cát", "Crushed Victoria Bitter beer cans littering filetype:bitmap"),
    ("kho_04_don_loai_bia_carton", "Thùng carton ẩm ướt rách nát ngoài đường", "Dumped cardboard - Devon Close filetype:bitmap"),
    ("kho_05_don_loai_quan_ao", "Đống quần áo cũ vứt bãi rác", "Dumping for Charity filetype:bitmap"),
    ("kho_06_don_loai_giay_dep", "Chiếc giày cũ lấm lem cát biển", "Discarded Nine West shoe on Delray Beach filetype:bitmap"),
    ("kho_07_don_loai_nhua", "Chai nhựa móp méo phơi nắng bạc màu", "Coca-Cola 2 liter bleaching filetype:bitmap"),
    ("kho_08_don_loai_giay", "Báo cũ giấy rác rơi vãi vỉa hè", "Dog sleeping on newspaper waste filetype:bitmap"),
    ("kho_09_don_loai_huu_co", "Rác hữu cơ vỏ chuối thức ăn mục rữa", "banana peel litter grass ground filetype:bitmap"),
    ("kho_10_don_loai_pin_dien_tu", "Pin bo mạch phồng rộp rỉ axit", "Polysix Battery corroded KLM367A filetype:bitmap"),
    ("kho_11_da_loai_bai_bien", "Bãi biển ngập rác chai nhựa, lon, xốp", "Water Pollution with Trash Disposal Garbage Beach filetype:bitmap"),
    ("kho_12_da_loai_ranh_nuoc", "Rãnh nước đô thị ứ đọng vỏ lon và chai", "An Unbearable Tidiness of Cans filetype:bitmap"),
    ("kho_13_da_loai_bo_song", "Bờ biển/bờ suối ngập rác trôi dạt", "004 Beach pollution in Tenerife Atlantic Ocean plastic debris filetype:bitmap"),
    ("kho_14_da_loai_bai_rac_tu_phat", "Đống rác tự phát đổ trộm ven đường", "Roadside flytip near West Dean geograph filetype:bitmap"),
    ("kho_15_da_loai_tham_co", "Rác picnic xả trên thảm cỏ công viên", "litter park grass cans bottles filetype:bitmap"),
    ("kho_16_da_loai_duong_mon", "Rác du lịch kẹt giữa khe đá đường mòn", "litter on trail forest mountain filetype:bitmap"),
    ("kho_17_da_loai_phe_thai_hon_hop", "Đống phế thải bao bì chai lọ ngổn ngang", "dumped rubbish pile road lane geograph filetype:bitmap"),
    ("kho_18_da_loai_kenh_rach", "Rác nhựa và bao bì nổi trên mặt nước", "floating garbage plastic river water filetype:bitmap"),
    ("kho_19_da_loai_thung_rac_tran", "Thùng rác công cộng quá tải vương vãi", "Litter bin rubbish Tottenham London England 1 filetype:bitmap"),
    ("kho_20_da_loai_goc_pho", "Góc phố đô thị bị xả rác bừa bãi", "Alley redumped - Update 19 February 2009 filetype:bitmap"),
]

if __name__ == "__main__":
    final_list = []
    for code, desc, q in TARGETS:
        print(f"Finding {code}: {desc}...")
        item = search_one(q)
        if item:
            item["code"] = code
            item["description"] = desc
            final_list.append(item)
            print(f"  FOUND: {item['title']} ({item['width']}x{item['height']}, {item['mime']})")
        else:
            print(f"  FAILED for query: {q}")
        time.sleep(0.5)

    with open("reports/verified_20_targets.json", "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    print(f"\nTotal verified: {len(final_list)}/{len(TARGETS)}")
