# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Tìm kiếm, tải về và mở rộng mẫu ảnh thực tế bổ sung cho `search_candidates`.
"""Search candidate waste images from Wikimedia Commons for 20 challenging categories."""

import json
import time
import urllib3
import requests

urllib3.disable_warnings()

HEADERS = {"User-Agent": "WasteDetectionResearchBot/1.0 (academic research; contact@university.edu)"}

CATEGORIES = [
    # 10 Single-Class Challenging
    ("kho_01_don_loai_pin", "battery", "discarded battery ground"),
    ("kho_02_don_loai_thuy_tinh", "glass", "broken bottle grass glass"),
    ("kho_03_don_loai_kim_loai", "metal", "crushed beverage can litter"),
    ("kho_04_don_loai_bia_carton", "cardboard", "discarded cardboard street"),
    ("kho_05_don_loai_quan_ao", "clothes", "discarded clothes rubbish textile"),
    ("kho_06_don_loai_giay_dep", "shoes", "discarded shoe beach mud"),
    ("kho_07_don_loai_nhua", "plastic", "plastic bottle litter sand"),
    ("kho_08_don_loai_giay", "paper", "discarded paper street pavement"),
    ("kho_09_don_loai_huu_co", "biological", "food waste compost apple peel"),
    ("kho_10_don_loai_pin_dien_tu", "battery/ewaste", "leaking battery corrosion cell"),
    
    # 10 Multi-Class Challenging
    ("kho_11_da_loai_bai_bien", "mixed", "beach plastic bottles cans debris"),
    ("kho_12_da_loai_ranh_nuoc", "mixed", "gutter drainage litter rubbish"),
    ("kho_13_da_loai_bo_song", "mixed", "riverbank plastic litter waste"),
    ("kho_14_da_loai_bai_rac_tu_phat", "mixed", "fly tipping dumped rubbish roadside"),
    ("kho_15_da_loai_tham_co", "mixed", "litter grass bottles cans park"),
    ("kho_16_da_loai_duong_mon", "mixed", "trail litter forest hiking rubbish"),
    ("kho_17_da_loai_phe_thai_hon_hop", "mixed", "waste heap plastic metal cardboard"),
    ("kho_18_da_loai_kenh_rach", "mixed", "floating garbage canal water plastic"),
    ("kho_19_da_loai_thung_rac_tran", "mixed", "overflowing litter bin rubbish street"),
    ("kho_20_da_loai_goc_pho", "mixed", "rubbish dumped corner alley street"),
]

def search_wikimedia(query, limit=5):
    q = f"{query} filetype:bitmap"
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&list=search&srnamespace=6&srlimit={limit}&srsearch={requests.utils.quote(q)}"
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=12)
        if r.status_code == 200:
            return r.json().get("query", {}).get("search", [])
    except Exception as e:
        print(f"Search error for {query}: {e}")
    return []

def get_image_info(title):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=url|size|mime&titles={requests.utils.quote(title)}"
    try:
        r = requests.get(url, headers=HEADERS, verify=False, timeout=12)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for pid, p in pages.items():
                ii_list = p.get("imageinfo", [])
                if ii_list:
                    return ii_list[0]
    except Exception as e:
        print(f"Info error for {title}: {e}")
    return None

if __name__ == "__main__":
    candidates = {}
    for code, waste_type, query in CATEGORIES:
        print(f"Searching: {code} ({waste_type}) -> '{query}'...")
        results = search_wikimedia(query, limit=4)
        found = []
        for res in results:
            title = res["title"]
            info = get_image_info(title)
            if info and info.get("mime", "").startswith("image/"):
                width = info.get("width", 0)
                height = info.get("height", 0)
                # require reasonable resolution (e.g. >= 400px)
                if width >= 400 and height >= 400:
                    found.append({
                        "title": title,
                        "url": info.get("url"),
                        "width": width,
                        "height": height,
                        "mime": info.get("mime"),
                        "size": info.get("size")
                    })
                    break  # Found best candidate
            time.sleep(0.4)
        candidates[code] = {
            "waste_type": waste_type,
            "query": query,
            "candidate": found[0] if found else None
        }
        time.sleep(0.4)

    with open("reports/candidate_20_images.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    
    print("\n=== SUMMARY ===")
    found_count = sum(1 for c in candidates.values() if c["candidate"] is not None)
    print(f"Successfully found {found_count}/20 candidates!")
