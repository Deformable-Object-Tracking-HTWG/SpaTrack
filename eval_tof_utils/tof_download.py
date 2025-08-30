import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from tqdm import tqdm

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
}


def _parse_share_url(share_url: str) -> Tuple[str, str, str]:
    """
    Splits a Nextcloud public share folder URL into (base_url, token, path_in_share).
    Expected shape: https://<host>/s/<TOKEN>?path=/some/folder
    """
    u = urlparse(share_url)
    base_url = f"{u.scheme}://{u.netloc}"
    parts = [p for p in u.path.split("/") if p]

    token = ""
    for i, p in enumerate(parts):
        if p == "s" and i + 1 < len(parts):
            token = parts[i + 1]
            break
    if not token:
        raise ValueError("Could not infer share token from URL (expected .../s/<token>).")

    q = parse_qs(u.query)
    path_in_share = q.get("path", ["/"])[0] or "/"
    if not path_in_share.startswith("/"):
        path_in_share = "/" + path_in_share
    return base_url, token, path_in_share


def _webdav_list_npy(
    base_url: str,
    token: str,
    password: str,
    path_in_share: str,
    session: requests.Session,
) -> List[Dict[str, Optional[str]]]:
    """
    Lists .npy files (non-recursive) in the given share path via WebDAV (Depth: 1).
    Returns a list of dicts: {"href": str, "name": str, "size": Optional[int]}
    """
    webdav_base = f"{base_url}/public.php/webdav"
    target_url = f"{webdav_base}{quote(path_in_share, safe='/%')}"

    body = """<?xml version="1.0" encoding="utf-8" ?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:resourcetype/>
    <d:getcontentlength/>
    <d:getlastmodified/>
    <d:getcontenttype/>
  </d:prop>
</d:propfind>""".strip()

    headers = {**_HEADERS, "Depth": "1", "Content-Type": "text/xml; charset=utf-8"}
    auth = requests.auth.HTTPBasicAuth(token, password or "")

    r = session.request("PROPFIND", target_url, headers=headers, data=body, auth=auth, timeout=30)
    if r.status_code != 207:
        raise RuntimeError(f"WebDAV PROPFIND failed: HTTP {r.status_code} - {r.text[:200]}")

    ns = {"d": "DAV:"}
    root = ET.fromstring(r.text)

    files: List[Dict[str, Optional[str]]] = []
    for resp in root.findall("d:response", ns):
        href_el = resp.find("d:href", ns)
        if href_el is None:
            continue
        href = href_el.text or ""
        href_decoded = unquote(href)

        propstat = resp.find("d:propstat", ns)
        if propstat is None:
            continue
        prop = propstat.find("d:prop", ns)
        if prop is None:
            continue

        rtype = prop.find("d:resourcetype", ns)
        is_collection = False
        if rtype is not None and list(rtype):
            is_collection = any(child.tag.endswith("collection") for child in rtype)
        if is_collection:
            continue  # only files

        name = os.path.basename(href_decoded.rstrip("/"))
        if not name.lower().endswith(".npy"):
            continue

        size_el = prop.find("d:getcontentlength", ns)
        size_val: Optional[int] = None
        if size_el is not None and size_el.text and size_el.text.isdigit():
            size_val = int(size_el.text)

        if href.startswith("/"):
            download_url = f"{base_url}{href}"
        else:
            download_url = f"{webdav_base}/{name}"

        files.append({"href": download_url, "name": name, "size": size_val})

    return files


def download_all_npy_from_nextcloud_folder(
    share_folder_url: str, target_dir: str, password: str = ""
) -> int:
    """
    Downloads all .npy files from the given Nextcloud public share folder URL into `target_dir`.
    Returns the number of files successfully downloaded or already present.
    """
    os.makedirs(target_dir, exist_ok=True)
    base_url, token, path_in_share = _parse_share_url(share_folder_url)

    with requests.Session() as s:
        try:
            listing = _webdav_list_npy(base_url, token, password, path_in_share, s)
        except Exception as e:
            print(f"❌ Failed to fetch .npy listing: {e}")
            return 0

        if not listing:
            print("❗ No .npy files found in the specified folder.")
            return 0

        print(f"🔎 Found .npy files: {len(listing)}")
        ok = 0
        auth = requests.auth.HTTPBasicAuth(token, password or "")

        for item in listing:
            url = item["href"]
            fname = item["name"]
            expected_size = item["size"]
            out_path = os.path.join(target_dir, fname)

            if os.path.exists(out_path):
                if expected_size is not None and os.path.getsize(out_path) == expected_size:
                    print(f"[=] Skipping (already exists, size matches): {fname}")
                    ok += 1
                    continue
                else:
                    print(f"[~] File exists but size differs/unknown: {fname} -> re-downloading")

            try:
                with s.get(url, headers=_HEADERS, stream=True, timeout=120, auth=auth) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    with open(out_path, "wb") as f, tqdm(
                        desc=fname,
                        total=total or expected_size or None,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                    ) as bar:
                        for chunk in resp.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                print(f"[✓] Saved: {out_path}")
                ok += 1
            except Exception as e:
                print(f"[x] Download error for '{fname}': {e}")

        print(f"📁 Done: {ok}/{len(listing)} .npy files in '{target_dir}'")
        return ok
