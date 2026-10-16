import json
import argparse
import os
from itertools import count
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the jsonline file")
    args = parser.parse_args()
    data = []
    preference = os.environ.get('CATEGORIES', 'cs.CV, cs.CL').split(',')
    preference = list(map(lambda x: x.strip(), preference))
    def rank(cate):
        if cate in preference:
            return preference.index(cate)
        else:
            return len(preference)

    def primary_category(item) -> str:
        categories = item.get("categories") or []
        return categories[0] if categories else "uncategorized"

    with open(args.data, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))

    categories = sorted({primary_category(item) for item in data}, key=rank)
    template = open("paper_template.md", "r").read()
    cnt = {cate: 0 for cate in categories}
    for item in data:
        cnt[primary_category(item)] += 1

    markdown = f"<div id=toc></div>\n\n# Table of Contents\n\n"
    for idx, cate in enumerate(categories):
        markdown += f"- [{cate}](#{cate}) [Total: {cnt[cate]}]\n"

    idx = count(1)
    for cate in categories:
        markdown += f"\n\n<div id='{cate}'></div>\n\n"
        markdown += f"# {cate} [[Back]](#toc)\n\n"
        papers = []
        for item in data:
            if primary_category(item) != cate:
                continue
            # Safely access AI fields with default values
            ai_data = item.get('AI', {})
            if not ai_data or not isinstance(ai_data, dict):
                print(f"Skipping item '{item.get('title', 'Unknown')}' due to missing or invalid AI data")
                continue

            # Check if all required AI fields are present
            required_fields = ['tldr', 'motivation', 'method', 'result', 'conclusion']
            if not all(field in ai_data for field in required_fields):
                print(f"Skipping item '{item.get('title', 'Unknown')}' due to incomplete AI fields")
                continue

            papers.append(
                template.format(
                    title=item["title"],
                    authors=",".join(item["authors"]),
                    summary=item["summary"],
                    url=item['abs'],
                    tldr=ai_data.get('tldr', ''),
                    motivation=ai_data.get('motivation', ''),
                    method=ai_data.get('method', ''),
                    result=ai_data.get('result', ''),
                    conclusion=ai_data.get('conclusion', ''),
                    cate=primary_category(item),
                    idx=next(idx)
                )
            )
        markdown += "\n\n".join(papers)
    output_path = Path(args.data).with_name(Path(args.data).stem.split('_')[0] + '.md')
    with output_path.open("w", encoding="utf-8") as f:
        f.write(markdown)
