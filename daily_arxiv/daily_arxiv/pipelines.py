# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import arxiv
import json
import os
import sys
from datetime import datetime, timedelta


class DailyArxivPipeline:
    def __init__(self):
        self.page_size = 100
        self.client = arxiv.Client(
            page_size=self.page_size,
            delay_seconds=float(os.environ.get("ARXIV_API_DELAY_SECONDS", "3")),
            num_retries=int(os.environ.get("ARXIV_API_NUM_RETRIES", "8")),
        )

    def process_item(self, item: dict, spider):
        item["pdf"] = f"https://arxiv.org/pdf/{item['id']}"
        item["abs"] = f"https://arxiv.org/abs/{item['id']}"
        search = arxiv.Search(
            id_list=[item["id"]],
        )
        try:
            paper = next(self.client.results(search))
            item["authors"] = [a.name for a in paper.authors]
            item["title"] = paper.title
            item["categories"] = paper.categories
            item["comment"] = paper.comment
            item["summary"] = paper.summary
        except Exception as exc:
            spider.logger.error("Failed to fetch arXiv metadata for %s: %s", item["id"], exc)
            item.setdefault("authors", [])
            item.setdefault("title", item["id"])
            item.setdefault("categories", item.get("categories", []))
            item.setdefault("comment", None)
            item.setdefault(
                "summary",
                "arXiv metadata fetch failed after retries. Original paper links and ID are preserved.",
            )
            item["metadata_status"] = "fallback"
            item["metadata_error"] = str(exc)[:500]
        return item
