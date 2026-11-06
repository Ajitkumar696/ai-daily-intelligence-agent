
import requests
from datetime import datetime, timezone

from .collection_result import CollectionResult


API_URL = "https://huggingface.co/api/daily_papers"


def collect_huggingface_papers(max_articles=None, api_url=API_URL):
    """
    Collect daily research papers from Hugging Face
    and convert them into our common article structure.
    """

    print("\nCollecting from Hugging Face Papers...")

    try:
        response = requests.get(
            api_url,
            params={
                "limit": (
                    max_articles if max_articles is not None else 20
                )
            },
            timeout=(10, 30),
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        papers = response.json()

        if not isinstance(papers, list):
            raise ValueError("API response must be a list of papers.")

        print("Papers received:", len(papers))

        articles = []
        for item in papers:
            if not isinstance(item, dict):
                raise ValueError("API response contains an invalid paper.")

            paper = item.get("paper", {})
            if not isinstance(paper, dict):
                raise ValueError("Paper details must be an object.")

            paper_id = paper.get("id", "")

            title = item.get("title") or paper.get(
                "title", ""
            )

            summary = item.get("summary") or paper.get(
                "summary", ""
            )

            published = item.get("publishedAt") or paper.get(
                "publishedAt", ""
            )

            if not paper_id or not title:
                raise ValueError(
                    "Paper entry is missing its required id or title."
                )

            paper_url = f"https://huggingface.co/papers/{paper_id}"

            authors = paper.get("authors", [])
            if not isinstance(authors, list):
                raise ValueError("Paper authors must be a list.")

            author_names = [
                author.get("name", "Unknown")
                for author in authors
                if isinstance(author, dict)
            ]
            if len(author_names) != len(authors):
                raise ValueError("Paper author entries must be objects.")

            article = {
                "source": "Hugging Face Papers",
                "publisher": "Hugging Face Papers",
                "source_type": "paper",
                "title": title.strip(),
                "url": paper_url,
                "published": published,
                "summary": summary.strip(),
                "authors": author_names,
                "metadata": {
                    "paper_id": paper_id,
                    "upvotes": item.get("upvotes"),
                    "github_repo": item.get("githubRepo"),
                    "submitted_at": item.get("submittedOnDailyAt")
                },
                "collected_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            articles.append(article)

        result = CollectionResult.from_articles(
            "Hugging Face Papers",
            articles
        )

    except ValueError as error:
        result = CollectionResult.failed(
            "Hugging Face Papers",
            f"Invalid API data: {error}"
        )

    except requests.exceptions.RequestException as error:
        result = CollectionResult.failed(
            "Hugging Face Papers",
            f"Network or HTTP request failed: {error}"
        )

    except Exception as error:
        result = CollectionResult.failed("Hugging Face Papers", str(error))

    if result.status.value == "FAILED":
        print(f"[ERROR] Hugging Face Papers: {result.error}")
    else:
        print(
            f"[{result.status.value}] Hugging Face Papers: "
            f"collected {len(result.articles)} papers"
        )

    return result
