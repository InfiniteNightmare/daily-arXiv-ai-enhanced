import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
AI_DIR = ROOT_DIR / "ai"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

import enhance  # noqa: E402


class TestNormalizeRepoName(unittest.TestCase):
    def test_keeps_repo_names_ending_in_git_letters(self):
        # rstrip(".git") treats its argument as a character set and used to
        # eat trailing g/i/t letters: "coding" -> "codin", "vit" -> "".
        self.assertEqual(enhance.normalize_repo_name("coding"), "coding")
        self.assertEqual(enhance.normalize_repo_name("vit"), "vit")
        self.assertEqual(enhance.normalize_repo_name("sparse-attention"), "sparse-attention")
        self.assertEqual(enhance.normalize_repo_name("smartgit"), "smartgit")

    def test_strips_git_suffix(self):
        self.assertEqual(
            enhance.normalize_repo_name("daily-arXiv-ai-enhanced.git"),
            "daily-arXiv-ai-enhanced",
        )
        self.assertEqual(
            enhance.normalize_repo_name("daily-arXiv-ai-enhanced.git."),
            "daily-arXiv-ai-enhanced",
        )

    def test_strips_trailing_url_punctuation(self):
        self.assertEqual(enhance.normalize_repo_name("repo."), "repo")
        self.assertEqual(enhance.normalize_repo_name("repo,)"), "repo")

    def test_plain_repo_unchanged(self):
        self.assertEqual(
            enhance.normalize_repo_name("daily-arXiv-ai-enhanced"),
            "daily-arXiv-ai-enhanced",
        )


if __name__ == "__main__":
    unittest.main()
