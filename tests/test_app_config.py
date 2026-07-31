import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


class AppConfigTests(unittest.TestCase):
    def test_loads_groq_api_key_from_dotenv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("GROQ_API_KEY=test-key\n", encoding="utf-8")

            old_cwd = os.getcwd()
            old_env = os.environ.get("GROQ_API_KEY")
            os.chdir(tmpdir)
            os.environ.pop("GROQ_API_KEY", None)
            sys.modules.pop("app", None)

            try:
                import app
                importlib.reload(app)
                self.assertEqual(app.GROQ_API_KEY, "test-key")
            finally:
                if old_env is None:
                    os.environ.pop("GROQ_API_KEY", None)
                else:
                    os.environ["GROQ_API_KEY"] = old_env
                os.chdir(old_cwd)
                sys.modules.pop("app", None)


if __name__ == "__main__":
    unittest.main()
