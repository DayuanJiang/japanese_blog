# Quarto writes ".../index.html" URLs into sitemap.xml, while canonical-url: true
# gives each page the folder URL ".../" (quarto-cli issue #11365). Search Console
# then counts the sitemap URLs as alternates and reports the posts as not indexed.
# List the canonical folder URLs instead.
import os

path = os.path.join(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"), "sitemap.xml")
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.replace("/index.html</loc>", "/</loc>"))
