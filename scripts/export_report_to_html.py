import markdown
import os

# Paths
source_file = r'research_diary/2026-01-28_progress_report.md'
output_file = r'research_diary/2026-01-28_progress_report.html'

if not os.path.exists(source_file):
    print(f"Error: File not found: {source_file}")
    exit(1)

# Read Markdown
with open(source_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to HTML
# Including 'extra' extension for tables support
html_content = markdown.markdown(text, extensions=['extra', 'codehilite'])

# Add minimal CSS for better viewing
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Progress Report - Campeones Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            max_width: 900px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        img {{
            max_width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1rem;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f8f8f8;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
        }}
        h1, h2, h3 {{ 
            color: #2c3e50; 
            margin-top: 1.5rem;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

# Write HTML
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_template)

print(f"Successfully converted report to: {os.path.abspath(output_file)}")
