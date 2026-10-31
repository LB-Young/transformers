import io, pathlib
base = pathlib.Path(r"E:\transformers\bayon-note\InternVL\internvl-1_5")
html_path = base / "InternVL-1_5-家族架构详解.html"
ec_path = base / "_tools" / "echarts.min.js"

html = html_path.read_text(encoding="utf-8")
ec = ec_path.read_text(encoding="utf-8")

old_tag = '<script src="https://fastly.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>'
assert old_tag in html, "script tag not found"
# 防止内联脚本里出现 </script> 字符串破坏 HTML
assert "</script>" not in ec.lower(), "echarts source contains </script>"
new_tag = "<script>/* echarts 5.5.0 inlined from fastly.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js */\n" + ec + "\n</script>"
html = html.replace(old_tag, new_tag)
html_path.write_text(html, encoding="utf-8")
print("inlined, new size bytes:", len(html.encode("utf-8")))
