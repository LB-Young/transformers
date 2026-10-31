import pathlib, re
p = pathlib.Path(r"E:\transformers\bayon-note\InternVL\internvl-1_5\InternVL-1_5-家族架构详解.html")
s = p.read_text(encoding="utf-8")

# 1) 替换内联 echarts 块为 CDN script
# 内联块形如: <script>/* echarts 5.5.0 inlined ... */\n...(1MB js)...\n</script>
pat = re.compile(r"<script>/\* echarts 5\.5\.0 inlined.*?</script>", re.DOTALL)
m = pat.search(s)
assert m, "inline echarts block not found"
cdn_tag = '<script src="https://fastly.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>'
s = s[:m.start()] + cdn_tag + s[m.end():]

# 2) 在 #chart 容器下加降级占位（hidden，失败时显示）
# 已有 <div id="chart"></div>，在其后追加 fallback
old_chart = '<div id="chart"></div>'
assert old_chart in s
new_chart = '<div id="chart"></div>\n  <div id="chart-fallback" style="display:none; padding:18px; background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); color:var(--sub); font-size:.86rem;">图表脚本未加载（离线或 CDN 不可达）。数据：Mini-2B = 视觉塔 0.3B + LLM 1.8B；Mini-4B = 0.3B + 3.8B；V1.5 = 6.0B + 20.0B（总参 2B/4B/26B）。</div>'
s = s.replace(old_chart, new_chart)

# 3) 替换底部 IIFE：加 typeof echarts 守卫
old_iife = """(function(){
  var chart = echarts.init(document.getElementById('chart'));
  chart.setOption({"""
new_iife = """(function(){
  var dom = document.getElementById('chart');
  if (typeof echarts === 'undefined' || !dom) {
    var fb = document.getElementById('chart-fallback');
    if (fb) { fb.style.display = 'block'; dom.style.display = 'none'; }
    return;
  }
  var chart = echarts.init(dom);
  chart.setOption({"""
assert old_iife in s, "IIFE not found"
s = s.replace(old_iife, new_iife)

p.write_text(s, encoding="utf-8")
print("OK, new size bytes:", len(s.encode("utf-8")))
