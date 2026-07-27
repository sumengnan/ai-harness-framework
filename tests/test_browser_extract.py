from harness.browser.extract import extract_main_text, extract_title_and_text

ARTICLE_HTML = """
<html><head><title>光合作用</title></head><body>
<nav>首页 关于 联系我们 登录 注册</nav>
<article>
<h1>光合作用的原理</h1>
<p>光合作用是绿色植物利用光能，把二氧化碳和水转化为储存能量的有机物，并释放氧气的过程。
这一过程主要发生在叶绿体中，是地球上几乎所有生命能量的最终来源，对维持大气中氧气与二氧化碳的平衡至关重要。</p>
<p>光合作用分为光反应和暗反应两个阶段。光反应在类囊体膜上进行，把光能转化为化学能并释放氧气；
暗反应在基质中进行，利用这些化学能固定二氧化碳，最终合成葡萄糖等有机物。</p>
</article>
<footer>版权所有 © 2026 保留所有权利 隐私政策 网站地图</footer>
</body></html>
"""


EN_ARTICLE_HTML = """
<html><head><title>Photosynthesis</title></head><body>
<nav>Home About Contact Login Sign up</nav>
<article>
<h1>How Photosynthesis Works</h1>
<p>Photosynthesis is the process by which green plants use light energy to convert
carbon dioxide and water into energy-rich organic compounds, releasing oxygen. It
takes place mainly in the chloroplasts and is the ultimate source of energy for
almost all life on Earth.</p>
<p>It proceeds in two stages. The light reactions on the thylakoid membranes turn
light energy into chemical energy and release oxygen, while the dark reactions in
the stroma fix carbon dioxide to build sugars such as glucose.</p>
</article>
<footer>Copyright © 2026 All rights reserved Privacy Policy Sitemap</footer>
</body></html>
"""


def test_extract_returns_main_text():
    text = extract_main_text(ARTICLE_HTML)
    assert "光合作用是绿色植物" in text
    assert "光反应和暗反应" in text


def test_extract_returns_english_main_text():
    # 我们只抽中英文正文；英文路径同样要去样板、留正文（也守住剥离 babel 非中英 locale 后不炸）
    title, text = extract_title_and_text(EN_ARTICLE_HTML)
    assert "Photosynthesis" in title
    assert "green plants use light energy" in text
    assert "light reactions" in text
    assert "Login Sign up" not in text
    assert "All rights reserved" not in text


def test_extract_strips_boilerplate():
    text = extract_main_text(ARTICLE_HTML)
    assert "登录 注册" not in text
    assert "版权所有" not in text


def test_extract_empty_html():
    assert extract_main_text("") == ""


def test_extract_title_and_text_returns_both():
    title, text = extract_title_and_text(ARTICLE_HTML)
    assert "光合作用" in title           # trafilatura 可能取 <title> 或 <h1>，均含此词
    assert "光合作用是绿色植物" in text
    assert "登录 注册" not in text


def test_extract_title_falls_back_to_title_tag():
    # 无正文可抽，但 <title> 仍能取到（转义实体被还原）
    html = "<html><head><title>Tom &amp; Jerry</title></head><body></body></html>"
    title, text = extract_title_and_text(html)
    assert title == "Tom & Jerry"


def test_extract_title_and_text_empty():
    assert extract_title_and_text("") == ("", "")


def test_extract_no_title_returns_empty_title():
    title, _ = extract_title_and_text("<html><body><p>no title here</p></body></html>")
    assert title == ""
