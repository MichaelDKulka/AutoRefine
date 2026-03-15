"""Embeddable HTML/JS feedback widget for web applications.

Generates a fully self-contained HTML snippet (inline CSS + JS, zero
external dependencies) that developers paste into their pages.  The
widget POSTs feedback to the AutoRefine dashboard API or any custom URL.

Three style presets:

- ``"minimal"`` — two small icon buttons (thumbs up/down)
- ``"standard"`` — buttons with labels and a collapsible comment box
- ``"detailed"`` — buttons, comment box, and category tags

Usage::

    from autorefine.widget import FeedbackWidget

    widget = FeedbackWidget(endpoint="http://localhost:8787")
    html = widget.render("interaction-id-here", style="standard")
    # Inject `html` into your web page template
"""

from __future__ import annotations

import html as _html

# ═══════════════════════════════════════════════════════════════════════
# CSS shared across all styles
# ═══════════════════════════════════════════════════════════════════════

_BASE_CSS = """\
.ar-widget{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
font-size:14px;display:inline-flex;flex-direction:column;gap:6px}
.ar-row{display:flex;gap:6px;align-items:center}
.ar-btn{border:1px solid #d0d7de;border-radius:6px;background:#fff;cursor:pointer;
transition:all .15s;display:inline-flex;align-items:center;gap:4px;color:#656d76}
.ar-btn:hover{border-color:#0969da;color:#0969da}
.ar-btn.ar-active-up{border-color:#1a7f37;color:#1a7f37;background:#dafbe1}
.ar-btn.ar-active-down{border-color:#cf222e;color:#cf222e;background:#ffebe9}
.ar-btn.ar-sent{opacity:.6;cursor:default}
.ar-comment{border:1px solid #d0d7de;border-radius:6px;padding:6px 8px;font-size:13px;
width:100%;resize:vertical;min-height:32px;font-family:inherit;display:none}
.ar-comment.ar-show{display:block}
.ar-tags{display:flex;flex-wrap:wrap;gap:4px;display:none}
.ar-tags.ar-show{display:flex}
.ar-tag{border:1px solid #d0d7de;border-radius:12px;padding:2px 10px;font-size:11px;
cursor:pointer;background:#fff;color:#656d76;transition:all .15s}
.ar-tag.ar-tag-on{border-color:#0969da;background:#ddf4ff;color:#0969da}
.ar-submit{border:none;border-radius:6px;background:#0969da;color:#fff;padding:4px 12px;
font-size:12px;cursor:pointer;display:none}
.ar-submit.ar-show{display:inline-block}
.ar-submit:hover{background:#0860ca}
.ar-done{font-size:12px;color:#1a7f37;display:none}
.ar-done.ar-show{display:inline}
"""

# ═══════════════════════════════════════════════════════════════════════
# SVG icons (inline, no external deps)
# ═══════════════════════════════════════════════════════════════════════

_THUMB_UP = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8.834.066c.763.087 1.5.295 2.01.884.505.581.656 1.378.656 2.3 0 .467-.087 1.119-.157 1.637H13.5c.7 0 1.252.245 1.607.673.34.408.443.93.393 1.413-.043.44-.236.9-.55 1.25.034.165.04.354.01.559-.063.42-.267.836-.61 1.15a1.67 1.67 0 0 1-.258 1.076 1.67 1.67 0 0 1-.63.58c.012.168-.01.354-.065.54-.18.6-.7 1.072-1.397 1.072H8.5c-.246 0-.498-.048-.727-.1l-.09-.022a6 6 0 0 1-.702-.204l-.295-.107a4 4 0 0 0-.58-.168c-.201-.042-.48-.074-.748-.074H4V5.89a4.5 4.5 0 0 0 1.252-1.15c.32-.412.663-.95.937-1.612.14-.337.269-.722.374-1.128h.001c.084-.32.153-.653.2-.953a.75.75 0 0 1 .542-.627c.56-.157 1.047-.157 1.528-.07zM3.5 6.5V13H2a.5.5 0 0 1-.5-.5V7a.5.5 0 0 1 .5-.5z"/></svg>'
_THUMB_DOWN = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M7.166 15.934c-.763-.087-1.5-.295-2.01-.884-.505-.581-.656-1.378-.656-2.3 0-.467.087-1.119.157-1.637H2.5c-.7 0-1.252-.245-1.607-.673-.34-.408-.443-.93-.393-1.413.043-.44.236-.9.55-1.25a1.5 1.5 0 0 1-.01-.559c.063-.42.267-.836.61-1.15a1.67 1.67 0 0 1 .258-1.076 1.67 1.67 0 0 1 .63-.58 1.3 1.3 0 0 1 .065-.54c.18-.6.7-1.072 1.397-1.072H7.5c.246 0 .498.048.727.1l.09.022c.236.06.478.136.702.204l.295.107c.222.08.39.132.58.168.201.042.48.074.748.074H12v7.11a4.5 4.5 0 0 1-1.252 1.15c-.32.412-.663.95-.937 1.612-.14.337-.269.722-.374 1.128h-.001a9 9 0 0 1-.2.953.75.75 0 0 1-.542.627c-.56.157-1.047.157-1.528.07zM12.5 9.5V3H14a.5.5 0 0 1 .5.5V9a.5.5 0 0 1-.5.5z"/></svg>'

# ═══════════════════════════════════════════════════════════════════════
# JS (shared logic, parameterized per instance)
# ═══════════════════════════════════════════════════════════════════════

_JS_TEMPLATE = """\
(function(){{
  var W=document.getElementById('{wid}');
  var sent=false;
  var signal='',comment='',tags=[];

  W.querySelectorAll('.ar-btn').forEach(function(b){{
    b.addEventListener('click',function(){{
      if(sent)return;
      signal=b.dataset.signal;
      W.querySelectorAll('.ar-btn').forEach(function(x){{x.classList.remove('ar-active-up','ar-active-down')}});
      b.classList.add(signal==='thumbs_up'?'ar-active-up':'ar-active-down');
      var cm=W.querySelector('.ar-comment');if(cm)cm.classList.add('ar-show');
      var tg=W.querySelector('.ar-tags');if(tg)tg.classList.add('ar-show');
      var sb=W.querySelector('.ar-submit');if(sb)sb.classList.add('ar-show');
      {auto_send}
    }});
  }});

  W.querySelectorAll('.ar-tag').forEach(function(t){{
    t.addEventListener('click',function(){{
      t.classList.toggle('ar-tag-on');
      var idx=tags.indexOf(t.textContent);
      if(idx>=0)tags.splice(idx,1);else tags.push(t.textContent);
    }});
  }});

  var sb=W.querySelector('.ar-submit');
  if(sb)sb.addEventListener('click',function(){{doSend()}});

  function doSend(){{
    if(sent||!signal)return;
    sent=true;
    var cm=W.querySelector('.ar-comment');
    comment=(cm?cm.value:'')+(tags.length?' ['+tags.join(', ')+']':'');
    W.querySelectorAll('.ar-btn').forEach(function(b){{b.classList.add('ar-sent')}});
    var dn=W.querySelector('.ar-done');if(dn)dn.classList.add('ar-show');
    if(sb)sb.style.display='none';

    fetch('{endpoint}/api/widget/feedback',{{
      method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{interaction_id:'{iid}',signal:signal,comment:comment}})
    }}).catch(function(e){{console.warn('AutoRefine widget: feedback POST failed',e)}});
  }}
}})();
"""

# ═══════════════════════════════════════════════════════════════════════
# Category tags for detailed style
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_TAGS = ["Wrong tone", "Inaccurate", "Too long", "Too short", "Off topic", "Unhelpful"]


class FeedbackWidget:
    """Generates embeddable HTML/JS feedback widgets.

    Args:
        endpoint: Base URL for the feedback POST (e.g.
            ``"http://localhost:8787"``).  The widget POSTs to
            ``{endpoint}/api/widget/feedback``.
        tags: Category tags for the ``"detailed"`` style preset.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8787",
        tags: list[str] | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._tags = tags or list(_DEFAULT_TAGS)

    def render(
        self,
        interaction_id: str,
        style: str = "minimal",
    ) -> str:
        """Return a self-contained HTML snippet for the feedback widget.

        Args:
            interaction_id: The ``response.id`` from an AutoRefine call.
            style: ``"minimal"``, ``"standard"``, or ``"detailed"``.

        Returns:
            A string of HTML with inline ``<style>`` and ``<script>``
            that can be pasted into any web page.
        """
        iid = _html.escape(interaction_id, quote=True)
        wid = f"ar-{iid[:12]}"

        # Build button sizes based on style
        if style == "minimal":
            btn_size = "padding:4px 8px;font-size:12px"
            btn_up = f'<button class="ar-btn" data-signal="thumbs_up" style="{btn_size}">{_THUMB_UP}</button>'
            btn_down = f'<button class="ar-btn" data-signal="thumbs_down" style="{btn_size}">{_THUMB_DOWN}</button>'
            body = f'<div class="ar-row">{btn_up}{btn_down}<span class="ar-done">Thanks!</span></div>'
            auto_send = "doSend();"
        elif style == "standard":
            btn_size = "padding:6px 12px;font-size:13px"
            btn_up = f'<button class="ar-btn" data-signal="thumbs_up" style="{btn_size}">{_THUMB_UP} Helpful</button>'
            btn_down = f'<button class="ar-btn" data-signal="thumbs_down" style="{btn_size}">{_THUMB_DOWN} Not helpful</button>'
            body = (
                f'<div class="ar-row">{btn_up}{btn_down}<span class="ar-done">Feedback sent!</span></div>'
                f'<textarea class="ar-comment" placeholder="What could be better? (optional)"></textarea>'
                f'<div class="ar-row"><button class="ar-submit">Send feedback</button></div>'
            )
            auto_send = ""
        elif style == "detailed":
            btn_size = "padding:6px 12px;font-size:13px"
            btn_up = f'<button class="ar-btn" data-signal="thumbs_up" style="{btn_size}">{_THUMB_UP} Helpful</button>'
            btn_down = f'<button class="ar-btn" data-signal="thumbs_down" style="{btn_size}">{_THUMB_DOWN} Not helpful</button>'
            tags_html = "".join(f'<span class="ar-tag">{_html.escape(t)}</span>' for t in self._tags)
            body = (
                f'<div class="ar-row">{btn_up}{btn_down}<span class="ar-done">Feedback sent!</span></div>'
                f'<div class="ar-tags">{tags_html}</div>'
                f'<textarea class="ar-comment" placeholder="Additional comments (optional)"></textarea>'
                f'<div class="ar-row"><button class="ar-submit">Send feedback</button></div>'
            )
            auto_send = ""
        else:
            raise ValueError(f"Unknown widget style '{style}'. Use: minimal, standard, detailed")

        js = _JS_TEMPLATE.format(
            wid=wid, iid=iid, endpoint=self._endpoint, auto_send=auto_send,
        )

        return (
            f'<div id="{wid}" class="ar-widget">'
            f'<style>{_BASE_CSS}</style>'
            f'{body}'
            f'<script>{js}</script>'
            f'</div>'
        )
