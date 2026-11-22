"""On-screen buttons: layout, hit-testing and the actions they trigger (no window)."""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neck_rom import app as A  # noqa: E402
from neck_rom import session as S  # noqa: E402
from neck_rom import ui as U  # noqa: E402
from neck_rom.i18n import T  # noqa: E402


def center(ui, bid):
    _, x0, y0, x1, y1 = next(b for b in ui.buttons if b[0] == bid)
    return (x0 + x1) // 2, (y0 + y1) // 2


def live(ui, flow, t=0.0):
    canvas = np.zeros((U.H, U.W, 3), np.uint8)
    ui.render_live(canvas, flow, S.Sample(t, False, reason="q_no_face"), t)
    return {b[0] for b in ui.buttons}


def test_buttons_per_screen():
    ui = U.UI(T("zh"))
    flow = S.Flow(S.AngleProcessor())
    assert live(ui, flow) == {"lang", "start", "history"}
    flow.start(0.0)
    assert live(ui, flow) == {"lang", "next"}            # calib_prep
    flow.phase = "measure"
    assert live(ui, flow) == {"lang", "next", "skip"}
    ui.render_result([])
    assert {b[0] for b in ui.buttons} == {"lang", "history", "home", "again"}
    ui.render_history([])
    assert ui.buttons == []


def test_buttons_inside_canvas_and_not_overlapping():
    ui = U.UI(T("en"))
    flow = S.Flow(S.AngleProcessor())
    for setup in (lambda: None, lambda: flow.start(0.0), lambda: setattr(flow, "phase", "measure")):
        setup()
        live(ui, flow)
        bs = ui.buttons
        for _, x0, y0, x1, y1 in bs:
            assert 0 <= x0 < x1 <= U.W and 0 <= y0 < y1 <= U.H - 44   # above the disclaimer
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                assert a[3] < b[1] or b[3] < a[1] or a[4] < b[2] or b[4] < a[2], (a, b)


def test_hit_and_miss():
    ui = U.UI(T("zh"))
    live(ui, S.Flow(S.AngleProcessor()))
    assert ui.hit(*center(ui, "start")) == "start"
    assert ui.hit(*center(ui, "lang")) == "lang"
    assert ui.hit(640, 360) is None


def press(bid, flow, ui, tr, now=1.0):
    return A.handle_key(A.BUTTON_KEYS[bid], flow, flow.proc, now, "live", None, False, ui,
                        tr, Namespace(user="__test_nobody__"))


def test_button_actions():
    tr = T("zh")
    ui = U.UI(tr)
    flow = S.Flow(S.AngleProcessor())
    press("lang", flow, ui, tr)
    assert tr.lang == "en"
    flow, *_ = press("start", flow, ui, tr)
    assert flow.phase == "calib_prep"
    flow.phase, flow.t_phase = "prep", 1.0
    flow, *_ = press("skip", flow, ui, tr)
    assert flow.results[-1].skipped and flow.idx == 1
    flow, screen, hist, _ = A.handle_key("h", flow, flow.proc, 1.0, "live", None, False, ui, tr,
                                         Namespace(user="__test_nobody__"))
    assert screen == "live"                       # history only from home / result
    flow.phase = "result"
    flow2, screen, hist, _ = press("history", flow, ui, tr)
    assert screen == "history" and hist is not None
    flow3, *_ = press("again", flow, ui, tr)
    assert flow3 is not flow and flow3.phase == "calib_prep"
    flow4, *_ = press("home", flow, ui, tr)
    assert flow4.phase == "home"
