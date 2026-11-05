/* EyeGlynt Live Monitoring — polls the Flask API and updates the page in place. */
(function () {
  "use strict";
  const root = document.getElementById("live");
  if (!root) return;

  const D = root.dataset;
  const CSRF = document.querySelector('meta[name="csrf-token"]').content;
  const POLL_MS = parseInt(D.poll, 10) || 2000;
  const $ = (id) => document.getElementById(id);

  const state = { afterSel: 0, afterAlert: 0, elapsed: parseInt(D.elapsed, 10) || 0,
                  failures: 0, ending: false, firstLoad: true };

  // ---------------------------------------------------------------- helpers
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }
  function icon(name) {
    const NS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(NS, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }
  function toast(msg) {
    const t = $("toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("show"), 2600);
  }
  async function post(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": CSRF },
      body: JSON.stringify(body || {}),
      credentials: "same-origin",
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.ok === false) throw new Error(data.error || "Request failed (" + res.status + ")");
    return data;
  }
  const withId = (tpl, id) => tpl.replace(/\/0\//, "/" + id + "/");
  function fmtClock(sec) {
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
    return h ? h + ":" + mm + ":" + ss : mm + ":" + ss;
  }

  // ---------------------------------------------------------------- timer
  setInterval(() => { state.elapsed += 1; $("timer").textContent = fmtClock(state.elapsed); }, 1000);
  $("timer").textContent = fmtClock(state.elapsed);

  // ---------------------------------------------------------------- activity log
  function renderSelection(s, isNew) {
    const li = el("li", (s.intent === "unintended" ? "unintended " : "") + (isNew ? "is-new" : ""));
    li.dataset.id = s.id;
    li.appendChild(el("span", "t", s.time));
    li.appendChild(el("span", "pill " + (s.interface === "quickfire" ? "qf" : "kb"),
                      s.interface === "quickfire" ? "Quick-Fire" : "Keyboard"));
    const txt = el("span", "txt", s.text);
    if (s.block_type === "accuracy" && D.mode === "test") {
      txt.appendChild(scoreButtons(s));
    } else if (s.target !== null) {
      txt.appendChild(document.createTextNode(" "));
      txt.appendChild(el("span", "tag pill " + (s.is_correct ? "ok" : "bad"),
                         s.is_correct ? "✓ target" : "✗ missed"));
    } else if (s.in_block && D.mode === "test") {
      txt.appendChild(document.createTextNode(" "));
      txt.appendChild(el("span", "tag pill warn", "free-look"));
    }
    li.appendChild(txt);

    const btn = el("button", "flag");
    btn.type = "button";
    btn.appendChild(icon("flag"));
    btn.appendChild(el("span", null, s.intent === "unintended" ? "Unintended" : ""));
    btn.title = s.intent === "unintended" ? "Marked unintended — tap to undo" : "Mark as unintended";
    btn.setAttribute("aria-label", btn.title);
    btn.addEventListener("click", () => toggleFlag(li, btn));
    li.appendChild(btn);
    return li;
  }

  // Accuracy test: the researcher says the target word out loud, then marks the selection.
  function scoreButtons(s) {
    const box = el("span", "score");
    const current = s.target === null ? null : (s.is_correct ? "correct" : "wrong");
    [["correct", "✓ Correct"], ["wrong", "✗ Wrong"]].forEach(([value, label]) => {
      const b = el("button", "score-btn " + value + (current === value ? " on" : ""), label);
      b.type = "button";
      b.setAttribute("aria-pressed", current === value ? "true" : "false");
      b.addEventListener("click", async () => {
        const turningOff = b.classList.contains("on");
        const result = turningOff ? "clear" : value;
        box.querySelectorAll("button").forEach((x) => (x.disabled = true));
        try {
          await post(withId(D.scoreUrl, s.id), { result: result });
          box.querySelectorAll("button").forEach((x) => {
            const on = !turningOff && x === b;
            x.classList.toggle("on", on);
            x.setAttribute("aria-pressed", on ? "true" : "false");
          });
          toast(result === "clear" ? "Score removed" : result === "correct" ? "Marked correct" : "Marked wrong");
          poll();
        } catch (e) { toast(e.message); }
        box.querySelectorAll("button").forEach((x) => (x.disabled = false));
      });
      box.appendChild(b);
    });
    return box;
  }

  async function toggleFlag(li, btn) {
    const nowUnintended = !li.classList.contains("unintended");
    btn.disabled = true;
    try {
      await post(withId(D.flagUrl, li.dataset.id), { intent: nowUnintended ? "unintended" : "intended" });
      li.classList.toggle("unintended", nowUnintended);
      btn.lastChild.textContent = nowUnintended ? "Unintended" : "";
      btn.title = nowUnintended ? "Marked unintended — tap to undo" : "Mark as unintended";
      btn.setAttribute("aria-label", btn.title);
      toast(nowUnintended ? "Marked as unintended" : "Flag removed");
      poll();
    } catch (e) { toast(e.message); }
    btn.disabled = false;
  }

  // ---------------------------------------------------------------- alerts
  const TONE = { recalibration_success: "ok", gaze_lock_restored: "ok",
                 recalibration_failed: "bad", gaze_lock_loss: "bad", calibration_required: "bad",
                 lighting_shift: "warn" };
  function renderAlert(a) {
    const li = el("li", TONE[a.type] || "");
    const left = el("div");
    left.appendChild(el("div", "a-title", a.label));
    const detail = a.message + (a.latency_ms !== null ? " · detected in " + a.latency_ms + " ms" : "");
    left.appendChild(el("div", "a-msg", detail));
    li.appendChild(left);
    const right = el("div");
    right.style.textAlign = "right";
    right.appendChild(el("div", "t", a.time));
    if (!a.acknowledged && (a.severity === "warning" || a.severity === "critical")) {
      const b = el("button", "btn btn-sm btn-ghost ack", "Acknowledge");
      b.type = "button";
      b.addEventListener("click", async () => {
        b.disabled = true;
        try { await post(withId(D.ackUrl, a.id)); b.replaceWith(el("div", "a-msg", "Seen")); }
        catch (e) { toast(e.message); b.disabled = false; }
      });
      right.appendChild(b);
    }
    li.appendChild(right);
    return li;
  }

  // ---------------------------------------------------------------- metrics
  function setTile(key, value, check, sub) {
    const tile = root.querySelector('[data-metric="' + key + '"]');
    if (!tile) return;
    const v = tile.querySelector(".v");
    v.textContent = "";
    if (value === null || value === undefined) v.textContent = "—";
    else if (typeof value === "object") { v.append(value.main); if (value.unit) v.append(el("small", null, value.unit)); }
    else v.textContent = value;
    tile.classList.remove("pass", "fail");
    if (check === "pass" || check === "fail") tile.classList.add(check);
    if (sub !== undefined) tile.querySelector(".sub").textContent = sub;
  }
  function renderMetrics(m, checks) {
    if (!root.querySelector(".live-tiles")) return;   // caregiver view has no metric tiles
    const pct = (x) => (x === null || x === undefined ? null : { main: Math.round(x), unit: "%" });
    setTile("total_selections", m.total_selections);
    setTile("gaze_accuracy_pct", pct(m.gaze_accuracy_pct), checks.gaze_accuracy_pct);
    setTile("avg_detection_latency_ms",
            m.avg_detection_latency_ms === null ? null : { main: Math.round(m.avg_detection_latency_ms), unit: "ms" },
            checks.avg_detection_latency_ms);
    setTile("recalibration_success_pct", pct(m.recalibration_success_pct), checks.recalibration_success_pct);
    setTile("false_positive_pct", pct(m.false_positive_pct), checks.false_positive_pct);
    const most = m.most_used_interface;
    const tile = root.querySelector('[data-metric="most_used_interface"]');
    tile.classList.toggle("kb", most === "keyboard");
    const count = most === "keyboard" ? m.keyboard_selections : m.quickfire_selections;
    setTile("most_used_interface", most ? (most === "keyboard" ? "Keyboard" : "Quick-Fire") : null, null,
            most ? count + "/" + m.total_selections + " selections" : " ");
  }

  // ---------------------------------------------------------------- device + test panel
  function renderDevice(dev) {
    const badge = $("live-badge");
    badge.classList.toggle("offline", !dev.online);
    badge.textContent = dev.online ? "Live" : "Device offline";
    const chip = $("device-chip");
    const label = { tracking: "tracking", calibrating: "calibrating", lost: "gaze lost", idle: "idle" }[dev.state] || "—";
    chip.textContent = dev.online ? "Device: " + label + (dev.fps ? " · " + Math.round(dev.fps) + " fps" : "") : "Device: offline";
    chip.className = "pill " + (!dev.online ? "bad" : dev.state === "tracking" ? "ok" : "warn");
  }
  function renderBlock(block) {
    if (!$("test-panel")) return;
    $("test-idle").hidden = !!block;
    $("test-running").hidden = !block;
    // Trials from a finished accuracy test can still be marked; remind the researcher.
    const unmarked = root.querySelectorAll("#activity .score:not(:has(.on))").length;
    $("test-unmarked").hidden = !!block || unmarked === 0;
    $("test-unmarked").textContent = unmarked + " accuracy trial(s) still need Correct or Wrong. Mark them in the activity log before ending the session.";
    if (!block) return;
    if (block.type === "accuracy") {
      const total = block.planned_trials || 30;
      $("test-name").textContent = "Accuracy trials";
      $("test-count").textContent = block.activations + " / " + total;
      $("test-bar").style.width = Math.min(100, (100 * block.activations) / total) + "%";
      $("test-detail").textContent = block.trials + " marked, " + block.correct
        + " correct. Say the target word, then mark each selection Correct or Wrong. The test stops after " + total + " selections.";
    } else {
      const total = block.planned_seconds || 120;
      const left = Math.max(0, total - block.elapsed_s);
      $("test-name").textContent = "Free-look observation";
      $("test-count").textContent = fmtClock(left) + " left";
      $("test-bar").style.width = Math.min(100, (100 * block.elapsed_s) / total) + "%";
      $("test-detail").textContent = block.activations + " unintended activation(s). Ask the user to look around without selecting.";
    }
  }

  // ---------------------------------------------------------------- polling
  let timer = null, inFlight = false;
  async function poll() {
    if (inFlight) return;
    inFlight = true;
    clearTimeout(timer);
    try {
      const url = D.updatesUrl + "?after_sel=" + state.afterSel + "&after_alert=" + state.afterAlert;
      const res = await fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" } });
      if (res.status === 401 || res.redirected) { location.reload(); return; }
      const data = await res.json();
      state.failures = 0;

      if (data.status === "ended") { location.href = data.summary_url; return; }
      state.elapsed = data.elapsed_s;

      const log = $("activity");
      data.selections.forEach((s) => {
        log.prepend(renderSelection(s, !state.firstLoad));
        state.afterSel = Math.max(state.afterSel, s.id);
      });
      $("activity-empty").hidden = log.children.length > 0;

      const alerts = $("alerts");
      data.alerts.forEach((a) => {
        alerts.prepend(renderAlert(a));
        state.afterAlert = Math.max(state.afterAlert, a.id);
        if (!state.firstLoad && a.severity !== "info") toast(a.label);
      });
      $("alerts-empty").hidden = alerts.children.length > 0;

      renderMetrics(data.metrics, data.checks);
      renderDevice(data.device);
      renderBlock(data.block);

      const recal = $("btn-recal");
      if (!recal.dataset.busy) {
        recal.disabled = data.recal_pending;
        recal.querySelector("span").textContent = data.recal_pending
          ? "Recalibration requested…" : "Trigger manual recalibration";
      }
      if (String(data.dwell_time_ms) !== $("dwell").value && document.activeElement !== $("dwell")) {
        $("dwell").value = String(data.dwell_time_ms);
      }
      state.firstLoad = false;
    } catch (e) {
      state.failures += 1;
      const badge = $("live-badge");
      badge.classList.add("offline");
      badge.textContent = "Reconnecting…";
    }
    inFlight = false;
    timer = setTimeout(poll, Math.min(POLL_MS * Math.pow(2, state.failures), 30000));
  }

  // ---------------------------------------------------------------- controls
  $("btn-recal").addEventListener("click", async (ev) => {
    const b = ev.currentTarget;
    b.disabled = true; b.dataset.busy = "1";
    try { await post(D.recalUrl); toast("Recalibration sent to the device"); }
    catch (e) { toast(e.message); }
    delete b.dataset.busy;
    poll();
  });

  const endBtn = $("btn-end");
  endBtn.addEventListener("click", async () => {
    if (!state.ending) {               // two-tap confirm, no blocking browser dialog
      state.ending = true;
      endBtn.querySelector("span").textContent = "Tap again to end session";
      endBtn.classList.add("btn-primary");
      setTimeout(() => {
        state.ending = false;
        endBtn.querySelector("span").textContent = "End session";
        endBtn.classList.remove("btn-primary");
      }, 4000);
      return;
    }
    endBtn.disabled = true;
    endBtn.querySelector("span").textContent = "Saving session…";
    try { const r = await post(D.endUrl); location.href = r.redirect; }
    catch (e) { toast(e.message); endBtn.disabled = false; }
  });

  $("dwell").addEventListener("change", async (ev) => {
    try { await post(D.dwellUrl, { dwell_time_ms: parseInt(ev.target.value, 10) });
          toast("Dwell time set to " + (ev.target.value / 1000).toFixed(1) + " s"); }
    catch (e) { toast(e.message); }
  });

  root.querySelectorAll("[data-block]").forEach((b) =>
    b.addEventListener("click", async () => {
      b.disabled = true;
      try { await post(D.blockUrl, { action: "start", type: b.dataset.block });
            toast(b.dataset.block === "accuracy" ? "Accuracy test started" : "Free-look test started"); }
      catch (e) { toast(e.message); }
      b.disabled = false;
      poll();
    })
  );
  if ($("test-stop")) $("test-stop").addEventListener("click", async () => {
    try { await post(D.blockUrl, { action: "end" }); toast("Test stopped"); } catch (e) { toast(e.message); }
    poll();
  });

  poll();
})();
