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
  // Each selection has two buttons:
  //   Incorrect  (database: "unintended") - wrong selection: counts against
  //              accuracy / false positives
  //   Unintended (database: "ignored")    - wasn't meant to be selected:
  //              left out of every metric
  // Only the button labels changed on 8 Nov; the database values stayed.
  function renderSelection(s, isNew) {
    const li = el("li", (isNew ? "is-new" : ""));
    li.dataset.id = s.id;
    li.appendChild(el("span", "t", s.time));
    li.appendChild(el("span", "pill " + (s.interface === "quickfire" ? "qf" : "kb"),
                      s.interface === "quickfire" ? "Quick-Fire" : "Keyboard"));
    const txt = el("span", "txt");
    if (s.type === "speak") {           // a sentence typed on the keyboard and spoken
      li.classList.add("spoken");
      txt.appendChild(el("span", "spoke pill ok", "Spoken"));
    }
    txt.appendChild(document.createTextNode(s.text));
    if (s.target !== null) {            // old Test Mode sessions only
      txt.appendChild(document.createTextNode(" "));
      txt.appendChild(el("span", "tag pill " + (s.is_correct ? "ok" : "bad"),
                         s.is_correct ? "✓ target" : "✗ missed"));
    }
    li.appendChild(txt);

    const marks = el("span", "marks");
    const flag = el("button", "flag unintended-btn");
    flag.type = "button";
    // Only the icons show; the word appears once the button is on.
    flag.appendChild(icon("flag"));
    flag.appendChild(el("span", "lbl", "Incorrect"));
    const ignore = el("button", "flag ignore");
    ignore.type = "button";
    ignore.appendChild(icon("ignore"));
    ignore.appendChild(el("span", "lbl", "Unintended"));
    flag.addEventListener("click", () => setIntent(li, s.id, "unintended"));
    ignore.addEventListener("click", () => setIntent(li, s.id, "ignored"));
    marks.append(flag, ignore);
    li.appendChild(marks);
    showIntent(li, s.intent);
    return li;
  }

  function showIntent(li, intent) {
    li.dataset.intent = intent;
    li.classList.toggle("unintended", intent === "unintended");
    li.classList.toggle("ignored", intent === "ignored");
    const flag = li.querySelector(".unintended-btn"), ignore = li.querySelector(".ignore");
    flag.title = intent === "unintended" ? "Marked incorrect. Tap to undo" : "Mark as incorrect";
    ignore.title = intent === "ignored" ? "Marked unintended. Tap to undo"
                                        : "Mark as unintended: left out of the results";
    flag.setAttribute("aria-pressed", intent === "unintended" ? "true" : "false");
    ignore.setAttribute("aria-pressed", intent === "ignored" ? "true" : "false");
    flag.setAttribute("aria-label", flag.title);
    ignore.setAttribute("aria-label", ignore.title);
  }

  // Tapping the button that is already on turns it off again.
  async function setIntent(li, id, wanted) {
    const intent = li.dataset.intent === wanted ? "intended" : wanted;
    const buttons = li.querySelectorAll(".marks button");
    buttons.forEach((b) => (b.disabled = true));
    try {
      await post(withId(D.flagUrl, id), { intent: intent });
      showIntent(li, intent);
      toast({ unintended: "Marked as incorrect", ignored: "Marked as unintended",
              intended: "Mark removed" }[intent]);
      poll();
    } catch (e) { toast(e.message); }
    buttons.forEach((b) => (b.disabled = false));
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

  // ---------------------------------------------------------------- device
  function renderDevice(dev) {
    const badge = $("live-badge");
    badge.classList.toggle("offline", !dev.online);
    badge.textContent = dev.online ? "Live" : "Device offline";
    const chip = $("device-chip");
    const label = { tracking: "tracking", calibrating: "calibrating", lost: "gaze lost", idle: "idle",
                    closed: "app closed" }[dev.state] || "—";
    chip.textContent = dev.online ? "Device: " + label + (dev.fps ? " · " + Math.round(dev.fps) + " fps" : "") : "Device: offline";
    chip.className = "pill " + (!dev.online ? "bad" : dev.state === "tracking" ? "ok" : "warn");
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

      // While the device is calibrating (or about to), offer Stop calibration.
      const calibrating = data.recal_pending || (data.device.online && data.device.state === "calibrating");
      const recal = $("btn-recal"), stop = $("btn-stop-recal");
      if (!recal.dataset.busy) {
        recal.disabled = calibrating;
        recal.querySelector("span").textContent = data.recal_pending
          ? "Recalibration requested…" : calibrating ? "Calibrating…" : "Trigger manual recalibration";
      }
      if (!stop.dataset.busy) stop.hidden = !calibrating;
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

  $("btn-stop-recal").addEventListener("click", async (ev) => {
    const b = ev.currentTarget;
    b.disabled = true; b.dataset.busy = "1";
    try { await post(D.stopRecalUrl); toast("Stopping calibration on the device"); }
    catch (e) { toast(e.message); }
    b.disabled = false; delete b.dataset.busy;
    poll();
  });

  poll();
})();
