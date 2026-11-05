#!/usr/bin/env python3
"""
EyeGlynt device simulator — a stand-in for the real gaze engine.

Writes realistic selections, alerts, recalibrations and heartbeats into MariaDB
through engine_bridge, and obeys dashboard commands (Recalibrate, test blocks,
dwell changes). Use it to demo or test the dashboard before the hardware is ready.

    python tools/simulator.py                 # normal pace
    python tools/simulator.py --speed 3       # 3x faster
    python tools/simulator.py --accuracy 0.9 --recal-success 0.88
"""
import argparse
import os
import random
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine_bridge.eyeglynt_db import EyeGlyntDB  # noqa: E402

QUICKFIRE = {
    "Emergency & Urgent": ["I need help", "Call the nurse", "I can't breathe well"],
    "Pain & Physical Discomfort": ["I am in pain", "My head hurts", "Please reposition me"],
    "Basic Needs": ["I want water", "I am hungry", "I need the bathroom"],
    "Social & Emotional": ["Thank you", "I love you", "I feel tired"],
    "People & Family": ["Where is my family?", "Call my daughter"],
    "Environment & Comfort": ["It's too cold", "Turn off the light", "Open the window"],
}
SHORTHAND = ["Yes", "No", "Thank you", "Please", "Wait", "Later"]
SENTENCES = ["I want to sleep", "Watch TV please", "Where is mama", "I feel better today"]
KEYS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def slug(text):
    return "".join(c for c in text.lower() if c.isalnum())[:24]


class Simulator:
    def __init__(self, args):
        self.a = args
        self.db = EyeGlyntDB()
        self.running = True
        self.state = "tracking"
        self.recal = None          # (recal_id, finish_at)
        self.next_sel = 0.0
        self.next_alert = time.time() + self._j(25)
        self.last_beat = 0.0
        self.last_poll = 0.0
        self.lock_lost_until = None
        self._pending_retry = False

    def _j(self, seconds):
        """Jittered interval, scaled by --speed."""
        return random.uniform(0.6, 1.4) * seconds / self.a.speed

    def log(self, msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    # ------------------------------------------------------------ commands
    def handle_commands(self):
        for cmd in self.db.poll_commands():
            c, p = cmd["command"], cmd["payload"]
            self.log(f"command: {c} {p or ''}")
            if c == "recalibrate":
                self.begin_recal("manual_caregiver")
            elif c == "start_session":
                self.next_sel = time.time() + self._j(3)
            self.db.complete_command(cmd["command_id"])

    def begin_recal(self, source, alert_id=None):
        if self.recal:
            return
        rid = self.db.start_recalibration(source, alert_id)
        if rid:
            self.state = "calibrating"
            self.recal = (rid, time.time() + self._j(4))

    def tick_recal(self):
        if self.recal and time.time() >= self.recal[1]:
            ok = random.random() < self.a.recal_success
            self.db.finish_recalibration(self.recal[0], ok)
            self.log(f"recalibration {'succeeded' if ok else 'FAILED'}")
            self.state = "tracking" if ok else "lost"
            if not ok:  # retry shortly, like the real engine would
                self.recal = None
                self.next_alert = time.time() + self._j(3)
                self._pending_retry = True
            else:
                self.recal = None

    # ------------------------------------------------------------ events
    def maybe_alert(self, session):
        now = time.time()
        if self._pending_retry and now >= self.next_alert:
            self._pending_retry = False
            self.begin_recal("auto_drift")
            return
        if self.lock_lost_until and now >= self.lock_lost_until:
            self.lock_lost_until = None
            self.db.raise_alert("gaze_lock_restored", "User back in view", severity="info")
            self.state = "tracking"
        if now < self.next_alert or self.recal or self.lock_lost_until:
            return
        self.next_alert = now + self._j(45)
        latency = random.uniform(0.28, 0.52)          # seconds between event and detection
        event_at = now - latency
        if random.random() < 0.55:
            lux = round(random.uniform(120, 420), 1)
            aid = self.db.raise_alert("lighting_shift", f"Ambient light changed to {lux} lux — recalibration triggered",
                                      event_at=event_at, detected_at=now, lux=lux)
            self.log(f"alert: lighting shift ({latency*1000:.0f} ms)")
            self.begin_recal("auto_lighting", aid)
        else:
            self.db.raise_alert("gaze_lock_loss", "Eyes not detected — user looked away or closed eyes",
                                event_at=event_at, detected_at=now)
            self.log(f"alert: gaze lock lost ({latency*1000:.0f} ms)")
            self.state = "lost"
            self.lock_lost_until = now + self._j(5)

    def maybe_select(self, session):
        now = time.time()
        if now < self.next_sel or self.state != "tracking":
            return
        block = self.db.active_test_block(session["session_id"]) if session["mode"] == "test" else None
        dwell = session["dwell_time_ms"] + random.randint(0, 250)

        if block and block["block_type"] == "accuracy":
            target = random.choice(KEYS)
            hit = random.random() < self.a.accuracy
            chosen = target if hit else random.choice([k for k in KEYS if k != target])
            # Like the real device, the simulator doesn't know the target: the researcher
            # says it out loud and marks each selection Correct or Wrong on the dashboard.
            self.db.log_selection("keyboard", "key", f"kb:{chosen}", chosen, dwell_ms=dwell)
            self.log(f"accuracy trial: target {target}, selected {chosen} (mark it on the dashboard)")
            self.next_sel = now + self._j(2.5)
            return
        if block and block["block_type"] == "free_look":
            if random.random() < 0.35:                     # occasional Midas-touch activation
                cat, items = random.choice(list(QUICKFIRE.items()))
                text = random.choice(items)
                self.db.log_selection("quickfire", "phrase", f"qf:{slug(cat)}:{slug(text)}", text,
                                      category=cat, dwell_ms=dwell)
                self.log("free-look activation")
            self.next_sel = now + self._j(12)
            return

        r = random.random()
        if r < 0.6:
            cat, items = random.choice(list(QUICKFIRE.items()))
            text = random.choice(items)
            self.db.log_selection("quickfire", "phrase", f"qf:{slug(cat)}:{slug(text)}", text,
                                  category=cat, dwell_ms=dwell, gaze_xy=(random.randint(0, 1024), random.randint(0, 600)))
        elif r < 0.8:
            word = random.choice(SHORTHAND)
            self.db.log_selection("keyboard", "shorthand", f"kb:sh:{slug(word)}", word, dwell_ms=dwell)
        else:
            sentence = random.choice(SENTENCES)
            self.db.log_selection("keyboard", "speak", "kb:speak", sentence, dwell_ms=dwell)
        self.log("selection logged")
        self.next_sel = now + self._j(6)

    # ------------------------------------------------------------ main loop
    def run(self):
        self.log("EyeGlynt simulator running. Start a session in the dashboard. Ctrl+C to stop.")
        while self.running:
            now = time.time()
            try:
                if now - self.last_poll >= 1:
                    self.handle_commands()
                    self.last_poll = now
                if now - self.last_beat >= 2:
                    self.db.heartbeat(self.state, fps=round(random.uniform(26, 30), 1),
                                      lux=round(random.uniform(230, 270), 1), version="sim-1.0")
                    self.last_beat = now
                session = self.db.active_session()
                if session:
                    self.tick_recal()
                    self.maybe_alert(session)
                    self.maybe_select(session)
                elif self.state != "tracking" and not self.recal:
                    self.state = "tracking"
            except Exception as exc:  # keep going like a real device would
                self.log(f"db error: {exc}")
                time.sleep(2)
            time.sleep(0.2)
        self.db.heartbeat("idle")
        self.log("stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=1.0, help="time multiplier (2 = twice as fast)")
    ap.add_argument("--accuracy", type=float, default=0.9, help="hit rate in accuracy trials (0-1)")
    ap.add_argument("--recal-success", type=float, default=0.88, help="recalibration success rate (0-1)")
    ap.add_argument("--seed", type=int, help="random seed for repeatable demos")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    sim = Simulator(args)
    signal.signal(signal.SIGINT, lambda *_: setattr(sim, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(sim, "running", False))
    sim.run()


if __name__ == "__main__":
    main()
