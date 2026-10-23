// Guest page: one tap starts a browser voice call with the drinks agent; the seat comes from the QR (?s=7).
import { createWonderful } from "@wonderful/web";
import { createAgentOrb } from "./orb/wonderfulOrb.js";
import config from "../config.json";

const $ = (id) => document.getElementById(id);
const app = $("app");
const CONNECT_TIMEOUT_MS = 30_000;
// The order tool's confirmation line (agent lib/drinks.ts LINES.sent); after it, the call ends on its own.
const SENT = [/^ส่งรายการแล้วค่ะ/, /^Done, your order is in/i];
const CLOSED = [/^ขออภัยค่ะ ตอนนี้รบกวนสั่งเครื่องดื่มกับพนักงาน/, /^Sorry, for now please ask your waiter/i];
const HANG_UP_AFTER_MS = 9000;

const params = new URLSearchParams(location.search);
// Hold-to-talk (?mode=hold): the mic is open only while the guest holds the orb, so table talk never reaches the agent.
const HOLD = params.get("mode") === "hold";
const RELEASE_TAIL_MS = 350;
const seatRaw = Number(params.get("s"));
const seat = Number.isInteger(seatRaw) && seatRaw >= 1 && seatRaw <= 20 ? seatRaw : null;

const STATUS = {
  idle: ["แตะเพื่อสั่งเครื่องดื่ม", "Tap to order a drink"],
  connecting: ["กำลังเชื่อมต่อ…", "Connecting…"],
  live: HOLD ? ["กดค้างที่วงกลมแล้วพูด", "Hold the circle and speak"] : ["พูดได้เลยค่ะ", "Go ahead, I'm listening"],
  holding: ["กำลังฟัง… ปล่อยเมื่อพูดจบ", "Listening… let go when you're done"],
  done: ["สั่งเพิ่มได้ทุกเมื่อ", "Order again any time"],
  sent: ["เรียบร้อยค่ะ", "All set"],
  failed: ["เชื่อมต่อไม่สำเร็จ แตะเพื่อลองใหม่", "Couldn't connect. Tap to try again"],
  mic: ["กรุณาอนุญาตไมโครโฟน แล้วแตะอีกครั้ง", "Please allow the microphone, then tap again"],
  noseat: ["ไม่พบเลขที่นั่ง", "No seat number found"],
};

function setState(state, statusKey = state) {
  app.dataset.state = state;
  const [th, en] = STATUS[statusKey] ?? STATUS.idle;
  $("status-th").textContent = th;
  $("status-en").textContent = en;
  $("end").hidden = !(state === "live" || state === "connecting");
  $("again").hidden = state !== "done";
  $("talk").disabled = (state === "live" && !HOLD) || state === "connecting" || !seat;
  if (state !== "live") agentUntil = heardAt = 0;
  syncOrb();
}

// ---- orb choreography, driven by real call events only
// live: listening by default; thinking once the guest's words stop for 0.6 s, until the agent replies (max 5 s);
// speaking for about as long as the agent's line takes to say.
let agentUntil = 0;
let heardAt = 0;
let orbTimer = 0;
let orbNow = "";
function syncOrb() {
  clearTimeout(orbTimer);
  const st = app.dataset.state;
  const now = Date.now();
  let next = st === "connecting" ? "thinking" : st === "done" ? "complete" : st === "failed" ? "idle" : "idle";
  if (st === "live") {
    if (agentUntil > now) next = "speaking";
    else if (heardAt && now - heardAt > 600 && now - heardAt < 5000) next = "thinking";
    else next = $("received").hidden ? "listening" : "complete";
    const wake = Math.min(...[agentUntil, heardAt && heardAt + 600, heardAt && heardAt + 5000].filter((t) => t > now));
    if (Number.isFinite(wake)) orbTimer = setTimeout(syncOrb, wake - now + 30);
  }
  if (next === orbNow) return;
  orbNow = next;
  app.dataset.orb = next;
  if (next !== "listening") orb?.setAudioLevel(0);
  orb?.setState(next);
}
const speakMs = (text) => 700 + text.length * (/[\u0e00-\u0e7f]/.test(text) ? 75 : 62);
const buzz = (p) => navigator.vibrate?.(p);

// ---- seat
document.querySelectorAll("[data-seat]").forEach((el) => (el.textContent = seat ?? ""));
$("seat").hidden = !seat;
$("no-seat").hidden = !!seat;

// ---- in-app browsers (LINE, Facebook, Instagram, WeChat) handle the microphone badly: offer the real browser
const ua = navigator.userAgent;
if (/\bLine\//i.test(ua) || /FBAN|FBAV|Instagram|MicroMessenger/i.test(ua)) {
  $("inapp").hidden = false;
  app.classList.add("inapp");
  const url = new URL(location.href);
  if (/\bLine\//i.test(ua)) url.searchParams.set("openExternalBrowser", "1");
  $("inapp-open").href = url.toString();
}

// ---- orb: live WebGL where available, else the still
let orb = null;
try {
  const canvas = document.querySelector(".orb-live");
  const gl = document.createElement("canvas").getContext("webgl2");
  if (gl) {
    orb = createAgentOrb(canvas, { theme: "all-blue", state: "idle", quality: "balanced", fill: 1 / 1.24, transition: 0.45, reducedMotion: "user", onError: () => showStill() });
    canvas.hidden = false;
    // The still stays under the canvas until the first frames are drawn, so the orb never blinks out.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (canvas.hidden) return;
      canvas.classList.add("on");
      setTimeout(() => !canvas.hidden && (document.querySelector(".orb-still").style.visibility = "hidden"), 450);
    }));
  }
} catch {
  orb = null;
}
function showStill() {
  orb = null;
  document.querySelector(".orb-live").hidden = true;
  document.querySelector(".orb-still").style.visibility = "visible";
}

// ---- call
let client = null;
let session = null;
let voice = null;
let stops = [];
let hangTimer = 0;
let wakeLock = null;
let gen = 0;
const seen = new Map();
const firstAt = new Map();

const textOf = (c) => (typeof c === "string" ? c : c && typeof c.text === "string" ? c.text : "");
const isEnd = (m) => m.type === "action" && ["end", "hangup"].includes(m.content?.action?.type ?? m.content?.type ?? "");

// Only the agent's line is shown: the mic also hears the table, and a guest's private talk must never appear on screen.
// The guest line is still kept (hidden) for diagnostics.
function showLine(role, text) {
  if (!$("received").hidden) return;
  $("lines").hidden = false;
  $(role === "user" ? "line-guest" : "line-agent").textContent = text;
}

function onMessage(g, m) {
  if (g !== gen) return;
  if (isEnd(m)) return finish();
  if (m.role !== "user" && m.role !== "assistant") return;
  if (m.type === "thinking" || m.type === "action") return;
  const text = textOf(m.content).replace(/[\u200b\ufeff]/g, "").trim();
  // Platform markers such as "[BACKCHANNEL_IGNORED] Hi." are not words the guest should see.
  if (!text || /^\[[A-Z_]+\]/.test(text)) return;
  const rev = m.revision ?? 0;
  if ((seen.get(m.id) ?? -1) > rev) return;
  seen.set(m.id, rev);
  showLine(m.role, text);
  if (app.dataset.state === "live") {
    if (m.role === "user") heardAt = Date.now();
    else {
      heardAt = 0;
      if (!firstAt.has(m.id)) firstAt.set(m.id, Date.now());
      agentUntil = Math.max(agentUntil, firstAt.get(m.id) + speakMs(text));
    }
    syncOrb();
  }
  if (m.role === "assistant" && SENT.some((r) => r.test(text))) {
    $("received-text").textContent = orderText(text);
    if (!$("received").hidden) return;
    $("received").hidden = false;
    $("lines").hidden = true;
    // The order is in: stop listening so a "thanks" cannot start another turn, and hang up after the line plays.
    voice?.setMuted(true);
    setState("live", "sent");
    buzz([18, 70, 18]);
    hangUpSoon();
  } else if (m.role === "assistant" && CLOSED.some((r) => r.test(text))) hangUpSoon();
}

// The card shows only the order; the seat is in the banner and the waiter line is the card's own note.
const orderText = (t) =>
  t
    .replace(SENT[0], "").replace(SENT[1], "").replace(/^[.\s]+/, "")
    .replace(/\s*(พนักงานจะ|A waiter will).*$/s, "")
    .replace(/\s*สำหรับที่นั่ง\s*\d+$/, "")
    .replace(/,?\s*for seat \d+\.?$/i, "")
    .trim();

function hangUpSoon() {
  clearTimeout(hangTimer);
  hangTimer = setTimeout(finish, HANG_UP_AFTER_MS);
}

function teardown() {
  gen += 1;
  clearTimeout(hangTimer);
  stops.forEach((s) => s());
  stops = [];
  voice?.close();
  voice = null;
  session?.close().catch(() => {});
  session = null;
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  orb?.setAudioLevel(0);
}

function finish() {
  if (app.dataset.state !== "live" && app.dataset.state !== "connecting") return;
  teardown();
  setState($("received").hidden ? "idle" : "done");
}

async function start() {
  if (!seat || app.dataset.state === "live" || app.dataset.state === "connecting") return;
  teardown();
  const g = gen;
  seen.clear();
  firstAt.clear();
  $("received").hidden = true;
  $("lines").hidden = true;
  $("line-agent").textContent = "";
  $("line-guest").textContent = "";
  setState("connecting");
  // Ask for the microphone inside the tap, so iOS shows its prompt now.
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    probe.getTracks().forEach((t) => t.stop());
  } catch (err) {
    console.warn("microphone", err);
    if (g === gen) {
      setState("failed", "mic");
      buzz(40);
    }
    return;
  }
  try {
    wakeLock = (await navigator.wakeLock?.request("screen").catch(() => null)) ?? null;
    client ??= createWonderful({ key: config.key, agentId: config.agent_id, baseUrl: config.base_url, storageKey: "dd-guest" });
    const old = await client.getActiveSessions("voice").catch(() => []);
    await Promise.all(old.map((s) => s.close().catch(() => {})));
    if (g !== gen) return;
    session = client.createSession({
      agentId: config.agent_id,
      modality: "voice",
      source: "in_website",
      locale: "th",
      metadata: { seat: String(seat), lang: (params.get("lang") || navigator.language || "").slice(0, 2).toLowerCase() },
    });
    voice = session.connectVoice();
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("timeout")), CONNECT_TIMEOUT_MS);
      let ok = false;
      stops.push(
        voice.subscribe((st) => {
          if (g !== gen) return;
          if (orbNow === "listening") orb?.setAudioLevel(st.audioLevel ?? 0);
          if (st.status === "connected" && !ok) {
            ok = true;
            clearTimeout(timer);
            resolve();
          } else if (st.status === "failed" || st.status === "ended") {
            clearTimeout(timer);
            if (!ok) reject(new Error(st.error?.message ?? st.status));
            else finish();
          }
        }),
      );
    });
    if (g !== gen) return;
    stops.push(session.subscribe((m) => onMessage(g, m), { onStatus: (s) => s === "closed" && g === gen && finish() }));
    if (HOLD) voice.setMuted(true);
    setState("live");
    buzz(12);
  } catch (err) {
    console.warn("call failed", err);
    if (g === gen) {
      teardown();
      setState("failed");
      buzz(40);
    }
  }
}

$("talk").addEventListener("click", () => {
  if (!(HOLD && app.dataset.state === "live")) start();
});
// Hold-to-talk: open the mic on press, close it a moment after release (so the last word is not cut).
let releaseTimer = 0;
const press = (e) => {
  if (!HOLD || app.dataset.state !== "live" || !voice || !$("received").hidden) return;
  e.preventDefault();
  clearTimeout(releaseTimer);
  voice.setMuted(false);
  buzz(8);
  app.dataset.holding = "1";
  setState("live", "holding");
};
const release = () => {
  if (!HOLD || app.dataset.holding !== "1") return;
  app.dataset.holding = "";
  releaseTimer = setTimeout(() => voice?.setMuted(true), RELEASE_TAIL_MS);
  if (app.dataset.state === "live" && $("received").hidden) setState("live");
};
$("talk").addEventListener("pointerdown", press);
for (const ev of ["pointerup", "pointercancel", "pointerleave"]) $("talk").addEventListener(ev, release);
$("talk").addEventListener("contextmenu", (e) => HOLD && e.preventDefault());
$("again").addEventListener("click", start);
$("end").addEventListener("click", () => {
  teardown();
  setState($("received").hidden ? "idle" : "done");
});
// A hidden tab (screen lock, app switch) loses the microphone on iOS; end cleanly so the next tap starts fresh.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && app.dataset.state === "connecting") {
    teardown();
    setState("idle");
  }
});

app.classList.toggle("no-seat", !seat);
setState("idle", seat ? "idle" : "noseat");
