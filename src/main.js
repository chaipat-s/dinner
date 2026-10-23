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
  done: ["แตะเพื่อสั่งอีก", "Tap to order again"],
  failed: ["เชื่อมต่อไม่สำเร็จ แตะเพื่อลองใหม่", "Couldn't connect. Tap to try again"],
  mic: ["กรุณาอนุญาตไมโครโฟน แล้วแตะอีกครั้ง", "Please allow the microphone, then tap again"],
  noseat: ["สแกน QR บนการ์ดอีกครั้ง", "Please scan your seat card again"],
};

function setState(state, statusKey = state) {
  app.dataset.state = state;
  const [th, en] = STATUS[statusKey] ?? STATUS.idle;
  $("status-th").textContent = th;
  $("status-en").textContent = en;
  $("end").hidden = !(state === "live" || state === "connecting");
  $("again").hidden = state !== "done";
  $("talk").disabled = (state === "live" && !HOLD) || state === "connecting" || !seat;
  orb?.setState(state === "live" ? "listening" : state === "connecting" ? "thinking" : state === "done" ? "complete" : "idle");
}

// ---- seat
document.querySelectorAll("[data-seat]").forEach((el) => (el.textContent = seat ?? ""));
$("seat").hidden = !seat;
$("no-seat").hidden = !!seat;

// ---- in-app browsers (LINE, Facebook, Instagram, WeChat) handle the microphone badly: offer the real browser
const ua = navigator.userAgent;
if (/\bLine\//i.test(ua) || /FBAN|FBAV|Instagram|MicroMessenger/i.test(ua)) {
  $("inapp").hidden = false;
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
    orb = createAgentOrb(canvas, { theme: "all-blue", state: "idle", quality: "balanced", fill: 1 / 1.24, reducedMotion: "user", onError: () => showStill() });
    canvas.hidden = false;
    document.querySelector(".orb-still").style.visibility = "hidden";
  }
} catch {
  orb = null;
}
function showStill() {
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

const textOf = (c) => (typeof c === "string" ? c : c && typeof c.text === "string" ? c.text : "");
const isEnd = (m) => m.type === "action" && ["end", "hangup"].includes(m.content?.action?.type ?? m.content?.type ?? "");

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
  if (!text) return;
  const rev = m.revision ?? 0;
  if ((seen.get(m.id) ?? -1) > rev) return;
  seen.set(m.id, rev);
  showLine(m.role, text);
  if (m.role === "assistant" && SENT.some((r) => r.test(text))) {
    $("received").hidden = false;
    $("lines").hidden = true;
    $("received-text").textContent = text.replace(SENT[0], "").replace(SENT[1], "").replace(/^[.\s]+/, "");
    // The order is in: stop listening so a "thanks" cannot start another turn, and hang up after the line plays.
    voice?.setMuted(true);
    hangUpSoon();
  } else if (m.role === "assistant" && CLOSED.some((r) => r.test(text))) hangUpSoon();
}

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
    if (g === gen) setState("failed", "mic");
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
          orb?.setAudioLevel(st.audioLevel ?? 0);
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
  } catch (err) {
    console.warn("call failed", err);
    if (g === gen) {
      teardown();
      setState("failed");
    }
  }
}

$("talk").addEventListener("click", () => {
  if (!(HOLD && app.dataset.state === "live")) start();
});
// Hold-to-talk: open the mic on press, close it a moment after release (so the last word is not cut).
let releaseTimer = 0;
const press = (e) => {
  if (!HOLD || app.dataset.state !== "live" || !voice) return;
  e.preventDefault();
  clearTimeout(releaseTimer);
  voice.setMuted(false);
  app.dataset.holding = "1";
  setState("live", "holding");
};
const release = () => {
  if (!HOLD || app.dataset.holding !== "1") return;
  app.dataset.holding = "";
  releaseTimer = setTimeout(() => voice?.setMuted(true), RELEASE_TAIL_MS);
  if (app.dataset.state === "live") setState("live");
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

setState(seat ? "idle" : "idle", seat ? "idle" : "noseat");
