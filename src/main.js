// Guest page: one tap starts a browser voice call with the drinks agent.
// Who is ordering comes from the link only: ?g=<attendee_id>&n=<display name> (email links), or the old ?s=<seat> (QR cards).
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
const GREETING = /^(สวัสดี|ยินดีต้อนรับ|Hi\b|Hello|Good evening|Welcome)/i;

const params = new URLSearchParams(location.search);
// Hold-to-talk: the mic is open only while the guest holds the orb, so table talk never reaches the agent.
// The mode is switched remotely from the waiter App's Settings (dd_settings.talk, served by the public Function
// dd-talk as a script: the API does not allow this origin for fetch) or by scripts/talk-mode.sh; mode.json on this
// site is the fallback. Read at load and again on every tap that starts a call; &mode=hold / &mode=free overrides it.
// Nothing within 1.5 s means hands-free.
const URL_MODE = ["hold", "free"].includes(params.get("mode")) ? params.get("mode") : null;
let HOLD = URL_MODE === "hold";
async function loadMode() {
  let hold = URL_MODE === "hold";
  if (!URL_MODE) {
    const live = await liveTalk();
    if (live) hold = live === "hold";
    else {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 1500);
      try {
        const r = await fetch(`mode.json?t=${Date.now()}`, { cache: "no-store", signal: ctl.signal });
        hold = (await r.json()).talk === "hold";
      } catch {
        hold = false;
      }
      clearTimeout(timer);
    }
  }
  HOLD = hold;
  app.classList.toggle("hold", HOLD);
  app.dataset.talk = HOLD ? "hold" : "hands-free";
  $("hint").innerHTML = HOLD
    ? 'แตะวงกลม แล้วแตะค้างไว้ขณะพูด<br><span class="en">Tap the circle, then hold it while you speak.</span>'
    : 'ถือโทรศัพท์ใกล้ปาก เหมือนส่งข้อความเสียง<br><span class="en">Hold the phone close, like a voice message.</span>';
}
/** The mode set in the waiter App, loaded as a script (window.__ddTalk); null when it does not answer within 1.5 s. */
const TALK_URL = "https://th.api.demo.wonderful.ai/api/public/functions/a2771683-39f0-4962-a28c-e06c6120360a/94c1ffd0-24bf-4fec-b7e8-2b318a0a8a26/dd-talk";
function liveTalk() {
  return new Promise((done) => {
    window.__ddTalk = null;
    const tag = document.createElement("script");
    const finish = (v) => {
      clearTimeout(timer);
      tag.remove();
      done(v === "hold" || v === "hands-free" ? v : null);
    };
    const timer = setTimeout(() => finish(null), 1500);
    tag.onload = () => finish(window.__ddTalk?.talk);
    tag.onerror = () => finish(null);
    tag.src = `${TALK_URL}?t=${Date.now()}`;
    document.head.appendChild(tag);
  });
}
loadMode(); // at load too, so the idle hint already matches the mode
const RELEASE_TAIL_MS = 350;
const seatRaw = Number(params.get("s"));
const seat = Number.isInteger(seatRaw) && seatRaw >= 1 && seatRaw <= 20 ? seatRaw : null;
const guest = /^[A-Za-z0-9_-]{4,64}$/.test(params.get("g") ?? "") ? params.get("g") : null;
const name = (params.get("n") ?? "").replace(/[\u0000-\u001f\u200b\ufeff]/g, "").replace(/\s+/g, " ").trim().slice(0, 60) || null;
const THAI = /[\u0e00-\u0e7f]/;
// A returning guest (an earlier call on this link, this page or before a reload) gets no second greeting.
const KEY = `dd-calls:${guest ?? (seat ? `s${seat}` : "-")}`;
const store = { get: () => { try { return Number(localStorage.getItem(KEY)) || 0; } catch { return 0; } }, bump() { try { localStorage.setItem(KEY, String(this.get() + 1)); } catch {} } };
// Walk-in link (no params): a house phone passes from guest to guest, so every call is a first call.
const WALKIN = !name && !seat && !guest;
const wasHere = () => !WALKIN && store.get() > 0;
let returning = wasHere();

const STATUS = {
  idle: WALKIN ? ["แตะ แล้วบอกชื่อกับเครื่องดื่มที่ต้องการ", "Tap, then say your name and your drink"] : ["แตะเพื่อสั่งเครื่องดื่ม", "Tap to order a drink"],
  again: ["แตะเพื่อสั่งเพิ่ม", "Tap to order another"],
  connecting: ["กำลังเชื่อมต่อ…", "Connecting…"],
  get live() { return HOLD ? ["แตะค้างไว้ขณะพูด", "Hold while you speak"] : ["พูดได้เลยค่ะ", "Go ahead, I'm listening"]; },
  holding: ["กำลังฟัง… ปล่อยเมื่อพูดจบ", "Listening… let go when you're done"],
  done: ["", ""],
  sent: ["", ""],
  failed: ["เชื่อมต่อไม่สำเร็จ แตะเพื่อลองใหม่", "Couldn't connect. Tap to try again"],
  mic: ["ใช้ไมโครโฟนไม่ได้ ดูวิธีแก้ด้านบน", "The microphone is blocked. See the steps above"],
};

function setState(state, statusKey = state === "idle" && returning ? "again" : state) {
  app.dataset.state = state;
  // Returning guest: no "สวัสดีค่ะ" line, no first-time hint; just the name.
  app.classList.toggle("returning", returning);
  const [th, en] = STATUS[statusKey] ?? STATUS.idle;
  $("status-th").textContent = th;
  $("status-en").textContent = en;
  $("end").hidden = !(state === "live" || state === "connecting");
  $("again").hidden = state !== "done";
  $("talk").disabled = (state === "live" && !HOLD) || state === "connecting";
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
  // While the agent talks or thinks, "go ahead" would be wrong: the status steps aside for the agent's line.
  if (st === "live" && $("received").hidden && app.dataset.holding !== "1") {
    const [th, en] = next === "listening" ? STATUS.live : ["", ""];
    $("status-th").textContent = th;
    $("status-en").textContent = en;
  }
  if (next === orbNow) return;
  orbNow = next;
  app.dataset.orb = next;
  if (next !== "listening") orb?.setAudioLevel(0);
  orb?.setState(next);
}
const speakMs = (text) => 700 + text.length * (/[\u0e00-\u0e7f]/.test(text) ? 75 : 62);
const buzz = (p) => navigator.vibrate?.(p);

// ---- welcome: by name on an email link, else the seat (old QR cards), else the walk-in welcome (no params)
$("hello").hidden = false;
if (name) {
  const bare = name.replace(/^คุณ\s*/, "");
  $("who").textContent = THAI.test(bare) ? `คุณ${bare}` : bare;
  $("who").lang = THAI.test(bare) ? "th" : "en";
  $("who-en").hidden = true;
} else if (seat) {
  $("who").textContent = `ที่นั่ง ${seat}`;
  $("who-en").textContent = `Seat ${seat}`;
} else {
  app.classList.add("walkin");
  $("who").textContent = "สั่งเครื่องดื่มด้วยเสียง";
  $("who-en").textContent = "Order a drink by voice";
}

// ---- in-app browsers: email, social and chat apps open links in their own web view, where the microphone
// is often blocked (the app must grant it; many never do). Detected by user agent; first match wins.
// Not detectable and not needed: Gmail and Outlook on Android open Chrome Custom Tabs, Telegram and WhatsApp on
// iOS open Safari (SFSafariViewController); those are the real browser engine with a Safari/Chrome user agent.
const ua = navigator.userAgent;
const IOS = /\b(iPhone|iPad|iPod)\b/.test(ua);
const ANDROID = /\bAndroid\b/.test(ua);
const BROWSER = IOS ? "Safari" : ANDROID ? "Chrome" : "Safari / Chrome";
const MENU = IOS ? "⋯" : "⋮";
// [app, UA test, Thai step, English step]
const APPS = [
  ["line", /\bLine\//i, `แตะปุ่มด้านล่าง หรือแตะ ${MENU} แล้วเลือกเปิดในเบราว์เซอร์`, `Tap the button below, or tap ${MENU} → Open in browser.`],
  ["wechat", /MicroMessenger/i, `แตะ ··· มุมบนขวา แล้วเลือกเปิดใน ${BROWSER}`, `Tap ··· at the top right → Open in ${BROWSER}.`],
  ["instagram", /\bInstagram\b/i, `แตะ ${MENU} มุมบนขวา แล้วเลือกเปิดในเบราว์เซอร์ภายนอก`, `Tap ${MENU} at the top right → Open in external browser.`],
  ["messenger", /MessengerForiOS|Orca-Android|MessengerLite/i, `แตะ ${MENU} มุมบนขวา แล้วเลือกเปิดใน ${BROWSER}`, `Tap ${MENU} at the top right → Open in ${BROWSER}.`],
  ["facebook", /FBAN|FBAV|FB_IAB|FBIOS|FB4A/, `แตะ ${MENU} มุมบนขวา แล้วเลือกเปิดใน ${BROWSER}`, `Tap ${MENU} at the top right → Open in ${BROWSER}.`],
  ["linkedin", /LinkedInApp/i, `แตะ ${MENU} มุมบนขวา แล้วเลือกเปิดในเบราว์เซอร์`, `Tap ${MENU} at the top right → Open in browser.`],
  ["google", /\bGSA\//, `แตะปุ่มแชร์หรือ ${MENU} แล้วเลือกเปิดใน ${BROWSER}`, `Tap the share button or ${MENU} → Open in ${BROWSER}.`],
  ["chat", /\bWhatsApp\b|\bTelegram\b|KAKAOTALK|\bTwitter\b|Snapchat|BytedanceWebview|musical_ly/i, `แตะ ${MENU} แล้วเลือกเปิดใน ${BROWSER}`, `Tap ${MENU} → Open in ${BROWSER}.`],
  // Android WebView (Outlook, Gmail with Custom Tabs off, any app): the "; wv)" token.
  ["webview", /\bAndroid\b.*;\s?wv\)/, `แตะ ${MENU} มุมบนขวา แล้วเลือกเปิดใน Chrome`, `Tap ${MENU} at the top right → Open in Chrome.`],
  // iOS WKWebView (Gmail and Outlook in-app browsers, most apps): no "Safari/" token. Real Safari, Chrome (CriOS),
  // Firefox (FxiOS), Edge and SFSafariViewController all carry it.
  ["webview", /\b(iPhone|iPad|iPod)\b(?!.*Safari\/)/, `แตะ ${MENU} หรือปุ่มแชร์ แล้วเลือกเปิดใน Safari`, `Tap ${MENU} or the share button → Open in Safari.`],
];
const hit = APPS.find(([, re]) => re.test(ua)) ?? (!navigator.mediaDevices?.getUserMedia ? ["nomic", null, `เปิดลิงก์นี้ใน ${BROWSER}`, `Open this link in ${BROWSER}.`] : null);
const PASTE = [`หรือคัดลอกลิงก์ แล้ววางใน ${BROWSER}`, `Or copy the link and paste it into ${BROWSER}.`];

// The banner: why, the app's own steps, a one-tap way out where one exists, and Copy link.
function showHelp(kind) {
  const [th, en] = kind === "mic" ? ["ใช้ไมโครโฟนไม่ได้", "The microphone is blocked"] : [`เปิดหน้านี้ใน ${BROWSER} เพื่อใช้ไมโครโฟน`, `Open this page in ${BROWSER} to use the microphone.`];
  const [t, e] = hit ? [hit[2], hit[3]] : micSteps();
  const step = kind === "mic" && !hit
    ? [`${t} แล้วแตะวงกลมอีกครั้ง ถ้ายังไม่ได้ คัดลอกลิงก์ แล้วเปิดใน ${BROWSER}`, `${e} Then tap the circle again. Still blocked? Copy the link and open it in ${BROWSER}.`]
    : [`${t} ${PASTE[0]}`, `${e} ${PASTE[1]}`];
  $("inapp-th").textContent = th;
  $("inapp-en").textContent = en;
  $("step-th").textContent = step[0];
  $("step-en").textContent = step[1];
  const open = $("inapp-open");
  const url = new URL(location.href);
  if (hit?.[0] === "line") {
    url.searchParams.set("openExternalBrowser", "1"); // LINE opens this in the phone's default browser
    open.href = url.toString();
    open.hidden = false;
  } else if (hit && ANDROID) {
    // Android: hand the link to Chrome; if Chrome is missing, stay here (the steps and Copy link still work).
    open.href = `intent://${url.host}${url.pathname}${url.search}#Intent;scheme=https;package=com.android.chrome;end`;
    $("open-th").textContent = "เปิดใน Chrome";
    $("open-en").textContent = "Open in Chrome";
    open.hidden = false;
  }
  $("inapp").hidden = false;
  $("inapp").dataset.kind = kind;
  app.classList.add("inapp");
}
// Mic blocked in a real browser: where that browser keeps the site's microphone switch.
function micSteps() {
  if (/CriOS/.test(ua)) return ["เปิด ตั้งค่า › Chrome › ไมโครโฟน", "Open Settings › Chrome › turn on Microphone."];
  if (IOS) return ["แตะ aA หรือ ⋯ ที่แถบที่อยู่ › การตั้งค่าเว็บไซต์ › ไมโครโฟน › อนุญาต", "Tap aA or ⋯ in the address bar › Website Settings › Microphone › Allow."];
  if (ANDROID) return ["แตะไอคอนหน้าที่อยู่เว็บ › สิทธิ์ › ไมโครโฟน › อนุญาต", "Tap the icon left of the address › Permissions › Microphone › Allow."];
  return ["อนุญาตไมโครโฟนให้เว็บนี้", "Allow the microphone for this site."];
}
if (hit) showHelp("inapp");

$("copy").addEventListener("click", async () => {
  const url = location.href.replace(/[?&]openExternalBrowser=1/, "");
  let ok = false;
  try { await navigator.clipboard.writeText(url); ok = true; } catch {}
  if (!ok) {
    const box = $("copy-url");
    box.value = url;
    box.hidden = false;
    box.focus();
    box.setSelectionRange(0, url.length);
    try { ok = document.execCommand("copy"); } catch {}
  }
  $("copy-th").textContent = ok ? "คัดลอกแล้ว" : "กดค้างที่ลิงก์ด้านล่างเพื่อคัดลอก";
  $("copy-en").textContent = ok ? "Copied" : "Press and hold the link below";
  buzz(10);
  if (ok) setTimeout(() => { $("copy-th").textContent = "คัดลอกลิงก์"; $("copy-en").textContent = "Copy link"; }, 2500);
});

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
let saidFirst = false;
const seen = new Map();
const firstAt = new Map();

const textOf = (c) => (typeof c === "string" ? c : c && typeof c.text === "string" ? c.text : "");
const isEnd = (m) => m.type === "action" && ["end", "hangup"].includes(m.content?.action?.type ?? m.content?.type ?? "");

// Only the agent's line is shown: the mic also hears the table, and a guest's private talk must never appear on screen.
// The guest line is still kept (hidden) for diagnostics.
function showLine(role, text) {
  if (!$("received").hidden) return;
  $("lines").hidden = false;
  const el = $(role === "user" ? "line-guest" : "line-agent");
  el.textContent = text;
  el.lang = THAI.test(text) ? "th" : "en";
}

function onMessage(g, m) {
  if (g !== gen) return;
  if (isEnd(m)) return finish();
  if (m.role !== "user" && m.role !== "assistant") return;
  if (m.type === "thinking" || m.type === "action") return;
  const text = textOf(m.content).replace(/[\u200b\ufeff]/g, "").trim();
  // Platform markers such as "[BACKCHANNEL_IGNORED] Hi." are not words the guest should see.
  if (!text || /^\[[A-Z_]+\]/.test(text)) return;
  // Second and later calls: the agent's short ack is shown, a full greeting is not shown again.
  if (m.role === "assistant" && !saidFirst) {
    saidFirst = true;
    if (returning && GREETING.test(text)) seen.set(m.id, Infinity);
  }
  if (seen.get(m.id) === Infinity) return;
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

// The card shows only the order; the waiter line is the card's own note.
const orderText = (t) =>
  t
    .replace(SENT[0], "").replace(SENT[1], "").replace(/^[.\s]+/, "")
    .replace(/\s*(พนักงานจะ|A waiter will).*$/s, "")
    .replace(/\s*สำหรับ(ที่นั่ง\s*\d+|คุณ\S*)$/, "")
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
  returning = wasHere();
  setState($("received").hidden ? "idle" : "done");
}

async function start() {
  if (app.dataset.state === "live" || app.dataset.state === "connecting") return;
  teardown();
  const g = gen;
  seen.clear();
  firstAt.clear();
  saidFirst = false;
  $("received").hidden = true;
  $("lines").hidden = true;
  $("line-agent").textContent = "";
  $("line-guest").textContent = "";
  setState("connecting");
  const modeP = loadMode();
  // Ask for the microphone inside the tap, so iOS shows its prompt now.
  try {
    const probe = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    probe.getTracks().forEach((t) => t.stop());
  } catch (err) {
    console.warn("microphone", err);
    if (g === gen) {
      setState("failed", "mic");
      showHelp("mic");
      buzz(40);
    }
    return;
  }
  try {
    await modeP;
    if (g !== gen) return;
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
      metadata: {
        ...(guest && { guest }),
        ...(name && { name }),
        ...(seat && { seat: String(seat) }),
        lang: (params.get("lang") || navigator.language || "").slice(0, 2).toLowerCase(),
      },
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
    store.bump();
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
  returning = wasHere();
  setState($("received").hidden ? "idle" : "done");
});
// A hidden tab (screen lock, app switch) loses the microphone on iOS; end cleanly so the next tap starts fresh.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && app.dataset.state === "connecting") {
    teardown();
    setState("idle");
  }
});

setState("idle");
