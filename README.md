# Guest drinks page

Static page for the dinner's voice drink ordering. Source in `src/`, built into `docs/` (served by GitHub Pages) with `npm run build`.

- Email links: `?g=<attendee_id>&n=<display name>` (URL-encoded). Welcomes the guest by name; sends session metadata `{guest, name, lang}`. Names and tokens come from the URL only, never from this repo.
- Old QR cards: `?s=<seat>` (1–20) still work; sends `seat`.
- No params (walk-in link and QR card): "สั่งเครื่องดื่มด้วยเสียง · Order a drink by voice"; the agent asks the guest's name. Every call counts as a first call (house phones pass between guests).
- `&mode=hold`: hold-to-talk. `&lang=th|en`: overrides the phone language (tests).
- A second call on the same link (this page or after a reload) hides a full greeting line; the agent says a short ack.
- Logo: `src/brand/` (from the intro deck, see `SOURCE.md`).
- In-app browsers (email, social, chat apps): detected by user agent in `src/main.js` (`APPS`); a banner gives the app's steps to open Safari / Chrome, an Open in Chrome button on Android (`intent://`), Open in browser on LINE (`openExternalBrowser=1`), and Copy link. The same banner, with the browser's microphone steps, shows whenever the microphone is refused.
