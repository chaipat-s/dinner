# Guest drinks page

Static page for the dinner's voice drink ordering. Source in `src/`, built into `docs/` (served by GitHub Pages) with `npm run build`.

- Email links: `?g=<attendee_id>&n=<display name>` (URL-encoded). Welcomes the guest by name; sends session metadata `{guest, name, lang}`. Names and tokens come from the URL only, never from this repo.
- Old QR cards: `?s=<seat>` (1–20) still work; sends `seat`.
- No params: a neutral welcome; the call still works.
- `&mode=hold`: hold-to-talk. `&lang=th|en`: overrides the phone language (tests).
- A second call on the same link (this page or after a reload) hides a full greeting line; the agent says a short ack.
- Logo: `src/brand/` (from the intro deck, see `SOURCE.md`).
