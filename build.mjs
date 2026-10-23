// Builds the static guest page into docs/ (GitHub Pages serves that folder).
import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";

const out = "docs";
const v = Date.now().toString(36);
rmSync(out, { recursive: true, force: true });
mkdirSync(`${out}/worklets`, { recursive: true });
await build({ entryPoints: ["src/main.js"], bundle: true, format: "esm", target: "es2020", minify: true, outfile: `${out}/app.js`, loader: { ".json": "json" } });
// The SDK loads its audio worklets next to the bundle (new URL("./worklets/…", import.meta.url)).
for (const f of ["microphone.js", "playback.js"]) cpSync(`node_modules/@wonderful/web/voice/worklets/${f}`, `${out}/worklets/${f}`);
cpSync("src/style.css", `${out}/style.css`);
cpSync("src/orb/all-blue.webp", `${out}/orb.webp`);
writeFileSync(`${out}/index.html`, readFileSync("src/index.html", "utf8").replaceAll("__V__", v));
writeFileSync(`${out}/.nojekyll`, "");
console.log(`built ${out}/ (v ${v})`);
