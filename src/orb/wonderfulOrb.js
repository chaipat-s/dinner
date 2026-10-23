/* ==========================================================================
 * Wonderful Agent Orb
 * A live, voice-reactive WebGL2 orb for Wonderful agents.
 *
 * The look is ported from the Blender "Live" scene in 03_Agents/01_States:
 *   - the orb material (two 4D Perlin noise waves + rim, shell mix) evaluated analytically per pixel,
 *   - Blender's compositor Bloom (Glare node) re-implemented pass for pass,
 *   - Blender's AgX "Very High Contrast" display transform, baked into a 3D LUT.
 * Colour themes and their approved still frames come straight from the .blend files
 * (see tools/). States follow the Blender state materials: Live, Thinking, Waiting, plus
 * Speaking (Talking Test), Listening, Complete and Error.
 *
 * No dependencies. ES module.
 *
 *   import { createAgentOrb } from "./wonderful-orb.js";
 *   const orb = createAgentOrb(document.querySelector("canvas"), { theme: "all-blue" });
 *   orb.setState("thinking");
 *   await orb.connectMicrophone();          // voice drives the orb
 *   orb.destroy();
 *
 * Or declaratively:  <wonderful-orb theme="pink" state="idle"></wonderful-orb>
 * ========================================================================== */

/* ------------------------------------------------------------------ *
 * 1. Constants
 * ------------------------------------------------------------------ */
const FPS = 24;                              // Blender timeline rate
const TAU = Math.PI * 2;
const K_WORLD = 0.98925;                     // EEVEE Translucent BSDF response to the uniform world: K * world * (T * weight)^2
const REF_RES = 1080;                        // resolution of the approved Blender renders
const REF_LEVELS = 10;                       // floor(log2(1080)) bloom levels at that resolution
const LUT_MIN_EV = -12.47393;                // AgX log2 domain used by the baked LUT
const LUT_MAX_EV = 4.026069;

export const AGENT_STATES = Object.freeze(["static", "idle", "listening", "waiting", "thinking", "speaking", "complete", "error"]);
export const QUALITY_PROFILES = Object.freeze({
  low:      { fps: 30, dpr: 1,   maxPx: 512,  matPx: 256 },
  balanced: { fps: 45, dpr: 1.5, maxPx: 896,  matPx: 448 },
  high:     { fps: 60, dpr: 2,   maxPx: 1280, matPx: 704 },
});
export const AUDIO_RESPONSE_DEFAULTS = Object.freeze({ sensitivity: 1, brightness: 1, motion: 1, pulse: 1 });

/* ------------------------------------------------------------------ *
 * 2. Blender's noise hash, precomputed on the CPU (BLI_hash, Jenkins lookup3)
 * ------------------------------------------------------------------ */
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
const floatBits = (f) => { _f32[0] = f; return _u32[0]; };
const rot32 = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;
function hashUint2(kx, ky) {
  let a = (0xdeadbeef + (2 << 2) + 13) >>> 0, b = a, c = a;
  a = (a + kx) >>> 0; b = (b + ky) >>> 0;
  c ^= b; c = (c - rot32(b, 14)) >>> 0;
  a ^= c; a = (a - rot32(c, 11)) >>> 0;
  b ^= a; b = (b - rot32(a, 25)) >>> 0;
  c ^= b; c = (c - rot32(b, 16)) >>> 0;
  a ^= c; a = (a - rot32(c, 4)) >>> 0;
  b ^= a; b = (b - rot32(a, 14)) >>> 0;
  c ^= b; c = (c - rot32(b, 24)) >>> 0;
  return c >>> 0;
}
const hashVec2ToFloat = (x, y) => hashUint2(floatBits(x), floatBits(y)) / 4294967295.0;
const randomFloat4Offset = (seed) => [0, 1, 2, 3].map((i) => 100.0 + hashVec2ToFloat(seed, i) * 100.0);
const NOISE_OFFSETS = new Float32Array([0, 1, 2, 3, 4, 5].flatMap(randomFloat4Offset));

/* ------------------------------------------------------------------ *
 * 3. GLSL
 * ------------------------------------------------------------------ */
const QUAD_VS = `#version 300 es
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

/* Pass 1: the orb material, evaluated analytically on the sphere. Output: linear HDR colour,
 * alpha = analytic silhouette coverage. Colour is edge-extended past the silhouette. */
const MATERIAL_FS = `#version 300 es
precision highp float;
precision highp int;
#define DEBUG_STAGE 0
in vec2 vUv;
out vec4 fragColor;

struct Wave {
  vec3 emit; float strength; float scale; float dist;
  vec3 loc; vec3 scl; vec4 range; vec2 ramp; float lw; vec3 transl;
  vec2 rotXY; float rot; float w;
};
uniform Wave uA;
uniform Wave uB;
uniform vec3 uRimEmit; uniform float uRimStrength, uRimPow, uRimLw; uniform vec4 uRimRange; uniform vec3 uRimTransl;
uniform float uMixWR, uMixB, uMixE;
uniform vec3 uVeil;           /* radiance of the empty slot (usually black); already divided by (1 - mix.e) */
uniform vec3 uBase;           /* world colour * strength * response */
uniform vec4 uNoiseOff[6];
uniform vec2 uRes;
uniform float uFill, uSide, uClip, uSigma;
uniform mat3 uOrient;         /* object-space orientation of the sphere (identity = Blender's) */
uniform float uWeight;        /* how much of this pattern counts: while one state dissolves into another, each is drawn with its own weight and the two are added */

/* ---- Blender BLI_hash (Jenkins lookup3) and 4D Perlin noise ---- */
uint rot32(uint x, uint k){ return (x << k) | (x >> (32u - k)); }
uint hash4u(uint kx, uint ky, uint kz, uint kw){
  uint a = 0xdeadbeefu + (4u << 2u) + 13u; uint b = a, c = a;
  a += kx; b += ky; c += kz;
  a -= c; a ^= rot32(c, 4u);  c += b;
  b -= a; b ^= rot32(a, 6u);  a += c;
  c -= b; c ^= rot32(b, 8u);  b += a;
  a -= c; a ^= rot32(c, 16u); c += b;
  b -= a; b ^= rot32(a, 19u); a += c;
  c -= b; c ^= rot32(b, 4u);  b += a;
  a += kw;
  c ^= b; c -= rot32(b, 14u);
  a ^= c; a -= rot32(c, 11u);
  b ^= a; b -= rot32(a, 25u);
  c ^= b; c -= rot32(b, 16u);
  a ^= c; a -= rot32(c, 4u);
  b ^= a; b -= rot32(a, 14u);
  c ^= b; c -= rot32(b, 24u);
  return c;
}
float nfade(float t){ return t*t*t*(t*(t*6.0 - 15.0) + 10.0); }
float negIf(float v, uint cond){ return cond != 0u ? -v : v; }
float grad4(uint h, float x, float y, float z, float w){
  uint hh = h & 31u;
  float u = hh < 24u ? x : y; float v = hh < 16u ? y : z; float s = hh < 8u ? z : w;
  return negIf(u, hh & 1u) + negIf(v, hh & 2u) + negIf(s, hh & 4u);
}
float perlin4d(vec4 p){
  ivec4 I = ivec4(floor(p)); vec4 f = p - vec4(I);
  uint X = uint(I.x), Y = uint(I.y), Z = uint(I.z), W = uint(I.w);
  float u = nfade(f.x), v = nfade(f.y), t = nfade(f.z), s = nfade(f.w);
  float g0000 = grad4(hash4u(X,Y,Z,W), f.x,f.y,f.z,f.w);
  float g1000 = grad4(hash4u(X+1u,Y,Z,W), f.x-1.0,f.y,f.z,f.w);
  float g0100 = grad4(hash4u(X,Y+1u,Z,W), f.x,f.y-1.0,f.z,f.w);
  float g1100 = grad4(hash4u(X+1u,Y+1u,Z,W), f.x-1.0,f.y-1.0,f.z,f.w);
  float g0010 = grad4(hash4u(X,Y,Z+1u,W), f.x,f.y,f.z-1.0,f.w);
  float g1010 = grad4(hash4u(X+1u,Y,Z+1u,W), f.x-1.0,f.y,f.z-1.0,f.w);
  float g0110 = grad4(hash4u(X,Y+1u,Z+1u,W), f.x,f.y-1.0,f.z-1.0,f.w);
  float g1110 = grad4(hash4u(X+1u,Y+1u,Z+1u,W), f.x-1.0,f.y-1.0,f.z-1.0,f.w);
  float g0001 = grad4(hash4u(X,Y,Z,W+1u), f.x,f.y,f.z,f.w-1.0);
  float g1001 = grad4(hash4u(X+1u,Y,Z,W+1u), f.x-1.0,f.y,f.z,f.w-1.0);
  float g0101 = grad4(hash4u(X,Y+1u,Z,W+1u), f.x,f.y-1.0,f.z,f.w-1.0);
  float g1101 = grad4(hash4u(X+1u,Y+1u,Z,W+1u), f.x-1.0,f.y-1.0,f.z,f.w-1.0);
  float g0011 = grad4(hash4u(X,Y,Z+1u,W+1u), f.x,f.y,f.z-1.0,f.w-1.0);
  float g1011 = grad4(hash4u(X+1u,Y,Z+1u,W+1u), f.x-1.0,f.y,f.z-1.0,f.w-1.0);
  float g0111 = grad4(hash4u(X,Y+1u,Z+1u,W+1u), f.x,f.y-1.0,f.z-1.0,f.w-1.0);
  float g1111 = grad4(hash4u(X+1u,Y+1u,Z+1u,W+1u), f.x-1.0,f.y-1.0,f.z-1.0,f.w-1.0);
  float w0 = mix(mix(mix(g0000,g1000,u), mix(g0100,g1100,u), v), mix(mix(g0010,g1010,u), mix(g0110,g1110,u), v), t);
  float w1 = mix(mix(mix(g0001,g1001,u), mix(g0101,g1101,u), v), mix(mix(g0011,g1011,u), mix(g0111,g1111,u), v), t);
  return mix(w0, w1, s);
}
float snoise4(vec4 p){ return 0.8344 * perlin4d(p); }
float noise4(vec4 p){ return 0.5 * snoise4(p) + 0.5; }

/* Noise Texture node (4D, FBM, roughness 0 = single octave, distortion): the three colour channels */
vec3 noiseColor(vec3 co, float w, float scale, float distortion){
  vec4 p = vec4(co, w) * scale;
  if (distortion != 0.0){
    p += vec4(snoise4(p + uNoiseOff[0]), snoise4(p + uNoiseOff[1]),
              snoise4(p + uNoiseOff[2]), snoise4(p + uNoiseOff[3])) * distortion;
  }
  return vec3(noise4(p), noise4(p + uNoiseOff[4]), noise4(p + uNoiseOff[5]));
}
/* colour -> float conversion by luminance */
float noiseGray(vec3 co, float w, float scale, float distortion){
  return dot(noiseColor(co, w, scale, distortion), vec3(0.2126729, 0.7151522, 0.0721750));
}
/* Mapping node, Point type: location + EulerXYZ(rotation) * (scale * vector) */
vec3 mapping(vec3 v, vec3 loc, vec3 e, vec3 scl){
  float cx = cos(e.x), cy = cos(e.y), cz = cos(e.z), sx = sin(e.x), sy = sin(e.y), sz = sin(e.z);
  mat3 m = mat3(cy*cz, cy*sz, -sy,
                sy*sx*cz - cx*sz, sy*sx*sz + cx*cz, cy*sx,
                sy*cx*cz + sx*sz, sy*cx*sz - sx*cz, cy*cx);
  return m * (v * scl) + loc;
}
float mapRange(float v, vec4 r){
  float o = r.z + (v - r.x) / (r.y - r.x) * (r.w - r.z);
  return r.w > r.z ? clamp(o, r.z, r.w) : clamp(o, r.w, r.z);
}
/* Layer Weight "Facing" */
float facingOf(float NdotV, float blend){
  blend = clamp(blend, 0.0, 0.99999);
  float b = blend < 0.5 ? 2.0 * blend : 0.5 / (1.0 - blend);
  return 1.0 - pow(NdotV, b);
}
/* Colour Ramp, two stops black -> white, B-spline interpolation. Blender evaluates three segments over the
 * whole [0,1] domain with virtual end stops at 0 and 1 (key_curve_position_weights), so the ramp is not
 * constant outside the stops: it eases in below the first stop and eases out above the last. */
float rampBSpline(float fac, vec2 st){
  if (fac <= 0.0) return 0.0;
  if (fac >= 1.0) return 1.0;
  if (fac < st.x){ float u = fac / max(st.x, 1e-4); return u*u*u / 6.0; }
  if (fac > st.y){ float u = (fac - st.y) / max(1.0 - st.y, 1e-4); float v = 1.0 - u; return 1.0 - v*v*v / 6.0; }
  float u = (fac - st.x) / (st.y - st.x);
  float u2 = u*u, u3 = u2*u;
  return (-0.5*u3 + 0.5*u2 + 0.5*u + 0.16666666) + 0.16666666*u3;
}
/* The two white area lights of the scene as seen by the translucent shell (fitted to EEVEE, ~0.4% of the world term) */
float lightsTerm(vec3 nt){
  float d1 = dot(nt, vec3(0.4750, 0.0, 0.8800)), d2 = dot(nt, vec3(-0.6609, 0.0, -0.7505));
  return 0.022038 * pow(max(0.5 + 0.5 * d1, 0.0), 2.85) + 0.028075 * pow(max(0.5 + 0.5 * d2, 0.0), 2.45);
}
/* Emission fraction of one wave: ramp(noise) mixes the translucent shell with the emission */
float waveRamp(Wave W, vec3 P, float NdotV){
  float n = noiseGray(mapping(P, W.loc, vec3(W.rotXY, W.rot), W.scl), W.w, W.scale, W.dist);
  return rampBSpline(mapRange(n, W.range) + facingOf(NdotV, W.lw), W.ramp);
}

void main(){
  float m = min(uRes.x, uRes.y);
  vec2 p = (vUv * 2.0 - 1.0) * (uRes / m) / uFill;       /* sphere units; +x right, +y up (= object +Z) */
  float rr = length(p);
  float px = 2.0 / (m * uFill);                          /* one pixel in sphere units */
  float Rc = sqrt(1.0 - uClip * uClip);                  /* silhouette radius: the camera's near clip trims the inner wall */
  if (rr > Rc + 4.0 * px * uSigma + px){ fragColor = vec4(0.0); return; }
  float cov = 1.0 / (1.0 + exp(1.702 * (rr - Rc) / (px * uSigma)));   /* edge filtered by a gaussian of sigma uSigma px */
  vec2 q = rr > Rc ? p * (Rc / rr) : p;                  /* edge extension */
  float NdotV = sqrt(max(1.0 - dot(q, q), 0.0));
  vec3 P = uOrient * vec3(q.x, uSide * NdotV, q.y);
  vec3 base = uBase + vec3(lightsTerm(vec3(uSide * q.x, NdotV, uSide * q.y)));          /* translucent lighting */

#if DEBUG_STAGE > 0
  {
    Wave Wv = uA;
    if (DEBUG_STAGE >= 6 && DEBUG_STAGE <= 9) Wv = uB;
    vec3 mp = mapping(P, Wv.loc, vec3(Wv.rotXY, Wv.rot), Wv.scl);
    vec3 nc = noiseColor(mp, Wv.w, Wv.scale, Wv.dist);
    float mr = mapRange(dot(nc, vec3(0.2126729, 0.7151522, 0.0721750)), Wv.range);
    float fc = facingOf(NdotV, Wv.lw);
    vec3 o = vec3(0.0);
    if (DEBUG_STAGE == 1 || DEBUG_STAGE == 6) o = nc;
    else if (DEBUG_STAGE == 2 || DEBUG_STAGE == 7) o = vec3(mr);
    else if (DEBUG_STAGE == 3 || DEBUG_STAGE == 8) o = vec3(mr + fc);
    else if (DEBUG_STAGE == 4 || DEBUG_STAGE == 9) o = vec3(rampBSpline(mr + fc, Wv.ramp));
    else if (DEBUG_STAGE == 5) o = vec3(fc);
    else if (DEBUG_STAGE == 10) o = vec3(mapRange(pow(max(facingOf(NdotV, uRimLw), 0.0), uRimPow), uRimRange));
    else if (DEBUG_STAGE == 11) o = mp;
    fragColor = vec4(o, cov); return;
  }
#endif
  float rA = waveRamp(uA, P, NdotV);
  float rB = waveRamp(uB, P, NdotV);
  float rR = mapRange(pow(max(facingOf(NdotV, uRimLw), 0.0), uRimPow), uRimRange);

  /* Shell mix tree: Mix.004( waveB, Mix.002( rim, waveA ) ). Emission mixes linearly. */
  float wB = 1.0 - uMixB, wR = uMixB * (1.0 - uMixWR), wA = uMixB * uMixWR;
  vec3 em = wB * rB * (uB.emit * uB.strength) + wR * rR * (uRimEmit * uRimStrength) + wA * rA * (uA.emit * uA.strength);
  /* EEVEE applies the translucent closure's colour*weight twice: radiance = K * world * (T * weight)^2, with
   * weight = (1 - empty-slot mix) * (translucent share of the mix tree). Measured exponents: T 2.00, weight 2.00. */
  float tB = wB * (1.0 - rB), tR = wR * (1.0 - rR), tA = wA * (1.0 - rA);
  vec3 tc = tA * uA.transl + tB * uB.transl + tR * uRimTransl;
  vec3 mix4 = em + (1.0 - uMixE) * tc * tc * base + uVeil;
  fragColor = vec4(mix4, cov) * uWeight;                 /* shell mix (1 - uMixE) is applied by the composite */
}`;

/* Bloom: highlight prefilter (Blender Glare "Bloom": soft knee on the max channel) */
const PREFILTER_FS = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTex; uniform float uThreshold, uKnee, uShell; uniform vec2 uTexel; uniform vec3 uFw;
vec3 tap(vec2 o){ vec4 m = texture(uTex, vUv + o * uTexel); return m.rgb * m.a; }
void main(){
  vec3 c = tap(vec2(0.0)) * uFw.x;                       /* premultiplied radiance, as Blender's render result */
  if (uFw.y > 0.0){                                      /* pixel filter (gaussian fitted to Blackman-Harris) */
    c += (tap(vec2(1.0, 0.0)) + tap(vec2(-1.0, 0.0)) + tap(vec2(0.0, 1.0)) + tap(vec2(0.0, -1.0))) * uFw.y;
    c += (tap(vec2(1.0, 1.0)) + tap(vec2(-1.0, 1.0)) + tap(vec2(1.0, -1.0)) + tap(vec2(-1.0, -1.0))) * uFw.z;
  }
  c *= uShell;
  float l = max(c.r, max(c.g, c.b));
  float soft = clamp(l - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-5);
  float contrib = max(soft, l - uThreshold) / max(l, 1e-5);
  fragColor = vec4(c * contrib, 1.0);
}`;

/* 13-tap downsample (Jimenez); first level uses the Karis average to tame fireflies */
const DOWN_FS = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTex; uniform vec2 uTexel; uniform bool uKaris;
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 karis(vec3 a, vec3 b, vec3 c, vec3 d){
  float wa = 1.0 / (1.0 + luma(a)), wb = 1.0 / (1.0 + luma(b)), wc = 1.0 / (1.0 + luma(c)), wd = 1.0 / (1.0 + luma(d));
  return (a*wa + b*wb + c*wc + d*wd) / (wa + wb + wc + wd);
}
void main(){
  vec3 e  = texture(uTex, vUv).rgb;
  vec3 a  = texture(uTex, vUv + uTexel * vec2(-2.0,  2.0)).rgb;
  vec3 b  = texture(uTex, vUv + uTexel * vec2( 0.0,  2.0)).rgb;
  vec3 c  = texture(uTex, vUv + uTexel * vec2( 2.0,  2.0)).rgb;
  vec3 d  = texture(uTex, vUv + uTexel * vec2(-2.0,  0.0)).rgb;
  vec3 f  = texture(uTex, vUv + uTexel * vec2( 2.0,  0.0)).rgb;
  vec3 g  = texture(uTex, vUv + uTexel * vec2(-2.0, -2.0)).rgb;
  vec3 h  = texture(uTex, vUv + uTexel * vec2( 0.0, -2.0)).rgb;
  vec3 i  = texture(uTex, vUv + uTexel * vec2( 2.0, -2.0)).rgb;
  vec3 j  = texture(uTex, vUv + uTexel * vec2(-1.0,  1.0)).rgb;
  vec3 k  = texture(uTex, vUv + uTexel * vec2( 1.0,  1.0)).rgb;
  vec3 l  = texture(uTex, vUv + uTexel * vec2(-1.0, -1.0)).rgb;
  vec3 m  = texture(uTex, vUv + uTexel * vec2( 1.0, -1.0)).rgb;
  vec3 r;
  if (uKaris){
    r = 0.5 * karis(j, k, l, m)
      + 0.125 * (karis(a, b, d, e) + karis(b, c, e, f) + karis(d, e, g, h) + karis(e, f, h, i));
  } else {
    r = e * 0.125 + (j + k + l + m) * 0.125 + (b + d + f + h) * 0.0625 + (a + c + g + i) * 0.03125;
  }
  fragColor = vec4(r, 1.0);
}`;

/* 3x3 tent upsample, offsets in OUTPUT pixels (Blender), added into the larger level by blending */
const UP_FS = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uTex; uniform vec2 uPixel; uniform float uScale;
void main(){
  vec3 s = texture(uTex, vUv).rgb * 4.0;
  s += (texture(uTex, vUv + vec2(-uPixel.x, 0.0)).rgb + texture(uTex, vUv + vec2(uPixel.x, 0.0)).rgb
      + texture(uTex, vUv + vec2(0.0, -uPixel.y)).rgb + texture(uTex, vUv + vec2(0.0, uPixel.y)).rgb) * 2.0;
  s += texture(uTex, vUv + vec2(-uPixel.x, -uPixel.y)).rgb + texture(uTex, vUv + vec2(uPixel.x, -uPixel.y)).rgb
     + texture(uTex, vUv + vec2(-uPixel.x,  uPixel.y)).rgb + texture(uTex, vUv + vec2(uPixel.x,  uPixel.y)).rgb;
  fragColor = vec4(s * (uScale / 16.0), 1.0);
}`;

/* Composite: shell mix + bloom, un-premultiply, AgX look via baked LUT, dither, premultiplied output */
const COMPOSITE_FS = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv; out vec4 fragColor;
uniform sampler2D uMat, uBloom;
uniform sampler3D uLut;
uniform vec2 uRes; uniform float uFill, uShell, uBloomK, uExposure, uLutN, uClip, uSigma, uDesat, uDim; uniform vec2 uMatTexel; uniform vec3 uFw;
uniform bool uOpaque; uniform vec3 uBg;
vec3 toSrgb(vec3 c){ c = max(c, vec3(0.0)); return mix(12.92 * c, 1.055 * pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c)); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main(){
  float m = min(uRes.x, uRes.y);
  vec2 p = (vUv * 2.0 - 1.0) * (uRes / m) / uFill;
  float px = 2.0 / (m * uFill);
  float Rc = sqrt(1.0 - uClip * uClip);
  float cov = 1.0 / (1.0 + exp(1.702 * (length(p) - Rc) / (px * uSigma)));
  if (cov < 0.002 && !uOpaque){ fragColor = vec4(0.0); return; }
  vec3 rad = texture(uMat, vUv).rgb * uFw.x;             /* radiance, edge-extended */
  if (uFw.y > 0.0){
    rad += (texture(uMat, vUv + vec2(uMatTexel.x, 0.0)).rgb + texture(uMat, vUv - vec2(uMatTexel.x, 0.0)).rgb
          + texture(uMat, vUv + vec2(0.0, uMatTexel.y)).rgb + texture(uMat, vUv - vec2(0.0, uMatTexel.y)).rgb) * uFw.y;
    rad += (texture(uMat, vUv + uMatTexel).rgb + texture(uMat, vUv - uMatTexel).rgb
          + texture(uMat, vUv + vec2(uMatTexel.x, -uMatTexel.y)).rgb + texture(uMat, vUv + vec2(-uMatTexel.x, uMatTexel.y)).rgb) * uFw.z;
  }
  rad *= uShell;
  vec3 bloom = texture(uBloom, vUv).rgb * uBloomK;
  vec3 hdr = (rad + bloom / max(cov, 1e-3)) * exp2(uExposure);
  vec3 t = clamp((log2(max(hdr, vec3(1e-6))) - ${LUT_MIN_EV.toFixed(5)}) / ${(LUT_MAX_EV - LUT_MIN_EV).toFixed(6)}, 0.0, 1.0);
  vec3 srgb = texture(uLut, t * ((uLutN - 1.0) / uLutN) + 0.5 / uLutN).rgb;
  srgb = mix(srgb, vec3(dot(srgb, vec3(0.2126, 0.7152, 0.0722))), uDesat) * uDim;
  srgb += (hash12(gl_FragCoord.xy) - 0.5) / 255.0;       /* dither, like Blender's 8-bit output */
  if (uOpaque){ fragColor = vec4(mix(toSrgb(uBg), srgb, cov), 1.0); return; }
  fragColor = vec4(srgb * cov, cov);
}`;

/* ------------------------------------------------------------------ *
 * 4. Renderer
 * ------------------------------------------------------------------ */
const ZERO2 = [0, 0];
const IDENTITY3 = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const mixv = (a, b, t) => a + (b - a) * t;
const clone = (o) => (Array.isArray(o) ? o.map(clone) : o && typeof o === "object" ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, clone(v)])) : o);

function decodeLut(lut) {
  const bin = atob(lut.data);
  const n = lut.size, out = new Uint8Array(n * n * n * 4);
  for (let i = 0, j = 0; i < n * n * n; i++) { out[j++] = bin.charCodeAt(i * 3); out[j++] = bin.charCodeAt(i * 3 + 1); out[j++] = bin.charCodeAt(i * 3 + 2); out[j++] = 255; }
  return out;
}

export class OrbRenderer {
  /**
   * @param {HTMLCanvasElement|OffscreenCanvas} canvas
   * @param {{quality?: string, background?: string|null}} [opts]
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.profile = QUALITY_PROFILES[opts.quality] || QUALITY_PROFILES.balanced;
    this.background = opts.background || null;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: true, premultipliedAlpha: true, depth: false, stencil: false });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    if (!gl.getExtension("EXT_color_buffer_float") && !gl.getExtension("EXT_color_buffer_half_float")) throw new Error("Floating-point render targets are not available");
    gl.getExtension("OES_texture_float_linear");
    this.sigma = 0.5;                      // silhouette softness in canvas pixels (Blender's film filter: 0.426)
    this.filter = 0;                       // interior pixel filter sigma in material pixels (0 = off; Blender's film filter: 0.426)
    this._debugStage = opts.debugStage | 0;
    this._init();
  }

  _shader(type, src) {
    const gl = this.gl, s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  _program(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    gl.attachShader(p, this._shader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, this._shader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace("[0]", "")] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  }

  _init() {
    const gl = this.gl;
    this.prog = {
      mat: this._program(QUAD_VS, MATERIAL_FS.replace("#define DEBUG_STAGE 0", `#define DEBUG_STAGE ${this._debugStage}`)), pre: this._program(QUAD_VS, PREFILTER_FS),
      down: this._program(QUAD_VS, DOWN_FS), up: this._program(QUAD_VS, UP_FS), comp: this._program(QUAD_VS, COMPOSITE_FS),
    };
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    // AgX Very High Contrast, baked from Blender, as a 3D LUT
    const lut = AGX_LUT;
    this.lutTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, lut.size, lut.size, lut.size, 0, gl.RGBA, gl.UNSIGNED_BYTE, decodeLut(lut));
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    for (const w of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) gl.texParameteri(gl.TEXTURE_3D, w, gl.CLAMP_TO_EDGE);
    this.lutSize = lut.size;
    this.targets = null;
  }

  _target(w, h) {
    const gl = this.gl, tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { tex, fb, w, h };
  }
  _free() {
    if (!this.targets) return;
    const gl = this.gl;
    for (const t of [this.targets.mat, ...this.targets.levels]) { gl.deleteFramebuffer(t.fb); gl.deleteTexture(t.tex); }
    this.targets = null;
  }
  /** Size of the material/bloom buffers for a canvas of w*h pixels. */
  _matSize(w, h) {
    const m = Math.max(2, Math.min(Math.min(w, h), this.profile.matPx));
    return [Math.max(2, Math.round(m * w / Math.min(w, h))), Math.max(2, Math.round(m * h / Math.min(w, h)))];
  }
  _ensure(w, h) {
    const [mw, mh] = this._matSize(w, h);
    const t = this.targets;
    if (t && t.mat.w === mw && t.mat.h === mh) return;
    this._free();
    const mat = this._target(mw, mh), levels = [];
    let lw = mw, lh = mh;
    const count = Math.max(1, Math.floor(Math.log2(Math.min(mw, mh))));
    for (let i = 0; i < count; i++) {
      levels.push(this._target(lw, lh));
      lw = Math.max(1, lw >> 1); lh = Math.max(1, lh >> 1);
    }
    this.targets = { mat, levels };
  }

  setQuality(q) {
    const p = QUALITY_PROFILES[q];
    if (p && p !== this.profile) { this.profile = p; this._free(); }
  }
  setSize(w, h) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  _quad() { const gl = this.gl; gl.bindVertexArray(this.vao); gl.drawArrays(gl.TRIANGLES, 0, 3); gl.bindVertexArray(null); }

  _wave(name, W) {
    const gl = this.gl, u = this.prog.mat.u;
    gl.uniform3fv(u[`${name}.emit`], W.emit); gl.uniform1f(u[`${name}.strength`], W.strength);
    gl.uniform1f(u[`${name}.scale`], W.scale); gl.uniform1f(u[`${name}.dist`], W.distortion);
    gl.uniform3fv(u[`${name}.loc`], W.loc); gl.uniform3fv(u[`${name}.scl`], W.scl);
    gl.uniform4fv(u[`${name}.range`], W.range); gl.uniform2fv(u[`${name}.ramp`], W.ramp);
    gl.uniform1f(u[`${name}.lw`], W.lw); gl.uniform3fv(u[`${name}.transl`], W.transl);
    const rxy = W.rotXY || ZERO2;
    gl.uniform2f(u[`${name}.rotXY`], rxy[0], rxy[1]); gl.uniform1f(u[`${name}.rot`], W.rot); gl.uniform1f(u[`${name}.w`], W.w);
  }

  /**
   * Render one frame.
   * @param {object} f  frame description: { A, B, R, mix, world:{color,strength}, fill, side, refFill, bloom, exposure, desat, dim }
   */
  render(f) {
    const gl = this.gl;
    if (gl.isContextLost()) return;
    const W = this.canvas.width, H = this.canvas.height;
    this._ensure(W, H);
    const { mat, levels } = this.targets;
    const L = levels.length, shell = 1 - f.mix.e;
    const fw = filterWeights(this.filter);

    // ---- pass 1: material ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, mat.fb);
    gl.viewport(0, 0, mat.w, mat.h);
    gl.disable(gl.BLEND);
    const M = this.prog.mat, mu = M.u;
    gl.useProgram(M.p);
    gl.uniform3fv(mu.uRimEmit, f.R.emit); gl.uniform1f(mu.uRimStrength, f.R.strength); gl.uniform1f(mu.uRimPow, f.R.pow);
    gl.uniform1f(mu.uRimLw, f.R.lw); gl.uniform4fv(mu.uRimRange, f.R.range); gl.uniform3fv(mu.uRimTransl, f.R.transl);
    gl.uniform1f(mu.uMixWR, f.mix.wr); gl.uniform1f(mu.uMixB, f.mix.b); gl.uniform1f(mu.uMixE, f.mix.e);
    // translucent lighting: world * response; the shell mix (1 - mix.e) multiplies the final colour
    const wk = K_WORLD * f.world.strength;
    gl.uniform3f(mu.uBase, f.world.color[0] * wk, f.world.color[1] * wk, f.world.color[2] * wk);
    const vk = f.mix.e / Math.max(1 - f.mix.e, 1e-4), veil = f.mix.veil || [0, 0, 0];
    gl.uniform3f(mu.uVeil, veil[0] * vk, veil[1] * vk, veil[2] * vk);
    const clip = f.clip === undefined ? (f.side > 0 ? 0.1 : 0) : f.clip;
    gl.uniform1f(mu.uClip, clip); gl.uniform1f(mu.uSigma, this.sigma);
    gl.uniform4fv(mu.uNoiseOff, NOISE_OFFSETS);
    gl.uniform2f(mu.uRes, mat.w, mat.h); gl.uniform1f(mu.uFill, f.fill); gl.uniform1f(mu.uSide, f.side);
    gl.uniformMatrix3fv(mu.uOrient, false, f.orient || IDENTITY3);
    // the pattern: usually one pass; while a state dissolves into another, one pass for each, weighted and added together
    const layers = f.layers && f.layers.length ? f.layers : null;
    if (layers) {
      layers.forEach((ly, i) => {
        if (i === 1) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE); }
        this._wave("uA", ly.A); this._wave("uB", ly.B); gl.uniform1f(mu.uWeight, ly.w);
        this._quad();
      });
      gl.disable(gl.BLEND);
    } else {
      this._wave("uA", f.A); this._wave("uB", f.B); gl.uniform1f(mu.uWeight, 1);
      this._quad();
    }

    // ---- bloom: prefilter -> downsample chain -> accumulate (Blender Glare "Bloom", Size 1) ----
    const k = f.bloomStrength === undefined ? 1 : f.bloomStrength;
    let weights = null;
    if (k > 0) {
      const sphereDia = f.fill * Math.min(mat.w, mat.h), refDia = f.refFill * REF_RES;
      weights = f.bloomWeights || bloomWeights(L, sphereDia, refDia);
      const P = this.prog.pre, D = this.prog.down, U = this.prog.up;
      gl.bindFramebuffer(gl.FRAMEBUFFER, levels[0].fb); gl.viewport(0, 0, levels[0].w, levels[0].h);
      gl.useProgram(P.p); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, mat.tex);
      gl.uniform1i(P.u.uTex, 0); gl.uniform1f(P.u.uThreshold, 0.5); gl.uniform1f(P.u.uKnee, 0.5); gl.uniform1f(P.u.uShell, shell);
      gl.uniform2f(P.u.uTexel, 1 / mat.w, 1 / mat.h); gl.uniform3fv(P.u.uFw, fw);
      this._quad();
      gl.useProgram(D.p); gl.uniform1i(D.u.uTex, 0);
      for (let i = 1; i < L; i++) {
        const src = levels[i - 1], dst = levels[i];
        gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb); gl.viewport(0, 0, dst.w, dst.h);
        gl.bindTexture(gl.TEXTURE_2D, src.tex);
        gl.uniform2f(D.u.uTexel, 1 / src.w, 1 / src.h); gl.uniform1i(D.u.uKaris, i === 1 ? 1 : 0);
        this._quad();
      }
      gl.useProgram(U.p); gl.uniform1i(U.u.uTex, 0);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
      for (let i = L - 2; i >= 0; i--) {
        const src = levels[i + 1], dst = levels[i];
        gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb); gl.viewport(0, 0, dst.w, dst.h);
        gl.bindTexture(gl.TEXTURE_2D, src.tex);
        gl.uniform2f(U.u.uPixel, 1 / dst.w, 1 / dst.h);
        gl.uniform1f(U.u.uScale, weights[i + 1] / weights[i]);       // Horner form of sum_k w_k * T^k(level_k)
        this._quad();
      }
      gl.disable(gl.BLEND);
    }

    // ---- composite to the canvas ----
    const C = this.prog.comp, cu = C.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(C.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, mat.tex);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, levels[0].tex);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_3D, this.lutTex);
    gl.uniform1i(cu.uMat, 0); gl.uniform1i(cu.uBloom, 1); gl.uniform1i(cu.uLut, 2);
    gl.uniform2f(cu.uRes, W, H); gl.uniform1f(cu.uFill, f.fill); gl.uniform1f(cu.uShell, shell);
    gl.uniform1f(cu.uBloomK, weights ? weights[0] * k : 0);
    gl.uniform1f(cu.uExposure, f.exposure || 0); gl.uniform1f(cu.uLutN, this.lutSize);
    gl.uniform1f(cu.uClip, f.clip === undefined ? (f.side > 0 ? 0.1 : 0) : f.clip); gl.uniform1f(cu.uSigma, this.sigma);
    gl.uniform2f(cu.uMatTexel, 1 / mat.w, 1 / mat.h); gl.uniform3fv(cu.uFw, fw); gl.uniform1f(cu.uDesat, f.desat || 0); gl.uniform1f(cu.uDim, f.dim === undefined ? 1 : f.dim);
    const bg = this.background ? hexToRgb(this.background) : [0, 0, 0];
    gl.uniform1i(cu.uOpaque, this.background ? 1 : 0); gl.uniform3fv(cu.uBg, bg);
    this._quad();
    gl.activeTexture(gl.TEXTURE0);
  }

  destroy() {
    this._free();
    const ext = this.gl.getExtension("WEBGL_lose_context");
    if (ext) ext.loseContext();
  }
}

/** 3x3 gaussian weights (centre, edge, corner) for pixel filter sigma `s`; s = 0 means a single tap. */
function filterWeights(s) {
  if (!(s > 0)) return new Float32Array([1, 0, 0]);
  const e = Math.exp(-0.5 / (s * s)), c = e * e, n = 1 + 4 * e + 4 * c;
  return new Float32Array([1 / n, e / n, c / n]);
}

/** Per-level weights mapping Blender's 10 equal levels at 1080px onto this buffer's chain, by relative scale. */
export function bloomWeights(levels, sphereDiameterPx, refDiameterPx) {
  const j = Math.log2(refDiameterPx / sphereDiameterPx);
  const w = new Float32Array(levels);
  for (let k = 0; k < REF_LEVELS; k++) {
    const p = clamp(k - j, 0, levels - 1), i0 = Math.floor(p), fr = p - i0;
    w[i0] += (1 - fr) / REF_LEVELS;
    if (fr > 0) w[i0 + 1] += fr / REF_LEVELS;
  }
  let s = 0;                                            // every level needs a little weight: the accumulation divides by it
  for (let i = 0; i < levels; i++) { w[i] = Math.max(w[i], 2e-3); s += w[i]; }
  for (let i = 0; i < levels; i++) w[i] /= s;
  return w;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const s2l = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  return [0, 2, 4].map((i) => s2l(parseInt(h.slice(i, i + 2), 16) / 255));
}

/* ------------------------------------------------------------------ *
 * 5. Themes
 * ------------------------------------------------------------------ */
export const DEFAULT_THEME = "all-blue";
/** The approved colours as a list: [{ id, name, group }]. */
export function listThemes() { return Object.values(THEMES).map(({ id, name, group }) => ({ id, name, group })); }

/** An approved theme by id, as a deep copy. Unknown ids fall back to the default. */
export function getTheme(id) {
  const t = clone(THEMES[id] || THEMES[DEFAULT_THEME]);
  t.mix.veil = t.mix.veil || [0, 0, 0];
  for (const k of ["A", "B"]) t[k].rotXY = t[k].rotXY || [0, 0];
  t.hero = t.hero || { rotA: 0, wA: 0, rotB: 0, wB: 0 };
  t.side = t.side || 1; t.refFill = t.refFill || 0.8264;
  return t;
}

/* ------------------------------------------------------------------ *
 * 6. Motion: the Blender state materials as velocity fields
 * ------------------------------------------------------------------ *
 * Every Blender state animates the same four numbers: the Z rotation and the W offset of the two noise
 * waves (the "drivers"). The orb keeps those four numbers as continuous state and integrates a velocity
 * per state, so changing state never jumps:
 *   Live      rot = 2*pi*f/860, W = sin(rot)               (the 860 frame loop)        -> 35.8 s period
 *   Thinking  rot = W = f/15                               (continuous drift)          -> 1.6 /s   (the orb turns at 0.4 of it, the noise drifts at 0.4 of that)
 *   Waiting   rot = W = f/80, and wave A stretched 4.5x in Z                            -> 0.3 /s
 *   Speaking  the "Talking Test": faster rotation, W drift, scale/distortion/location breathing
 * f is the frame number at 24 fps.
 */
const LIVE_RATE = (TAU * FPS) / 860;
const TALK_RATE = (TAU * FPS) / 250;
/* The Blender drivers (live, lin, talk, scl, rangeShift) are the motion of each state. On top of them every state has its own
 * character, and only inside the orb: how the pattern moves, how fast, what colour treatment it gets and how it breathes or beats.
 * Nothing is drawn around the orb and the orb never changes size (apart from the voice pulse and Complete's pop).
 *   live, lin, talk   the speed of the pattern (the Blender drivers)
 *   wDrift            how fast the noise itself drifts, as a share of lin (1 = the Blender drift). Lower is steadier: the features keep their shape
 *                     and turn, instead of boiling, which is what makes a quick pattern blink
 *   swirl             0..1  wave B turns against wave A (1 = the opposite way)
 *   fine, turb        noise scale and distortion multipliers: a finer, more turbulent pattern
 *   open              how far the glowing patches open with the voice
 *   breath            { hz, light, cover, flow }  a breath: the light swells, the glowing patches open and the flow surges, then it eases back.
 *                     A little irregular, like a person's: the rate and the depth drift from one breath to the next.
 *   beat              [amount, Hz, shape]  pulses of light for the busy states; shape 1 heartbeat, 2 syllables (Speaking also runs in pushes)
 *   boil, rimBeat     how far the patches swell and shrink, and the rim light pulses, with the beat
 *   expo, dim, sat, mute, bloom, str, rim, glow   colour: brighter or dimmer, vivid or muted, rim light
 */
const STATE_DEFS = {
  static:    { live: 0 },
  idle:      { live: 1, breath: { hz: 0.2, light: 0.08, cover: 0.025, flow: 0.25 } },
  listening: { live: 1.4, open: 0.12, fine: -0.25, turb: -0.18, rim: 1.2, expo: 0.14, str: 1.1, sat: 0.15, bloom: 1.2, breath: { hz: 0.3, light: 0.05, cover: 0.02, flow: 0.2 } },
  waiting:   { lin: (FPS / 80) * 0.8, scl: [1, 1.1, 4.5], sclB: 2.4, rangeShift: -0.03, expo: -0.16, dim: 0.8, mute: 0.35, bloom: 0.65, str: 0.82, breath: { hz: 0.14, light: 0.26, cover: 0.07, flow: 0.5 } },
  thinking:  { lin: (FPS / 15) * 0.4, wDrift: 0.4, rangeShift: -0.03, swirl: 0.6, fine: 0.06, turb: 0.05, glow: 0.08, sat: 0.22, rim: 0.3, bloom: 1.05, beat: [0.035, 0.8, 1], boil: 0.01, rimBeat: 1.0 },
  speaking:  { live: 0.25, talk: 0.75, rim: 0.45, sat: 0.1, expo: 0.04, bloom: 1.1, beat: [0.08, 2.4, 2], boil: 0.02 },
  complete:  { live: 1.6, oneShot: 1.4, then: "idle" },
  error:     { live: 0.12, gray: 1, dim: 0.6 },
};
/* the shapes of a beat, from -1 to about 1: a soft heartbeat (a slow lub-dub, wide enough to read as a swell and never as a flash) and a run of syllables */
function beatShape(shape, x) {
  const ph = x - Math.floor(x);
  if (shape === 1) { const d1 = (ph - 0.12) / 0.14, d2 = (ph - 0.42) / 0.16; return (Math.exp(-d1 * d1) + 0.6 * Math.exp(-d2 * d2) - 0.42) * 1.6; }
  if (shape === 2) return Math.sin(TAU * x) * (0.6 + 0.4 * Math.sin(TAU * x * 0.19 + 1));
  return Math.sin(TAU * x);
}
/* a breath from -1 to 1: the inhale a little shorter than the exhale, eased at the top and at the bottom */
const sstep = (u) => u * u * (3 - 2 * u);
function breathShape(x) {
  const ph = x - Math.floor(x), up = 0.4;
  return 2 * (ph < up ? sstep(ph / up) : 1 - sstep((ph - up) / (1 - up))) - 1;
}
/* How long a change of state takes, in seconds, for each part of the look. It is slow on purpose and eased in and out, so the orb
 * never jumps or lurches: the colour and light move first, then the speed, and the shape of the pattern last. */
function blendTimes(from, to) {
  if (to === "complete") return { look: 0.35, mot: 0.35, str: 0.35 };                 // a flash is an event, not a mood
  if (to === "error" || from === "error") return { look: 4, mot: 4, str: 4 };         // the colour drains out, like powering down
  // the more the two states differ, the longer it takes: into or out of Waiting (nodding off, waking up) and Thinking are the biggest changes
  const f = Math.max(from === "waiting" || to === "waiting" ? 1.25 : 1, from === "thinking" || to === "thinking" ? 1.1 : 1);
  return { look: 4 * f, mot: 5 * f, str: 7 * f };
}
const smoother = (u) => u * u * (3 - 2 * u);                                           // zero speed at both ends

/* ------------------------------------------------------------------ *
 * 7. Voice level
 * ------------------------------------------------------------------ */
/**
 * Adaptive voice level 0..1 from audio samples, for a microphone and for the agent's own speech.
 *
 *   1. DC removal and RMS to dBFS for each block.
 *   2. A noise floor that learns the room: it falls at once and rises slowly, and only while nobody is talking.
 *   3. A speech gate, relative to that floor, with a hangover so short gaps do not flicker.
 *   4. The speaker's own loudness (the reference): where their speech usually peaks. It follows a whisper or a shout within a
 *      second or two, and jumps to a new speaker at the start of an utterance. The level is where a block sits against that
 *      reference, in dB.
 *   5. A soft attack and release.
 *
 * So a quiet talker and a loud talker move the orb the same way. Speech at the speaker's usual loudness reads about 0.6,
 * a block 14 dB below it reads 0, and the loudest syllables reach about 0.95.
 */
export class VoiceLevel {
  constructor(o = {}) {
    this.floor = o.initialNoiseFloorDb ?? null;            // null: learn the room from the first moments
    this.ref = o.initialSpeechReferenceDb ?? null;         // null: take the speaker's level at their first syllable
    this.range = o.range ?? 14;                            // dB below the reference that still moves the orb
    this.headroom = o.headroom ?? 7;                       // dB above the reference that reach the ceiling
    this.gamma = o.outputGamma ?? 1.1;                     // the reference itself reads about 0.6
    this.ceiling = o.ceiling ?? 0.95;
    this.open = false; this.hang = 0; this.level = 0; this.t = null;
    this.warm = this.floor == null ? 0.4 : 0;              // seconds spent settling on the room before the gate can open
    this.silent = 0;                                       // seconds since the gate closed
    this.peak = -120;                                      // the speaker's recent peak, in dB, falling 18 dB per second
    this.onset = -1;                                       // seconds into the current utterance while it is being judged, else -1
    this.onsetPeak = -120; this.longGap = false;
  }
  process(samples, now) {
    const dt = this.t == null ? 1 / 60 : Math.min(0.1, Math.max(1e-3, now - this.t));
    this.t = now;
    let mean = 0;
    for (let i = 0; i < samples.length; i++) mean += samples[i];
    mean /= samples.length;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) { const v = samples[i] - mean; sum += v * v; }
    const db = 10 * Math.log10(sum / samples.length + 1e-12);
    const k = (tau) => 1 - Math.exp(-dt / tau);

    if (this.floor == null) this.floor = db;
    if (this.warm > 0) { this.warm -= dt; this.floor += (db - this.floor) * k(0.08); }
    else if (db < this.floor) this.floor += (db - this.floor) * k(0.12);
    else this.floor += (db - this.floor) * k(this.open ? 90 : 2.5);
    this.floor = clamp(this.floor, -110, -18);

    const wasOpen = this.open;
    if (this.warm <= 0) {
      if (db > this.floor + 10 && db > -85) { this.open = true; this.hang = 0.25; }
      else if (db < this.floor + 6) { this.hang -= dt; if (this.hang <= 0) this.open = false; }
    }

    if (this.open) {
      if (!wasOpen) { this.onset = 0; this.onsetPeak = db; this.longGap = this.silent > 3; if (this.ref == null) this.ref = db; }
      this.peak = wasOpen ? Math.max(db, this.peak - 18 * dt) : db;
      if (this.onset >= 0) {                               // the first 0.2 s of an utterance: the same speaker, or a new one at another loudness?
        this.onset += dt; this.onsetPeak = Math.max(this.onsetPeak, db);
        if (this.onset >= 0.2) { this.onset = -1; if (this.longGap || Math.abs(this.onsetPeak - this.ref) > 10) this.ref = this.onsetPeak - 1; }
      }
      // a running 85th percentile: up for blocks above it, down for blocks below
      this.ref += (db > this.ref ? 0.85 : -0.15) * (db > this.ref + 10 ? 24 : 6) * dt;
      if (this.peak < this.ref - 7) this.ref += (this.peak + 1 - this.ref) * k(0.35);    // the speaker got quieter: follow their peaks down
      this.ref = clamp(this.ref, this.floor + 12, -3);
      this.silent = 0;
    } else { this.silent += dt; this.onset = -1; }

    const x = this.open ? clamp((db - this.ref + this.range) / (this.range + this.headroom), 0, 1) : 0;
    const target = this.ceiling * Math.pow(x, this.gamma);
    this.level += (target - this.level) * (1 - Math.exp(-dt / (target > this.level ? 0.02 : 0.09)));
    return this.level;
  }
}

function runAnalyser(analyser, setLevel, calibration) {
  const buf = new Float32Array(analyser.fftSize), meter = new VoiceLevel(calibration);
  let raf = 0;
  const tick = () => { raf = requestAnimationFrame(tick); analyser.getFloatTimeDomainData(buf); setLevel(meter.process(buf, performance.now() / 1000)); };
  raf = requestAnimationFrame(tick);
  return () => { cancelAnimationFrame(raf); setLevel(0); };
}
const mediaGraphs = new WeakMap();

/* ------------------------------------------------------------------ *
 * 8. Controller
 * ------------------------------------------------------------------ */
const ARR = ["emit", "loc", "scl", "range", "ramp", "transl", "rotXY"];
function newFrame(theme) {
  const f = clone(theme);
  for (const k of ["A", "B"]) { f[k].rot = 0; f[k].w = 0; }
  return f;
}
function copyWave(dst, src) {
  for (const k of ARR) { const d = dst[k], s = src[k]; if (d && s) for (let i = 0; i < d.length; i++) d[i] = s[i]; }
  dst.strength = src.strength; dst.lw = src.lw;
  if (src.scale !== undefined) { dst.scale = src.scale; dst.distortion = src.distortion; }
}

/**
 * createAgentOrb(canvas, options) -> controller
 *
 * options
 *   theme          an approved colour id (see listThemes). Default "all-blue".
 *   state          "static" | "idle" | "listening" | "waiting" | "thinking" | "speaking" | "complete" | "error"
 *   quality        "low" | "balanced" | "high"  (render resolution and frame rate)
 *   speed          overall motion speed multiplier (1)
 *   intensity      emission strength multiplier (1)
 *   glow           bloom strength multiplier (1)
 *   exposure       exposure in stops (0)
 *   fill           orb diameter as a fraction of the canvas (0.9; leaves room for glow and pulses)
 *   seed           0 starts exactly on the approved still; any number starts elsewhere in the motion; "random"
 *   reducedMotion  true | false | "user": reduced motion shows the approved still and no movement
 *   background     null for a transparent canvas, or "#rrggbb" to composite on a colour
 *   audioResponse  { sensitivity, brightness, motion, pulse }, 0..2 each. sensitivity scales the incoming voice level first.
 *   beat           how strong the breathing and the pulses of light are, as a multiple of the default: 0 turns them off, 2 doubles them
 *   transition     how long a change of state takes, as a multiple of the default (about 4 s for colour and light, 5 s for speed, 7 s for the
 *                  pattern, eased in and out; longer for the biggest changes): 0.5 is twice as fast, 2 is twice as slow
 *   onError(err)   rendering problems (WebGL unavailable, context lost)
 *   onStateChange(state)
 */
export function createAgentOrb(canvas, options = {}) {
  const opts = {
    theme: DEFAULT_THEME, state: "idle", quality: "balanced", speed: 1, intensity: 1, glow: 1, exposure: 0, fill: 0.9,
    seed: 0, reducedMotion: "user", background: null, audioResponse: {}, transition: 1, beat: 1, ...options,
  };
  let renderer = new OrbRenderer(canvas, { quality: opts.quality, background: opts.background });
  let base = getTheme(opts.theme);
  let frame = newFrame(base);
  let from = null;                                  // previous theme during a colour transition
  let themeBlend = 1;
  const resp = { ...AUDIO_RESPONSE_DEFAULTS, ...opts.audioResponse };

  let state = AGENT_STATES.includes(opts.state) ? opts.state : "idle";
  // how much each state counts: for the colour and light (weight), the speed (wMot) and the pattern (wStr). They change at their own pace.
  const mkWeights = () => Object.fromEntries(AGENT_STATES.map((s) => [s, s === state ? 1 : 0]));
  const weight = mkWeights(), wMot = mkWeights(), wStr = mkWeights();
  let trans = null;                                 // a change of state in progress: where it started and how long each part takes
  const phase = Object.fromEntries(AGENT_STATES.map((s, i) => [s, i * 0.37]));   // where each state's breath or beat is in its cycle
  const drift = (i, t) => 1 + 0.62 * Math.sin(TAU * 0.041 * t + i * 1.7) + 0.38 * Math.sin(TAU * 0.113 * t + 2.1 + i * 0.9);   // -0..2, slow
  const depth = (i, t) => 1 + 0.25 * Math.sin(TAU * 0.027 * t + i * 1.1) + 0.12 * Math.sin(TAU * 0.071 * t + 1.7 + i);          // how deep this breath is
  let oneShotT = state === "complete" ? 0 : -1, thenState = "idle";
  const S = { rotA: 0, wA: 0, rotB: 0, wB: 0, phi: 0, phiT: 0 };       // motion state: the four Blender drivers + phases
  let audioTarget = 0, audio = 0, disconnect = null;
  let raf = 0, last = 0, lastDraw = 0, running = false, visible = true, paused = false, destroyed = false, clock = 0;

  const mq = typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : null;
  let reducedPref = opts.reducedMotion;
  const reduced = () => (reducedPref === "user" ? !!(mq && mq.matches) : !!reducedPref);

  function snapToHero(seed = opts.seed) {
    const h = base.hero || { rotA: 0, wA: 0, rotB: 0, wB: 0 };
    const sd = seed === "random" ? Math.random() * 40 : +seed || 0;
    S.rotA = h.rotA + sd * 2.399963; S.wA = h.wA + sd * 1.7; S.rotB = h.rotB + sd * 2.399963; S.wB = h.wB + sd * 1.7;
    S.phi = (S.rotA % TAU + TAU) % TAU; S.phiT = sd;
  }
  snapToHero();

  /* advance motion, weights and audio by dt seconds */
  function step(dt) {
    clock += dt;
    // state weights: each part of the look eases from where it was to the new state, slowly at the start and at the end
    if (trans) {
      trans.t += dt;
      let done = true;
      for (const [W, key] of [[weight, "look"], [wMot, "mot"], [wStr, "str"]]) {
        const u = clamp(trans.t / trans.dur[key], 0, 1), e = smoother(u);
        for (const s of AGENT_STATES) W[s] = trans.from[key][s] * (1 - e) + (s === state ? e : 0);
        if (u < 1) done = false;
      }
      if (done) trans = null;
    }
    if (oneShotT >= 0) {
      oneShotT += dt;
      if (oneShotT >= STATE_DEFS.complete.oneShot) { oneShotT = -1; opts.onStateComplete && opts.onStateComplete("complete"); api.setState(thenState); }
    }
    // ease the voice so speech breathes the orb instead of twitching it
    const heard = clamp(audioTarget * resp.sensitivity, 0, 1);
    audio += (heard - audio) * (1 - Math.exp(-dt / (heard > audio ? 0.035 : 0.14)));
    if (reduced()) return;
    // every breath and beat keeps its own phase and its rate drifts a little (about 15 % for a breath), so it never ticks like a metronome
    let flow = 0;
    AGENT_STATES.forEach((s, i) => {
      const d = STATE_DEFS[s], hz = d.breath ? d.breath.hz : d.beat ? d.beat[1] : 0;
      if (!hz) return;
      phase[s] += hz * (1 + (d.breath ? 0.15 : 0.07) * (drift(i, clock) - 1)) * dt;
      if (d.breath) flow += wMot[s] * d.breath.flow * breathShape(phase[s]) * depth(i, clock);   // the flow surges and eases with the breath
    });
    let live = 0, lin = 0, linW = 0, talk = 0, swirl = 0;
    for (const s of AGENT_STATES) { const d = STATE_DEFS[s], w = wMot[s]; live += w * (d.live || 0); lin += w * (d.lin || 0); linW += w * (d.lin || 0) * (d.wDrift ?? 1); talk += w * (d.talk || 0); swirl += w * (d.swirl || 0); }
    const a = audio * (weight.speaking * 0.9 + weight.listening * 0.55 + 0.15) * (1 - weight.error);
    const gain = opts.speed * (1 + 1.55 * a * resp.motion) * (1 + flow * opts.beat);
    // Speaking runs in pushes at the syllable rate: the pattern speeds up and eases off, which reads as rhythm without jitter
    const push = 1 + 0.8 * Math.max(0, beatShape(2, phase.speaking)) * wMot.speaking * (reduced() ? 0 : 1);
    const talkG = talk * (0.5 + 0.5 * clamp(audio * 1.2, 0, 1)) * push;
    const rl = LIVE_RATE * live * gain, wl = rl * Math.cos(S.phi);
    S.phi = (S.phi + rl * dt) % TAU;
    S.phiT += TALK_RATE * talkG * opts.speed * dt;
    S.rotA += (rl + lin * opts.speed * gain + talkG * 1.2064 * opts.speed) * dt;
    S.wA += (wl + linW * opts.speed * gain + talkG * 0.288 * opts.speed) * dt;
    S.rotB += (rl + lin * (1 - 2 * swirl) * opts.speed * gain - talkG * 0.9048 * opts.speed) * dt;
    S.wB += (wl + linW * opts.speed * gain + talkG * 0.24 * opts.speed) * dt;
    // static: glide to the approved still when it is close, otherwise hold
    if (weight.static > 0.01 && base.hero) {
      const h = base.hero, kk = Math.min(1, dt * 3.2) * weight.static;
      const dr = ((h.rotA - S.rotA + Math.PI) % TAU + TAU) % TAU - Math.PI;
      if (Math.abs(h.wA - S.wA) < 3 && Math.abs(h.wB - S.wB) < 3 && Math.abs(dr) < 3) {
        S.rotA += dr * kk; S.wA += (h.wA - S.wA) * kk; S.wB += (h.wB - S.wB) * kk;
        S.rotB += (((h.rotB - S.rotB + Math.PI) % TAU + TAU) % TAU - Math.PI) * kk;
      }
    }
  }

  /* compose the frame description the renderer needs */
  function compose() {
    const f = frame;
    const t = themeBlend;
    f.fill = opts.fill; f.side = base.side; f.refFill = base.refFill;
    for (const k of ["A", "B", "R"]) {
      const b = base[k], o = from ? from[k] : null, d = f[k];
      copyWave(d, b);
      if (o && t < 1) for (const key of ["emit", "transl"]) for (let i = 0; i < 3; i++) d[key][i] = mixv(o[key][i], b[key][i], t);
      if (o && t < 1) d.strength = mixv(o.strength, b.strength, t);
    }
    f.R.pow = base.R.pow;
    f.mix.wr = base.mix.wr; f.mix.b = base.mix.b; f.mix.e = base.mix.e;
    for (let i = 0; i < 3; i++) {
      f.mix.veil[i] = base.mix.veil[i]; f.world.color[i] = base.world.color[i];
      if (from && t < 1) { f.mix.veil[i] = mixv(from.mix.veil[i], base.mix.veil[i], t); f.world.color[i] = mixv(from.world.color[i], base.world.color[i], t); }
    }
    f.world.strength = from && t < 1 ? mixv(from.world.strength, base.world.strength, t) : base.world.strength;

    // the character of each state, blended by the state weights so that a change eases from one look to the next
    const calm = reduced() ? 0 : 1;
    let expo = 0, mute = 0, sat = 0, bloomM = 0, strM = 0, dimM = 1, open = 0, rimM = 0, glow = 0, beat = 0, boil = 0, breathRim = 0;
    AGENT_STATES.forEach((s, i) => {
      const d = STATE_DEFS[s], w = weight[s];
      expo += w * (d.expo || 0); mute += w * (d.mute || 0); sat += w * (d.sat || 0); bloomM += w * (d.bloom || 1); strM += w * (d.str || 1);
      if (s !== "error") dimM *= 1 - w * (1 - (d.dim || 1));          // Error has its own dim below
      open += w * (d.open || 0); rimM += w * (d.rim || 0); glow += w * (d.glow || 0);
      if (d.beat) { const v = beatShape(d.beat[2], phase[s]) * calm; beat += w * d.beat[0] * v; boil += w * (d.boil || 0) * v; breathRim += w * d.beat[0] * (d.rimBeat || 0) * v * opts.beat; }
      if (d.breath) { const v = breathShape(phase[s]) * depth(i, clock) * calm; beat += w * d.breath.light * v; boil += w * d.breath.cover * v; breathRim += w * d.breath.light * 2.5 * v * opts.beat; }
    });
    beat *= opts.beat; boil *= opts.beat;
    const a = audio * (0.9 * weight.speaking + 0.55 * weight.listening + 0.15) * (1 - weight.error);

    // motion drivers
    f.A.rot = S.rotA; f.A.w = S.wA; f.B.rot = S.rotB; f.B.w = S.wB;
    // the glowing patches: their size follows the beat, the breath and the voice (the same for every pattern layer below)
    const grow = boil + open * clamp(audio * resp.sensitivity, 0, 1) * (1 - weight.error);
    // brightness, colour and the voice
    let pulse = 0, flash = 0;
    if (oneShotT >= 0) { const u = oneShotT / STATE_DEFS.complete.oneShot; flash = Math.sin(Math.PI * Math.min(1, u * 1.6)) * (1 - u); pulse = Math.sin(Math.PI * Math.min(1, u * 1.25)) * Math.exp(-2.5 * u); }
    const inten = opts.intensity * strM * (1 + beat + 0.38 * a * resp.brightness + glow + flash * 0.5);
    f.A.strength *= inten; f.B.strength *= inten;
    f.R.strength *= opts.intensity * (1 + rimM + breathRim + 1.1 * a * resp.pulse + flash);
    f.bloomStrength = opts.glow * bloomM * (1 + 0.44 * a * resp.brightness + flash * 1.2 + beat * 0.8);
    f.exposure = opts.exposure + expo + 0.15 * a * resp.brightness + flash * 0.4;
    // error: fully grey and dimmed, like a disabled control; the voice no longer reaches it
    const we = weight.error;
    f.desat = clamp(we * STATE_DEFS.error.gray + mute - sat * (1 - we), -0.6, 1);   // below zero the colours get more vivid
    f.dim = (1 - we * (1 - STATE_DEFS.error.dim)) * dimM;
    f.fill = opts.fill * (1 + 0.038 * a * resp.pulse + 0.055 * pulse);

    // The shape of the pattern is never morphed from one state's to the next: stretching, zooming and warping the noise makes its features
    // slide around and the orb looks as if it has lost its mind. Each state that still counts keeps its own whole pattern (the same drivers,
    // so the patterns move together) and the renderer dissolves between them over the same slow ease.
    const layers = f.layers || (f.layers = []);
    layers.length = 0;
    let wsum = 0;
    for (const s of AGENT_STATES) if (wStr[s] > 0.002) wsum += wStr[s];
    for (const s of AGENT_STATES) {
      if (wStr[s] <= 0.002) continue;
      layers.push({ w: wStr[s] / wsum, A: patternOf("A", s, f, grow), B: patternOf("B", s, f, grow) });
    }
    layers.sort((x, y) => y.w - x.w);
    return f;
  }

  /* wave k as state s draws it: the colours and strength of the frame, the shape of the pattern of the state */
  function patternOf(k, s, f, grow) {
    const d = STATE_DEFS[s], b = base[k], W = { ...f[k] };
    const fine = d.fine || 0, turb = d.turb || 0, A = k === "A", sgn = A ? 1 : -1;
    W.scl = b.scl.slice(); W.loc = b.loc.slice(); W.range = b.range.slice();
    if (s === "waiting") { if (A) { W.scl[1] = d.scl[1]; W.scl[2] = d.scl[2]; } else W.scl[2] = d.sclB; }   // Waiting stretches the waves along Z into bands (Blender: 1, 1.1, 4.5)
    W.range[0] = A ? b.range[0] + (d.rangeShift || 0) + grow : b.range[0] - grow * 0.7;                    // wave A maps high noise to dark, wave B the other way round
    W.scale = b.scale * (1 + fine); W.distortion = b.distortion * (1 + turb);
    if (s === "speaking") {                                       // the Talking Test: scale, distortion and location breathe with the voice
      const sp = 0.6 + 0.4 * clamp(audio * 1.2, 0, 1), p = S.phiT;
      W.scale += sgn * 0.1 * Math.sin(p) * sp;
      W.distortion += A ? 0.5 * Math.sin(p) * sp : 0.45 * Math.sin(p + 1.0472) * sp;
      W.loc[1] += A ? (1.5 * Math.sin(p) + 0.5 * Math.sin(2 * p)) * sp : (1.2 * Math.sin(p + 2.0944) + 0.4 * Math.sin(2 * p)) * sp;
    }
    return W;
  }

  function measure() {
    const prof = renderer.profile;
    const dpr = Math.min((typeof devicePixelRatio === "number" && devicePixelRatio) || 1, prof.dpr);
    /* @wonderful-patch:start — the TrueMoney Care OS stage is a CSS scale() of a 1920×1080 canvas: size the buffer
     * from the box on screen (getBoundingClientRect includes the transform), not the layout box, so the orb stays
     * sharp at 2560×1440 and cheaper at 1280×720. */
    const box = canvas.getBoundingClientRect();
    let w = (box.width || canvas.clientWidth || 1) * dpr, h = (box.height || canvas.clientHeight || 1) * dpr;
    /* @wonderful-patch:end */
    const m = Math.max(w, h);
    if (m > prof.maxPx) { w *= prof.maxPx / m; h *= prof.maxPx / m; }
    renderer.setSize(w, h);
  }

  function draw() { try { renderer.render(compose()); } catch (err) { opts.onError && opts.onError(err); } }

  function loop(now) {
    raf = 0;
    if (!running) return;
    schedule();
    const gap = 1000 / renderer.profile.fps - 2;
    if (last && now - last < gap) return;
    const dt = last ? Math.min(0.1, (now - last) / 1000) : 1 / 60;
    last = now;
    if (themeBlend < 1) { themeBlend = Math.min(1, themeBlend + dt / 0.6); if (themeBlend >= 1) from = null; }
    step(dt);
    draw();
  }
  function schedule() { if (!raf && running) raf = requestAnimationFrame(loop); }
  function update() {
    const should = !destroyed && !paused && visible && !(typeof document !== "undefined" && document.hidden);
    if (should && !running) { running = true; last = 0; schedule(); }
    if (!should && running) { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
  }

  const ro = new ResizeObserver(() => { measure(); if (!running && !destroyed) draw(); });
  ro.observe(canvas);
  const io = new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; update(); });
  io.observe(canvas);
  const onVis = () => update();
  document.addEventListener("visibilitychange", onVis);
  const onLost = (e) => { e.preventDefault(); running = false; };
  const onRestored = () => {
    try { renderer = new OrbRenderer(canvas, { quality: opts.quality, background: opts.background }); measure(); draw(); }
    catch (err) { opts.onError && opts.onError(err); }
    update();
  };
  canvas.addEventListener("webglcontextlost", onLost);
  canvas.addEventListener("webglcontextrestored", onRestored);
  measure();
  draw();
  update();

  const api = {
    get state() { return state; },
    get theme() { return base; },
    get audioLevel() { return audio; },
    get options() { return { ...opts }; },
    /** Change state. Smooth by default; "complete" plays once (1.4 s) and then returns to `then` (default "idle"). */
    setState(next, o = {}) {
      if (!AGENT_STATES.includes(next)) throw new Error(`Unknown state "${next}". Use one of: ${AGENT_STATES.join(", ")}`);
      if (next === state && next !== "complete") return;
      const bt = blendTimes(state, next), m = reduced() ? 0 : Math.max(0, +opts.transition);
      const dur = { look: Math.max(0.001, bt.look * m), mot: Math.max(0.001, bt.mot * m), str: Math.max(0.001, bt.str * m) };
      trans = { t: 0, dur, from: { look: { ...weight }, mot: { ...wMot }, str: { ...wStr } } };
      thenState = o.then || STATE_DEFS[next].then || "idle";
      state = next;
      oneShotT = next === "complete" ? 0 : -1;
      opts.onStateChange && opts.onStateChange(state);
      if (!running) { step(0); draw(); }
    },
    /** Switch colour (an approved id). Colours blend over 0.6 s; use snapshot() for a cross-dissolve. */
    setTheme(theme, o = {}) {
      const next = getTheme(theme);
      const hardCut = o.transition === false || next.side !== base.side || reduced();
      from = hardCut ? null : base; themeBlend = hardCut ? 1 : 0;
      base = next; frame = newFrame(base);
      if (!running) draw();
    },
    /** Change speed, intensity, glow, exposure, fill, audioResponse, quality or reducedMotion. */
    setOptions(o) {
      const { quality, reducedMotion, audioResponse, background, ...rest } = o;
      Object.assign(opts, rest);
      if (audioResponse) Object.assign(resp, audioResponse);
      if (quality && quality !== opts.quality) { opts.quality = quality; renderer.setQuality(quality); measure(); }
      if (reducedMotion !== undefined) reducedPref = reducedMotion;
      if (background !== undefined) { opts.background = background; renderer.background = background; }
      if (!running) draw();
    },
    /** Jump back to the approved still of the current theme. */
    snapToHero(seed) { snapToHero(seed); if (!running) draw(); },
    /* @wonderful-patch:start — not in the design pack; upstream to apps/agent-orb before the next pack refresh. */
    /** Where the motion is: the four drivers, their phases and every breath/beat phase. Plain data, safe to keep after destroy(). */
    getMotion() { return { S: { ...S }, phase: { ...phase }, clock }; },
    /** Continue from a getMotion() result, so a remounted orb picks up exactly where the last one stopped. */
    setMotion(m) {
      if (!m) return;
      Object.assign(S, m.S); Object.assign(phase, m.phase); clock = +m.clock || 0;
      if (!running) draw();
    },
    /* @wonderful-patch:end */
    /** Manual voice activity 0..1 (use when you already have a level, e.g. from your own analyser). */
    setAudioLevel(v) { audioTarget = clamp(+v || 0, 0, 1); },
    /** Use the microphone. Resolves to a function that stops it. Needs HTTPS (or localhost) and a user gesture. */
    async connectMicrophone({ constraints, calibration } = {}) {
      api.disconnectAudio();
      const stream = await navigator.mediaDevices.getUserMedia(constraints || { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 } });
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 80;
      const an = ctx.createAnalyser(); an.fftSize = 1024; an.smoothingTimeConstant = 0;
      src.connect(hp); hp.connect(an);
      const stop = runAnalyser(an, (v) => { audioTarget = v; }, calibration);
      disconnect = () => { stop(); stream.getTracks().forEach((tr) => tr.stop()); ctx.close(); disconnect = null; };
      return () => disconnect && disconnect();
    },
    /** Drive the orb from an <audio>/<video> element (the agent's own voice) or an AnalyserNode. */
    attachAudioSource(source, { calibration } = {}) {
      api.disconnectAudio();
      let an;
      if (typeof AnalyserNode !== "undefined" && source instanceof AnalyserNode) an = source;
      else {
        let g = mediaGraphs.get(source);
        if (!g) {
          const ctx = new AudioContext();
          const ms = ctx.createMediaElementSource(source);
          const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 80;
          const a = ctx.createAnalyser(); a.fftSize = 1024; a.smoothingTimeConstant = 0;
          ms.connect(ctx.destination); ms.connect(hp); hp.connect(a);
          g = { ctx, an: a }; mediaGraphs.set(source, g);
        }
        g.ctx.resume();
        an = g.an;
      }
      const stop = runAnalyser(an, (v) => { audioTarget = v; }, calibration);
      disconnect = () => { stop(); disconnect = null; };
      return () => disconnect && disconnect();
    },
    releaseAudioSource(el) { const g = mediaGraphs.get(el); if (g) { g.ctx.close(); mediaGraphs.delete(el); } },
    disconnectAudio() { if (disconnect) disconnect(); audioTarget = 0; },
    /** Advance the animation by `dt` seconds and draw one frame. For tests and frame-by-frame capture; the loop calls the same step. */
    advance(dt = 1 / 30) {
      if (themeBlend < 1) { themeBlend = Math.min(1, themeBlend + dt / 0.6); if (themeBlend >= 1) from = null; }
      step(dt); draw();
    },
    /** Render now and return the canvas content as an ImageBitmap-able snapshot (for cross-dissolves). */
    snapshot() { draw(); return canvas; },
    resize: measure,
    pause() { paused = true; update(); },
    resume() { paused = false; update(); },
    destroy() {
      destroyed = true; api.disconnectAudio(); update();
      ro.disconnect(); io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      renderer.destroy();
    },
  };
  return api;
}

/* ------------------------------------------------------------------ *
 * 9. <wonderful-orb>
 * ------------------------------------------------------------------ */
/* @wonderful-patch: the pack's <wonderful-orb> custom element (registerElement) is removed. @wonderful/ui renders
 * the orb from React on its own canvas (SphereCanvasWebGL), never as a custom element, and the element built its
 * shadow DOM from an innerHTML template. Re-add it from the design pack if a non-React host ever needs it. */

/* ------------------------------------------------------------------ *
 * 10. Generated data: themes + AgX LUT (tools/build_pack.py rewrites this block)
 * ------------------------------------------------------------------ */
/* @generated:start */
export const THEMES = {
  "all-blue": {"id":"all-blue","name":"All Blue","group":"Blue","source":"Live/AllBlue.blend @ frame 94","refFill":0.82645,"side":1,"hero":{"rotA":-0.229997,"wA":0.238835,"rotB":-0.321835,"wB":-0.079765},"A":{"emit":[0.174103,0.369936,1.0],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.68,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.011498,0.167774,0.610036],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.0,0.051421,0.241519],"strength":3.1}},
  "crystal": {"id":"crystal","name":"Crystal","group":"Blue","source":"Live/Final - Crystal.blend @ frame 98","refFill":0.82645,"side":1,"hero":{"rotA":-0.252805,"wA":0.214546,"rotB":-0.312205,"wB":-0.111663},"A":{"emit":[0.174103,0.369936,1.0],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.68,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.037443,0.610014,0.437207],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.0,0.130941,0.665684],"strength":3.1}},
  "pink": {"id":"pink","name":"Pink","group":"Pink","source":"Live/Archive/Pink_Live.blend @ frame 125","refFill":0.8642,"side":1,"hero":{"rotA":0.913254,"wA":0.791497,"rotB":0.913254,"wB":0.791497},"A":{"emit":[1.0,0.0,0.067599],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[1.0,0.174021,0.681559],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.913088,0.070567,0.245177],"strength":3.1}},
  "purple": {"id":"purple","name":"Purple","group":"Purple","source":"Live/Archive/try.blend @ frame 104","refFill":0.8642,"side":1,"hero":{"rotA":4.356342,"wA":-0.937282,"rotB":4.356342,"wB":-0.937282},"A":{"emit":[0.398548,0.0,1.0],"strength":7.0,"scale":-0.18,"distortion":-5.0,"loc":[0.0,0.0,0.0],"rotXY":[0.3,0.5],"scl":[0.7,0.7,1.0],"range":[0.62,0.18,0.0,1.0],"ramp":[0.3,0.7],"lw":0.04,"transl":[1.0,1.0,1.0]},"B":{"emit":[0.6,0.0,1.0],"strength":7.0,"scale":0.22,"distortion":5.0,"loc":[0.0,0.0,0.0],"rotXY":[-0.4,0.3],"scl":[0.45,0.45,1.0],"range":[0.45,0.72,0.0,1.0],"ramp":[0.3,0.7],"lw":0.04,"transl":[1.0,1.0,1.0]},"R":{"emit":[0.768501,0.589649,0.899993],"strength":7.0,"pow":0.7,"range":[0.05,0.55,0.0,1.0],"lw":0.04,"transl":[1.0,1.0,1.0]},"mix":{"wr":0.5,"b":0.5,"e":0.5,"veil":[0.0,0.0,0.0]},"world":{"color":[0.397924,0.349738,1.0],"strength":0.5}},
  "blacky": {"id":"blacky","name":"Blacky","group":"Neutral","source":"Live/Archive/Gray_live.blend @ frame 50","refFill":0.8642,"side":-1,"hero":{"rotA":0.365301,"wA":0.357231,"rotB":0.365301,"wB":0.357231},"A":{"emit":[0.475762,0.318775,0.577773],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.324358,0.324358,0.324358],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.8,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.625,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.110956,0.110956,0.110956],"strength":3.1}},
  "green": {"id":"green","name":"Green","group":"Green","source":"Live/Archive 2/GreenTree_ive.blend @ frame 207","refFill":0.8642,"side":-1,"hero":{"rotA":2.167699,"wA":0.827081,"rotB":2.167699,"wB":0.827081},"A":{"emit":[0.151115,0.577781,0.064421],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.225881,1.0,0.552518],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.0,0.238214,0.119039],"strength":3.1}},
  "aurora": {"id":"aurora","name":"Aurora","group":"Green","source":"Live/Archive/Bdolach live_Live.blend @ frame 57","refFill":0.8642,"side":1,"hero":{"rotA":0.416444,"wA":0.404511,"rotB":0.416444,"wB":0.404511},"A":{"emit":[0.082741,1.0,0.243012],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.418,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.0,0.015724,0.653149],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.368,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.198726,0.0,0.44797]},"world":{"color":[0.123156,0.45,0.488657],"strength":0.0}},
  "ember": {"id":"ember","name":"Ember","group":"Warm","source":"Live/Archive/Roaring Earth1_Live.blend @ frame 0","refFill":0.8642,"side":1,"hero":{"rotA":0.0,"wA":0.0,"rotB":0.0,"wB":0.0},"A":{"emit":[0.462077,0.332452,1.0],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.973445,0.181164,0.028426],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.175293,0.075281,0.0],"strength":3.1}},
  "berry": {"id":"berry","name":"Berry","group":"Pink","source":"Live/Archive/Roaring Earth2_Live.blend @ frame 112","refFill":0.8642,"side":1,"hero":{"rotA":0.818275,"wA":0.729968,"rotB":0.818275,"wB":0.729968},"A":{"emit":[0.462077,0.332452,1.0],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.973389,0.0,0.311756],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.175293,0.075281,0.0],"strength":3.1}},
  "yellow": {"id":"yellow","name":"Yellow","group":"Warm","source":"Live/Archive/Yellow_Live.blend @ frame 720","refFill":0.8642,"side":1,"hero":{"rotA":5.260341,"wA":-0.853593,"rotB":5.260341,"wB":-0.853593},"A":{"emit":[1.0,0.522511,0.0],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.85,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[1.0,0.522494,0.0],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.374344,0.098451,0.004176],"strength":3.1}},
  "roaring-earth": {"id":"roaring-earth","name":"Roaring Earth","group":"Warm","source":"Waiting/Roaring Earth_Waiting.blend @ frame 1 (Live)","refFill":0.8642,"side":1,"hero":{"rotA":2.241667,"wA":2.241667,"rotB":2.241667,"wB":2.241667},"A":{"emit":[0.462077,0.332452,1.0],"strength":8.0,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.973445,0.181164,0.028426],"strength":8.0,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.584074,0.651527,1.0],"strength":8.0,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.45,"veil":[0.0,0.0,0.0]},"world":{"color":[0.283149,0.035601,0.0185],"strength":2.0}},
  "sunset-red": {"id":"sunset-red","name":"Sunset Red","group":"Warm","source":"Thinking/Archive/Sunset red_Thinking.blend @ frame 17 (colours fitted to the approved still; its exact state is not saved in any .blend)","refFill":0.8642,"side":1,"hero":{"rotA":1.16843,"wA":1.16541,"rotB":1.16662,"wB":1.16675},"A":{"emit":[0.9991,0.4559,0.0],"strength":7.015,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,0.0,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[1.0,0.2523,0.2631],"strength":5.126,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.3335,0.6735,0.8128],"strength":9.834,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.42936,"veil":[0.0,0.0,0.0]},"world":{"color":[0.89086,0.09362,0.03013],"strength":2.61818}},
  "avatar": {"id":"avatar","name":"Avatar","group":"Pastel","source":"Live/Final - Sunrise Blue_Live.blend @ frame 190 (colours fitted to the approved still; its exact state is not saved in any .blend)","refFill":0.9901,"side":1,"hero":{"rotA":-2.74541,"wA":3.14713,"rotB":-2.60507,"wB":-0.92282},"A":{"emit":[0.3737,0.3621,1.0],"strength":4.394,"scale":-0.24445,"distortion":-4.99252,"loc":[-0.44465,2.41692,0.00491],"rotXY":[0.41439,-0.45992],"scl":[1.0,1.8,1.0],"range":[0.68,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.6884,0.2572,0.4021],"strength":4.116,"scale":0.42363,"distortion":3.36611,"loc":[-0.13979,0.37911,0.21657],"rotXY":[0.13394,-0.4614],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[1.0,0.4485,0.8352],"strength":49.318,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.42631,"veil":[0.0,0.0,0.0]},"world":{"color":[0.04036,0.25176,1.03465],"strength":2.52101}},
  "moon-blue": {"id":"moon-blue","name":"Moon Blue","group":"Blue","source":"Thinking/Archive/Roaring Earth_Thinking.blend @ frame 181 (colours fitted to the approved still; its exact state is not saved in any .blend)","refFill":0.8642,"side":1,"hero":{"rotA":12.08539,"wA":12.08027,"rotB":12.08246,"wB":12.08356},"A":{"emit":[0.2099,0.2757,0.9729],"strength":12.874,"scale":-0.3,"distortion":-3.3,"loc":[0.0,2.4,0.0],"rotXY":[0.0,0.0],"scl":[1.0,1.8,1.0],"range":[0.65,0.02,-0.1,1.8],"ramp":[0.572545,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"B":{"emit":[0.7361,0.1367,0.0203],"strength":6.331,"scale":0.5,"distortion":3.0,"loc":[0.0,0.0,0.0],"rotXY":[0.0,0.0],"scl":[0.5,-0.9,1.0],"range":[0.35,0.72,-0.2,1.0],"ramp":[0.513455,0.945455],"lw":0.08,"transl":[0.8,0.8,0.8]},"R":{"emit":[0.5832,0.6403,1.0],"strength":12.694,"pow":0.7,"range":[0.1,5.6,0.0,0.5],"lw":0.08,"transl":[0.8,0.8,0.8]},"mix":{"wr":0.5,"b":0.5,"e":0.48816,"veil":[0.0,0.0,0.0]},"world":{"color":[0.07352,0.29481,0.38678],"strength":3.57645}},
};
const AGX_LUT = { size: 33, data: "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w/22w/22w/22w/22xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//V+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///7///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAEgAAIpwALzgEF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAEgQAIpwALzgEF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAEgQAIpwALzgAE7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgAAMgABRgACXwAEgQAIpwALzgAE7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFwAAIgAAMgABRgACYAAEgQAHpwAKzgAE7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADgAAFwAAIwAAMgABRgACYAADgQAHpwAKzgAE7C8n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwAAIwAAMgAARgABYAADgQAGqAAIzgEC7C8n/l9W/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAAACAAADgAAFwAAIwAAMwAARwAAYQACggAFqAAHzgEC7DAn/l9W/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAAACAAADwEAGAEAJAEAMwAARwAAYQABggAEqAAFzwIB7DAn/mBW/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAEACQEAEAEAGQEAJAIANAIASAEAYgAAgwABqQADzwQA7DEn/mBW/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAQEAAQEAAwEABQEACgEAEQIAGgIAJQMANQMASQMAYwIAhAEAqgEAzwgA7DMn/mFW/4Z+/6Se/7y3/8/L/9/b/+vo//TyAAEAAAEAAAEAAAEAAAEAAAEAAAEAAAEAAAEAAQEAAQEAAgIAAwIABgIACwMAEgMAGwQAJgQANgUASgYAZAYAhQQAqwYA0BAA7DUn/mJW/4d+/6We/7y3/8/L/9/b/+vo//TyAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAQMAAQMAAgMABAQABwQADAUAEwUAHAYAKAcANwkASwoAZQsAhgwAqw0A0BcA7Dko/WNX/4h+/6We/723/9DL/9/b/+vo//TyAAYAAAYAAAYAAAYAAAYAAAYAAAYAAAYAAAYAAQYAAgYAAwcABQcACQgADggAFQkAHgsAKgwAOQ4ATRAAZxIAhxMArBYA0R4A7Dwp/WVX/4l//6ae/723/9DL/9/b/+vo//TyAAwAAAwAAAwAAAwAAAwAAAwAAAwAAAwAAQwAAQwAAw0ABA0ABw0ACw4AEA8AFxAAIBEALBMAOxUATxcAaRkAiRwArh8A0SYA7UEq/WhY/4p//6ae/724/9DL/9/b/+vo//TyABQAABQAABQAABQAABQAABQAABQAABQAARUAARUAAxUABRUACBYADRYAExcAGhgAIxkALxsAPh0AUh8AbCIAjCUArykA0jAA7Ugr/WtZ/4yA/6if/764/9DM/9/b/+vo//XyAB8BAB8BAB8AAB8AAB8AAB8AAB8AAB8AAR8AASAAAyAABiAACiAADyEAFSIAHSIAJiQAMiUAQigAVisAcC4AjzEAsjUA0zwA7VEt/HBb/4+B/6mf/7+4/9HM/9/c/+vo//XyAC0BAC0BAC0BAC0BAC4BAC4BAC4AAS4AAS4AAS4AAy4ABS4ACy4AES8AGDAAITAAKjIANzMARzYAWzkAdTwAlEAAtEQA1EoA7Fww/Hdd/5KC/6ug/8C4/9HM/+Dc/+vo//XyAEADAEADAEACAEACAUACAUACAUABAUABAUAAAkEAA0EABUEACkEAEkIAG0IAJEMAL0QAPEUATEgAYUsAe04AmFIAt1UA1VoA7Gg0+39g/5eE/66h/8G5/9LM/+Dc/+zo//XzAVgFAVgFAVgEAVgEAVgEAVgEAVgDAVkDAVkCAlkBA1kABlkAC1kAEloAHVoAKFsANFwAQl0AU18AaGIAgWUAnWgAumoA1m4D7Hg6+4lk/52G/7Kj/8S6/9PN/+Hc/+zp//XzAnYIAnYIAnYHAnYHAnYHAnYGA3cGA3cFA3cEBHcDBXcBCHcADncAFXgAH3gAK3kAOnoASHsAWnwAb34Ah4AAoYEAvIIA1oMO6olD+ZVq/6WK/7al/8e7/9XO/+Ld/+3p//bzCJcKCJcKCJcKCJcKCZcJCZcJCpcICpcHDJcGDZcFD5cDE5gCGJgAIJgAKJgAMpgAQJkAUJkAYZoAdZsAjJwApZ0AvpsA1poi6JxO+KRx/6+O/72o/8q9/9fP/+Pd/+3p//bzG7QCG7QCG7QCG7QCHLQCHLQCHbQBHrQBH7QBIbQBI7QAJ7QALLQAMrQAO7QAQ7QATLQAWbUAabUAerUAjrUApbYAvrUK1LI65rFd9bR6/ruV/8Sr/8+//9rQ/+Xe/+7q//fzPswCPswCPswCPswCPswCPswCP8wCP8wCQMwCQcwCQswCRMwCR8wCSswCT8wDVMwFW8wGZcwHcMwKf8wOkMsgpMsyu8pE0slW48Vu8sWG/sic/82w/9XC/97S/+ff//Dq//j0a95La95La95La95La95La95La95La95LbN5LbN5Lbd5Lbt5Mb95McN5Mct5Mdd5Ned5Oft1Pht1Rj91Tm9xXqtxhu9tqzdt24dmC79aT+9Wl/9i2/93G/+PU/+vg//Lr//n0kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6ket6ket6kut6kut6k+t7let7l+t7mut8nup9o+p+quqAs+qDv+mJzeiQ3eeZ7eai+eSw/+O9/+XL/+nX/+7i//Ts//r1rPWcrPWcrPWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrvWcrvWdr/WdsPWdsvWdtPWet/Wfu/WgwPShx/Sj0fOo3POt6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//V+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///7///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAEgAAIpwALzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAEgQAIpwALzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMQABRQACXwAEgQAIpwALzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMgABRgACXwAEgQAIpwALzgAE7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFwAAIgAAMgABRgACYAAEgQAHpwAKzgAE7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADgAAFwAAIwAAMgABRgACYAAEgQAHpwAKzgAE7C8n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwAAIwAAMgAARgABYAADgQAGqAAJzgEC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/+vg//Lr//n0kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6ket6ket6kut6kut6k+t7let7l+t7mut8nup9o+p+quqAs+qDv+mJzeiQ3eeZ7eai+eSw/+O9/+XL/+nX/+7i//Ts//r1rPWcrPWcrPWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrvWcrvWdr/WdsPWdsvWdtPWet/Wfu/WgwPShx/Sj0fOo3POt6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///7///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAFgAAIpwALzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQACXwAFgQAIpwALzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMQABRQACXwAEgQAIpwALzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMgABRgACXwAEgQAIpwALzgAF7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFwAAIgAAMgABRgACYAAEgQAIpwALzgAE7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwAAIwAAMgABRgACYAAEgQAHpwAKzgAE7C4n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwAAIwAAMgAARgABYAADgQAGqAAJzgED7C8n/l9W/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAAACAAADgAAGAAAIwAAMwAARwABYQADggAFqAAIzgEC7DAn/l9W/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAAACAAADwEAGAEAJAEAMwAARwAAYQABggAEqAAGzwIB7DAn/mBW/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABQEACQEAEAEAGQEAJAEANAEASAEAYgAAgwABqQAEzwQA7DEn/mBW/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAQEAAQEAAwEABQEACgEAEQIAGgIAJQMANQMASQMAYwEAhAAAqgEBzwgA7DMn/mFW/4Z+/6Se/7y3/8/L/9/b/+vo//TyAAEAAAEAAAEAAAEAAAEAAAEAAAEAAAEAAAEAAQEAAQEAAgIAAwIABgIACwMAEgMAGwQAJgQANgUASgYAZAYAhQQAqwUA0BAA7DUn/mJW/4d+/6We/7y3/8/L/9/b/+vo//TyAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAQMAAQMAAgMABAQABwQADAUAEwUAHAYAKAcANwkASwoAZQsAhgsAqw0A0BYA7Dko/WNX/4h+/6We/723/9DL/9/b/+vo//TyAAYAAAYAAAYAAAYAAAYAAAYAAAYAAAYAAAYAAQYAAgYAAwcABQcACQgADggAFQkAHgsAKgwAOQ4ATRAAZxEAhxMArBYA0R0A7Twp/WVX/4l//6ae/723/9DL/9/b/+vo//TyAAwAAAwAAAwAAAwAAAwAAAwAAAwAAAwAAQwAAQwAAg0ABA0ABg0ACw4AEA4AFxAAIBEALBMAOxQATxcAaRkAiRwArh8A0SYA7UEq/WhY/4p//6ae/724/9DL/9/b/+vo//TyABQAABQAABQAABQAABQAABQAABQAABUAARUAARUAAxUABRUACBYADRYAExcAGhgAIxkALxsAPh0AUh8AbCIAjCUAsCkA0jAA7Ugr/WtZ/4yA/6ef/764/9DM/9/b/+vo//XyAB8BAB8BAB8BAB8AAB8AAB8AAB8AAB8AAR8AASAAAyAABSAACiAADyEAFSIAHSIAJiQAMiUAQigAVioAcC4AkDEAsjUA0zwA7VEt/HBb/4+B/6mf/7+4/9HM/9/c/+vo//XyAC0BAC0BAC0BAC0BAC4BAC4BAC4AAS4AAS4AAS4AAi4ABS4ACi8AES8AGDAAIDAAKjIANzMARzYAWzkAdTwAlEAAtEQA1EoA7Fww/Hdd/5KC/6ug/8C5/9HM/+Dc/+vo//XyAEADAEADAEACAEACAUACAUACAUABAUABAUABAUEAA0EABUEACUEAEkIAG0IAJEMAL0QAPEUATEgAYUsAe04AmFIAt1UA1VoA7Gg0+39g/5eE/66h/8G5/9LM/+Dc/+zo//XzAVgFAVgFAVgFAVgEAVgEAVgEAVgDAVkDAlkCAlkBA1kABVkAClkAEloAHFoAKFsANFwAQl0AU18AZ2IAgWUAnWgAumoA1m0D7Hc6+4lk/52G/7Kj/8S6/9PN/+Hc/+zp//XzAnYIAnYIAnYHAnYHAnYHAnYGA3cGA3cFA3cEBHcDBXcBB3cADXgAFXgAHngAK3kAOnoASHsAWnwAb34Ah4AAoYEAvIIA1oMO6olD+ZVq/6WK/7al/8e7/9XO/+Ld/+3p//bzCJcKCJcKCJcKCZcKCZcKCZcJCpcICpcIDJcGDZcFD5cDE5gCGJgAIJgAKJgAMpgAQJkAUJkAYZoAdZsAjJwApZ0AvpsA1poi6JxO+KRx/6+O/72o/8q9/9fP/+Pd/+3p//bzGrQCG7QCG7QCG7QCG7QCHLQCHLQCHbQBH7QBILQBI7QBJ7QALLQAMrQAO7QAQ7QATLQAWbUAabUAerUAjrUApbYAvrUK1LI65rFd9bR6/ruV/8Sr/8+//9rQ/+Xe/+7q//fzPswCPswCPswCPswCPswCPswCP8wCP8wCQMwCQcwCQswCRMwCRswCSswDTswDVMwFW8wGZcwIcMwKf8wOkMsgpMsyu8pE0slW48Vu8sWG/sic/82w/9XC/97S/+ff//Dq//j0a95La95La95La95La95La95La95La95LbN5LbN5Lbd5Mbt5Mb95McN5Mct5Mdd5Ned5Oft1Qht1Rj91Tm9xXqtxhu9tqzdt24dmC79aT+9Wl/9i2/93G/+PU/+vg//Lr//n0kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6ket6ket6kut6kut7k+t7let7l+t7mut8nup9o+p+quqAs+qDv+mKzeiQ3eeZ7eai+eSw/+O9/+XL/+nX/+7i//Ts//r1rPWcrPWcrPWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrvWcrvWdr/WdsPWdsvWdtPWet/Wfu/WgwPShx/Sj0fOo3POt6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/23xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///7///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABgAADQAAFgAAIgABMQABRQADXwAFgAAJpwAMzgAF7C0m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMQABRQADXwAFgQAIpwAMzgAF7C4m/l9V/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMQABRQACXwAFgQAIpwAMzgAF7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgAAIgABMgABRgACXwAEgQAIpwALzgAF7C4m/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFwAAIgABMgABRgACYAAEgQAIpwALzgAF7C4n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwAAIwAAMgABRgACYAAEgQAHpwAKzgAE7C4n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwAAIwAAMgABRgACYAADgQAHqAAJzgAD7C8n/l9W/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAAACAAADwAAGAAAJAAAMwAARwABYQADggAGqAAIzgEC7C8n/l9W/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABAAACAAADwAAGAEAJAAAMwAARwAAYQABggAFqAAGzwIB7DAn/mBW/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgAABQEACQEAEAEAGQEAJQEANAEASAAAYgAAgwABqQAEzwQA7DEn/mBW/4Z+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAQEAAQEAAwEABQEACgEAEQIAGgIAJQIANQMASQMAYwEAhAAAqgEBzwgA7DMn/mFW/4Z+/6Se/7y3/8/L/9/b/+vo//TyAAEAAAEAAAEAAAEAAAEAAAEAAAEAAAEAAAEAAQEAAQEAAgIAAwIABgIACwMAEgMAGwQAJgQANgUASgUAZAUAhQMAqwUA0A8A7DUo/mJW/4d+/6We/7y3/8/L/9/b/+vo//TyAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAAMAAQMAAQMAAgMABAQABwQADAUAEwUAHAYAKAcANwgASwoAZQsAhgsAqwwA0BYA7Dko/WNX/4h+/6We/723/9DL/9/b/+vo//TyAAYAAAYAAAYAAAYAAAYAAAYAAAYAAAYAAAYAAQYAAgYAAwcABQcACQgADggAFQkAHgsAKgwAOQ4ATRAAZxEAhxMArBYA0R0A7Twp/WVX/4l//6ae/723/9DL/9/b/+vo//TyAAwAAAwAAAwAAAwAAAwAAAwAAAwAAAwAAAwAAQwAAg0ABA0ABg0ACg4AEA4AFxAAIBEALBMAOxQATxcAaRkAihsArh8A0SYA7UEq/WhY/4p//6ae/724/9DL/9/b/+vo//TyABQAABQAABQAABQAABQAABQAABQAABUAARUAARUAAxUABRUACBYADRYAEhcAGhgAIxkALxsAPh0AUh8AbCIAjCUAsCkA0jAA7Ugr/WtZ/4yA/6ef/764/9DM/9/b/+vo//XyAB8BAB8BAB8BAB8BAB8AAB8AAB8AAB8AAR8AASAAAyAABSAACiAADyEAFSIAHSIAJiQAMiUAQigAVioAcC4AkDEAsjUA0zsA7VEt/HBb/4+B/6mf/7+4/9HM/9/c/+vo//XyAC0BAC0BAC0BAC0BAC4BAC4BAC4BAS4AAS4AAS4AAi4ABS4ACi8AES8AGDAAIDAAKjIANzMARzYAWzgAdTwAlEAAtEQA1EkA7Fsw/Hdd/5KC/6ug/8C5/9HM/+Dc/+vo//XyAEADAEADAEADAEACAUACAUACAUABAUABAUABAUEAAkEABUEACUEAEkIAG0IAJEMAL0QAPEUATEgAYUsAe04AmFIAt1UA1VoA7Gg0+39g/5eE/66h/8G5/9LM/+Dc/+zo//XzAVgFAVgFAVgFAVgEAVgEAVgEAVgDAVkDAlkCAlkBA1kABVkAClkAEVoAHFoAKFsANFwAQl0AU18AZ2IAgWUAnWgAumoA1m0D7Hc6+4lk/52G/7Kj/8S6/9PN/+Hc/+zp//XzAnYIAnYIAnYIAnYHAnYHAnYHA3cGA3cFA3cEBHcDBXcCB3cADXgAFHgAHngAK3kAOnoASHsAWnwAb34Ah4AAoYEAvIIA1oMO6olD+ZVq/6WK/7al/8e7/9XO/+Ld/+3p//bzCJcLCJcKCJcKCZcKCZcKCZcJCpcJC5cIDJcHDZcFEJcDE5gCGJgAH5gAJ5gAMpgAQJkAUJkAYZoAdZsAjJwApZ0AvpsA1pki6JxO+KRx/6+O/72o/8q9/9fP/+Pd/+3p//bzGrQCGrQCG7QCG7QCG7QCG7QCHLQCHbQCHrQBILQBI7QBJrQALLQAMrQAOrQAQrQATLQAWbUAaLUAerUAjrUApbYAvrUK1LI65rFd9bR6/ruV/8Sr/8+//9rQ/+Xe/+7q//fzPswCPswCPswCPswCPswCPswCP8wCP8wCQMwCQcwCQswCRMwCRswDSswDTswDVMwFW8wGZcwIcMwKf8wOkMshpMsyu8pE0slW48Vu8sWG/sic/82w/9XC/97S/+ff//Dq//j0a95La95La95La95La95La95La95La95LbN5MbN5Mbd5Mbt5Mb95McN5Mct5Mdd5Ned5Oft1Qht1Rj91TmtxXqtxhu9tqzdt24dmC79aT+9Wl/9i2/93G/+PU/+vg//Lr//n0kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6ket6ket6kut6kut7k+t7let7l+t7met8nup9o+p+quqAs+qDv+mKzeiQ3eeZ7eai+eSw/+O9/+XL/+nX/+7i//Ts//r1rPWcrPWcrPWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrvWdrvWdr/WdsPWdsvWdtPWet/Wfu/WgwPShx/Sj0fOo3POt6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/23xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y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/+7i//Ts//r1rPWcrPWcrPWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrvWdrvWdr/WdsPWdsvWdtPWet/Wfu/WgwPShx/Sj0fOo3POt6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/22w/23xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///7///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgABIgABMQACRQADXwAGgAAJpwANzgAH7C0n/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgABIgABMQACRQADXwAFgAAJpwANzgAH7C0n/l9W/4V+/6Se/7y3/8/L/97b/+vo//Ty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4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///7///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQABFgABIgABMQACRQAEXwAGgAAKpwAOzgAI7C0n/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQABFgABIgABMQACRQADXwAGgAAKpwAOzgAI7C0n/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQABFgABIgABMQACRQADXwAGgQAKpwANzgAI7S0n/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAwAABwAADQAAFgABIgABMgACRgADXwAGgQAKpwANzgAH7S0n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwAADgAAFwABIgABMgACRgADYAAGgQAJpwANzgAH7S4n/l9W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA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cMwLf8wQkMsjpMsyu8pF0clW48Vu8sWG/sic/82w/9XC/97S/+ff//Dq//f0a95Ma95Ma95Ma95Ma95Ma95Ma95Ma95MbN5MbN5Mbd5Mbt5Mb95McN5Mct5Ndd5NeN5Pft1Qht1Rj91Tm9xYqtxhu9tqzdt24dmC79aT+9Wl/9i2/93G/+PU/+vg//Lr//n0kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6kOt6ket6ket7ket7kut7k+t7let7l+t8met9nup9o+p+quqAs+qDv+mKzeiR3eeZ7eai+eSw/+O9/+XL/+nX/+7i//Ts//r1rPWcrPWcrPWcrPWcrfWcrfWcrfWcrfWcrfWcrfWcrfWcrfWdrfWdrvWdrvWdr/WdsPWdsvWdtPWet/Wfu/WgwPShx/Sj0fOo3POt6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/22w/22w/22w/22w/22w/22w/22w/22w/22w/23w/23w/23w/23xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7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//Pl//ju//z2w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23xP23xP23xP23xf23xv23x/23yf24y/24zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwABDQABFgACIgACMQADRQAFXwAIgAANpwAQzgAL7Swn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwABDQABFgACIgACMQADRQAFXwAIgAANpwAQzgAL7Swn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQAAAwAABwABDgABFwACIgACMgADRQAFXwAIgAANpwAQzgAL7Swn/l5W/4V+/6Se/7y3/8/L/97b/+vo//Ty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7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAQABAwABBwABDgACFwACIgADMQAERQAGXwAKgAAOpgASzgAO7Ssn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgABAwABBwABDgACFwACIgADMgAERQAGXwAKgAAOpwASzgAO7Ssn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgABAwABBwABDgACFwACIgADMgAERQAGXwAKgAAOpwASzgAN7Ssn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgABAwABBwABDgACFwACIwADMgAERgAGXwAKgAAOpwASzgAM7Ssn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgABBAABBwABDgACFwACIwADMgAERgAGXwAJgQAOpwASzgAL7Ssn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAgABBAABCAABDgACFwACIwADMgAERgAGYAAJgQAOpwARzgAJ7Ssn/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAB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ht1Sj91Vm9xaqtxju9trzdt24dmC79aT+9Wl/9i2/93G/+PU/+vg//Lr//n0kOt7kOt7kOt7kOt7kOt7kOt7kOt7kOt7kOt7kOt7ket7ket7ket7kut7k+t7let8l+t8met9net+o+p/quqAs+qEv+mLzeiR3eeZ7eai+eSw/+O9/+XL/+nX/+7i//Ts//r1rPWdrPWdrPWdrPWdrfWdrfWdrfWdrfWdrfWdrfWdrfWdrfWdrfWdrvWdrvWdr/WdsPWdsvWetPWet/Wfu/WgwPShx/Sk0fOp3POu6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23xP23xP23xP23xf23xv23x/23yP24y/25zv250vy62Py84PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//M3v/N4f/O5f/P6//S8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/e9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+//z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACBwACDgADFwADIwAEMgAGRQAJXwAMgAARpgAVzQAS7Skn/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACCAACDgADFwADIwAEMgAGRQAJXwAMgAARpgAVzQAS7Skn/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACCAACDgADFwADIwAEMgAGRgAIXwAMgAARpgAVzQAS7Skn/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACCAACDwADFwADIwAEMgAGRgAIXwAMgAARpgAUzQAR7Skn/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACCAACDwADGAADIwAEMgAGRgAIXwAMgAARpwAUzQAR7Skn/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACCAACDwADGAADIwAEMgAGRgAIYAAMgQAQpwAUzgAP7Soo/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAgABBAACCAACDwADGAADJAAEMwAFRgAIYAALgQAQpwATzgAN7Soo/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAQABAgABBQACCQACEAACGQADJAAEMwAFRwAHYAALgQAPpwATzgAM7Soo/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQABAQACAwACBQACCgACEQACGQADJQADNAAFRwAHYQAKggAOqAARzgAM7Sso/l5W/4V+/6Se/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAABAAABAAABAAABAQACAQACAwACBgACCwACEgACGwACJgADNQAESAAGYgAIgwANqAAQzwAJ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//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//N3f/N4f/O5f/P6//T8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/f9f/h+//j///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+v/z+v/z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAABAAABAAABAAABAAABAAABAAABAAACAAACAAACAAACAQACAgACBAADCAADDwAEGAAFIwAHMgAJRgAMXwAQgAAUpgAYzQAX7SYo/lxW/4R+/6Oe/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAACAAACAAACAAACAAACAQACAgADBAADCAAEDwAEGAAFIwAHMgAJRgAMXwAPgAAUpgAYzQAX7SYo/lxW/4R+/6Oe/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAABAAACAAACAAACAAACAAACAQACAgADBAADCAAEDwAEGAAFIwAHMgAJRgAMXwAPgAAUpgAYzQAX7SYo/lxW/4R+/6Oe/7y3/8/L/97b/+vo//TyAAABAAABAAABAAABAAABAAACAAACAAACAAACAAACAAACAQACAgADBAADCAAEDwAEGAAFJAAHMwAJRgALXwAPgAAUpgAYzQAW7SYo/lxW/4R+/6Oe/7y3/8/L/97b/+vo//TyAAACAAACAAACAAACAAACAAACAAACAAACAAACAAACAQACAQACAgADBAADCAAEDwAEGAAFJAAHMwAJRgALYAAPgAAUpgAYzQAW7Sco/l1W/4R+/6Oe/7y3/8/L/97b/+vo//TyAAACAAACAAACAAACAAACAAACAAACAAACAAACAAACAQACAQADAgADBAADCQAEEAAEGAAFJAAHMwAIRgALYAAPgQAUpgAXzQAW7Sco/l1W/4R+/6Oe/7y3/8/L/97b/+vo//TyAAACAAACAAACAAACAAACAAACAAACAAACAAACAAACAQADAQADAgADBQADCQAEEAAEGQAFJAAGMwAIRwALYAAPgQATpwAXzQAV7Sco/l1W/4R+/6Oe/7y3/8/L/97b/+vo//TyAAACAAACAAACAAACAAACAAACAAACAAACAAACAAADAQADAQADAwADBQADCQAEEAAEGQAFJQAGNAAIRwALYAAOgQATpwAWzgAU7Sgo/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAACAAACAAACAAACAAACAAACAAACAAACAAADAAADAQADAQADAwADBgAECgAEEQAEGgAFJQAGNAAISAAKYQANggASpwAVzgAT7Sko/l1W/4V+/6Se/7y3/8/L/97b/+vo//TyAAECAAECAAECAAECAAECAAECAAEDAAEDAAADAAADAQADAgADAwAEBgAECwAEEgAEGwAFJgAFNQAHSQAJYgAMggARqAAUzwAQ7Sso/l5X/4V+/6Se/7y3/8/L/97b/+vo//TyAAEDAAEDAAEDAAEDAAEDAAEDAAEDAAEDAAEDAAEDAQEDAgEEAwEEBwEEDAEEEwAEHQAEKAAFNwAGSgAHYwALgwAPqQASzwAN7S0p/l9X/4V+/6Se/7y3/8/L/97b/+vo//TyAAIDAAIDAAIDAAIDAAIDAAIDAAIDAAIDAAIEAAIEAQIEAgIEBAIEBwIEDQIEFAIEHgEEKgADOQADTAAFZQAIhQAMqgAO0AAF7S8p/mBX/4Z+/6Se/7y3/8/L/9/b/+vo//TyAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAAQEAgQEBAQECAQEDgQEFQQEHwQELAMCOwAATwAAaAABhwACrAAF0QMB7TMp/mFX/4d//6We/7y3/8/L/9/b/+vo//TyAAcFAAcFAAcFAAcFAAcFAAcFAAcFAAcFAAcFAAgFAAgEAggEBQgECQgEDwkEFwkEIAkDLQgCPQcAUQQAagMAigIArgUA0gwB7Tgq/mNY/4h//6We/723/9DL/9/b/+vo//TyAA4GAA4GAA4GAA4GAA4GAA4GAA4GAA4GAA4GAA4FAA4FAg4FBQ4ECg8EEA8EGBAEIhADLxABPxAAUhAAbBEAjBIAsBMA0xsA7T0r/mZZ/4l//6af/724/9DM/9/b/+vo//TyABYIABYIABYIABYIABYIABYIABYIABYHABYHABYGABYFABcFBBcEChcEERgEGhgDJBkCMRkAQRoAVRsAbx0Ajh8AsSIA1CkA7UQs/Wla/4uA/6ef/764/9DM/9/c/+vo//XyACELACELACELACELACELACEKACEKACEKACEJACEHACEFACEDASIDByICEiMBGyMAJiQAMyUARCYAWCgAcioAkS0AszAA1TYA7U4u/W9b/46B/6ig/7+4/9HM/9/c/+vo//XyAC8OAC8OAC8OAC8OAC8OAC8NAC8NAC8NAC8MAC8KAC8IAC8FATABBzAAEzEAHTEAKTIANzMASDUAXDcAdjoAlT0AtUAA1kYA7Vkx/HVd/5KC/6qg/8C5/9HM/+Dc/+vo//XyAEETAEETAEESAEESAEESAEESAEESAEERAEEQAEIPAEILAEIHAkICCEMAEkMAH0QALEUAO0YATEgAYkoAfE0AmVAAuFMA1lgA7WY1/H5g/5aE/62h/8G5/9LM/+Dc/+zo//XzAFkXAFkXAFkXAFkXAFkWAFkWAFkWAFkVAFoUAFoTAFoRAFoNAVsHBVsDE1wAIlwAMF0AQF4AUmAAZ2IAgWQAnmcAumgA1mwD7HY7+4hk/52H/7Gj/8O6/9PN/+Hc/+zp//XzAHcbAHcbAHcbAHcbAHcbAHcaAHcaAHcZAHgZAHgXAHgWAHgSAHgMBHkHD3kDInoANXsARXsAWH0Abn4Ah4AAooEAvIEA1oEP64hD+pVq/6WK/7al/8a7/9XO/+Ld/+3p//XzAJgZAJgZAJgZAJgZAJgYAJgYAJgXAJgWAJgVAJgTAZgQAZgMA5kHB5kDFJkBJ5kAPJoATJoAX5sAc5sAi5wApZ0AvpoA1pki6ZxO+KNx/6+P/7yo/8q9/9fP/+Pd/+3p//bzDrQJDrQJDrQJDrQID7QID7QIELQIEbQHErUGFLUFF7UEGrUDH7UBJ7UAMLUAPLUASbUAVrUAZrUAeLYAjbYApLYBvrUM1LE+5rBd9bN6/7qV/8Sr/8+//9rQ/+Xe/+7q//fzOcwROcwROcwROswROswROswROswRO8wRO8wRPMwRPswRP8wSQswSRswSS8wTUswUWswVY8wYb8wcf8wgkMstpMs4uspH0clV48Vw8sSG/sec/82w/9XC/97S/+ff//Dq//f0a95Pa95Pa95Pa95Pa95Pa95Pa95PbN5PbN5PbN5Pbd5Qbt5Qb95QcN5Qct5Rdd5Red5Sft1Uht1Wj91Zm9xdqtxmu9tszdt34dmC79aU+9Wl/9i2/93G/+PU/+rg//Lr//n0kOt8kOt8kOt8kOt8kOt8kOt8kOt8kOt8kOt8kOt8ket8ket8ket8kut8k+t8let8l+t9met9nep+o+qAquqCs+qFv+mMzeiR3OeZ7eai+eOw/+O9/+XL/+nX/+7i//Ts//r1rPWdrPWdrPWdrPWdrPWdrfWdrfWdrfWdrfWdrfWdrfWdrfWdrfWdrvWdrvWdr/WesPWesvWetPWft/Wfu/WgwPSix/Sk0fOp3POu6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23xP23xP23xP23xf23xv23x/24yP24y/25zv250vy62Py94PzA6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//L2P/M2f/M2//N3f/N4f/O5f/Q6//T8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/f9f/h+//k///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y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nuqApOqCq+qEtOmHwOmNzeiS3Oea7eai+eOw/+O9/+XL/+nX/+7i//Ts//r1rPWerPWerPWerPWerPWerPWerfWerfWerfWerfWerfWerfWerfWervWervWer/WesPWesvWetPWft/Wgu/WhwfSjyPSl0fOq3POu6fK09vG7/+/G/+7Q//Db//Pl//ju//z2w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23w/23xP23xP23xP23xf23xv24x/24yP24y/25zv250vy72Py94PzB6fvE8/rJ/frP//jX//ff//no//vw//731f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1f/L1v/L1v/L1//M2P/M2f/M2//N3f/N4f/O5f/Q6//T8//W+//a///e//7k//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/b5f/c5v/c6P/c6v/d7P/d8P/f9f/h+//k///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+f/z+v/z/P/z/v/0/v/0///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAAGAAAHAAAHAAAHAAAHAAAHAAAHAAAHAAAIAAAJAAAJAQAKAwAKBQALCgAMEQANGgAOJQAQNAASSAAVYQAZgQAepQAiywAj7xMq/1pX/4N+/6Oe/7u3/8/L/97b/+vo//TyAAEHAAAHAAAHAAAHAAAHAAAHAAAHAAAHAAAIAAAJAAAJAQAKAwAKBQALCgAMEQANGgAOJQAQNAASSAAVYQAZ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//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/b5P/b5P/c5f/c5v/c6P/c6v/d7P/d8P/f9f/h+//k///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/y+f/z+f/z+v/z/P/z/v/0/v/1///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAENAAENAAENAAENAAEOAAEOAAEOAAEOAAEPAAEQAAARAQARAwASBgATCwAUEgAUGwAWJgAXNQAaSQAdYgAgggAlpgApywAr7wwu/1dX/4J//6Ke/7u3/8/L/97b/+ro//TyAAENAAENAAENAAENAAEOAAEOAAEOAAEOAAEPAAEQAAARAQARAwASBgATCwAUEgAVGwAWJwAXNQAaSQAdYgAgggAlpgApywAr7wwu/1dX/4J//6Ke/7u3/8/L/97b/+ro//TyAAENAAENAAENAAEOAAEOAAEOAAEOAAEOAAEPAAEQAAARAQARAwASBgATCwAUEgAVGwAWJwAXNQAaSQAdYgAgggAlpgApywAr7w0u/1dX/4J//6Ke/7u3/8/L/97b/+ro//TyAAEOAAEOAAEOAAEOAAEOAAEOAAEOAAEPAAEPAAEQAAARAQASAwASBgATCwAUEgAVGwAWJwAXNgAaSQAdYgAgggAlpgApywAr7w0u/1dY/4J//6Ke/7u3/8/L/97b/+ro//TyAAEOAAEOAAEOAAEOAAEOAAEOAAEOAAEPAAEPAAEQAAARAQASAwATBgATCwAUEgAVGwAWJwAXNgAaSQAdYgAgggAlpgApywAr7w4u/1dY/4J//6Ke/7u3/8/L/97b/+ro//TyAAEOAAEOAAEOAAEOAAEOAAEOAAEPAAEPAAEQAAEQAAARAQASAwATBgAUDAAUEwAVHAAWJwAXNgAaSQAcYwAgggAlpwApywAq7w8u/1dY/4J//6Ke/7u3/8/L/97b/+ro//TyAAEOAAEOAAEOAAEPAAEPAAEPAAEPAAEPAAEQAAERAAESAQATAwATBgAUDAAVEwAVHAAWJwAYNgAaSgAcYwAgggAlpwApywAq7xEu/1hY/4J//6Ke/7u3/8/L/97b/+ro//TyAAIPAAIPAAIPAAIPAAIPAAIPAAEQAAEQAAERAAERAAESAQATAwAUBgAUDAAVEwAVHAAWKAAYNwAaSgAcYwAggwAkpwApywAq7xIu/1hY/4J//6Ke/7u3/8/L/97b/+ro//TyAAIQAAIQAAIQAAIQAAIQAAIQAAIQAAIRAAERAAESAAETAQEUAwEUBgEVDAAVFAAWHQAXKAAYNwAaSwAcZAAggwAkpwAozAAp7xMv/1hY/4N//6Oe/7u3/8/L/97b/+ro//TyAAIQAAIQAAIQAAIQAAIRAAIRAAIRAAIRAAISAAISAAETAQEUAwEVBwEVDQEWFAAWHgAXKQAYOAAaSwAcZAAfhAAkqAAnzAAo7xQu/1lY/4N//6Oe/7u3/8/L/97b/+vo//TyAAMRAAMRAAMRAAMRAAMRAAMRAAMSAAMSAAMSAAMTAAIUAQIVAwIVBwIVDQIWFQEXHgAYKgAYOQAaTQAcZgAfhQAjqQAnzQAn7xQt/1pY/4N//6Oe/7u3/8/L/97b/+vo//TyAAUSAAUSAAUSAAUSAAUSAAQSAAQTAAQTAAQTAAQUAAQVAQMVAwMWBwQWDgMXFQMXHwMYLAEZOwAaTgAbZwAehgAiqgAlzgAl7xgu/1tY/4R//6Oe/7y3/8/L/97b/+vo//TyAAcTAAcTAAcTAAcTAAcTAAcUAAcUAAcUAAcUAAYVAAYWAQYWBAYWCAYXDgYXFgYYIAYYLQQZPQEaUQAaagAdiQAgrAAjzwAi7x0u/1xZ/4R//6Se/7y4/8/L/97b/+vo//TyAAsVAAsVAAsVAAsVAAsVAAsVAAsVAAsVAAsWAAsWAAsWAQoXBAoXCQsYDwsYFwsYIQsZLgoaPwcaVAEZbQAajAAcrwAe0gAb8Cct/15Z/4aA/6Sf/7y4/8/M/97b/+vo//TyABEXABEXABEXABEXABEXABEXABEXABEXABEXABEXABEYABEYAxEYCREZEBEZGRIZIxIaMBIaQRAaVgsYcgAVkQATswAV1QAQ8DAu/2Fa/4eA/6Wf/724/9DM/9/b/+vo//TyABkZABkZABkZABkZABkZABkZABkZABkZABkZABkZABkZARkZAxkZCBkaERoaGhoaJRsaMhsaQxsaWRkYdRQTlwUDtwYB1xEC8Dkv/mVb/4mB/6af/724/9DM/9/c/+vo//TyACMcACMcACMcACMcACMcACMcACMcACMcACMcACMcACMcACQbAiQbCCQbESUbGyUbJyYbNCYbRicaXCcXeCURmSIBuSIB2CkC70Ux/mtc/4yC/6ig/764/9DM/9/c/+vo//XyADEgADEgADEgADEgADEgADEgADEfADEfADEfADEfADEeADIeADIdAzIcDzMbGzMbKDQbNzUaSTYZXzcVezcMmzcCuzgB2D8H71Mz/XJe/5CD/6qh/7+5/9HM/9/c/+vo//XyAEMlAEMlAEMlAEMlAEMlAEMkAEMkAEMkAEQjAEQjAEQiAEQgAEQfAEUcBEUbGEYaKUcZOkgYTUkWZEoQf0sEnk0DvE4B2VMP7mI2/Hth/5WF/6yi/8G5/9LN/+Dc/+vo//XzAFsqAFsqAFsqAFsqAFsqAFsqAFsqAFsqAFwpAFwpAFwoAFwlAFwiAF0dAV4VC18TJl8RO2AOUGEIaGMEg2QEoWUBvmYC2GkI7XM//IZl/5yH/7Cj/8O6/9PN/+Hc/+zp//XzAHkwAHkwAHkwAHkwAHkwAHkwAHkwAHkvAHkvAHkuAHktAHorAHooAXsjBHsaEXwJJn0EPX4EVH4DbYADiIEAo4EAvoAF2IAf7IdH+5Nq/6SK/7Wl/8a7/9XO/+Ld/+zp//XzAJkzAJkyAJkyAJkyAJkyAJkyAJkyAJkxAJkxAJkwAJkuAJkrAJomAZoeB5oTF5sFLJwAQpwAWZwBcJ0Cip0ApZ0BvpoJ1pk06ZpP+aJy/66P/7yo/8q9/9fP/+Pd/+3p//bzCLUeCLUdCLUdCbUdCbUdCbUcCrYcCrYbC7YaDLYYDrYWEbYTFbYOGrYII7YEL7cEPbYGT7YIY7YMdrYTjLYapLYlvrUy07FF5rBi9rJ7/7qV/8Sr/8+//9rQ/+Xe/+7q//fzP802P802P802P802P802P802QM02QM02Qc02Qc02Q802RM02R803Ssw3T8w4Vsw4Xsw5Z8w7c8w+gcxBkctGpctLuspT0clh48Vz88SG/sec/82w/9XC/97S/+ff//Dq//f0b95hb95hb95hcN5hcN5hcN5hcN5hcN5hcN5hcd5hcd5hct5hc95hdN5idt1ied1ifd1jgt1kid1lkt1nndxqqtxuu9tzzdt64NmF79aW/NWl/9e2/9zG/+PU/+rg//Lr//n0kuuDkuuDkuuDkuuDkuuDkuuDkuuDkuuDkuuDk+uDk+uDk+uDlOuDleuEluuEl+uEmeuFnOqFoOqGpeqHrOqKtemMwOmQzeiV3Oeb7Oai+eOx/+O9/+XL/+nX/+7i//Ts//r1rfWgrfWgrfWgrfWgrfWgrfWgrfWgrfWgrvWgrvWgrvWgrvWgrvWgr/Whr/WhsPWhsfWhs/WhtfWiuPWjvPSkwfSmyPSo0vOs3POw6fK19vG7/+/G/+7Q//Db//Pl//fu//z2w/24w/24w/24w/24w/24w/24w/24w/24w/24w/24w/24xP24xP24xP24xP24xf24xf25xv25x/25yf26y/26zv270/y82Py+4PvC6fvF8/rK/frO//jX//ff//no//vw//731f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1v/M1v/M1//M2P/M2f/M2//N3v/O4f/P5f/Q7P/T8//W+//a///e//7l//7r///y///44//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b4//b5P/c5P/c5P/c5f/c5v/c6P/c6v/d7P/e8P/f9f/h+//k///n///q///v///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/o8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/y+P/z+f/z+f/z+v/z/P/z/v/0/v/1///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7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tvWluPWlvPSmwvSoyfSq0vOt3POx6fK19vG7/+/G/+7Q//Db//Pl//fu//z2xP26xP26xP26xP26xP26xP26xP26xP26xP26xP26xP26xP26xP26xP26xf26xf26xv26x/26yP27yf27zP28z/280/y+2fy/4PvC6fvG8/rK/frO//jX//ff//no//vw//731f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1f/M1v/M1v/M1v/N1//N2P/N2f/N2//O3v/P4f/Q5v/R7P/U8//W+//a///e//7l//7r///y///44//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c5P/c5P/c5P/c5f/c5v/c6P/d6v/d7f/e8P/f9f/i+//k///n///q///w///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//o8P/p8f/p8v/p9P/p9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y+P/y+P/y+P/y+P/z+P/z+f/z+f/z+v/z/P/z/v/0/v/1///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///8///+///+////////AAQjAAQjAAQjAAQjAAQjAAQjAAQjAAQkAAQkAAMlAAMmAQMmAwMnBwMoDQIqFQIrHgEtKgAvOQAwTQAzZgA2hgA7qgA/zgBC8BhH/1Nf/3+B/6Cf/7q4/87L/97b/+ro//TyAAQjAAQjAAQjAAQjAAQjAAQjAAQjAAQkAAQkAAMlAAMmAQMmAwMnBwMoDQIqFQIrHgEtKgAvOQAwTQAzZgA2hgA7qgA/zgBC8BhH/1Nf/3+B/6Cf/7q4/87M/97b/+ro//TyAAQjAAQjAAQjAAQjAAQjAAQjAAQjAAQkAAQkAAMlAAMmAAMmAwMnBwMoDQIqFQIrHgEtKwAvOgAwTQAzZgA2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/7Sm/8W8/9TO/+Hd/+zp//XzAJpLAJpLAJpLAJpLAJpLAJpLAJpLAJtLAJtKAJtKAJtJAJtIAJtHAptFCJxCFZw+K507QZ06WZ07cZ47ip49pZ4/vpxH1ZtV6J1k+aJ3/62Q/7uo/8m9/9fP/+Pd/+3p//bzEbZKEbZKEbZKErZKErZJErZJErZJE7ZJFLZJFbdIF7dHGbdGHbdFI7dDK7dENLdEQbdEVLdFZbdGeLdIjbdLpLZQvbZW0rNi5rFw9rOA/7mX/8Os/87A/9rQ/+Xe/+7q//bzSsxZSsxZSsxZSsxZSsxZS8xZS8xZS8xZTMxZTMxZTcxZT8xZUcxZVMxZWcxaXsxaZcxbbcxcd8xdhcxflMtipstmu8pr0cly48Z+8sWN/8ac/8yx/9XC/97S/+ff//Dq//f0d910d910d910d910d910d910d910d910d910eN10eN10ed10ed10e910fN11f911gt11h912jt13ldx5oNx7rNx+vNuBzduG4NqO79aY+9Wn/9e2/9zG/+PU/+rg//Lr//n0luuPluuPluuPluuPluuPl+uPl+uPl+qPl+qPl+qPl+qPmOqPmOqPmeqPmuqPm+qPneqQn+qQo+qRqOqSruqTt+mVwemYzuib3Oeg7Oao+OOy/+O9/+XL/+nX/+7i//Ts//r1sPWnsPWnsPWnsPWnsPWnsPWnsPWnsPWnsPWnsPWnsPWnsPWnsfWnsfWnsfWnsvWns/WntfWot/WouvSpvvSqw/SryfSt0vOv3fOy6fK29vG9/+/H/+7Q//Db//Pl//fu//z2xf28xf28xf28xf28xf28xf28xf28xf28xf28xf28xf28xf28xf28xf28xv28xv28xv28x/28yP29yv29zP2+z/2+1PzA2fzB4PvD6fvH8/rK/frP//jX//ff//no//vw//731f/N1f/N1f/N1f/N1f/N1f/N1f/N1f/N1f/N1f/N1v/N1v/N1v/N1v/N1v/O1v/O1v/O1//O2P/O2P/O2v/P3P/P3v/Q4f/R5v/S7P/U8//X+v/a///e//7l//7r///y///44//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c4//c5P/c5P/c5P/c5f/c5f/d5v/d6P/d6v/e7f/f8P/g9f/i+//k///n///q///w///1///67v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7v/o7//o7//o7//o7//p8P/p8f/p8v/p9P/q9v/q+f/r/f/t///v///x///0///4///99//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//y9//z+P/z+P/z+P/z+P/z+P/z+f/z+f/z+v/z/P/z/v/0/v/1///2///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///9///+///+////////AAgyAAgyAAgyAAgyAAgzAAgzAAgzAAgzAAg0AAc0AAc1AAY2AQY3BgY4DgY5FgU7IAU9LANAPABCUABEaQBHiQBLrABP0ABS7yRZ/1tv/36F/5+g/7m4/87M/97b/+ro//TyAAgyAAgyAAgyAAgyAAgzAAgzAAgzAAgzAAg0AAc0AAc1AAY2AQY3BgY4DgY6FgU7IAU9LANAPABCUABEaQBHiQBLrABP0ABS7yRZ/1tv/36F/5+g/7m4/87M/97b/+ro//TyAAgyAAgyAAgyAAgzAAgzAAgzAAgzAAgzAAg0AAc0AAc1AAY2AQY3BgY4DgY6FgU7IAU9LANAPQBCUABEaQBHiQBLrABP0ABS7yRZ/1tv/36F/5+g/7m4/87M/97b/+ro//TyAAkyAAkyAAgzAAgzAAgzAAgzAAgzAAgzAAg0AAc1AAc2AAY3AQY3BgY4DgY6FgU7IAU9LANAPQBCUABEagBHiQBLrABP0ABS7yRZ/1tv/36F/5+g/7m4/87M/97b/+ro//TyAAkzAAkzAAkzAAkzAAkzAAgzAAgzAAg0AAg0AAc1AAc2AAY3AQY4BgY5DgY6FgY8IAU9LQNAPQBCUABEagBHiQBLrABP0ABS7yRZ/1tv/36F/5+g/7m4/87M/97b/+ro//TyAAkzAAkzAAkzAAkzAAkzAAkzAAkzAAg0AAg0AAg1AAc2AAY3AQY4BgY5DgY6FgY8IAU+LQNAPQBCUABEagBHiQBLrABP0ABS7yVZ/1tv/36F/5+g/7m4/87M/97b/+ro//Ty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e92AfN2AfN2AfN2BfN2Bfd2Bft2Bf92Bgd2Bg92Bht2Ci92CkdyDmNyEotyGrtyIvduLzduQ4NqV7taf+9Wr/9e2/9zG/+PU/+rg//Lr//n0muqWmuqWmuqWmuqWmuqWmuqWmuqWmuqWmuqWmuqWmuqWm+qWm+qXnOqXneqXnuqXn+qXouqYpeqYquqZsOmauOmcwumfzuih3eel7Oer+OOz/+O//+TL/+nX/+7i//Ts//r1svWssvWssvWssvWssvWssvWssvWssvWssvWssvWssvWssvWss/Wss/Wss/WstPWstfWst/StufStu/Suv/SuxPSvyvOx0/Oz3fO26fK59vG//+/H/+7Q/+/b//Pl//fu//z2xv2+xv2+xv2+xv2+xv2+xv2+xv2+xv2+xv2/xv2/xv2/xv2/xv2/xv2/x/2/x/2/yP2/yP2/yf2/y/3Azf3A0PzB1PzC2vzD4PvF6fvI8/rL/frQ//jY//ff//no//vw//731v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1v/P1//P1//P1//P2P/P2P/Q2f/Q2v/Q3P/R3//R4v/S5v/T7P/V8//Y+v/a///e//7l//7r///y///44//d4//d4//d4//d4//d4//d4//d4//d4//d4//d4//d4//d5P/d5P/d5P/d5P/d5P/d5P/d5f/d5f/d5v/e5//e6P/e6v/f7f/g8f/h9f/j+//k///n///q///w///1///67v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7v/p7//p7//p7//p7//p8P/p8P/p8f/p8v/q9P/q9v/r+f/s/f/t///v///x///0///4///99//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z+P/z+P/z+P/z+P/z+P/z+f/z+f/z+v/z/P/0/v/0/v/1///3///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///9///+///+////////AA9HAA9HAA9HAA9HAA9HAA9HAA9HAA9IAA5IAA5JAA5KAA1LAAxMAgxNDAxOFgxQIQtSLglVPwZYVABbbQBdjQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyAA9HAA9HAA9HAA9HAA9HAA9HAA9HAA9IAA9IAA5JAA5KAA1LAAxMAgxNDAxOFgxQIQtSLglVPwZYVABbbQBdjQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyAA9HAA9HAA9HAA9HAA9HAA9HAA9HAA9IAA9IAA5JAA5KAA1LAAxMAgxNDAxPFgxQIQtSLglVPwZYVABbbQBdjQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyAA9HAA9HAA9HAA9HAA9HAA9HAA9IAA9IAA9IAA5JAA5KAA1LAAxMAgxNDAxPFgxQIQtSLglVPwZYVABbbQBdjQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyABBHABBHABBHAA9HAA9HAA9HAA9IAA9IAA9IAA5JAA5KAA1LAAxMAgxODAxPFgxQIQtTLglVPwZYVABbbQBejQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyABBHABBHABBHABBHABBHABBIAA9IAA9IAA9IAA9JAA5KAA1LAAxNAgxODAxPFgxRIQtTLgpVPwZYVABbbQBejQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyABBHABBHABBHABBIABBIABBIABBIABBIAA9JAA9JAA5KAA5LAA1NAgxODAxPFgxRIQtTLgpWPwZYVABbbgBejQBhsABk0wBm7y5t/mB9/4SR/5+j/7m5/83M/93c/+ro//TyABFIABFIABFIABFIABBIABBIABBIABBJABBJAA9KAA9KAA5MAA1NAg1PCwxQFgxRIQtTLgpWPwZZVQBcbgBejgBhsABk0wBm7y5t/mB9/4WR/5+j/7m5/83M/93c/+ro//TyABFIABFIABFIABFIABFIABFIABFJABFJABFJABBKABBLAA9MAA5OAg1PCw1QFgxSIQxULgpWQAZZVQBcbgBejgBhsQBk0wBm7y9t/mB9/4WR/5+j/7m5/83M/93c/+ro//TyABJJABJJABJJABJJABJJABJJABJJABJKABJKABFLABBMABBNAA5OAQ1QCg1RFg1TIQxVLwpXQAZaVQBcbwBejgBhsQBk1ABm7zBt/mF9/4WR/5+j/7m5/83M/93c/+ro//TyABRJABRJABRKABRKABRKABRKABNKABNKABNLABJLABJMABFOABBPAQ5RCQ5TFQ1UIQxWLwpYQAdaVgBdcABfjwBhsgBk1ABm7zNt/mF9/4WR/5+j/7m5/83M/93c/+ro//TyABVLABVLABVLABVLABVLABVLABVLABVMABVMABRNABRNABNPABJQAhBSCA5VEw5WIQ1XMAxZQQlbVwFecQBgkABiswBk1QBm7zZt/mJ9/4WQ/5+j/7m5/83M/97c/+ro//TyABhNABhNABhNABhNABhNABhNABdNABdNABdOABdOABdPABZQABRRARNUBhFWExBXIhBYMA9aQg1cWARecgBhkgBitABk1gRm7zlt/mN9/4WQ/5+i/7m5/87M/97c/+ro//TyABtOABtOABtOABtOABtOABtPABtPABtPABtPABtQABpQABlSABhTABdVAxVXFBVZIhVaMRRbQxNdWQ9fdQNhlQBjtwBl1xBm7z1u/mV9/4aQ/5+i/7m5/87M/97c/+ro//TyACBQACBQACBQACBRACBRACBRACBRACBRACBRACBSAB9TAB5UAB1VABxXAxxZExtaIhtbMhtdRBpeWhhgdhFilwNkuQVl2B1n7kVv/Wp//4iR/6Ci/7q5/87M/97c/+ro//TyACdTACdTACdTACdTACdTACdTACdUACZUACZUACZUACZVACZWACVXASVYBiRaFSNcIyNdMyRfRiNgXCNieSBjmhpluxpm2C5o7k5x/W6A/4qR/6Gi/7q5/87M/97c/+ro//TyADBWADBWADBWADBWADBWADBXADBXADBXADBXADBYADBYAC9ZAC9aAS9cAy5eEC5fJS5gNS9hSC9iXy9jey5lnCxmvDBn1z9q7Vlz/HWC/42S/6Oj/7u5/8/M/97c/+vo//TzAD1bAD1bAD1bAD1bAD1bAD1bAD1bAD1bAD1bAD1bAD1cAD1cAD1dADxeBDxgEzxhJD1iNz1jSz5kYj5lfj9mnj9nvERo11Ft7GZ2/HyE/5GT/6ak/7y6/8/N/9/c/+vo//TzAE5fAE5fAE5fAE5fAE5fAE5gAE5gAE5gAE5gAE5gAE5gAE5hAE5hAE5iAE5jBE5kH09lOFBlTlFmZVJngVNooFVpvVpr1mVw63N6+4SF/5aU/6ml/766/9DN/9/c/+vp//XzAGRmAGRmAGRmAGRmAGRmAGRmAGRmAGRmAGRmAGRmAGRmAGRmAGVmAGVnA2VnDmZnImdoN2doUGhpampqhWtqoW1svHFu1Xh06YF8+o6I/52W/66n/8C7/9LN/+Dc/+vp//XzAIBsAIBsAIBsAIBsAIBsAIBsAIBsAIBsAIBsAIBsAIBsAIFsAIFsAIFsAIJrBIJrHINrPINqU4RrbYVsiIZto4ZvvIhy1It46JGA+JqM/6aZ/7Sq/8S8/9TO/+Hd/+zp//XzAJ1xAJ1xAJ1xAJ1xAJ1xAJ1xAJ1xAJ1xAZ1wAZ1wAZ5wAp5wA55vBp5uDp5uHp9tMp9tSZ9uXp9vdJ9wjKBxpaBzvKB30qB95qOG9qiQ/7Cd/7ut/8i+/9bP/+Ld/+3p//bzJrhzJrhzJrhzJrhzJrhzJ7hzJ7hzKLhzKrhzK7hzLbhzL7h0Mrh0N7h0Prh0R7h1U7h1YLd2brd3frd4kbd5prd7vLd+0baE47WM87eW/7yi/8Ow/83A/9nQ/+Te/+7q//bzXcyAXcyAXcyAXsyAXsyAXsyAXsyAXsyAX8yAX8yAYMyAYcyAYsyAZcyAaMyAbMyBccyBeMyCgcyCjMyDmcuEqcuGvMuJ0MqO4ciV8ced/cmo/8yz/9TD/93S/+ff/+/q//f0gd2Pgd2Pgd2Pgd2Pgd2Pgt2Pgt2Pgt2Pgt2Pgt2Pg92Pg92PhN2Phd2Pht2PiN2Qi9yQkNyQldyRnNySpdyTsNyVvtuXztua39qf7ten+9av/9e5/9zH/+LU/+rg//Hr//j0nuqgnuqgnuqgnuqgnuqgnuqgnuqgnuqgnuqgnuqgnuqgnuqgn+qgn+qgoOqhoeqho+qhpeqhqeqiremis+mjuumlxOim0Oip3eis7Oex+OS4/+PC/+TL/+nX/+7i//Ts//r1tPSytPSytPSytPSytPSytPSytPSytPSytPSytfSytfSytfSytfSytfSytvSyt/SyuPSyufSyu/SzvfSzwfS0xvS1zPO21PO43fK66fK+9vHC/+/I/+7S/+/b//Pl//fu//z2x/3Cx/3Cx/3Cx/3Cx/3Cx/3Cx/3Cx/3Cx/3Cx/3Cx/3CyP3CyP3CyP3CyP3Cyf3Cyf3Dyv3Dy/3DzP3Dz/zE0vzE1fzF2vzH4fvI6fvK8/rN/frS//jY//ff//jo//vw//731//R1//R1//R1//R1//R1//R1//R1//R1//R1//R1//R1//R1//R1//R1//R2P/R2P/R2P/S2f/S2v/S2//S3f/T3//T4v/U5//V7P/W8//Z+v/b///f//7l//7r///y///45P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5P/e5f/f5f/f5v/f5v/f5//f6f/f6//g7f/h8f/i9f/j+//l///n///q///w///1///67//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//p7//q7//q8P/q8P/q8P/q8f/q8v/q9P/r9v/s+f/s/f/u///v///y///0///5///99//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z9//z+P/z+P/z+P/z+P/z+P/z+P/z+f/z+v/z+//0/P/0/v/1///1///3///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///7///8///8///9///+///+////////ABhhABhhABhhABhhABhhABhiABhiABhiABdiABdjABdkABZlABVnABRoBhRpFhRrIhRtMBNwQRFzVgx3cgF6kQB9tAB/1gKA7jmC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABhhABhhABhhABhhABhhABhiABhiABhiABhiABdjABdkABZlABVnABRoBhRpFhRrIhRtMBNwQRFzVgx3cgF6kQB9tAB/1gKA7jmC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABhhABhhABhhABhhABhhABhiABhiABhiABhjABdjABdkABZlABVnABVoBhVpFhRrIhRtMBNwQRFzVgx3cgF6kQB9tAB/1gKA7jmC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABhhABhhABhhABhhABhiABhiABhiABhiABhjABdjABdkABZlABVnABVoBhVqFRVrIhRtMBNwQRFzVgx3cgF6kQB9tAB/1gKA7jmC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABlhABlhABlhABhiABhiABhiABhiABhiABhjABdjABdkABZlABVnABVoBhVqFRVrIhRtMBNwQRFzVwx3cgF6kgB9tAB/1gKA7jqC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABliABliABliABliABliABhiABhiABhiABhjABhjABdkABdlABZnABVpBhVqFRVsIhRuMBRwQRFzVwx3cgF6kgB9tAB/1gJ/7jqC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABliABliABliABliABliABliABliABhjABhjABhkABdkABdlABZnABVpBRVqFRVsIhRuMBRxQRJ0Vwx3cgF7kgB9tAB/1gN/7jqC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABliABliABliABliABliABliABljABljABljABhkABhlABdmABZnABVpBRVrFRVsIhVuMBRxQRJ0Vwx3cgF7kgB9tAB/1gN/7jqC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABpiABpiABpiABpiABpjABpjABpjABpjABlkABlkABllABhmABdoABZqBRVrFBVtIhVvMBRxQRJ0Vwx4cwF7kgB9tQB/1gR/7jqC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABtjABtjABtjABtjABtjABtjABtjABtkABpkABplABplABlnABhoABZqBRZsFBVuIhVvMBRyQhJ1Vwx4cwF7kwB+tQB/1wZ/7jqC/WWN/4ic/6St/7i7/83N/93c/+ro//TyABxkABxkABxkABxkABxkABxkABxkABxkABxlABtlABtmABpnABlpARhrBBZtExZvIhVxMBVzQhJ2WAx5dAJ8kwB+tgB/1wl/7jqC/WaN/4mc/6Ss/7i7/83N/93c/+ro//TyAB5lAB5lAB5lAB5lAB5lAB5lAB5lAB5lAB1mAB1mAB1nABxoABtqARlsBBdvEhdxIBZyMBV0QhN3WQ56dAN8lAB+twB/1wt/7j2C/WeN/4mc/6Ss/7i7/83N/93c/+ro//TyACBmACBmACBmACBmACBmACBmACBnACBnACBnACBoAB9oAB5qAB1rARxtBBlwEBdzHhd0MBZ2QxV4WRJ6dQh9lgF/uACA2BGA7kGD/WmN/4qc/6Ss/7m7/83N/93c/+ro//TyACNoACNoACNoACNoACNoACNpACNpACNpACNpACNqACJqACJrACFtAB9vAx1yDRp1Hhp2MBp4Qxl5Whd8dg9+lwSAugCA2ByA7kaD/WuN/4qc/6Ss/7m7/83N/93c/+ro//TyAChqAChqAChqAChqAChrAChrAChrAChrACdrACdsACdsACZtACVvACRxACJ0BiB2HyB4MSB5RR97XB59eBp/mRKBugyB2CuB7kuE/W6O/4yd/6Ws/7m7/83N/93c/+ro//TyAC5tAC5tAC5tAC5tAC5tAC5tAC5tAC5uAC5uAC5uAC5vAC1wACxxACt0ACl2BSh4Hyh6Mih7Rih8XSd+eSWAmiKCuyaC1zeB7VSF/HOP/4+e/6at/7m6/83N/93c/+ro//TzADdwADdwADdwADdwADdxADdxADdxADdxADdxADZyADZyADZzADZ0ADV2AjR3DTN6ITN8MzN9SDR+YDSAfDOCnDODvDeC10aD7F6H/HmR/5Of/6it/7q7/87N/97c/+rp//TzAEN0AEN0AEN0AEN0AEN0AEN0AEN0AEN0AEN1AEN1AEN2AEJ2AEJ3AUJ5AkF7CkF+HUF/NkF/SkKBYkOCfkODnkWEvEqE1leE62uK+oGT/5eg/6qu/7y7/87N/97c/+rp//TzAFN5AFN5AFN5AFN5AFN5AFN5AFN5AFN6AFN6AFN6AFN6AFN7AFN7AFN8AVN+CVN/IFOBNlSCTVWDZVaEgleFoFmFvF+F1WqH6XmN+YuW/52i/62u/768/8/N/9/c/+vp//TzAGl/AGl/AGl/AGl/AGl/AGl/AGl/AGl/AGl/AGl/AGmAAGmAAGmAAGmBAWmCA2qDGWqENWuFT2yFam2GhW6HoXGHu3aI036L54iR95WZ/6Oj/7Gw/8C9/9HO/9/d/+vp//XzAIOFAIOFAIOFAISFAISFAISFAISFAISFAISFAISFAISFAISGAISGAISGAoSGDoWGIYWHO4aHU4eHboeIiIiIooqJu42L0pGP5ZeU9qCc/6qm/7ex/8S//9PO/+Dd/+zp//XzAKCJAKCJAKCJAKCJAKCJAKCJAKCJAKCJAKCJAKCIAKCIAKCIAaCIBKCIC6CIGKGIMqGISaGIYaGJdqGKjaGLpKKMu6KO0aSS5KeY9Kyf/rSo/72z/8nB/9XP/+Ld/+3p//bzILmKIbmKIbmKIbmKIbmKIrmKIrmKI7mKJLmKJrmKKLmKK7mKMLmKNriLP7iLS7iLWbiMZbiNc7iOg7iPlLiQp7iRvLiT0LiX4ric8rqj/r+s/8W3/87D/9nQ/+Te/+7q//bzZcyTZcyTZsyTZsyTZsyTZsyTZsyTZsyTZ8yTaMyTaMyTacyTa8yTbcyTcMyUc8yUeMyUfsyVh8yWkcyXncuXrMuZvMubz8ud4Mmj78mp/Mux/866/9XG/93S/+bf/+/q//f0iNyeiNyeiNyeiNyeiNyeiNyeiNyeidyeidyeidyeidyeityeityei9yfjdyfj9yfktyfltyfmtygoNyhqdyis9ujwNukztun39qq7diw+ti3/9m//9zI/+LV/+rh//Hr//j0ouqrouqrouqrouqrouqrouqrouqrouqro+qro+qro+qro+qro+qrpOqrpeqrpumsp+msqemsrOmssemttumuvemvxeiw0Oiy3ei16+e49+W+/+TF/+XN/+jY/+7i//Ts//r1t/S5t/S5t/S5t/S5t/S5t/S5t/S5uPS5uPS5uPS5uPS5uPS5uPS5uPS5ufS5uvS5u/S6vPS6vfS6wPS6w/S7yPO8zfO91fO+3vLA6fLD9fHH/+/M/+7U/+/b//Pl//fu//z2yf3Hyf3Hyf3Hyf3Hyf3Hyf3Hyf3Hyf3Hyf3Hyf3Hyf3Hyv3Hyv3Hyv3Hyv3Hyv3Hy/3HzPzHzfzIzvzI0PzI0/zJ1/zK2/zL4vvM6vvO8/rR/frU//jZ//fg//jo//vw//732P/U2P/U2P/U2P/U2P/U2P/U2P/U2P/U2P/U2P/U2P/U2P/U2P/U2f/U2f/U2f/U2f/U2v/U2v/V2//V3P/V3v/V4P/W4//X5//Y7f/Z8//b+v/d///h//7m//7r///y///45f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5f/g5v/g5v/g5//h6P/h6f/h6//i7v/i8f/j9v/k+//m///o///r///w///1///67//q7//q7//q7//q7//q7//q7//q7//q7//q7//q7//q7//q7//q7//q7//q7//r7//r7//r8P/r8P/r8P/r8f/r8v/r8//r9P/s9//s+f/t/f/u///w///y///0///5///9+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/z+P/0+P/0+P/0+P/0+f/0+f/0+v/0+//0/P/1/v/1///2///3///4///6///8/////v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7/v/7///7///7///8///8///8///8///9///+////////////ACSCACSCACSCACSCACSCACSCACSCACSDACODACODASOEASKFAiGGAyCIByCKEiCLIyCNMiCPQx+SWR6VcxqZkxGctwud1iSc7Emb/Gue/4yo/6a1/7zD/8zP/9zc/+no//TzACSCACSCACSCACSCACSCACSCACSCACSDACODACODASOEASKFAiGGAyCIByCKEiCLIyCNMiCPQx+SWR6VcxqZkxGctwud1iSc7Emb/Gye/4yo/6a1/7zD/8zP/9zc/+no//TzACSCACSCACSCACSCACSCACSCACSCACSDACSDASODASOEASKFAiGHAyCIByCKEiCLIyCNMiCQQx+SWR6VcxqZkxGctwud1iSc7Emb/Gye/4yo/6a1/7zD/8zP/9zc/+no//TzACSCACSCACSCACSCACSCACSCACSCACSDACSDASODASOEASKFAiKHAyCIByCKEiCMIyCNMiCQQx+SWR6VcxqZkxGctwue1iSc7Emb/Gye/4yo/6a1/7zD/8zP/9zc/+no//TzACSCACSCACSCACSCACSCACSCACSCACSDACSDASOEASOEASKFAiKHAyCJBiCKEiCMIiCNMiCQQx+SWR6WcxqZkxGctwue1iWc7Emb/Gye/4yo/6a1/7zD/8zP/9zc/+no//TzACWCACSCACSCACSCACSCACSCACSDACSDACSDASSEASOEASOFAiKHAyGJBiCLEiCMIiCOMiCQQx+TWR6WcxqZlBGctwue1iWc7Emb/Gye/4yo/6a1/7zD/8zP/9zc/+no//TzACWCACWCACWCACWCACWCACSDACSDACSDACSDASSEASOFASOFAiKHAyGJBiCLEiCMIiCOMiCQQx+TWR6WcxqZlBGctwue1iWc7Emb/Gye/4yo/6a1/7zD/8zP/9zc/+no//TzACWCACWCACWDACWDACWDACWDACWDACWDACSEASSEASSFASOGAiKHAiGJBiCLEiCMIiCOMiCQQyCTWR6WcxqZlBGctwue1iac7Emb/Gye/4yo/6e1/7zD/8zP/9zc/+no//TzACWDACWDACWDACWDACWDACWDACWDACWDACWEASWEASSFASSGASOHAiKJBiCMESGNISGPMiCRQyCTWR6WdBqZlBKctwye1iac7Eqb/Gye/4yo/6e1/7zD/8zP/9zc/+no//TzACaDACaDACaDACaDACaDACaDACaEACaEACaEACWFASWFASSGASSIAiOJBiGMESGOISGPMiGRQyCUWR6XdBqalBKcuAye1iec7Uqb/Gye/4yo/6e1/7zD/8zP/9zc/+no//TzACeEACeEACeEACeEACeEACeEACeEACeEACeFACaFACaGASaHASWIAiSKBSKMECGPICGQMiGSQyCVWR+XdBualRKduA+e1imc7Uqb/Gye/4yo/6e1/7zD/8zP/9zc/+no//TzACmEACmEACmEACmEACmFACmFACmFACmFACiFACiGACiHACeIASaJASWLBSSNDyKQHyKRMSKTQyGWWR+YdRublRSduBOe1iuc7Uub/Gye/42o/6e1/7zD/8zO/9zc/+no//TzACuFACuFACuFACuFACuGACuGACuGACuGACuGACqHACqIACmIACiKASeMBCaODiORHiOTMCKVQyKXWiCZdRyclhieuRie1i2c7U2b/G6e/42p/6e1/7zD/8zO/93c/+np//TzAC6HAC6HAC6HAC6HAC2HAC2HAC2HAC2IAC2IAC2IAC2JACyKACuLASqNBCiPDCaSHCSWLiSXQyOZWyOadiGclx6euR+f1zGc7VCb/HCf/4+p/6i1/7zD/83O/93c/+np//TzADGJADGJADGJADGJADGJADGJADGJADGJADGKADGKADCLADCLAC+NAC6PASyRBiqUGSiYLiiZRCiaWyicdyedmCWfuiaf1zic7FWc/HSg/5Cp/6i1/73C/83O/93d/+rp//TzADeLADeLADeLADeLADeLADeLADeLADeLADeMADaMADaNADaOADWPADSRADKUATCXEy+ZMC+aRDCcXTCdeS+emS+fujOf1kKd7Fyc/Heg/5Kq/6q2/73D/83O/93d/+rp//TzAD+OAD+OAD+O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BjSoCjOpEDOrGTGuJTGvNDKwRzKyWzK0dDO2kjO3szW500G46lq0+3az/5G2/6m//77K/8/V/9ze/+np//PzATamATamATamATamATamATamATamATamATamAjanAjWnAzWoBTWpCDSqDTOsFzKuJDKwNDKxRzOyWzO0dDO2kjO4szW500K461u0+3ez/5G2/6q//7/K/8/U/9ze/+np//PzADimADemADemADemADemADenADenADenATenATenATeoAjaoAzapBTarCTWsFDSvJDOxMzOyRzOzWzS1dTS3kzS4sze500S461y0+3ez/5G2/6q//7/K/8/U/9ze/+np//PzADmnADmnADmnADmnADmnADmoADmoADmoADmoADmoADmpADipATiqAjesBTetEDavIjSyMTS0RTS1WzW2dTS4kzW5tDq500e46160+3iz/5K3/6u//7/K/9DU/9ze/+np//PzADyoADyoADyoADyoADyoADypADypADypADypADypADuqADurADusADqtAjmuDDiwHTezMDa2RDa3Wze4dji4lDq5tD+61Eu562G0+3qz/5S3/6y//7/K/9DU/9ze/+np//PzAECpAECpAECpAECpAECqAECqAECqAECqAECqAECqAD+rAD+sAD+tAD6uAD2wBjyyFju1Lzu3RTy4XT25dz+5lUG6tUa61FG56ma0+n60/5e3/63A/8DK/9DU/9ze/+np//TzAEirAEirAEirAEirAEirAEirAEirAEirAEerAEesAEesAEetAEauAEawAkWyB0S1F0O3LkO5R0W5Xka6eUi6l0q6tlC601u56m61+YO0/5q4/67A/8HK/9DU/9zd/+np//TzAFKtAFKtAFKtAFKtAFKtAFKtAFKtAFKtAFGuAFGuAFGvAFGvAFGwAFCyAlC0B0+2GE65Mk+6SVG6YVO6fFW7mFi7tl6602i56Hi1+Iu1/565/7HB/8PL/9HU/93d/+np//TzAF+xAF+xAF+xAF+xAF+xAF+xAF+xAF+xAF+xAF+yAF+yAF+zAF60AF61AV63A165Fl66MWC7T2K7ZmS7f2a7mmq7t2+60ne554O295O3/6S7/7XC/8XM/9LV/93d/+rp//TzAHG0AHK0AHK0AHK0AHK0AHK0AHK0AHK0AHK0AHK1AHK1AHG2AHG2AHG4A3G4DnK5I3K6PnO7U3W7bXi7hHu7nX66uIG60Yi55ZG39Z24/qu9/7rE/8jN/9PV/97e/+rp//TzAYi4AYi4AYi4AYi4AYi4AYi4AYi4AYi4Aoi4Aoi4A4i4BYi5CIi5DYi6FYm6IIm7Mom8RIq8XIy7co67ipC6oZK6uJW6z5m54qC58qm7/bS//7/G/8vO/9XW/9/e/+vp//XzAaG7AaG7AaG7AaG7AaG7AaG7AaG7AaG7AqG7AqG7BKG7BqG8C6G8E6G8HqK8LqK8Q6K8WKO8aaO7fKS7kKW7pae6uqm7zqy64LC78La+/L3C/8bI/8/P/9jW/+Hf/+zq//XzNru8Nru8Nru8Nru8Nru8Nru8N7u8OLu8OLu8Orq8PLq8Prq8Qrq8SLq8T7q8WLu8Ybq8a7q8eLq7h7u7l7u7qbu7vLy8zb283r++7sLA+8bE/8zK/9PQ/9vY/+Pg/+3q//bza866a866a866a866bM66bM66bM66bM66bM66bc66bs66b866cM66cs66dc66ec66f826hc27jc27l827o828scy9v82+z82/3s3A7M7D+dDH/9TM/9nS/9/Z/+bh/+7r//f0kN28kN28kN28kN28kN28kN28kN28kd28kd28kd28kd28kt29k929lN29ld29l929mt29ndy+oty+qdy/sNzAutzBxNzB0NvC3tvE69vG99vK/93P/+DU/+Tb/+rj//Hr//j0rOnDrOnDrOnDrOnDrOnDrOnDrOnDrOnDrOnDrOnDrOnDrenErenErunErunEr+nEsenEs+nEtunEuenFvejFw+jGyujH0+jI3ujJ6+jL9ubO/ubS/+fX/+rd/+7k//Ps//r1wPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwPPLwfPLwfPLwvPMw/PMxPPMx/PMyvPMzfPN0vPN2PPO4PLP6vLR9fLT/vDX//Db//Hg//Pm//fu//z2z/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zUz/zU0PzU0PzU0fzU0vzU0/zU1fzU1/zV2vvV3vvW5PvX6/vY8/ra/frc//jg//jk//jp//vw//733P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/c3P/d3P/d3f/d3f/d3f/d3v/d3//d4f/d4//e5v/e6f/f7v/g9P/h+//j///l//7p//7t///y///55//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l5//l6P/l6P/m6P/m6f/m6v/m6//m7f/n7//n8v/o9//p+//q///r///u///x///2///68P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8P/u8f/u8f/u8f/u8f/u8f/u8f/u8v/u8v/u8//u9P/v9v/v+P/v+v/w/f/x///y///z///2///5///9+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+P/1+f/1+f/1+f/1+f/2+f/2+f/2+v/2+v/2+//2/P/2/f/2///3///4///4///5///7///9/////v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8/v/8///8///9///9///9///9///9///+///+////////////C0TEC0TEC0TEDETEDETEDETEDETEDUTEDUTEDkTEEETFEkTFFkTFG0TGI0XGLUXHN0bIQkfIUEjJYknLd0vMkE7NrFPOylzO5mzN+IHI/5jH/63K/8DR/9Ha/9/j/+nq//PzC0TEC0TEC0TEC0TEDETEDETEDETEDUTEDUTEDkTED0TFEkTFFkTFG0TGIkXGLUXHN0bIQkfIUEjJYknLd0vMkE7NrFPOylzO5mzN+IHI/5jH/63K/8DR/9Ha/9/j/+nq//PzC0TEC0TEC0TEC0TEDETEDETEDETEDUTEDUTEDkTED0TFEkTFFUTFG0TGIkXGLUXHN0bIQkfJUEjKYknLd0vMkE7NrFPOylzO5mzN+IHI/5jH/63K/8DR/9Ha/9/j/+nq//PzC0TEC0TEC0TEC0TEC0TEDETEDETEDETEDUTEDkTED0TFEkTFFUTFG0TGIkXGLUXHN0bIQkfJUEjKYknLd0vMkE7NrFPOyl3O5mzN+IHI/5jH/63K/8DR/9Ha/9/j/+nq//PzC0TEC0TEC0TEC0TEC0TEC0TEDETEDETEDUTEDkTED0TFEUTFFUTFGkTGIkXGLUXHN0bIQkfJUEjKYknLd0vM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/9DU/9fY/97e/+Xk/+3r//X0dM7PdM7PdM7PdM7PdM7PdM7PdM7PdM7Pdc7Pdc7Pds7Pds7PeM7Pec7PfM7Pf87Pg87Pic7Pkc7Pms7Opc7Oss7OwM7Ozs7O3c/P69HQ99TS/tfW/9za/+Lf/+jl/+/s//b0k97Nk97Nk97Nk97Nk97Nk97Nk97Nk97Nk97Nk97NlN7NlN7Nld7Nlt7Nl97Nmd7NnN3NoN3Npd3Oqt3Ost3Ou9zPxtzP0dzQ3tzR6t3S9t3V/9/Y/+Lc/+bg/+vm//Ht//j0rerPrerPrerPrerPrerPrerPrurPrurPrurPrurPrunPrunPr+nPr+nPsOnPsenPs+nQtenQuOnQu+nRwOnRxunSzejT1ejT3+jU6ujV9ejX/uja/+ne/+zi/+/o//Tu//n1w/PVw/PVw/PVw/PVw/PVw/PVw/PVw/PVw/PVw/PVw/PVw/PVw/PVxPPVxPPVxfPVxfPVx/PVyPPVyvPWzfPW0PPX1fPX2vLX4vLY6vLZ9PLb/vHd//Hh//Ll//Tq//fv//z20vzb0vzb0vzb0vzb0vzb0vzb0vzb0vzb0vzb0vzb0vzb0vzb0vzb0/zb0/zb0/zb0/zb1Pzb1fzc1vzc2Pvc2vvc3fvc4fvd5vve7Pve9Pvg/fri//nk//no//ns//vx//733v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3v/i3//i3//i3//i4P/i4P/i4f/i4//j5f/j5//j6//k7//k9f/l+//n///p//7s///v///z///56P/p6P/p6P/p6P/p6P/p6P/p6P/p6P/p6P/p6P/p6P/p6P/p6f/p6f/p6f/p6f/p6f/p6f/p6f/p6v/p6v/p6//p7f/q7v/q8P/q8//r9//s/P/t///u///w///z///3///78f/w8f/w8f/w8f/w8f/w8f/w8f/w8f/w8f/w8f/w8f/w8f/w8f/w8v/w8v/w8v/w8v/w8v/w8v/w8v/w8//w8//w9P/x9f/x9v/x+P/x+//y/v/z///0///1///3///6///9+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+f/3+v/3+v/3+v/3+//3+//3/P/4/f/4///4///5///5///6///8///9/////v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9/v/9///9///9///9///9///+///+///+///+////////////////Nl3cNl3cNl3cNl3cNl3cNl3cN13cN13cN13cOF3cOV3cO13cPV7cQF7cRF/dSWDdUWHdWGLdY2PecGXegWjflGvfqnDfwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0Nl3cNl3cNl3cNl3cNl3cNl3cN13cN13cN13cOF3cOV3cO13cPV7cQF7cRF/dSWDdUWHdWGLdY2PecGXegWjflGvfqnDfwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0Nl3cNl3cNl3cNl3cNl3cNl3cN13cN13cN13cOF3cOV3cO13cPV7cQF7cRF/dSWDdUGHdWGLdY2PecGXegWjflGvfqnHfwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0Nl3cNl3cNl3cNl3cNl3cNl3cN13cN13cN13cOF3cOV3cO13cPV7cQF7cRF/dSWDdUGHdWGLdY2PecGXegWjflGvfqnHfwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0Nl3cNl3cNl3cNl3cNl3cNl3cNl3cN13cN13cOF3cOV3cO13cPV7cQF7cRF/dSWDdUGHdWGLdY2TecGXegWjflGvfqnHfwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0Nl3cNl3cNl3cNl3cNl3cNl3cNl3cN13cN13cOF3cOV3cO13cPV7cQF7dRF/dSWDdUGHdWGLdY2TecGXegWjflGvfqnHgwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0Nl3cNl3cNl3cNl3cNl3cNl3cNl3cN13cN13cOF3cOV3cO13cPV7cQF7dRF/dSWDdUGHdWGPdY2TecGXegWjflGvfqnHgwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0NV3cNV3cNl3cNl3cNl3cNl3cNl3cN13cN13cOF3cOV3cOl3cPV7dQF7dRF/dSWDdUGHdWGPdY2TecGbegWjflGvfqnHgwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0NV3cNV3cNV3cNV3cNl3cNl3cNl3cNl3cN13cOF3dOV3dOl7dPV7dQF7dRF/dSWDdUGHdWGPdY2TecGbegWjflGzfqnHgwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0NV3dNV3dNV3dNV3dNV3dNV3dNl3dNl3dN13dOF3dOV7dOl7dPF7dP17dQ1/dSGDdT2LdWGPeY2TecGbegWjflGzfqnHgwnjf3ILf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0NV3dNV3dNV3dNV3dNV3dNV3dNV3dNl3dNl3dN17dOF7dOl7dPF7dP1/dQ1/dR2DdTmLeWGPeY2Teb2bfgWjflGzgqnHgw3ng3IPf85Dd/qHY/7PX/8Ta/9Pg/+Dn/+vu//P0NF7dNF7dNF7dNF7dNF7dNF7dNF7dNV7dNV7dNl7dN17dOV7dO1/dPl/dQmDeRmHeTWLeWGTeYmXfb2ffgWnflGzgqnHgw3ng3IPf85Dd/qLY/7PX/8Ta/9Pg/+Dn/+vu//P0Ml7eMl7eMl7eMl7eMl7eM17eM17eM17eNF7eNV7eNl7eN1/eOl/ePV/eQGDeRWHeS2PeV2TfYmXfb2ffgWnglG3gq3Lgw3ng3IPf85Dd/qLY/7TX/8Ta/9Pg/+Dn/+vu//P0MF/eMF/eMF/eMF/eMF/eMV/eMV/eMV/eMl/eM1/eNF/eNV/eOGDfOmDfPmHfQmLfSWPfVWXfYmbfb2jggWrglG7gq3Pgw3rg3IPf85Hd/qLY/7TX/8Ta/9Pg/+Dn/+vu//P0LV/fLV/fLV/fLV/fLV/fLmDfLmDfLmDfL2DfMGDfMWDfMmDfNGDgN2HgO2LgP2PgRmTgUmbgYmjgbmnggWvhlG7hq3Phw3rg3YTf85Ld/qPZ/7TX/8Ta/9Pg/+Dn/+vu//P0KWHgKWHgKWHgKmHgKmHgKmHhKmHhKmHhK2HhLGHhLWHhLmHhMGLhMmLhNmPhPGThQmXhUGfhYWnhbmvhgG3hlXDiq3XhxHzh3Yff85Td/qTZ/7XX/8Xa/9Tg/+Dn/+vu//P0H2TgIGTgIGTgIGTgIGTgIGTgIGTgIWTgImThI2ThJGThJmThKWTiLWTiMmTjN2XjP2fjTGjjXGrjbm3jgG/jlXLirHjixIDh3Yvf85fd/qbZ/7bX/8bb/9Tg/+Hn/+vu//P0EmjgEmjgE2jgE2jgE2jgE2jgE2jgFGjgFWjgFmjgF2jgGWjhHGjhIGjiJmjjL2nkPGrkSWzkWm/kb3Pjgnbjl3rirX/hxYfg3Y/f85rd/qnZ/7jY/8fb/9Xg/+Hn/+vu//P0DXLhDXLhDnLhDnLhDnLiDnLiDnLiD3LiEHLiEXLiEnLiFHLjF3LjHHLkInLkLnPlO3XlTXfkYHrkcn3jhoHjmoTir4nhxo7g3Zbf8qDd/q3Z/7vY/8nb/9bh/+Hn/+vu//P0EH7kEH7kEH7lEH7lEH7lEH7lEX7lEX7lEX7lEn7lFH7lFn7lG3/lIX/lKIDlMYHlPoLlUITkZ4fkd4rji47inpHhsZXgx5rg3aDf8Kjc/bPZ/7/Z/8zc/9jh/+Po/+zu//P0GJDjGJDjGJDjGJDjGZDjGZDjGpDjGpHjG5HjHZHjH5HjIpHjJpHjLJLjNZLjQZPkUpTkYpXjb5fjf5rikZ3ho6DgtaPfyKff3Kve77Hc/Lra/8Ta/8/d/9ri/+To/+zu//T0SKTjSKTjSKTjSKTjSKTjSKTjSaTjSaTjSqTjS6TjTKTjTqTjUaTjVKXjWaXjX6bjZqbib6fie6niiavhmK3gqbDfubLeyrXe3Ljd7bzc+sPb/8vb/9Tf/93j/+bp/+3u//T0ZrjiZrjiZrjiZrjiZrjiZrjiZ7jhZ7jhZ7jhaLjhabjharjha7jhbrnhcbnhdbnhernhgLrhibvglLzgoL3frr/evcHezMPd3MXd68jc+Mzc/9Ld/9ng/+Hl/+jp/+7v//X0gMzggMzggczggczggczggczggczggczggczggszggszgg8zghMzghszgiMzgiszfjszfkszfmM3foM3fqs7etc7ewc/dz9Dd3dHd6tPd9tbd/9rf/9/i/+Tm/+rq//Dv//b1mN7emN7emN7emN7emN7emN7emN7emN7emd7emd7emd7emt7emt7em97end7ent3eoN3eo93eqN3erd3etN3evd3ex93d0t3d3t7e6d/e9eDf/uLh/+Xj/+nn/+3r//Lw//j1sOrcsOrcsOrcsOrcsOrcsOrcsOrcsOrcsOrcsOrcsOrcsercsercsercsurcs+rdterdturduerdverdwendx+ndz+ne1+ne4Onf6ung9Onh/erj/+vl/+3o//Hs//Xx//r2xPTexPTexPTexPTexPTexPTexPTexPTexPTexPTexPTexPTexfTexfTexfTexvTex/PeyPPfyvPfzPPfz/Pf0vPg1/Pg3fPh4/Lh6/Li9PPj/fLl//Ln//Pq//Xu//jy//z31fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1fzj1vzj1vzj1/zj2Pvj2fvj2vvk3fvk3/vk4/vl5/vl7fvm9Pvm/Pro//rq//rt//rw//z0//744f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4f/o4v/o4v/o4//o5P/o5f/p5//p6f/p7P/p8P/q9v/r+//s///t///v///y///2///66v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6v/t6//t6//t6//u6//u6//u7P/u7P/u7f/u7v/u8P/u8v/u9f/v+P/v/P/w///x///z///1///4///88//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z8//z9P/z9P/z9f/09v/09//0+f/0/P/1/v/1///2///3///5///7///++v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+v/5+//5+//5+//5/P/5/f/5/v/6///6///6///7///8///9///+///////+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+///+////////////////////////////////////X3vtX3vtX3vtX3vtX3vtYHvtYHztYHztYHztYXztYXztYnztY3ztZX3tZ33ta37tcH/tdYHtfILthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3vtX3vtX3vtX3vtX3vtYHvtYHztYHztYHztYXztYXztYnztY3ztZX3tZ33ta37tcH/tdYHtfILthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3vtX3vtX3vtX3vtX3vtYHvtYHztYHztYHztYXztYXztYnztY3ztZX3tZ33ta37tcH/tdYHtfILthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3vtX3vtX3vtX3vtX3vtYHvtYHztYHztYHztYXztYXztYnztY3ztZX3tZ33ta37tcH/tdYHtfILthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3vtX3vtX3vtX3vtX3vtYHztYHztYHztYHztYXztYXztYnztY3ztZX3tZ33ta37tb3/tdYHtfILthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3vtX3vtX3vtX3vtX3ztX3ztYHztYHztYHztYXztYXztYnztY3ztZX3tZ33ta37tb3/tdYHtfILthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3vtX3vtX3ztX3ztX3ztX3ztYHztYHztYHztYHztYXztYnztY3ztZX3tZ33ta37tb3/tdYHtfIPthYTtkIbtnojtrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3ztX3ztX3ztX3ztX3ztX3ztYHztYHztYHztYHztYXztYnztY3ztZX3tZ33ta37tb3/tdYHtfIPthYTtkIbtnontrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3ztX3ztX3ztX3ztX3ztX3ztX3ztYHztYHztYHztYXztYnztY33tZX3tZ37ta37tb3/tdYHtfIPthYTtkIbtnontrYztwJHt1Jjs6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3ztX3ztX3ztX3ztX3ztX3ztX3ztYHztYHztYHztYXztYnztY33tZX3tZ37tan7tb4DtdYHtfIPthYTtkIbtnontroztwJLt1Jjt6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3X3ztX3ztX3ztX3ztX3ztX3ztX3ztX3ztYHztYHztYXztYnztY33tZH3tZ37tan/tb4DtdIHtfIPthYTtj4btnontro3twJLt1Jnt6aHs/Kzp/7rm/8jl/9Xn/+Hs/+zx//X3XnztXnztXnztXnztX3ztX3ztX3ztX3ztX3ztYHztYHztYX3tY33tZH3tZ37tan/tboDtdIHtfIPthYXtj4ftnontro3twJLt1Jnt6aHs/Kzq/7rm/8jl/9Xn/+Hs/+zx//X3XnztXnztXnztXnztXnztXnztXnztXnztX3ztX33tYH3tYX3tYn3tZH7tZn7taX/tboDtc4LtfITthYXtj4funonuro3uwJLt1Jnt6aLs/Kzq/7rm/8jl/9Xn/+Hs/+zx//X3XX3uXX3uXX3uXX3uXX3uXX3uXX3uXX3uXn3uXn3uX33uYH3uYX7uY37uZX/uaH/ubIDucoLue4TuhYbuj4funoruro7uwJLt1Znt6qLs/K3q/7vm/8nl/9Xn/+Hs/+zx//X3W33uW33uW33uW33uW33uW33uXH3uXH3uXH3uXH3uXX7uXn7uX37uYX7uY3/uZ4Dua4HucIPueoXuhYbuj4junovuro7uwJPu1Zrt6qLs/K3q/7vm/8nl/9Xn/+Hs/+zx//X3WX7vWX7vWX7vWX7vWX7vWX7vWX7vWn7vWn7vWn7vW37vXH7vXX/vX3/vYYDvZYHvaYLvboPveIXvhYjvj4nvnozvro/uwZTu1Zrt6qPs/K7q/7vm/8nl/9Xn/+Ls/+zx//X3VX/wVX/wVX/wVX/wVX/wVn/wVn/wVn/wV3/wWH/wWH/wWX/wWoDwXIDwXoHwYYHwZoPwa4TwdYbwhInvjovvno3vrpDvwZXu1Zvt6qXs/LDq/7zm/8nl/9bn/+Ls/+zx//X3R4HxR4HxR4HxR4HxR4HxR4HxSIHxSIHxSYHxSoHxS4HxTYHyUIHyVIHyWYLyXoPyYoTyaIXxc4fxgYrxjo3wnY/wr5Lvwpjv1p/t6qjs/LLp/77m/8rl/9fn/+Ls/+zy//X3NIL0NIL0NYL0NYL0NYL0NYL0NYL0NoL0N4L0OIP0OYP0O4P0PoP0QoT0SITzUYXzXIfzZ4jzcovyf47ykJLxn5XwsJnvw57u1qTt6qvs/LTp/8Dm/8zl/9jo/+Ps/+3y//X3Poz0Poz0Poz0Poz0Poz0Poz0P4z0P430QI30QY30Qo30RY30R430TI7zUY7zWY/zYJDza5LydpTyhJfxk5vwo57vs6HvxKXu16rt6rHr+7np/8Pm/8/m/9no/+Ts/+3y//X3T5nzT5nzT5nzT5nzT5nzUJnzUJnzUZnzUZnzUpnzU5rzVZrzV5rzWZrzXZvzYpvzaJ3ycJ7yfKDxiqPxmKbwp6nvtqzuxq/t17Ps6bnr+r/p/8jm/9Lm/9zp/+Xt/+7y//X3YqnyYqnyYqnyYqnyY6nyY6nyY6nyY6nyY6nyZKnyZanyZqryZ6ryaarybKrxcKvxdqzxfa3xiK7wkrDvnrPvrbbuurjtybvs2L7r6cLq+Mfp/87n/9bn/97q/+fu/+/y//b3eLrweLrweLrweLrweLrweLrweLrwebrwebrwebrwerrwe7rwfLrwfrvwgbvwhLvviLzvjb3vk77unL/upsDtssLsv8XrzMfr2snq6czq98/p/9To/9vo/+Lr/+nu//Dz//f3jsvujsvujsvujsvujsvujsvujsvujsvuj8vuj8vuj8vukMvukMvukczuk8zulcztmMztnMztoc3tp87sr87sudDrxNHr0NPq3NTq6dbq9djp/tzo/+Hp/+bs/+zv//L0//j4otvsotvsotvsotvsotvsotvsotvsotvsotvsotvso9vso9vso9vspNvspdvsp9zsqdzsq9zsr9zrs9zrudzrwd3ryd3q097q3t/p6eDp9OLp/eTq/+fr/+vt/+/x//T0//n4s+rrs+rrs+rrtOrrtOrrtOrrtOrrtOrrtOrrtOrrtOrrtOrrterrterrturrt+rruOrruurqvOrqv+rqxOrqyerq0Orq2Onq4Orq6urq8+vq/ezr/+3t//Dv//Px//b1//v5xvTpxvTpxvTpxvTpxvTpxvTpxvTpxvTpxvTpxvTpxvTpxvTpx/Tpx/Tpx/TpyPTpyfTpyvTpy/TpzfTp0PTp1PPp2PPq3vPq5PPr7PPr9PPs/PPs//Tu//Xw//fz//n2//351vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1vzq1/zq1/zq2Pzr2fzr2vzr3Pvr3vvr4fvs5fvs6fvt7/vt9fvu/Pvu//rw//vy//v0//33///74//u4//u4//u4//u4//u4//u4//u4//u4//u4//u4//u4//u4//u4//u4//u4//u5P/u5P/u5f/u5f/u5v/v5//v6f/v6//v7v/w8v/w9//w/P/x///y///0///2///5///87P/y7P/y7P/y7P/y7P/y7P/y7P/y7P/y7P/y7P/y7P/y7P/y7f/y7f/y7f/y7f/y7f/y7f/y7f/y7v/y7v/z7//z8P/z8f/z8//z9v/z+f/0/P/0///1///2///4///7///+9P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39P/39f/39f/39f/39f/39v/39v/39//3+f/3+v/4/f/4/v/4///5///6///7///9///++//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7+//7/P/8/P/8/P/8/f/8/f/8/v/8///8///8///9///9///+////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6L4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6L4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6L4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6L4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6L4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6L4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpn4hpr4h5r4iJr4iZr4i5v4jZz4kZ34lp74m6D4oqH4q6P4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpr4hpr4h5r4iJr4iZv4i5v4jZz4kZ34lp74m6D4oqH4q6P4tqX4w6j30q334rL29Lr2/8L0/83w/9nw/+Px/+31//X6hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hZn4hpr4hpr4h5r4iJr4iZv4i5v4jZz4kZ34lp74m6D4oqH4q6P4tqX4w6j30q334rL29Lr2/8P0/87w/9nw/+Px/+31//X6hJn4hJn4hJn4hZn4hZn4hZn4hZn4hZn4hZn4hZr4hpr4hpr4h5r4h5r4iZv4i5v4jZz4kZ34lp74m6D4oqH4q6P4tqX4w6j40q334rL29Lr2/8P0/87w/9nw/+Px/+31//X6hJn5hJn5hJn5hJn5hJn5hJn5hZn5hZn5hZr5hZr5hZr5hpr5hpr4h5r4iZv4ipv4jZz4kZ34lp/4m6D4oqH4q6P4tqX4w6j40q334rP29Lr2/8P0/87w/9nw/+Px/+31//X6hJn5hJn5hJn5hJn5hJr5hJr5hJr5hJr5hZr5hZr5hZr5hpr5hpr5h5r5iJv5ipv4jZz4kJ34lZ/4m6D4oqH4q6P4tqX4w6n40q334rP39Lr2/8P0/87w/9nw/+Px/+31//X6hJr5hJr5hJr5hJr5hJr5hJr5hJr5hJr5hJr5hZr5hZr5hZr5hpr5h5v5iJv5ipv5jZz5kJ34lZ/4m6D4oqL4q6P4tqX4w6n40q334rP39Lr2/8P0/87w/9nw/+Px/+31//X6g5r5g5r5g5r5g5r5g5r5g5r5hJr5hJr5hJr5hJr5hJr5hZr5hpr5h5v5iJv5ipz5jJz5kJ35lJ/5m6H4oqL4q6T4tqb4w6n40q334rP39Lr2/8P0/87w/9nw/+Px/+31//X6g5r5g5r5g5r5g5r5g5r5g5r5g5r5g5r5g5r5g5r5hJr5hJv5hZv5hpv5h5v5iZz5i535j575k5/5m6H5oqL4q6T4tqb4w6n40q3347P39Lr2/8P0/87w/9nw/+Px/+31//X6gZv6gZv6gZv6gZv6gZv6gpv6gpv6gpv6gpv6gpv6gpv6g5v6hJv6hZv6hpz6iJz6ip36jp75kqD5mqH5oqP5qqX5t6f4w6r40q7447P39Lv2/8P0/87w/9nw/+Px/+31//b6gJv7gJv7gJv7gJv7gJv7gJv7gJv7gJv6gJv6gZv6gZv6gZv6gpz6g5z6hJz6hp36iZ76jJ/6kaD6mKL6oqT5qqb5t6f5w6r40q7447T39Lv2/8T0/8/w/9nw/+Px/+31//b6fZz7fZz7fZz7fZz7fZz7fZz7fpz7fpz7fpz7f5z7f5z7f5z7gJz7gZ37gp37hJ77h577ip/7jqH7lqP6oqX6qqf6tqn5xKv506/447X39L32/8X0/8/w/9rw/+Ty/+31//b6dZ39dZ39dZ39dZ39dZ39dZ39dZ39dZ39dp39dp39d539d539eZ39ep79fZ78gZ/8hJ/8h6D8jKL8lKP7n6b7qqj6tqr6xK3507L45Lj39L/2/8f0/9Dw/9vw/+Xy/+72//b6bZ79bp79bp79bp79bp79bp79bp79bp79b579b579cJ/9cZ/9cp/9dJ/9dqD9eaD9fqH9hKL9i6T8lKb8n6n7q636uLD6xbL51Lf45Lz39ML2/8n0/9Lx/9zw/+Xy/+72//b6dKf9dKf9dKf9dKf9dKf9dKf9dKf9dKf9dKf9daf9daf9dqf9d6j9eaj9e6j9fqn9gqr9h6v8jqz8mK77orH7rrT6u7f5x7n41b345MH39Mb1/83z/9Xx/97w/+fy/+/2//b6fbL9fbL9fbL9fbL9fbL9fbL9fbL9frL9frL9frL9f7L9f7L9gLL9gbL9g7L9hrP8ibT8jrX8lLb7nbj7p7v6sr35vsD4ysL318X35cj29Mz1/9Lz/9nx/+Hx/+jz//D2//f7jL/7jL/7jL/7jL/7jL/7jL/7jL/7jL/7jL/7jL/7jb/7jb/7jr/7j7/7kMD7ksD7lcD7mcH6n8L6psT6rsX5t8f4w8r3zcz22c725tD189P0/9jz/93x/+Ty/+v0//H3//j7m836m836m836m836m836m836m836m836m835nM35nM35nM35nc35ns35n835oc75o875p875qs/4r9D4tdH4vtL3x9T20tb13Nf159n089v0/d7z/+Py/+jy/+30//P4//n7rNv3rNv3rNv3rNv3rNv3rNv3rNv3rNv3rNv3rNv3rNv3rdv3rdv3rtv3rtv3r9v3sdv3s9z3ttz3utz2v932xd72zd711eD13+H06OL08+Tz/Obz/+nz/+zz//H2//b4//v8u+j2u+j2u+j2u+j2u+j2u+j2u+j2u+j2u+j2u+j2u+j2vOj2vOj2vOj2vej2vuj2v+j1wOj1wuj1xej1yen1zen10+n02ur04ur06uvz8+zz/O3z/+/0//L1//X3//j5//z8yfT1yfT1yfT1yfT1yfT1yfT1yfT1yfT1yfT1yfT1yfT1yfT1yfT1yvT1yvT1yvT0y/T0zPT0zvT00PT00vT01fT02vT03/T05fT07PT09PT0/PT0//X1//f2//n4//v6//791/zz1/zz1/zz1/zz1/zz1/zz1/zz1/zz1/zz1/zz1/zz1/zz1/zz2Pzz2Pzz2Pzz2fzz2fzz2vzz2/zz3fzz3/zz4vzz5vv06vv07/v09fv1/Pv1//v2//z3//35//77///+5P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05P/05f/05f/05v/15//16P/16v/17f/18P/28//2+P/2/P/3///3///5///6///8///+7v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37v/37//37//37//38P/38P/48f/48v/48//49f/49//4+v/5/f/5///6///6///8///9////9v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79v/79//79//79//7+P/7+f/7+v/7+//7/v/8///8///8///9///+/////////P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/P/+/f/+/f/+/f/+/f/+/f/+/f/+/v/+///+///+////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sbf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sbf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sbf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sbf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sLf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sLf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLL+pLL+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbX/sLf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLP+pLP+pLP+pLP+pbP+prP+p7P+qLT+qrX/rbb/sLf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+pLP+pLP+pLP+pLP+pbP+pbP+prP+qLT+qrX/rbb/sLf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7L+o7P+pLP+pLP+pLP+pLP+pbP+pbP+prT+qLT+qrX/rbb/sLf/tbj/urn/wbr/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7L+o7L+o7L+o7L+o7L+o7P+o7P+o7P+o7P+o7P+pLP+pLP+pLP+pbP+pbP+prT+qLT/qrX/rLb/sLf/tbj/urn/wbv/yrz/1L//4MP+7sf+/M39/9T8/935/+b4/+76//b9o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+pLP+pLP+pLP+pbP+pbP+prT+qLT/qrX/rLb/sLf/tbj/urn/wbv/yr3/1L//4MP+7sf+/M3+/9T8/935/+b4/+76//b9o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+pLP+pLP+pbP+pbP+prT/qLT/qbX/rLb/sLf/tbj/urn/wbv/yr3/1L//4MP+7sj+/M3+/9T8/935/+b4/+76//b9o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+o7P+pLP+pLP/pbT/prT/p7T/qbX/rLb/r7f/tbn/urr/wbv/yr3/1L//4MP+7sj+/M3+/9T8/935/+b4/+76//b9orP/orP/orP/orP/orP/orP/orP/orP/orP/o7P/o7P/o7P/o7P/pLP/pbT/pbT/p7T/qbX/q7b/r7f/tbn/urr/wbv/yr3/1MD/4MP+7sj+/M7+/9X8/935/+b4/+76//b9obP/obP/obP/obP/obP/obP/obP/orP/orP/orP/orP/orP/o7T/o7T/pLT/pbT/prX/qLX/q7b/rrf/tLn/urr/wbz/yr3/1MD/4MP+7sj+/M7+/9X8/975/+b4/+/6//b9oLP/oLP/oLP/oLP/oLP/oLP/oLP/obT/obT/obT/obT/obT/orT/orT/o7T/pLX/pbX/p7b/qrf/rrj/tLn/urv/wbz/yr7/1MD/4cT/7sj+/M7+/9X8/975/+b4/+/6//b9n7T/n7T/n7T/n7T/n7T/n7T/n7T/n7T/oLT/oLT/oLT/oLT/oLT/obT/orX/o7X/pLX/prb/qbf/rbj/s7r/urz/wb3/yr7/1MH/4cT+7sn+/M7+/9b8/975/+f4/+/6//f9nbT/nbX/nbX/nrX/nrX/nrX/nrX/nrX/nrX/nrX/nrX/n7X/n7X/oLX/oLX/obb/o7b/pbf/qLj/q7n/sbr/urz/wb7/yr//1cL/4cX+7sn+/ND+/9b8/9/5/+f5/+/6//f9mLX/mLX/mLX/mLX/mLX/mLX/mLX/mLX/mbX/mbX/mbX/mbX/mrX/m7b/nLb/nrb/oLf/o7f/pbj/qbn/r7v/uL3/wb//ysH/1cP/4cf+78z+/NH+/9j8/9/5/+f5/+/6//f9krb/krb/krb/krb/krb/krb/krb/krb/krb/k7b/k7b/k7b/lLf/lbf/lrf/mLj/m7j/nrn/o7r/qLv/r73/uMD/wsP/zMX/1sf+4sv+78/++9T+/9n8/+H5/+n5//D6//f+l73/l73/l73/l73/l73/mL3/mL3/mL3/mL3/mL3/mL3/mb3/mb7/mr7/m77/nb7/n7//osD/psH/q8L/ssT/u8b/xMj/zsv/2M3+48/+79P++9f9/9z7/+P5/+r5//H7//j+n8b/n8b/n8b/n8b/n8b/n8b/n8b/n8b/n8b/n8b/n8b/oMb/oMb/ocb/osb/o8f/pcf/p8j/q8n/sMr/t8v/v83/x9D+0NL+2tT+5Nb+8Nn++9z9/+D7/+b6/+z6//P7//n+qdH/qdH/qtH/qtH/qtH/qtH/qtH/qtH/qtH/qtH/qtH/qtH/q9H/q9H/rNH/rtL/r9L/sdL/tNP/uNT/vtX/xNb+y9j+1Nr+3dz+5t3+8N/9++L8/+X7/+r6/+/6//T8//r+ttz+ttz+ttz+ttz+ttz+ttz+ttz+ttz+ttz+ttz+ttz+ttz+t9z+t9z+uNz+ud3/ut3/vN3/v97+wt7+xd/+yt/+0eH+2OL+4OT96OX98eb8++j8/+v7/+76//L7//f8//z/w+j/w+j/w+j/w+j/w+j/w+j/w+j/w+j/w+j/w+j/w+j/w+j/w+j/xOj/xOj/xej/xej+x+j+yOj+yuj+zun+0un+1ur93Ov94+z86+388+78++/8//D7//P7//b8//r9//3/z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L9z/L90PL90PL90fL90fL90/P91PP91vP92fP83fP84fP85/T87fT89PX8+/X8//b8//j8//r9//3+////2vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82vz82/z82/z83Pz83fz83/z84fz84/z85/z86/z88Pz89vz8/Pz8//z8//39//7+///+////5f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75f/75v/75v/75v/75//76P/76f/76//87f/88P/89P/8+P/8/f/9///9///+///+////////7//87//87//87//87//87//87//87//87//87//87//87//87//87//87//87//87//88P/88P/88P/88f/98v/98//99P/99v/9+P/9/P/+/v/+///+///+///+////////9//+9//+9//+9//+9//+9//+9//+9//+9//+9//+9//+9//+9//+9//+9//+9//++P/++P/++P/++P/++P/++f/++f/++v/++//+/f/+/v/+/////////////////////////f///f///f///f///f///f///f///f///f///f///f///f///f///f///v///v///v///v///v///v///v///v///v///v//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycv/zsz/083/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycv/zsz/083/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycv/zsz/083/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycv/zsz/083/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycv/zsz/083/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycv/zsz/087/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zsz/087/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zsz/087/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcf/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zsz/087/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vMf/vcj/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zsz/087/2s//4tH/7NT/99j//93//+L//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vMf/vcj/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zcz/087/2s//4tH/7NT/99j//93//+P//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vcj/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zc3/087/2s//4tH/7NT/99j//93//+P//+n///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMf/vMf/vcj/vcj/vsj/v8j/wcn/w8r/xsv/ycz/zc3/087/2s//4tL/7NT/99j//93//+P//+r///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMf/vMf/vMj/vcj/vcj/vsj/v8n/wMn/wsr/xcv/ycz/zc3/087/2s//4tL/7NT/99j//93//+P//+r///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/vMf/vMj/vMj/vMj/vcj/vsj/v8n/wMn/wsr/xcv/ycz/zc3/087/2tD/4tL/7NX/99j//93//+P//+r///H+//j/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8f/u8j/u8j/u8j/u8j/u8j/vMj/vMj/vcj/vcj/vsn/wMn/wsr/xcv/ycz/zc3/087/2tD/4tL/7NX/99j//93//+P//+r///H+//j/usj/usj/usj/usj/usj/usj/usj/usj/usj/usj/u8j/u8j/u8j/u8j/u8j/vMj/vcn/vsn/v8n/wcr/xMv/ycz/zc3/08//2tD/4tL/7NX/99n//93//+P//+r///H+//j/ucj/ucj/ucj/ucj/ucj/ucj/ucj/ucj/usj/usj/usj/usj/usj/usj/u8j/u8j/vMn/vcn/v8r/wcr/w8v/yM3/zc7/08//2tD/4tL/7NX/99n//93//+P//+r///H+//j/uMj/uMj/uMj/uMj/uMj/uMj/uMj/uMj/uMj/uMj/ucj/ucj/ucj/ucj/usn/usn/u8n/vMn/vsr/wMv/w8z/yM3/zc7/09D/2tH/49P/7Nb/99n//97//+T//+r///H+//j/t8n/t8n/t8n/t8n/t8n/t8n/t8n/t8n/t8n/t8n/t8n/uMn/uMn/uMn/uMn/ucn/usr/u8r/vMr/v8v/wsz/x83/zc//09D/2tL/49P/7db/+Nr//9///+T//+v///H+//j/s8n/s8n/s8n/s8n/s8n/s8n/s8n/s8n/s8n/s8n/tMn/tMn/tMn/tcn/tcr/tsr/t8r/ucv/u8v/vcz/wM3/xc7/zND/09H/2tP/49X/7dj/+Nz//+D//+X//+v///L+//j/rsr/rsr/rsr/rsr/rsr/rsr/rsr/rsr/rsr/r8r/r8r/r8r/r8r/sMr/sMv/scv/s8v/tcz/t8z/u83/v87/xND/y9L/1NT/29b/5Nj/7tv/+N7//+L//+f//+3///P///n/ss//ss//ss//ss//ss//ss//ss//ss//stD/s9D/s9D/s9D/s9D/tND/tND/tdD/t9H/udH/u9L/vtL/wtP/x9X/ztf/1dn/3dv/5dz/7t7/+OH//+X//+n//+7///T///r/uNf/uNf/uNf/uNf/uNf/uNf/uNf/uNf/uNf/udf/udf/udf/udf/udf/utf/u9f/u9f/vdj/v9j/wtn/xtr/y9v/0d3/2N7/3+D/5+L/7+T/+eb//+n//+z///H///b///v/wd//wd//wd//wd//wd//wd//wd//wd//wd//wd//wd//weD/weD/wuD/wuD/w+D/xOD/xeD/x+H/yeH/zOL/0eP/1eT/2+X/4uf/6ej/8er/+ev//+3///D///T///j///z/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/y+n/zOn/zOn/zen/zun/z+n/0er/0+r/1+v/2uv/4Oz/5e3/7O//8/D/+vH///L///T///f///v///7/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1fL/1vL/1vL/1vL/1/L/2PL/2fL/2/P/3fP/4PP/5PT/6fT/7vX/9fb/+/f///j///n///v///7/////3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3vv/3/v/3/v/3/v/4Pv/4Pv/4fv/4vv/5Pv/5vv/6fv/7fz/8fz/9vz//P3///3///7/////////////5///5///5///5///5///5///5///5///5///5///5///5///5///5///5///5///5///6P//6P//6f//6v//6///7P//7///8f//9f//+f///f//////////////////////8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8P//8f//8f//8v//8v//8///9f//9///+f///P//////////////////////////+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+P//+f//+f//+f//+v//+////P///v///v///////////////////////////v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1dr/19v/2tv/3dz/4d3/597/7uD/9uL//uX//+n//+7///P///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1Nr/19v/2tv/3dz/4d3/597/7uD/9uL//ub//+n//+7///P///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0Nj/0dj/0tn/09n/1Nr/19v/2tv/3dz/4d3/597/7uD/9uP//ub//+n//+7///T///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/z9j/0Nj/0dn/0tn/09n/1Nr/19v/2tv/3dz/4d3/597/7uD/9uP//ub//+n//+7///T///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/z9j/0Nj/0dn/0tn/09n/1Nr/1tv/2tz/3dz/4d3/597/7uD/9uP//ub//+n//+7///T///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/z9j/0Nj/0dn/0dn/09n/1Nr/1tv/2tz/3dz/4d3/59//7uD/9uP//ub//+n//+7///T///n/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9j/z9j/z9j/0Nn/0Nn/0dn/0tn/1Nr/1tv/2tz/3d3/4d3/59//7uD/9uP//ub//+n//+7///T///r/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/ztj/z9n/z9n/0Nn/0dn/0tr/09r/1tv/2dz/3d3/4d7/59//7uH/9uP//ub//+r//+7///T///r/zdj/zdj/zdj/zdj/zdj/zdj/zdj/zdj/zdj/zdj/zdj/zdj/zdj/ztn/ztn/ztn/z9n/z9n/0Nn/0dr/09r/1dv/2dz/3d3/4d7/59//7uH/9uP//ub//+r//+7///T///r/zNn/zNn/zNn/zNn/zNn/zNn/zNn/zNn/zNn/zNn/zNn/zdn/zdn/zdn/zdn/zdn/ztn/ztn/z9r/0Nr/0tv/1Nv/2Nz/3d7/4d//5+D/7uH/9uP//ub//+r//+////T///r/y9n/y9n/y9n/y9n/y9n/y9n/y9n/y9n/y9n/y9n/y9n/y9n/zNn/zNn/zNn/zNn/zdn/zdr/ztr/z9r/0dv/09z/193/3d7/4d//5+D/7uL/9uT//uf//+v//+////T///r/yNn/yNn/yNn/yNn/yNn/yNn/yNn/yNn/yNn/yNn/ydn/ydn/ydn/ydn/ydr/ytr/ytr/y9r/zdr/ztv/0Nv/0tz/193/3N//4eD/5+H/7uP/9uX//uj//+z///D///X///r/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xdr/xtr/xtr/x9v/yNv/ydv/zNz/zt3/0t7/1t//2+D/4uL/6OT/7+X/9+j//ur//+3///H///b///v/x97/x97/x97/x97/x97/yN7/yN7/yN7/yN7/yN7/yN7/yN7/yN//yN//yd//yd//yt//y9//zOD/zuD/0eH/1OL/2OP/3eT/4+b/6uj/8On/9+v//u3///D///P///f///z/zOT/zOT/zOT/zOT/zOT/zOT/zOT/zOT/zOT/zeT/zeT/zeT/zeT/zeT/zeT/zuT/zuX/z+X/0OX/0eX/1Ob/1+f/2+j/4On/5ev/6+z/8u3/+O///vH///P///b///n///7/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/0+v/1Ov/1Oz/1ez/1ez/1uz/2Oz/2u3/3O3/4O7/4+//6PD/7vH/8/L/+vT///X///b///n///z///7/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/2/P/3PP/3PP/3fP/3fP/3/P/4PT/4vT/5PT/5/X/6/b/8Pf/9vj/+/n///r///v///z/////////4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/4/r/5Pr/5Pr/5Pv/5fv/5fv/5vv/6Pv/6vv/7Pz/7/z/8/z/+P3//f7///7///7/////////////6v//6v//6v//6v//6v//6v//6v//6v//6v//6v//6v//6v//6v//6v//6///6///6///6///7P//7P//7f//7v//7///8f//8///9v//+v///f//////////////////////8f//8f//8f//8f//8f//8f//8f//8f//8f//8f//8f//8f//8f//8f//8v//8v//8v//8v//8v//8v//8///9P//9f//9v//+P//+v///f//////////////////////////+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+f//+v//+v//+////P///f///v///v///////////////////////////v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+X/3+b/4Ob/4eb/4+f/5ef/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+b/3+b/4Ob/4eb/4+f/5Of/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+b/3+b/4Ob/4eb/4+f/5Of/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+b/3+b/4Ob/4eb/4+f/5Of/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+b/3+b/4Ob/4eb/4+f/5Of/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+b/3+b/4Ob/4eb/4+f/5Of/5+j/6en/7en/8ur/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3+X/3+b/3+b/4Ob/4eb/4+f/5Oj/5+j/6en/7en/8uv/9+z//e7///D///P///f///z/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3ub/3+b/3+b/4Ob/4eb/4+f/5Oj/5+j/6en/7er/8uv/9+z//e7///D///P///f///z/3eX/3eX/3eX/3eX/3eX/3eX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3uX/3ub/3+b/3+b/4Ob/4eb/4uf/5Oj/5+j/6en/7er/8uv/9+z//e7///D///P///f///z/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3uX/3uX/3uX/3ub/3ub/3+b/3+b/4Ob/4ef/4uf/5Oj/5+n/6en/7er/8uv/9+z//e7///H///P///f///z/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eX/3eb/3eb/3eb/3eb/3ub/3ub/3ub/3+b/4Ob/4ef/4uf/4+j/5un/6en/7er/8uv/9+z//e7///H///T///f///z/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3eb/3ub/3ub/3+b/4Of/4ef/4+j/5un/6er/7er/8uv/9+3//u7///H///T///f///z/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3Ob/3eb/3eb/3eb/3ub/3+f/4Of/4ef/4uj/5en/6er/7ev/8uv/9+3//u////H///T///j///z/2+b/2+b/2+b/2+b/2+b/2+b/2+b/2+b/2+b/2+b/2+b/2+b/2+b/3Ob/3Ob/3Ob/3Ob/3Of/3ef/3uf/3+f/4Oj/4uj/5en/6er/7ev/8uz/9+3//u////H///X///j///z/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2eb/2uf/2uf/2uf/2+f/2+f/3Of/3uj/3+j/4en/5Or/6Ov/7ez/8u3/9+7//vD///P///X///n///3/1uf/1uf/1uf/1uf/1uf/1uf/1uf/1uf/1uf/1uf/1uf/1uf/1+f/1+f/1+f/1+f/2Of/2Of/2ej/2uj/3Oj/3un/4er/5Ov/6Oz/7e7/8u//+PD//fL///T///f///r///3/2Ov/2Ov/2Ov/2Ov/2Ov/2Ov/2Ov/2Ov/2Ov/2Ov/2Ov/2ev/2ev/2ev/2ev/2ev/2uv/2uv/2+v/3Oz/3uz/4Oz/4u3/5e7/6u//7vH/9PL/+fP//vT///b///j///v///7/3O//3O//3O//3O//3O//3O//3O//3O//3O//3O//3O//3O//3O//3O//3e//3e//3e//3vD/3vD/3/D/4PD/4vH/5PH/6PL/7PP/8PT/9fb/+vf//vj///n///v///3/////4fX/4fX/4fX/4fX/4fX/4fX/4fX/4fX/4fX/4vX/4vX/4vX/4vX/4vX/4vX/4vX/4vX/4/X/4/X/5Pb/5fb/5/b/6ff/6/f/7vj/8vn/9/r//Pv///z///3///7/////////6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6Pv/6fv/6fv/6vz/6/z/7Pz/7fz/7/z/8v3/9f3/+f7//f7/////////////////////7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7v//7///7///7///8P//8P//8f//8v//9P//9v//+P//+////v//////////////////////9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9P//9f//9f//9f//9v//9v//9///+P//+v//+////v///v//////////////////////+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+///+///+////P///P///f///v//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/7PH/7fH/7vH/8PL/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/7PH/7fH/7vH/7/L/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/7PH/7fH/7vH/7/L/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/D/7PH/7fH/7vH/7/L/8fL/8/P/9vP/+vT//vX///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/H/7PH/7fH/7vH/7/L/8fL/8/P/9vP/+vT//vb///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/H/7PH/7fH/7vH/7/L/8fP/8/P/9vT/+vT//vb///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/D/6/D/6/H/7PH/7fH/7vH/7/L/8fP/8/P/9vT/+vT//vb///f///n///z///7/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6vD/6/H/6/H/7PH/7PH/7vL/7/L/8fP/8/P/9vT/+vX//vb///f///n///z///7/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6vD/6vD/6vD/6vD/6vD/6vD/6vH/6/H/6/H/7PH/7fL/7/L/8fP/8/P/9vT/+vX//vb///f///n///z///7/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6fD/6vH/6vH/6vH/6/H/6/H/7PH/7fL/7vL/8fP/8/T/9vT/+vX//vb///f///n///z///7/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6fH/6vH/6vH/6/H/6/L/7fL/7vL/8PP/8/T/9vT/+vX//vb///j///r///z///7/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6PH/6fH/6fH/6fH/6vH/6/L/7PL/7fP/8PP/8/T/9vX/+vb//vf///j///r///z/////5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5vH/5/H/5/H/5/H/6PL/6fL/6vL/6/L/7PP/7/T/8/X/9vX/+vb//vf///n///v///3/////5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5PH/5fL/5fL/5fL/5vL/5vL/5/L/6PP/6vP/7PT/7/X/8vb/9/f/+/j///n///r///z///7/////5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vT/5vX/5vX/5/X/5/X/6PX/6fX/6vb/6/b/7vf/8Pf/9Pj/+Pn//Pr///v///z///7///7/////6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6fj/6vj/6vj/6vn/6/n/7Pn/7vn/7/r/8vv/9fz/+fz//f3///7///7/////////////7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7f3/7v3/7v3/7v3/7/3/7/3/8P3/8f7/8/7/9f7/+P//+////v//////////////////////8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8v//8///8///8///9P//9f//9f//9///+P//+v///f///v//////////////////////9///9///9///9///9///9///9///9///9///9///9///9///9///9///9///9///+P//+P//+P//+P//+P//+f//+v//+////P///f///v///////////////////////////P///P///P///P///P///P///P///P///P///P///P///P///P///P///P///P///P///P///f///f///f///f///v///v//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fn/9fn/9vr/9/r/+Pr/+vv//Pv//vz///z///3///7/////////9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fr/9vr/9/r/+Pv/+vv//Pv//vz///3///3///7/////////8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9Pn/9fn/9fr/9vr/9/r/+Pv/+vv//Pz//vz///3///3///7/////////8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/9Pn/9Pn/9Pn/9Pr/9fr/9fr/9vr/+Pv/+fv//Pz//vz///3///3///7/////////8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/8/n/9Pr/9Pr/9Pr/9fr/9vr/9/v/+fv//Pz//fz///3///7///7/////////8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vn/8vr/8vr/8/r/8/r/8/r/8/r/8/r/8/r/9Pr/9fr/9vv/9/v/+fz//Pz//f3///3///7///7/////////8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8fr/8vr/8vr/8vr/8/r/9Pr/9fv/9vv/+Pz/+/3//f3///3///7/////////////7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/7/r/8Pr/8Pr/8Pr/8fr/8vv/8/v/9Pv/9vz/+Pz/+/3//v7///7/////////////////8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8fz/8f3/8f3/8f3/8v3/8v3/8/3/9P3/9f7/9/7/+f///P///v//////////////////////8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/8/7/9P7/9P7/9P7/9f7/9v//9///+P//+////f//////////////////////////9v//9v//9v//9v//9v//9v//9v//9v//9v//9v//9///9///9///9///9///9///9///9///9///9///+P//+P//+f//+v//+////f///v//////////////////////////+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+v//+///+///+///+///+///+///+///+////P///P///f///v///v///////////////////////////////v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v///v//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////" };
/* @generated:end */

/* @wonderful-patch: all-blue and crystal start 9.5 s later in their idle motion than the pack's approved frame
 * (AllBlue.blend @ 94, Crystal @ 98): each hero is the pose seed 0 reaches after advance(9.5) from the pack's hero.
 * Their stills in ../orbs are rendered from the live orb at this pose, not from Blender. Upstream to apps/agent-orb
 * (frames 322 and 326 at 24 fps) before the next pack refresh, which rewrites the generated block above. */
THEMES["all-blue"].hero = { rotA: 1.416724, wA: 1.457323, rotB: 1.324886, wB: 1.138723 };
THEMES.crystal.hero = { rotA: 1.393916, wA: 1.451338, rotB: 1.334516, wB: 1.125129 };

/* @wonderful-patch: the pack calls registerElement() here; the element is removed above (section 9). */
