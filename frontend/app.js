"use strict";

const API_BASE = location.protocol.startsWith("http") ? location.origin : "http://localhost:8000";
const MAX_TRACKS = 50;
const SUGGESTED_MIN_TRACKS = 10;
const SOUND_MAP_SAMPLE = 3500;
const FALLBACK_DATASET_SIZE = 127900;
const DIVERSITY_THRESHOLDS = { focused: 0.35, eclectic: 0.65 };

const PERSONA_COLOR_VARS = {
  "Heavy & Live": "--persona-heavy-live",
  "Bright Pop & Groove": "--persona-bright-pop",
  "Vocal & Rhythmic": "--persona-vocal-rhythmic",
  "Synth & Club": "--persona-synth-club",
  "Driven Alt & Punk": "--persona-driven-alt",
  "Unplugged & Intimate": "--persona-unplugged",
  "Ambient & Drone": "--persona-ambient",
};
const PERSONA_COLORS = {};

// engine.py picks the dominant persona with mode(), which breaks ties by the lowest
// cluster id; sorting the same way keeps headline, ring and recommendations in agreement.
// Keep in sync with the notebook's persona_names; k-means renumbers clusters on retrain.
const PERSONA_CLUSTER_ID = {
  "Synth & Club": 0,
  "Vocal & Rhythmic": 1,
  "Bright Pop & Groove": 2,
  "Unplugged & Intimate": 3,
  "Heavy & Live": 4,
  "Ambient & Drone": 5,
  "Driven Alt & Punk": 6,
};
const PERSONA_ORDER = Object.keys(PERSONA_CLUSTER_ID).sort((a, b) => PERSONA_CLUSTER_ID[a] - PERSONA_CLUSTER_ID[b]);

// Keyed by the data name so the backend never changes; a rename is a one-line edit here.
const PERSONA_DISPLAY = {
  "Bright Pop & Groove": { name: "Bright Pop & Groove", blurb: "Happy, danceable, built for singing along in the car.", traits: ["happy", "danceable", "fairly energetic"], stats: "mood (valence) +0.83, danceability +0.67, energy +0.46" },
  "Vocal & Rhythmic": { name: "Vocal & Rhythmic", blurb: "Lots of lyrics and lots of rhythm. The voice is the instrument.", traits: ["lots of spoken or rapped words", "danceable", "upbeat"], stats: "speechiness +1.35, danceability +0.80, mood (valence) +0.41" },
  "Heavy & Live": { name: "Heavy & Live", blurb: "Loud, high-energy, often recorded in front of a crowd.", traits: ["high energy", "loud", "sounds live"], stats: "energy +0.93, loudness +0.66, liveness +0.62" },
  "Synth & Club": { name: "Synth & Club", blurb: "Instrumental, driving and a little dark. Made for dance floors, not singalongs.", traits: ["mostly instrumental", "energetic", "darker mood"], stats: "instrumentalness +1.62, energy +0.59, mood (valence) −0.46" },
  "Unplugged & Intimate": { name: "Unplugged & Intimate", blurb: "Slow, acoustic and close, like someone playing for the room instead of a stage.", traits: ["slow", "acoustic", "gentle"], stats: "tempo −1.00, acousticness +0.86, energy −0.85" },
  "Ambient & Drone": { name: "Ambient & Drone", blurb: "Barely there. Very quiet, mostly instrumental, more atmosphere than song.", traits: ["very quiet", "mostly instrumental", "very low energy"], stats: "loudness −2.25, instrumentalness +2.05, energy −1.77" },
  "Driven Alt & Punk": { name: "Driven Alt & Punk", blurb: "Quick tempo, driving guitar riffs, alternative rock and punk energy.", traits: ["quick tempo", "punchy guitars", "driving energy"], stats: "tempo +0.80, energy +0.65, speechiness +0.56" },
};

// Hard-coded from the PCA loadings (pca_2d.components_ in models/pca_2d.joblib) over
// [danceability, speechiness, instrumentalness, liveness, valence, tempo, intensity]:
//   PC1: intensity +0.82, valence +0.30;  PC2: danceability +0.58, valence +0.51, intensity −0.39, instrumentalness −0.33
// A retrain changes these (signs can flip), so re-check the loadings and update the captions.
const AXIS_LABELS = {
  x: { low: "quieter, acoustic", high: "louder, energetic" },
  y: { low: "instrumental, intense", high: "danceable, upbeat" },
};

const FEATURE_COPY = {
  danceability: { label: "Danceability", hint: "How easy it is to dance to" },
  energy: { label: "Energy", hint: "How intense and dynamic it feels" },
  valence: { label: "Mood", hint: "How happy or bright it sounds" },
  acousticness: { label: "Acousticness", hint: "How acoustic vs. produced it sounds" },
};

const state = {
  tracks: [],
  status: "empty",         // empty | loading | results | error
  profile: null,
  analyzedIds: [],
  error: null,
  map: null,
  mapStatus: "loading",    // loading | ready | error
  datasetSize: FALLBACK_DATASET_SIZE,
  focusPersona: null,
  hoverTrackId: null,
  hoverSource: null,       // "map" | "list"
  search: { query: "", results: [], active: -1, status: "idle" },
};

/* ---------- helpers ---------- */
const $ = (sel) => document.querySelector(sel);
const el = {
  songCount: $("#song-count"), analyzeBtn: $("#analyze-btn"), clearBtn: $("#clear-btn"),
  hero: $("#hero"), compare: $("#compare"), recs: $("#recs"), recsHint: $("#recs-hint"), howBody: $("#how-body"),
  plot: $("#plot"), mapCanvas: $("#map-canvas"), mapLabels: $("#map-labels"), mapTip: $("#map-tip"), mapStatus: $("#map-status"),
  mapHint: $("#map-hint"), axesTip: $("#axes-tip"), yHigh: $("#y-high"), yLow: $("#y-low"), xLow: $("#x-low"), xHigh: $("#x-high"),
  searchInput: $("#search-input"), searchPop: $("#search-pop"), searchMsg: $("#search-msg"), searchOptions: $("#search-options"),
  songList: $("#song-list"), songHint: $("#song-hint"), songsCount: $("#songs-count"),
  tip: $("#tip"), announcer: $("#announcer"),
};

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const colorOf = (p) => PERSONA_COLORS[p] || "#998A9B";
const nameOf = (p) => (PERSONA_DISPLAY[p] && PERSONA_DISPLAY[p].name) || p;
const blurbOf = (p) => (PERSONA_DISPLAY[p] && PERSONA_DISPLAY[p].blurb) || "";
const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
const announce = (msg) => { el.announcer.textContent = ""; requestAnimationFrame(() => { el.announcer.textContent = msg; }); };
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const hexToRgb = (h) => { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };
const relLum = (h) => hexToRgb(h).map((v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; })
  .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
const contrastRatio = (a, b) => { const [x, y] = [relLum(a), relLum(b)].sort((m, n) => n - m); return (x + 0.05) / (y + 0.05); };
const mixHex = (a, b, t) => { const A = hexToRgb(a), B = hexToRgb(b); return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join(""); };
const ordinal = (n) => { const r = n % 100; return r >= 11 && r <= 13 ? `${n}th` : `${n}${["th", "st", "nd", "rd"][n % 10] || "th"}`; };

const ICON_X = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>';
const ICON_INFO = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg>';
const MARK_SVG = '<svg viewBox="96 20 320 456" aria-hidden="true"><circle cx="247.8" cy="68.2" r="47.2" fill="#F6C343"/><circle cx="336.6" cy="119.4" r="47.1" fill="#EA6C5A"/><circle cx="168.9" cy="134.3" r="47.2" fill="#F8A9C9"/><circle cx="255.5" cy="246.9" r="60" fill="#7DF29C"/><circle cx="342.2" cy="313.5" r="47.2" fill="#8398FE"/><circle cx="174.4" cy="374.5" r="47.1" fill="#42B194"/><circle cx="263.2" cy="425.7" r="47.2" fill="#DC6ED0"/></svg>';

function loadPersonaColors() {
  for (const [persona, v] of Object.entries(PERSONA_COLOR_VARS)) PERSONA_COLORS[persona] = cssVar(v) || "#998A9B";
}

// Percentages use largest-remainder rounding so they add up to exactly 100.
function breakdownEntries(breakdown) {
  const entries = Object.entries(breakdown || {})
    .filter(([, frac]) => frac > 0)
    .map(([persona, frac]) => ({ persona, frac, exact: frac * 100, pct: Math.floor(frac * 100) }));
  const remaining = 100 - entries.reduce((sum, e) => sum + e.pct, 0);
  [...entries].sort((a, b) => (b.exact - b.pct) - (a.exact - a.pct)).slice(0, Math.max(0, remaining)).forEach((e) => { e.pct += 1; });
  return entries.sort((a, b) => b.frac - a.frac || (PERSONA_CLUSTER_ID[a.persona] ?? 99) - (PERSONA_CLUSTER_ID[b.persona] ?? 99));
}
const currentEntries = () => (state.status === "results" ? breakdownEntries(state.profile.persona_breakdown) : []);

function diversityBand(score) {
  if (score < DIVERSITY_THRESHOLDS.focused) return { label: "Focused", desc: "You keep coming back to one sound." };
  if (score > DIVERSITY_THRESHOLDS.eclectic) return { label: "Eclectic", desc: "No single sound has a hold on you." };
  return { label: "Balanced", desc: "A home base, with regular detours." };
}
const isStale = () => state.status === "results" &&
  (state.tracks.length !== state.analyzedIds.length || state.tracks.some((t, i) => t.track_id !== state.analyzedIds[i]));

/* ---------- album art ---------- */
const artRequests = new Map();
const artUrls = new Map();

// Hydrate from localStorage so previously fetched covers load instantaneously
try {
  const savedArt = JSON.parse(localStorage.getItem("sonality_art_cache") || "{}");
  for (const [id, url] of Object.entries(savedArt)) {
    if (url) artUrls.set(id, url);
  }
} catch (_) {}

function saveArtToStorage(trackId, url) {
  artUrls.set(trackId, url);
  try {
    const saved = JSON.parse(localStorage.getItem("sonality_art_cache") || "{}");
    saved[trackId] = url;
    const keys = Object.keys(saved);
    if (keys.length > 300) delete saved[keys[0]];
    localStorage.setItem("sonality_art_cache", JSON.stringify(saved));
  } catch (_) {}
}

function getArt(trackId) {
  if (artUrls.has(trackId)) {
    return Promise.resolve(artUrls.get(trackId));
  }
  if (!artRequests.has(trackId)) {
    const req = fetch(`https://open.spotify.com/oembed?url=spotify:track:${encodeURIComponent(trackId)}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((j) => {
        const u = (j && j.thumbnail_url) || null;
        if (u) {
          saveArtToStorage(trackId, u);
          return u;
        }
        artRequests.delete(trackId);
        return null;
      })
      .catch(() => {
        artRequests.delete(trackId); // allow retry on network hiccups or rate limits
        return null;
      });
    artRequests.set(trackId, req);
  }
  return artRequests.get(trackId);
}

function artHtml(trackId, persona) {
  const known = artUrls.get(trackId);
  return `<span class="art ${known ? "has-art" : ""}" style="--c:${colorOf(persona)}" data-art="${esc(trackId)}">` +
    (known ? `<img src="${esc(known)}" alt="" decoding="async">` : "") + `</span>`;
}

function hydrateArt(root) {
  root.querySelectorAll("[data-art]").forEach((box) => {
    const id = box.dataset.art;
    if (!id) return;
    
    // If already in memory or localStorage, render immediately
    if (artUrls.has(id)) {
      const url = artUrls.get(id);
      if (!box.querySelector("img")) {
        const img = new Image();
        img.alt = "";
        img.decoding = "async";
        img.onload = () => {
          if (!box.querySelector("img")) box.appendChild(img);
          requestAnimationFrame(() => box.classList.add("has-art"));
        };
        img.src = url;
      }
      return;
    }

    getArt(id).then((url) => {
      if (!url) return;
      // Search for any matching boxes currently mounted in DOM (prevents rapid re-render drops)
      const matches = root.querySelectorAll(`[data-art="${id}"], [data-id="${id}"] .art`);
      matches.forEach((target) => {
        if (target.querySelector("img")) return;
        const img = new Image();
        img.alt = "";
        img.decoding = "async";
        img.onload = () => {
          if (!target.querySelector("img")) target.appendChild(img);
          requestAnimationFrame(() => target.classList.add("has-art"));
        };
        img.src = url;
      });
    });
  });
}

/* ---------- header + songs ---------- */
function renderHeader() {
  const n = state.tracks.length;
  el.songCount.textContent = n ? `${n} of ${MAX_TRACKS} songs` : "No songs yet";
  el.songsCount.textContent = `${n} / ${MAX_TRACKS}`;
  const loading = state.status === "loading";
  el.analyzeBtn.disabled = n === 0 || loading;
  el.analyzeBtn.setAttribute("aria-busy", String(loading));
  el.analyzeBtn.textContent = loading ? "Analyzing…" : "Analyze taste";
  el.clearBtn.disabled = n === 0 || loading;
  const hasResults = state.status === "results";
  document.body.classList.toggle("has-results", hasResults || loading);
}

function renderSongs() {
  const n = state.tracks.length;
  const hints = [];
  if (n > 0 && n < SUGGESTED_MIN_TRACKS) hints.push(`Add ${SUGGESTED_MIN_TRACKS - n} more for a steadier read.`);
  if (n >= MAX_TRACKS) hints.push("That's the most one analysis takes.");
  if (isStale()) hints.push("Changed since your last analysis. Analyze again to update.");
  el.songHint.innerHTML = hints.map((h) => `<span style="display:block">${esc(h)}</span>`).join("");
  if (!n) {
    el.songList.innerHTML = `<li class="song-empty">Search above and pick ${SUGGESTED_MIN_TRACKS} to ${MAX_TRACKS} songs you'd happily play on repeat.</li>`;
    return;
  }
  el.songList.innerHTML = state.tracks.map((t) => {
    const isPlaying = Boolean(player.track && player.track.track_id === t.track_id && player.isPlaying);
    return `
    <li class="song" data-id="${esc(t.track_id)}">
      ${artHtml(t.track_id, t.persona)}
      <span class="song-text">
        <span class="song-name" title="${esc(t.track_name)}">${esc(t.track_name)}</span>
        <span class="song-artist"><span class="dot" style="--c:${colorOf(t.persona)}" aria-hidden="true"></span><span>${esc(t.artists)}</span><span class="sr-only">, ${esc(nameOf(t.persona))}</span></span>
      </span>
      <div class="song-actions">
        <button class="icon-btn btn-row-preview ${isPlaying ? "is-playing" : ""}" type="button" data-play-preview="${esc(t.track_id)}" data-name="${esc(t.track_name)}" data-artist="${esc(t.artists)}" aria-label="${isPlaying ? "Pause" : "Play"} 10s preview of ${esc(t.track_name)}" title="${isPlaying ? "Pause preview" : "Play 10s preview"}">
          <span class="preview-icon-wrap" aria-hidden="true">
            <svg class="icon-play" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
            <svg class="icon-pause" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
          </span>
        </button>
        <button class="icon-btn btn-remove" type="button" data-remove="${esc(t.track_id)}" aria-label="Remove ${esc(t.track_name)}" title="Remove song">${ICON_X}</button>
      </div>
    </li>`;
  }).join("");
  hydrateArt(el.songList);
  syncLinkedRow();
}
function syncLinkedRow() {
  el.songList.querySelectorAll(".song").forEach((row) => {
    const linked = state.hoverSource === "map" && row.dataset.id === state.hoverTrackId;
    row.classList.toggle("is-linked", linked);
    if (linked) row.scrollIntoView({ block: "nearest" });
  });
}

/* ---------- hero: headline + ring ---------- */
let freshResults = false;

// The headline is the one place a persona colour is used as text; keep it >= 3:1 on the page.
function headlineColor(persona) {
  const base = colorOf(persona), bg = cssVar("--page") || "#17121B";
  let c = base;
  for (let t = 0.1; contrastRatio(c, bg) < 3 && t <= 1; t += 0.1) c = mixHex(base, "#ffffff", t);
  return c;
}
function headlineParts(entries) {
  const [a, b] = entries;
  if (!a) return { full: "We couldn't place these songs.", compact: "", streak: "" };
  const colA = headlineColor(a.persona);
  const nameA = esc(nameOf(a.persona));
  const A = `<span class="nowrap" style="color:${colA}">${nameA}</span>`;
  const A_dot = `<span class="nowrap" style="color:${colA}">${nameA}.</span>`;
  if (!b) return { full: `All ${A_dot}`, compact: `All ${A_dot}`, streak: "" };
  const B = esc(nameOf(b.persona));
  return {
    full: `Mostly ${A}, with a streak of <span class="nowrap">${B}.</span>`,
    compact: `Mostly ${A_dot}`,
    streak: `With a streak of ${B}. `
  };
}
// Two lines at most: shrink to a 34px floor first; on narrow screens move the
// "streak" clause down into the muted line instead of allowing a third line.
function fitHeadline() {
  const h = el.hero.querySelector(".headline");
  if (!h || !h.dataset.full) return;
  const streak = el.hero.querySelector(".lede-streak");
  h.innerHTML = h.dataset.full;
  if (streak) streak.hidden = true;
  const fit = () => {
    h.style.fontSize = "";
    let size = parseFloat(getComputedStyle(h).fontSize);
    // Desktop keeps the headline at 56px or more; narrow screens may go down to 34px.
    const floor = size >= 56 ? 56 : 34;
    const lines = () => Math.round(h.getBoundingClientRect().height / (size * 1.02));
    while (lines() > 2 && size > floor) { size -= 2; h.style.fontSize = `${size}px`; }
    return lines();
  };
  if (fit() > 2 && h.dataset.compact) { h.innerHTML = h.dataset.compact; if (streak) streak.hidden = false; fit(); }
}

const RING = { size: 200, r: 78, stroke: 24, gap: 4, minLen: 3 };

function ringSegments(entries) {
  const C = 2 * Math.PI * RING.r;
  if (entries.length === 1) return [{ e: entries[0], start: 0, len: C, solo: true }];
  // Round caps extend each visible end by half the stroke; reserve that plus the gap per segment.
  const overhead = RING.stroke + RING.gap;
  const avail = C - overhead * entries.length;
  let lens = entries.map((e) => Math.max(RING.minLen, e.frac * avail));
  const over = lens.reduce((s, l) => s + l, 0) - avail;
  if (over > 0) {
    const big = lens.map((l) => (l > RING.minLen ? l - RING.minLen : 0));
    const bigSum = big.reduce((s, l) => s + l, 0) || 1;
    lens = lens.map((l, i) => l - over * (big[i] / bigSum));
  }
  let cursor = 0;
  return entries.map((e, i) => {
    const seg = { e, start: cursor + RING.gap / 2 + RING.stroke / 2, len: lens[i], solo: false };
    cursor += lens[i] + overhead;
    return seg;
  });
}

function renderHero() {
  const { status } = state;
  const hero = el.hero;
  hero.style.removeProperty("--c");
  if (status === "empty") {
    hero.innerHTML = `
      <div class="hero-empty">
        <div class="hero-empty-mark" aria-hidden="true">
          ${MARK_SVG}
        </div>
        <div class="hero-empty-text">
          <h1 class="headline">What does your music have in common?</h1>
          <p class="lede">Add ${SUGGESTED_MIN_TRACKS} to ${MAX_TRACKS} songs you love and see which of seven sounds they share.</p>
        </div>
      </div>`;
    return;
  }
  const p = state.profile;
  const entries = breakdownEntries(p.persona_breakdown);
  const top = entries[0];
  const matched = (p.user_songs || []).length;
  const sent = state.analyzedIds.length || matched;
  const absent = PERSONA_ORDER.filter((x) => !entries.some((e) => e.persona === x));
  const parts = headlineParts(entries);
  const lede = matched < sent
    ? `Based on ${matched} of your ${sent} songs (${sent - matched} ${sent - matched === 1 ? "wasn't" : "weren't"} in the dataset), placed among ${fmtInt(state.datasetSize)} tracks.`
    : `Based on ${matched} ${matched === 1 ? "song" : "songs"}, placed among ${fmtInt(state.datasetSize)} tracks.`;
  hero.style.setProperty("--c", colorOf(top.persona));

  const C = 2 * Math.PI * RING.r, c = RING.size / 2;
  const segs = ringSegments(entries);
  const defs = [], paths = [];
  for (const s of segs) {
    const id = `grad-${PERSONA_CLUSTER_ID[s.e.persona]}`;
    const a0 = (s.start / C) * Math.PI * 2 - Math.PI / 2, a1 = ((s.start + s.len) / C) * Math.PI * 2 - Math.PI / 2;
    const base = colorOf(s.e.persona), light = mixHex(base, "#ffffff", 0.12);
    defs.push(`<linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="${(c + RING.r * Math.cos(a0)).toFixed(1)}" y1="${(c + RING.r * Math.sin(a0)).toFixed(1)}" x2="${(c + RING.r * Math.cos(a1)).toFixed(1)}" y2="${(c + RING.r * Math.sin(a1)).toFixed(1)}"><stop offset="0" stop-color="${base}"/><stop offset="1" stop-color="${light}"/></linearGradient>`);
    paths.push(`<circle class="ring-seg ${s.solo ? "is-solo" : ""}" data-persona="${esc(s.e.persona)}" stroke="url(#${id})"
      cx="${c}" cy="${c}" r="${RING.r}" stroke-dasharray="${s.len.toFixed(2)} ${(C - s.len).toFixed(2)}" stroke-dashoffset="${(-s.start).toFixed(2)}" transform="rotate(-90 ${c} ${c})"></circle>`);
  }
  const score = p.diversity_score;
  const band = diversityBand(score);
  const x = Math.max(0, Math.min(1, score)) * 100;

  hero.innerHTML = `
    <div class="hero-grid">
      <div class="${freshResults ? "reveal" : ""}">
        <h1 class="headline">${parts.full}</h1>
        <p class="lede">${parts.streak ? `<span class="lede-streak" hidden>${parts.streak}</span>` : ""}${esc(lede)}</p>
        <div class="hero-actions">
          <button class="btn-share-trigger" id="btn-hero-share" type="button" aria-label="Share your Sonality">
            <svg viewBox="0 0 20 20" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="15" cy="5" r="2.5"/>
              <circle cx="5" cy="10" r="2.5"/>
              <circle cx="15" cy="15" r="2.5"/>
              <path d="M7.3 11.2l5.4 2.6M12.7 6.2L7.3 8.8"/>
            </svg>
            <span>Share your Sonality</span>
          </button>
        </div>
      </div>
      <div class="ring-block">
        <div class="ring-wrap">
          <svg class="ring ${freshResults ? "is-fresh" : ""}" viewBox="0 0 ${RING.size} ${RING.size}" role="img"
               aria-label="${esc("Persona breakdown: " + entries.map((e) => `${nameOf(e.persona)} ${e.pct}%`).join(", "))}">
            <defs>${defs.join("")}
              <mask id="ring-mask"><circle class="ring-reveal" cx="${c}" cy="${c}" r="${RING.r}" stroke-dasharray="${C.toFixed(2)}" style="--c-len:${C.toFixed(2)}" transform="rotate(-90 ${c} ${c})"></circle></mask>
            </defs>
            ${segs[0].solo ? "" : `<circle class="ring-track" cx="${c}" cy="${c}" r="${RING.r}"></circle>`}
            <g ${freshResults ? 'mask="url(#ring-mask)"' : ""}>${paths.join("")}</g>
          </svg>
          <div class="ring-center" aria-hidden="true">
            <span class="ring-pct" id="ring-pct">${top.pct}%</span>
            <span class="ring-name" id="ring-name">${esc(nameOf(top.persona))}</span>
          </div>
        </div>
        <div class="legend-col">
          <ul class="legend" aria-label="Your mix, by share">
            ${entries.map((e) => `
              <li class="legend-row" data-persona="${esc(e.persona)}" tabindex="0" data-tip="${esc(blurbOf(e.persona))}" aria-label="${esc(nameOf(e.persona))}, ${e.pct} percent">
                <span class="dot" style="--c:${colorOf(e.persona)}" aria-hidden="true"></span>
                <span class="name">${esc(nameOf(e.persona))}</span>
                <span class="pct" aria-hidden="true">${e.pct}%</span>
              </li>`).join("")}
          </ul>
          ${absent.length ? `<p class="legend-absent">Not in your songs: ${absent.map((a) => esc(nameOf(a))).join(", ")}.</p>` : ""}
        </div>
        <div class="diversity ${freshResults ? "is-fresh" : ""}">
          <div class="range-scale">
            <span>Focused</span>
            <div class="meter" aria-hidden="true">
              <span class="meter-track"></span>
              <span class="meter-tick" style="left:${DIVERSITY_THRESHOLDS.focused * 100}%"></span>
              <span class="meter-tick" style="left:${DIVERSITY_THRESHOLDS.eclectic * 100}%"></span>
              <span class="meter-mark" style="--x:${x}%"></span>
            </div>
            <span>Eclectic</span>
          </div>
          <p class="diversity-text">
            Your songs cover ${entries.length} of the 7 sounds.
            <span class="sr-only">Diversity score ${score.toFixed(2)}, ${band.label.toLowerCase()}.</span>
            <button class="tip-btn" type="button" aria-label="What the diversity scale means"
              data-tip="${band.label}, ${score.toFixed(2)}. ${esc(band.desc)} The scale is the normalized Shannon entropy of your mix: 0 means every song is in one sound, 1 means an even spread across all seven.">${ICON_INFO}</button>
          </p>
        </div>
      </div>
    </div>`;
  const h = hero.querySelector(".headline");
  h.dataset.full = parts.full;
  h.dataset.compact = parts.compact;
  fitHeadline();
  syncFocus();
}

function setRingCenter(persona) {
  const pctEl = document.getElementById("ring-pct"), nameEl = document.getElementById("ring-name");
  if (!pctEl) return;
  const entries = currentEntries();
  const shown = persona ? (entries.find((e) => e.persona === persona) || { persona, pct: 0 }) : entries[0];
  if (!shown) return;
  pctEl.textContent = `${shown.pct}%`;
  nameEl.textContent = nameOf(shown.persona);
}

/* ---------- Audio Telemetry Comparison Panel ---------- */
const METRIC_DENSITIES = {
  danceability: [0, 4, 7, 14, 16, 20, 27, 37, 51, 66, 82, 91, 100, 100, 93, 78, 55, 27, 12, 4],
  energy: [25, 16, 17, 20, 22, 28, 33, 37, 46, 54, 60, 69, 74, 85, 94, 90, 97, 100, 99, 80],
  valence: [51, 61, 68, 82, 87, 85, 99, 100, 93, 97, 99, 92, 90, 92, 84, 74, 68, 59, 53, 38],
  acousticness: [100, 28, 20, 16, 13, 11, 10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 15]
};

const compareState = {
  selectedId: "danceability",
  metrics: []
};

function formatDelta(percentile, typical = 50) {
  const diff = percentile - typical;
  if (diff === 0) return "Typical";
  return diff > 0 ? `+${diff} pts` : `\u2212${Math.abs(diff)} pts`;
}

function updateContextAndInspector() {
  if (!compareState.metrics.length) return;
  const meanPct = Math.round(
    compareState.metrics.reduce((acc, item) => acc + item.percentile, 0) / compareState.metrics.length
  );
  const strongest = compareState.metrics.reduce((best, item) => {
    const bestDelta = Math.abs(best.percentile - best.typical);
    const itemDelta = Math.abs(item.percentile - item.typical);
    return itemDelta > bestDelta ? item : best;
  }, compareState.metrics[0]);

  const strongestDiff = strongest.percentile - strongest.typical;
  const strongestSign = strongestDiff > 0 ? "+" : "\u2212";
  const strongestDir = strongestDiff > 0 ? "above typical" : strongestDiff < 0 ? "below typical" : "at median";

  const meanEl = document.getElementById("summary-mean-pct");
  const strongestEl = document.getElementById("summary-strongest");
  const summaryTextEl = document.getElementById("radar-summary-text");

  if (meanEl) meanEl.textContent = `${ordinal(meanPct)}`;
  if (strongestEl) {
    strongestEl.textContent = strongestDiff !== 0 
      ? `${strongest.title} (${strongestSign}${Math.abs(strongestDiff)} pts ${strongestDir})`
      : `${strongest.title} (Typical)`;
  }

  if (summaryTextEl) {
    const topDelta = strongest.percentile - strongest.typical;
    const absDelta = Math.abs(topDelta);
    if (absDelta < 6) {
      summaryTextEl.textContent = "Your listening maintains an even balance across all four dimensions, tracking close to the dataset typical.";
    } else if (topDelta > 0) {
      const positiveNarratives = {
        danceability: "Your mix is driven by physical groove and danceable rhythm, well above the dataset average.",
        energy: "Your mix is fast and high-intensity, carrying significantly more sonic drive than typical tracks.",
        valence: "Your mix skews bright and emotionally uplifting, well above the dataset's median mood.",
        acousticness: "Your mix leans heavily into organic instruments and unplugged arrangements.",
      };
      summaryTextEl.textContent = positiveNarratives[strongest.id] || `Your music leans distinctly higher in ${strongest.title.toLowerCase()}.`;
    } else {
      const negativeNarratives = {
        danceability: "Your mix favors free-flowing, ambient rhythms over conventional dance beats.",
        energy: "Your mix is calm and atmospheric, sitting well below typical loudness and tempo.",
        valence: "Your mix carries a darker, more contemplative tone than most tracks in the dataset.",
        acousticness: "Your mix is heavily electronic and modern, with minimal acoustic instrumentation.",
      };
      summaryTextEl.textContent = negativeNarratives[strongest.id] || `Your music leans distinctly lower in ${strongest.title.toLowerCase()}.`;
    }
  }
}

function selectMetric(metricId) {
  compareState.selectedId = metricId;
  const container = document.getElementById("metrics-container");
  if (container) {
    container.querySelectorAll(".metric-row").forEach((item) => {
      item.setAttribute("aria-selected", String(item.dataset.metricId === metricId));
    });
    container.setAttribute("aria-activedescendant", `metric-${metricId}`);
  }
  renderRadarChart();
}

function renderRadarChart() {
  const svg = document.getElementById("radar-chart-svg");
  if (!svg || !compareState.metrics.length) return;

  const top = currentEntries()[0];
  const accentColor = (top && colorOf(top.persona)) || "var(--accent)";

  const cx = 155, cy = 96, maxRadius = 68;
  const count = compareState.metrics.length;
  const angleFor = (idx) => -Math.PI / 2 + (idx * 2 * Math.PI) / count;
  const pointFor = (idx, pct) => {
    const r = (Math.max(0, Math.min(100, pct)) / 100) * maxRadius;
    const a = angleFor(idx);
    return {
      x: cx + r * Math.cos(a),
      y: cy + r * Math.sin(a),
    };
  };

  const rings = [25, 50, 75, 100];
  const ringPolygons = rings
    .map((level) => {
      const pts = compareState.metrics.map((_, i) => {
        const p = pointFor(i, level);
        return `${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      }).join(" ");
      const strokeClass = level === 50
        ? 'stroke="rgba(244, 237, 232, 0.38)" stroke-dasharray="3 3" stroke-width="1.25"'
        : 'stroke="rgba(255, 235, 220, 0.08)" stroke-width="1"';
      return `<polygon points="${pts}" fill="none" ${strokeClass} />`;
    })
    .join("");

  const axes = compareState.metrics.map((m, i) => {
    const outer = pointFor(i, 100);
    const isSelected = m.id === compareState.selectedId;

    let labelX = cx;
    let labelY = cy;
    let anchor = "middle";
    if (i === 0) {
      labelX = cx;
      labelY = cy - maxRadius - 12;
      anchor = "middle";
    } else if (i === 1) {
      labelX = cx + maxRadius + 8;
      labelY = cy;
      anchor = "start";
    } else if (i === 2) {
      labelX = cx;
      labelY = cy + maxRadius + 14;
      anchor = "middle";
    } else if (i === 3) {
      labelX = cx - maxRadius - 8;
      labelY = cy;
      anchor = "end";
    }

    return `
      <line
        x1="${cx}" y1="${cy}"
        x2="${outer.x.toFixed(1)}" y2="${outer.y.toFixed(1)}"
        stroke="rgba(255, 235, 220, 0.08)" stroke-width="1"
      />
      <text
        x="${labelX.toFixed(1)}"
        y="${labelY.toFixed(1)}"
        text-anchor="${anchor}"
        dominant-baseline="central"
        font-size="11.5"
        font-weight="${isSelected ? "600" : "500"}"
        fill="${isSelected ? accentColor : "var(--text-2)"}"
        style="cursor:pointer"
        data-radar-id="${esc(m.id)}"
      >${esc(m.title)}</text>
    `;
  }).join("");

  const userPoints = compareState.metrics.map((m, i) => pointFor(i, m.percentile));
  const userPolygonStr = userPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  const nodes = compareState.metrics.map((m, i) => {
    const p = userPoints[i];
    const isSelected = m.id === compareState.selectedId;
    const isAbove = m.percentile >= 50;
    const nodeColor = isAbove ? accentColor : "#94A3B8";
    return `
      <circle
        cx="${p.x.toFixed(1)}"
        cy="${p.y.toFixed(1)}"
        r="${isSelected ? "5.5" : "4"}"
        fill="${nodeColor}"
        stroke="var(--card)"
        stroke-width="1.5"
        style="cursor:pointer"
        data-radar-id="${esc(m.id)}"
      />
    `;
  }).join("");

  svg.setAttribute("viewBox", "0 0 310 196");
  svg.innerHTML = `
    ${ringPolygons}
    ${axes}
    <polygon
      points="${userPolygonStr}"
      fill="${accentColor}"
      fill-opacity="0.20"
      stroke="${accentColor}"
      stroke-width="1.75"
    />
    ${nodes}
  `;

  svg.querySelectorAll("[data-radar-id]").forEach((el) => {
    el.addEventListener("click", () => {
      const targetId = el.getAttribute("data-radar-id");
      if (targetId) selectMetric(targetId);
    });
  });
}

function renderDetailedRows() {
  const container = document.getElementById("metrics-container");
  if (!container) return;

  container.innerHTML = compareState.metrics.map((metric) => {
    const pct = Math.max(0, Math.min(100, metric.percentile));
    const typical = 50;
    const isAbove = pct >= typical;
    const spanStart = Math.min(pct, typical);
    const spanWidth = Math.abs(pct - typical);
    const deltaLabel = formatDelta(pct, typical);
    const isSelected = metric.id === compareState.selectedId;

    return `
      <div id="metric-${esc(metric.id)}" class="metric-row" data-metric-id="${esc(metric.id)}" role="button" tabindex="0" aria-selected="${isSelected}">
        <div class="metric-info">
          <h3 class="metric-name">${esc(metric.title)}</h3>
          <p class="metric-desc">${esc(metric.description)}</p>
        </div>

        <div class="metric-plot-cell" role="img" aria-label="${esc(metric.title)}: ${esc(metric.ordinal)}, ${esc(deltaLabel)} relative to typical track">
          <div class="plot-track-bg"></div>
          <div class="divergence-span ${isAbove ? "is-above" : "is-below"}" style="left:${spanStart}%;width:${spanWidth}%;"></div>
          <div class="user-node ${isAbove ? "is-above" : "is-below"}" style="left:${pct}%;"></div>
        </div>

        <div class="metric-readouts tabular-nums">
          <span class="percentile-primary">${ordinal(pct)}</span>
          <span class="delta-secondary ${isAbove ? "is-above" : "is-below"}">${esc(deltaLabel)}</span>
        </div>
      </div>
    `;
  }).join("");

  container.querySelectorAll(".metric-row").forEach((row) => {
    const id = row.dataset.metricId;
    row.addEventListener("click", () => selectMetric(id));
    row.addEventListener("focus", () => selectMetric(id));
    row.addEventListener("keydown", (e) => {
      const idx = compareState.metrics.findIndex((m) => m.id === id);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        const next = compareState.metrics[(idx + 1) % compareState.metrics.length];
        selectMetric(next.id);
        const nextEl = document.getElementById(`metric-${next.id}`);
        if (nextEl) nextEl.focus();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        const prev = compareState.metrics[(idx - 1 + compareState.metrics.length) % compareState.metrics.length];
        selectMetric(prev.id);
        const prevEl = document.getElementById(`metric-${prev.id}`);
        if (prevEl) prevEl.focus();
      }
    });
  });
}

function renderCompare() {
  const top = currentEntries()[0];
  const compareCard = document.querySelector(".card-compare");
  if (compareCard) {
    if (top && colorOf(top.persona)) {
      compareCard.style.setProperty("--user-accent", colorOf(top.persona));
    } else {
      compareCard.style.removeProperty("--user-accent");
    }
  }

  const pct = (state.profile && state.profile.percentiles) || {};
  const order = ["danceability", "energy", "valence", "acousticness"];

  compareState.metrics = order.map((key) => {
    const f = FEATURE_COPY[key] || { label: key, hint: "" };
    const raw = Number(pct[key]);
    const p = Number.isFinite(raw) ? Math.max(0, Math.min(100, Math.round(raw))) : 50;
    return {
      id: key,
      title: f.label,
      description: f.hint,
      percentile: p,
      typical: 50,
      ordinal: `${ordinal(p)} percentile`,
      density: METRIC_DENSITIES[key] || []
    };
  });

  const strongest = compareState.metrics.reduce((best, item) => {
    const bestDelta = Math.abs(best.percentile - best.typical);
    const itemDelta = Math.abs(item.percentile - item.typical);
    return itemDelta > bestDelta ? item : best;
  }, compareState.metrics[0]);
  if (strongest && !compareState.selectedId) {
    compareState.selectedId = strongest.id;
  }

  renderDetailedRows();
  renderRadarChart();
  updateContextAndInspector();
}

function renderRecs() {
  const recs = state.profile.recommendations || [];
  const top = currentEntries()[0];
  el.recsHint.innerHTML = top
    ? `<span class="dot" style="--c:${colorOf(top.persona)}" aria-hidden="true"></span> Closest matches to your ${esc(nameOf(top.persona))} sound.`
    : "";
  if (!recs.length) { el.recs.innerHTML = `<p class="recs-empty">No recommendations came back for this mix.</p>`; return; }
  el.recs.innerHTML = recs.map((r) => {
    const isPlaying = Boolean(player.track && player.track.track_id === r.track_id && player.isPlaying);
    const curTime = isPlaying ? formatPlayerTime(player.audio.currentTime) : "0:00";
    const curPct = isPlaying ? Math.min(100, (player.audio.currentTime / player.durationLimit) * 100) : 0;

    const matchPct = r.match_pct || 97;
    const bpm = r.tempo ? `${r.tempo} BPM` : null;
    const acoustic = r.acoustic_pct !== undefined ? `Acoustic ${r.acoustic_pct}%` : null;
    const energy = r.energy_pct !== undefined ? `Energy ${r.energy_pct}%` : null;
    const mood = r.valence_pct !== undefined ? `Mood ${r.valence_pct}%` : null;
    const statItems = [bpm, acoustic, energy, mood].filter(Boolean).slice(0, 3);
    const artistsFormatted = (r.artists || "").replace(/;/g, ", ");

    return `
    <article class="rec-card ${isPlaying ? "is-active" : ""}" data-track-id="${esc(r.track_id)}" data-name="${esc(r.track_name)}" data-artist="${esc(r.artists)}">
      <!-- Top Header Row: 48x48 Art + Title/Artist + Open in Spotify -->
      <div class="rec-header">
        ${artHtml(r.track_id, top ? top.persona : null)}
        <div class="rec-info">
          <span class="rec-title" data-tip="${esc(r.track_name)}" tabindex="0" role="text">${esc(r.track_name)}</span>
          <span class="rec-artist" data-tip="${esc(artistsFormatted)}" tabindex="0" role="text">${esc(artistsFormatted)}</span>
        </div>
        <a class="rec-spotify-link" href="https://open.spotify.com/track/${encodeURIComponent(r.track_id)}" target="_blank" rel="noopener"
           aria-label="Open ${esc(r.track_name)} by ${esc(artistsFormatted)} in Spotify" title="Open in Spotify">
          <svg viewBox="0 0 16 16" width="13" height="13" fill="currentColor" aria-hidden="true">
            <path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.669 11.538a.498.498 0 0 1-.686.165c-1.879-1.147-4.243-1.407-7.028-.77a.499.499 0 0 1-.222-.973c3.049-.696 5.662-.397 7.77.892a.5.5 0 0 1 .166.686zm.979-2.178a.624.624 0 0 1-.857.205c-2.15-1.321-5.428-1.704-7.972-.932a.625.625 0 0 1-.362-1.194c2.905-.882 6.517-.456 8.986 1.064a.624.624 0 0 1 .205.857zm.084-2.268C10.155 5.57 5.748 5.426 3.197 6.2a.749.749 0 1 1-.437-1.433c2.936-.89 7.809-.724 10.82 1.065a.749.749 0 1 1-.848 1.26z"/>
          </svg>
          <span>Open in Spotify</span>
        </a>
      </div>

      <!-- Middle Row: Flush-left Monospace Metadata Row -->
      <div class="rec-traits">
        <span class="rec-trait-match">${matchPct}% MATCH</span>
        ${statItems.map((s) => `
          <span class="rec-trait-sep" aria-hidden="true">&middot;</span>
          <span class="rec-trait-item">${esc(s)}</span>
        `).join("")}
      </div>

      <!-- Bottom Row: Audio Preview Section with Balanced Hairline Divider -->
      <div class="rec-player-section">
        <div class="rec-preview-head">
          <span class="rec-preview-label ${isPlaying ? "is-playing" : ""}">CHORUS PREVIEW</span>
          <a class="rec-apple-link" href="https://music.apple.com/search?term=${encodeURIComponent(r.track_name + ' ' + (r.artists || ''))}" target="_blank" rel="noopener" aria-label="Listen to ${esc(r.track_name)} on Apple Music" title="Listen on Apple Music">
            <svg viewBox="0 0 16 16" width="10" height="10" fill="currentColor" aria-hidden="true">
              <path d="M11.5 1.5v8.5a2.5 2.5 0 1 1-2.5-2.5c.26 0 .5.04.72.12V3.7L5 5.5v7a2.5 2.5 0 1 1-2.5-2.5c.26 0 .5.04.72.12V4.5a1 1 0 0 1 .78-.98l7-1.75a1 1 0 0 1 1.22.98z"/>
            </svg>
            <span>Apple Music</span>
          </a>
        </div>
        <div class="rec-player-row">
          <button class="rec-play-btn ${isPlaying ? "is-playing" : ""}" type="button" aria-label="${isPlaying ? "Pause" : "Play"} 10s preview of ${esc(r.track_name)}">
            <span class="btn-icon-wrap" aria-hidden="true">
              <svg class="icon-play" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
              <svg class="icon-pause" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
            </span>
          </button>
          <div class="rec-progress-wrap">
            <span class="rec-time rec-time-cur">${curTime}</span>
            <div class="rec-progress-bar" role="progressbar" aria-valuemin="0" aria-valuemax="10" aria-valuenow="${isPlaying ? Math.round(player.audio.currentTime) : 0}">
              <div class="rec-progress-fill" style="width: ${curPct}%"></div>
            </div>
            <span class="rec-time rec-time-total">0:10</span>
          </div>
        </div>
      </div>
    </article>`;
  }).join("");
  hydrateArt(el.recs);
  preloadRecPreviews(recs);
}

/* ---------- linked highlight ---------- */
function setFocusPersona(persona) {
  if (state.focusPersona === persona) return;
  state.focusPersona = persona;
  syncFocus();
  drawBackground();
  drawMap();
}
function syncFocus() {
  const f = state.focusPersona;
  const ring = el.hero.querySelector(".ring");
  if (ring) {
    ring.classList.toggle("has-focus", !!f);
    ring.querySelectorAll(".ring-seg").forEach((s) => s.classList.toggle("is-focus", s.dataset.persona === f));
  }
  const legend = el.hero.querySelector(".legend");
  if (legend) {
    legend.classList.toggle("has-focus", !!f);
    legend.querySelectorAll(".legend-row").forEach((r) => r.classList.toggle("is-focus", r.dataset.persona === f));
  }
  setRingCenter(f);
  el.mapLabels.classList.toggle("has-focus", !!f);
  el.mapLabels.querySelectorAll(".map-label").forEach((l) => l.classList.toggle("is-focus", l.dataset.persona === f));
}
function setHoverTrack(id, source = null) {
  if (state.hoverTrackId === id && state.hoverSource === source) return;
  state.hoverTrackId = id;
  state.hoverSource = id ? source : null;
  syncLinkedRow();
  drawMap();
}

/* ---------- map ---------- */
const MAP_PAD = { x: 18, top: 36, bottom: 36 };
const view = { w: 0, h: 0, dpr: 1, bounds: null, points: null, centroids: null, userPts: [], bg: document.createElement("canvas"), layer: document.createElement("canvas") };

function prepareMap() {
  const pts = (state.map && state.map.points) || [];
  const sums = {};
  for (const p of pts) { const s = (sums[p.persona] ||= { x: 0, y: 0, n: 0 }); s.x += p.pca_x; s.y += p.pca_y; s.n += 1; }
  view.centroids = Object.fromEntries(Object.entries(sums).map(([k, s]) => [k, { x: s.x / s.n, y: s.y / s.n }]));
  // A seeded shuffle of the draw order so no persona always ends up on top in the dense core.
  let seed = 7;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const order = pts.slice();
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  view.points = order;
}
function computeBounds() {
  const pts = (state.map && state.map.points) || [];
  if (!pts.length) { view.bounds = null; return; }
  const xs = pts.map((p) => p.pca_x).sort((a, b) => a - b), ys = pts.map((p) => p.pca_y).sort((a, b) => a - b);
  const q = (arr, f) => arr[Math.floor(f * (arr.length - 1))];
  // Trim the extreme 0.5% each side so a few outliers don't squash the map.
  let minX = q(xs, 0.005), maxX = q(xs, 0.995), minY = q(ys, 0.005), maxY = q(ys, 0.995);
  if (state.status === "results") for (const u of state.profile.user_songs || []) {
    minX = Math.min(minX, u.pca_x); maxX = Math.max(maxX, u.pca_x); minY = Math.min(minY, u.pca_y); maxY = Math.max(maxY, u.pca_y);
  }
  const px = (maxX - minX) * 0.02 || 1, py = (maxY - minY) * 0.02 || 1;
  view.bounds = { minX: minX - px, maxX: maxX + px, minY: minY - py, maxY: maxY + py };
}
const toX = (x) => MAP_PAD.x + ((x - view.bounds.minX) / (view.bounds.maxX - view.bounds.minX)) * (view.w - 2 * MAP_PAD.x);
const toY = (y) => view.h - MAP_PAD.bottom - ((y - view.bounds.minY) / (view.bounds.maxY - view.bounds.minY)) * (view.h - MAP_PAD.top - MAP_PAD.bottom);

function resizeMap() {
  const rect = el.plot.getBoundingClientRect();
  view.w = rect.width; view.h = rect.height;
  view.dpr = Math.min(window.devicePixelRatio || 1, 2);
  for (const c of [el.mapCanvas, view.bg, view.layer]) {
    c.width = Math.max(0, Math.round(view.w * view.dpr));
    c.height = Math.max(0, Math.round(view.h * view.dpr));
  }
}
const DOT_R = 1.35;
function paintDots(ctx, filter, color) {
  if (color) ctx.fillStyle = color;
  for (const p of view.points) {
    if (!filter(p)) continue;
    if (!color) ctx.fillStyle = colorOf(p.persona);
    ctx.beginPath();
    ctx.arc(toX(p.pca_x), toY(p.pca_y), DOT_R, 0, Math.PI * 2);
    ctx.fill();
  }
}
// Dots are painted opaque on a scratch layer and faded in one step.
// Low globalAlpha ensures dots appear as subtle, delicate ambient points
// rather than a congested mass, allowing the user's songs to shine.
function drawBackground() {
  const bg = view.bg.getContext("2d"), layer = view.layer.getContext("2d");
  for (const ctx of [bg, layer]) { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, view.bg.width, view.bg.height); }
  if (!view.points || !view.bounds || !view.w || !view.h || !view.bg.width) return;
  layer.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  bg.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  const results = state.status === "results", focus = state.focusPersona;
  paintDots(layer, (p) => !focus || p.persona !== focus, results ? (cssVar("--text-3") || "#998A9B") : null);
  bg.save();
  bg.setTransform(1, 0, 0, 1, 0, 0);
  // Lower opacity to eliminate clustered muddy patches: 0.16 in results mode, 0.30 in empty mode
  bg.globalAlpha = focus ? (results ? 0.08 : 0.06) : (results ? 0.16 : 0.30);
  bg.drawImage(view.layer, 0, 0);
  bg.restore();
  if (focus) { bg.globalAlpha = 0.85; paintDots(bg, (p) => p.persona === focus); bg.globalAlpha = 1; }
}
let POINT_RING = "#F4EDE8";
function drawUserPoint(ctx, pt, r) {
  ctx.beginPath();
  ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
  ctx.fillStyle = colorOf(pt.song.persona);
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = POINT_RING;
  ctx.stroke();
}
function drawMap() {
  const ctx = el.mapCanvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, el.mapCanvas.width, el.mapCanvas.height);
  if (!view.bounds || !view.bg.width || !view.bg.height) { hideMapTip(); return; }
  ctx.drawImage(view.bg, 0, 0);
  view.userPts = [];
  if (state.status !== "results") { hideMapTip(); return; }
  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  const focus = state.focusPersona;
  let hovered = null;
  for (const s of state.profile.user_songs || []) {
    const pt = { x: toX(s.pca_x), y: toY(s.pca_y), song: s };
    view.userPts.push(pt);
    if (s.track_id === state.hoverTrackId) { hovered = pt; continue; }
    ctx.globalAlpha = focus && s.persona !== focus ? 0.3 : 1;
    drawUserPoint(ctx, pt, 5.5);
  }
  ctx.globalAlpha = 1;
  if (hovered) {
    ctx.beginPath();
    ctx.arc(hovered.x, hovered.y, 12.5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(244, 237, 232, 0.5)";
    ctx.lineWidth = 1;
    ctx.stroke();
    drawUserPoint(ctx, hovered, 7.5);
    showMapTip(hovered);
  } else hideMapTip();
}
function showMapTip(pt) {
  const s = pt.song;
  const wasOpen = el.mapTip.classList.contains("is-open");
  el.mapTip.innerHTML = `
    ${artHtml(s.track_id, s.persona)}
    <span class="map-tip-text">
      <span class="map-tip-name">${esc(s.track_name)}</span>
      <span class="map-tip-artist">${esc(s.artists)}</span>
      <span class="map-tip-persona"><span class="dot" style="--c:${colorOf(s.persona)}"></span>${esc(nameOf(s.persona))}</span>
    </span>`;
  hydrateArt(el.mapTip);
  el.mapTip.hidden = false;
  el.mapTip.dataset.instant = String(wasOpen);
  const tw = el.mapTip.offsetWidth, th = el.mapTip.offsetHeight;
  let left = pt.x + 14, top = pt.y - th - 10;
  if (left + tw > view.w - 8) left = pt.x - tw - 14;
  if (top < 8) top = pt.y + 14;
  left = Math.max(8, left);
  el.mapTip.style.translate = `${left}px ${top}px`;
  el.mapTip.style.transformOrigin = `${pt.x < left ? "left" : "right"} ${pt.y < top ? "top" : "bottom"}`;
  if (!wasOpen) requestAnimationFrame(() => el.mapTip.classList.add("is-open"));
}
function hideMapTip() { el.mapTip.classList.remove("is-open"); el.mapTip.hidden = true; }

function renderMapLabels() {
  if (!view.centroids || !view.bounds || !view.w) { el.mapLabels.innerHTML = ""; return; }
  const results = state.status === "results";
  const active = new Set(results ? currentEntries().map((e) => e.persona) : PERSONA_ORDER);
  // The user's personas get full-weight labels and are placed first; the rest are the first to go.
  const order = [...PERSONA_ORDER.filter((p) => active.has(p)), ...PERSONA_ORDER.filter((p) => !active.has(p))].filter((p) => view.centroids[p]);
  el.mapLabels.innerHTML = order.map((p) => `
    <span class="map-label halo ${active.has(p) ? "" : "is-minor"}" data-persona="${esc(p)}" data-tip="${esc(blurbOf(p))}">
      <span class="dot" style="--c:${colorOf(p)}"></span>${esc(nameOf(p))}
    </span>`).join("");

  // Axis captions and the user's dots are fixed obstacles; labels move around them.
  const plotRect = el.plot.getBoundingClientRect();
  const fixed = [el.yHigh, el.yLow].map((node) => {
    const r = node.getBoundingClientRect();
    return { x: r.left - plotRect.left + r.width / 2, y: r.top - plotRect.top + r.height / 2, w: r.width, h: r.height, fixed: true };
  });
  if (results) for (const u of state.profile.user_songs || []) fixed.push({ x: toX(u.pca_x), y: toY(u.pca_y), w: 30, h: 26, fixed: true });
  const boxes = [...el.mapLabels.children].map((node) => {
    const c = view.centroids[node.dataset.persona];
    return { node, x: toX(c.x), y: toY(c.y), w: node.offsetWidth, h: node.offsetHeight, minor: node.classList.contains("is-minor") };
  });
  const overlap = (a, b, padX = 8, padY = 3) => {
    const ox = (a.w + b.w) / 2 + padX - Math.abs(a.x - b.x), oy = (a.h + b.h) / 2 + padY - Math.abs(a.y - b.y);
    return ox > 0 && oy > 0 ? { ox, oy } : null;
  };
  const clamp = (b) => {
    b.x = Math.max(b.w / 2 + 6, Math.min(view.w - b.w / 2 - 6, b.x));
    b.y = Math.max(b.h / 2 + 6, Math.min(view.h - b.h / 2 - 6, b.y));
  };
  boxes.forEach(clamp);
  const all = [...boxes, ...fixed];
  for (let iter = 0; iter < 120; iter++) {
    let moved = false;
    for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
      const a = all[i], b = all[j];
      if (a.fixed && b.fixed) continue;
      const o = overlap(a, b);
      if (!o) continue;
      // Labels are wide and short, so the push is usually vertical; sideways only when clearly smaller.
      const axis = o.ox < o.oy * 2 ? "x" : "y";
      const amt = (axis === "x" ? o.ox : o.oy) + 0.5;
      const dir = (axis === "x" ? a.x <= b.x : a.y <= b.y) ? -1 : 1;
      if (a.fixed) b[axis] -= dir * amt;
      else if (b.fixed) a[axis] += dir * amt;
      else { a[axis] += dir * amt / 2; b[axis] -= dir * amt / 2; }
      moved = true;
    }
    boxes.forEach(clamp);
    if (!moved) break;
  }
  // A label caught between two obstacles can oscillate above; if it still collides,
  // try positions around its centroid at growing radius instead.
  const collides = (b, others) => others.some((o) => overlap(b, o, 4, 1));
  const placed = [...fixed];
  for (const b of boxes) {
    b.hidden = false;
    if (collides(b, placed)) {
      const c = view.centroids[b.node.dataset.persona];
      const cx = toX(c.x), cy = toY(c.y), nx = b.x, ny = b.y;
      let found = false;
      for (let r = 14; r <= 98 && !found; r += 14) for (let k = 0; k < 8 && !found; k++) {
        const ang = (k / 8) * Math.PI * 2 + Math.PI / 2;
        b.x = cx + Math.cos(ang) * r; b.y = cy + Math.sin(ang) * r;
        clamp(b);
        found = !collides(b, placed);
      }
      if (!found) { b.x = nx; b.y = ny; b.hidden = b.minor; }
    }
    b.node.classList.toggle("is-hidden", b.hidden);
    if (!b.hidden) placed.push(b);
  }
  for (const b of boxes) b.node.style.translate = `${Math.round(b.x - b.w / 2)}px ${Math.round(b.y - b.h / 2)}px`;
  syncFocus();
}
function renderMapStatus() {
  let html = "";
  if (state.mapStatus === "loading") html = `<div>Loading the sound map…</div>`;
  else if (state.mapStatus === "error") html = `<div>Couldn't load the sound map.<button class="btn-quiet" type="button" data-action="retry-map">Try again</button></div>`;
  else if (state.status === "loading") html = `<div>Placing your songs…</div>`;
  el.mapStatus.innerHTML = html;
  el.mapStatus.hidden = !html;
}
function describeMap() {
  const sample = (state.map && state.map.points.length) || SOUND_MAP_SAMPLE;
  let label = `Sound map: a sample of ${fmtInt(sample)} tracks placed by how they sound, from ${AXIS_LABELS.x.low} on the left to ${AXIS_LABELS.x.high} on the right, and from ${AXIS_LABELS.y.low} at the bottom to ${AXIS_LABELS.y.high} at the top. Each of the 7 sounds is labelled at its centre.`;
  if (state.status === "results") label += ` Your ${(state.profile.user_songs || []).length} songs are the outlined dots: ` + currentEntries().map((e) => `${e.pct}% ${nameOf(e.persona)}`).join(", ") + ".";
  el.mapCanvas.setAttribute("aria-label", label);
  el.mapHint.textContent = state.status === "results" ? "Outlined dots are your songs" : `${fmtInt(sample)} sampled tracks`;
}
function refreshMap() {
  renderMapStatus();
  describeMap();
  if (!view.w) resizeMap();
  computeBounds();
  drawBackground();
  drawMap();
  renderMapLabels();
}
function pointAt(e, radius) {
  const rect = el.mapCanvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bestD = radius * radius;
  for (const pt of view.userPts) { const d = (pt.x - mx) ** 2 + (pt.y - my) ** 2; if (d < bestD) { bestD = d; best = pt; } }
  return best;
}
function onMapPointerMove(e) {
  if (e.pointerType !== "mouse" || state.status !== "results" || !view.userPts.length) return;
  const best = pointAt(e, 14);
  el.mapCanvas.style.cursor = best ? "pointer" : "default";
  setHoverTrack(best ? best.song.track_id : null, best ? "map" : null);
}
function onMapTap(e) {
  if (e.pointerType === "mouse" || state.status !== "results" || !view.userPts.length) return;
  const best = pointAt(e, 22);
  setHoverTrack(best ? best.song.track_id : null, best ? "map" : null);
}

/* ---------- shared tooltip ---------- */
let tipOwner = null, tipTimer = null, tipLastHidden = 0;
function showTip(target, instant) {
  const text = target.dataset.tip;
  if (!text) return;
  tipOwner = target;
  el.tip.textContent = text;
  el.tip.hidden = false;
  el.tip.classList.remove("is-open");
  el.tip.dataset.instant = String(instant);
  const r = target.getBoundingClientRect(), t = el.tip.getBoundingClientRect();
  const left = Math.max(8, Math.min(window.innerWidth - t.width - 8, r.left + r.width / 2 - t.width / 2));
  let top = r.top - t.height - 8;
  const below = top < 8;
  if (below) top = r.bottom + 8;
  el.tip.style.translate = `${Math.round(left)}px ${Math.round(top)}px`;
  el.tip.style.transformOrigin = `${r.left + r.width / 2 - left}px ${below ? "0" : "100%"}`;
  target.setAttribute("aria-describedby", "tip");
  requestAnimationFrame(() => el.tip.classList.add("is-open"));
}
function hideTip() {
  clearTimeout(tipTimer);
  if (!tipOwner) return;
  tipOwner.removeAttribute("aria-describedby");
  tipOwner = null;
  el.tip.classList.remove("is-open");
  el.tip.hidden = true;
  tipLastHidden = performance.now();
}
// First tooltip waits; neighbours open instantly while one is (or was just) open.
function onTipEnter(target, viaFocus) {
  if (target === tipOwner) return;
  clearTimeout(tipTimer);
  const instant = viaFocus || !!tipOwner || performance.now() - tipLastHidden < 400;
  if (tipOwner) hideTip();
  if (instant) showTip(target, true);
  else tipTimer = setTimeout(() => showTip(target, false), 350);
}

/* ---------- how this works ---------- */
function renderHow() {
  const ev = (state.map && state.map.explained_variance) || [0.343, 0.175];
  const v1 = Math.round(ev[0] * 100), v2 = Math.round(ev[1] * 100);
  el.axesTip.dataset.tip =
    `Each dot is a track, placed by how it sounds; your songs are the outlined dots. ` +
    `Horizontal: principal component 1, ${v1}% of the variation in the audio features. Vertical: principal component 2, ${v2}%. ` +
    `Together they keep ${v1 + v2}%, so nearby dots sound alike but the picture is a simplification.`;
  el.howBody.innerHTML = `
    <div>
      <h3>The sounds</h3>
      <p>${fmtInt(state.datasetSize)} Spotify tracks were grouped by nine audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, mood and tempo), with no genre labels involved. Energy, loudness and acousticness move together, so they were combined into a single “intensity” feature first.</p>
      <p>k-means clustering then split the tracks into 7 groups. Each group was named after the features that set it apart from the average track.</p>
    </div>
    <div>
      <h3>The map</h3>
      <p>Principal component analysis squeezes the seven clustering features into two dimensions so they can be drawn. The two axes keep ${v1}% and ${v2}% of the variation (${v1 + v2}% together), so the map is a useful sketch rather than an exact picture.</p>
      <p>Horizontal mostly tracks intensity: ${AXIS_LABELS.x.low} to ${AXIS_LABELS.x.high}. Vertical mostly tracks danceability and mood against intensity and instrumentalness: ${AXIS_LABELS.y.low} to ${AXIS_LABELS.y.high}.</p>
    </div>
    <div>
      <h3>Your mix and diversity</h3>
      <p>Each of your songs already belongs to one sound, so your mix is the share of your songs in each. Diversity is the normalized Shannon entropy of that mix: 0 means every song is in one sound, 1 means an even spread across all seven.</p>
    </div>
    <div>
      <h3>How you compare</h3>
      <p>We average each feature across your songs and rank that average against every individual track in the dataset. It compares your songs to <em>tracks</em>, not to other listeners.</p>
      <p>Recommendations are the three tracks closest on the map to the centre of your songs in your top sound.</p>
    </div>
    <div class="how-wide">
      <h3>A caveat</h3>
      <p>Real music doesn't fall into neat boxes. The clusters overlap (silhouette score about 0.17 to 0.20, where 1 would be perfectly separated), so a song near a border could reasonably sit in either sound.</p>
    </div>
    <div class="how-wide">
      <h3>What sets each sound apart</h3>
      <table class="how-table">
        <thead><tr><th scope="col">Sound</th><th scope="col">In plain words</th><th scope="col">vs. the average track (standard deviations)</th></tr></thead>
        <tbody>${PERSONA_ORDER.map((p) => { const d = PERSONA_DISPLAY[p]; return `<tr><td><span class="dot" style="--c:${colorOf(p)}" aria-hidden="true"></span>${esc(d.name)}</td><td>${esc(d.traits.join(", "))}</td><td>${esc(d.stats)}</td></tr>`; }).join("")}</tbody>
      </table>
    </div>`;
}

/* ---------- search & song actions ---------- */
let searchTimer = null;

async function runSearch(query) {
  const trimmed = query.trim();
  if (!trimmed) {
    closeSearch();
    return;
  }
  state.search.status = "loading";
  el.searchPop.hidden = false;
  el.searchInput.setAttribute("aria-expanded", "true");
  el.searchMsg.hidden = false;
  el.searchMsg.textContent = "Searching…";
  el.searchOptions.innerHTML = "";

  try {
    const data = await getJSON(`/api/search?q=${encodeURIComponent(trimmed)}&limit=30`);
    if (state.search.query.trim() !== trimmed) return;
    state.search.results = data || [];
    state.search.status = "done";
    state.search.active = -1;
    renderSearchPop();
  } catch (err) {
    if (state.search.query.trim() !== trimmed) return;
    state.search.status = "error";
    el.searchMsg.hidden = false;
    el.searchMsg.textContent = "Could not fetch search results.";
    el.searchOptions.innerHTML = "";
  }
}

function renderSearchPop() {
  const { results, query, active } = state.search;
  if (!results.length) {
    el.searchMsg.hidden = false;
    el.searchMsg.innerHTML = `No songs found for "<strong>${esc(query)}</strong>"`;
    el.searchOptions.innerHTML = "";
    el.searchPop.hidden = false;
    return;
  }
  el.searchMsg.hidden = true;
  el.searchPop.hidden = false;
  el.searchOptions.innerHTML = results.map((t, idx) => {
    const isAdded = state.tracks.some((s) => s.track_id === t.track_id);
    const isMax = state.tracks.length >= MAX_TRACKS && !isAdded;
    const disabled = isAdded || isMax;
    const selected = idx === active;
    const tag = isAdded ? "Added" : isMax ? "Full" : nameOf(t.persona);
    return `
      <li class="option" role="option" data-idx="${idx}" data-id="${esc(t.track_id)}"
          aria-selected="${selected}" ${disabled ? 'aria-disabled="true"' : ""}>
        <span class="dot" style="--c:${colorOf(t.persona)}" aria-hidden="true"></span>
        <span class="option-text">
          <span class="option-name">${esc(t.track_name)}</span>
          <span class="option-meta">${esc(t.artists)} • ${esc(t.album_name)}</span>
        </span>
        <span class="option-tag">${esc(tag)}</span>
      </li>`;
  }).join("");
}

function closeSearch() {
  el.searchPop.hidden = true;
  el.searchInput.setAttribute("aria-expanded", "false");
  state.search.active = -1;
  state.search.results = [];
}

function addTrack(track) {
  if (!track || state.tracks.length >= MAX_TRACKS) return;
  if (state.tracks.some((t) => t.track_id === track.track_id)) return;
  state.tracks.push(track);
  el.searchInput.value = "";
  state.search.query = "";
  closeSearch();
  renderAll();
  announce(`Added ${track.track_name}`);
}

function removeTrack(trackId) {
  const idx = state.tracks.findIndex((t) => t.track_id === trackId);
  if (idx < 0) return;
  const removed = state.tracks.splice(idx, 1)[0];
  renderAll();
  announce(`Removed ${removed.track_name}`);
}

function clearTracks() {
  state.tracks = [];
  state.status = "empty";
  state.profile = null;
  state.analyzedIds = [];
  renderAll();
  announce("Cleared all songs");
}

async function analyzeTaste() {
  if (state.tracks.length === 0 || state.status === "loading") return;
  const trackIds = state.tracks.map((t) => t.track_id);
  state.status = "loading";
  renderHeader();
  try {
    const res = await fetch(API_BASE + "/api/taste-profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_ids: trackIds }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const profile = await res.json();
    state.profile = profile;
    state.analyzedIds = [...trackIds];
    state.status = "results";
    freshResults = true;
    renderAll();
    announce("Taste analysis complete");
    requestAnimationFrame(() => { freshResults = false; });
  } catch (err) {
    state.status = isStale() ? "results" : "empty";
    renderHeader();
    alert("Could not analyze taste: " + err.message);
  }
}

/* ---------- 10-Second Music Chorus Preview Player ---------- */
const player = {
  audio: new Audio(),
  track: null,
  isPlaying: false,
  durationLimit: 10.0, // 10s preview
};

// Cache preloaded audio objects and URLs in memory for 0ms instant playback
const previewCache = new Map(); // trackId -> { url, audio }

function setupAudioEvents(audio) {
  audio.ontimeupdate = () => {
    if (audio.currentTime >= player.durationLimit) {
      triggerPreviewFinish();
    } else {
      updatePlayerProgress();
    }
  };
  audio.onended = () => {
    triggerPreviewFinish();
  };
  audio.onpause = () => {
    if (player.isPlaying && player.audio === audio) {
      player.isPlaying = false;
      syncPlayerUI();
    }
  };
  audio.onerror = () => {
    if (player.audio === audio) {
      player.isPlaying = false;
      syncPlayerUI();
    }
  };
}

// Bind events to the initial audio instance
setupAudioEvents(player.audio);

function preloadRecPreviews(recs) {
  if (!Array.isArray(recs)) return;
  for (const r of recs) {
    if (!r || !r.track_id) continue;
    const trackId = r.track_id;
    if (previewCache.has(trackId)) continue;

    // Immediately fetch and pre-buffer MP3 audio into memory in the background
    getJSON(`/api/preview?track_name=${encodeURIComponent(r.track_name)}&artist=${encodeURIComponent(r.artists)}`)
      .then((data) => {
        if (data && data.status === "available" && data.preview_url) {
          const a = new Audio();
          a.preload = "auto";
          a.src = data.preview_url;
          previewCache.set(trackId, { url: data.preview_url, audio: a });
        }
      })
      .catch(() => {});
  }
}

function formatPlayerTime(seconds) {
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function updatePlayerProgress() {
  const current = player.audio.currentTime;
  const pct = Math.min(100, (current / player.durationLimit) * 100);
  const timeStr = formatPlayerTime(current);

  document.querySelectorAll(".rec-card").forEach((card) => {
    const isThis = Boolean(player.track && card.dataset.trackId === player.track.track_id);
    const fill = card.querySelector(".rec-progress-fill");
    const curTime = card.querySelector(".rec-time-cur");
    const bar = card.querySelector(".rec-progress-bar");
    if (fill) fill.style.width = isThis ? `${pct}%` : "0%";
    if (curTime) curTime.textContent = isThis ? timeStr : "0:00";
    if (bar) bar.setAttribute("aria-valuenow", isThis ? Math.round(current) : 0);
  });
}

function syncPlayerUI() {
  const isPlaying = Boolean(player.isPlaying);
  const activeId = player.track ? player.track.track_id : null;

  document.querySelectorAll(".rec-card").forEach((card) => {
    const isThisTrack = Boolean(activeId && card.dataset.trackId === activeId);
    const playing = isThisTrack && isPlaying;

    card.classList.toggle("is-active", playing);

    const playBtn = card.querySelector(".rec-play-btn");
    if (playBtn) {
      playBtn.classList.toggle("is-playing", playing);
      playBtn.setAttribute("aria-label", playing ? `Pause preview of ${card.dataset.name || "track"}` : `Play 10s preview of ${card.dataset.name || "track"}`);
    }

    const label = card.querySelector(".rec-preview-label, .rec-badge");
    if (label) {
      label.classList.toggle("is-playing", playing);
    }
  });

  // Keep any row preview buttons in sync
  document.querySelectorAll(".btn-row-preview").forEach((btn) => {
    const isThisTrack = Boolean(activeId && btn.dataset.playPreview === activeId);
    btn.classList.toggle("is-playing", isThisTrack && isPlaying);
  });
}

function triggerPreviewFinish() {
  pausePreview();
  player.audio.currentTime = 0;
  updatePlayerProgress();

  if (player.track) {
    const currentId = player.track.track_id;
    document.querySelectorAll(`.rec-card[data-track-id="${CSS.escape(currentId)}"] .rec-play-btn`).forEach((btn) => {
      btn.classList.add("just-stopped");
      setTimeout(() => btn.classList.remove("just-stopped"), 450);
    });
    document.querySelectorAll(`[data-play-preview="${CSS.escape(currentId)}"]`).forEach((btn) => {
      btn.classList.add("just-stopped");
      setTimeout(() => btn.classList.remove("just-stopped"), 450);
    });
  }
}

async function playPreview(trackId, trackName, artist) {
  // If clicking on the currently active track, toggle play/pause
  if (player.track && player.track.track_id === trackId) {
    if (player.isPlaying) {
      pausePreview();
    } else {
      if (player.audio.currentTime >= player.durationLimit - 0.1) {
        player.audio.currentTime = 0;
        updatePlayerProgress();
      }
      player.audio.play().then(() => {
        player.isPlaying = true;
        syncPlayerUI();
      }).catch(() => {
        player.isPlaying = false;
        syncPlayerUI();
      });
    }
    return;
  }

  // Stop previous track if playing
  if (player.isPlaying && player.audio) {
    player.audio.pause();
    player.isPlaying = false;
  }

  // Instant visual feedback: mark active and start play animation immediately on click
  player.track = { track_id: trackId, track_name: trackName, artists: artist };
  player.isPlaying = true;
  syncPlayerUI();
  updatePlayerProgress();

  // If already preloaded into memory, play INSTANTLY with 0ms delay
  const cached = previewCache.get(trackId);
  if (cached && cached.audio) {
    player.audio = cached.audio;
    player.audio.currentTime = 0;
    setupAudioEvents(player.audio);
    try {
      await player.audio.play();
      syncPlayerUI();
      announce(`Playing 10 second preview of ${trackName}`);
    } catch (_) {
      player.isPlaying = false;
      syncPlayerUI();
    }
    return;
  }

  // Fast on-demand fallback if clicked before background preload finished
  try {
    const data = await getJSON(`/api/preview?track_name=${encodeURIComponent(trackName)}&artist=${encodeURIComponent(artist)}`);
    if (data.status === "available" && data.preview_url) {
      const a = new Audio(data.preview_url);
      a.preload = "auto";
      previewCache.set(trackId, { url: data.preview_url, audio: a });

      if (player.track && player.track.track_id === trackId) {
        player.audio = a;
        player.audio.currentTime = 0;
        setupAudioEvents(player.audio);
        await player.audio.play();
        player.isPlaying = true;
        syncPlayerUI();
        announce(`Playing 10 second preview of ${trackName}`);
      }
    } else {
      if (player.track && player.track.track_id === trackId) {
        player.isPlaying = false;
        syncPlayerUI();
        alert(`Audio preview not available for "${trackName}". You can open it on Spotify!`);
      }
    }
  } catch (err) {
    if (player.track && player.track.track_id === trackId) {
      player.isPlaying = false;
      syncPlayerUI();
    }
  }
}

function pausePreview() {
  player.audio.pause();
  player.isPlaying = false;
  syncPlayerUI();
}

/* ==============================================================================
   SONALITY SHARE CARD (4:5 Editorial Portrait Card & Modal)
   ============================================================================== */

const isIOS = () => {
  return (
    typeof navigator !== "undefined" &&
    (/iPad|iPhone|iPod/.test(navigator.userAgent) ||
      (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1))
  );
};

function formatLeadIn(rawName) {
  const clean = (rawName || "").trim();
  if (!clean) return "My sound is mostly";
  const poss = clean.toLowerCase().endsWith("s") ? `${clean}’` : `${clean}’s`;
  return `${poss} sound is mostly`;
}

function hexToRgba(hex, alpha) {
  const clean = (hex || "#7DF29C").replace("#", "");
  const n = parseInt(clean, 16);
  if (isNaN(n)) return `rgba(125, 242, 156, ${alpha})`;
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function computeConvexHull(pts) {
  if (pts.length <= 2) return pts.map((p) => ({ x: p.px, y: p.py }));
  const sorted = pts
    .map((p) => ({ x: p.px, y: p.py }))
    .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

/**
 * Clean Category Centroids + Anchor + Outlier Radar
 * Clamped strictly to safeR = 34px (63% of outer R=54) so dots and rings never collide with axis text.
 */
function computeRefinedRadarMap(userSongs, anchorTrack, outlierTrack) {
  const size = 132;
  const mid = 66;
  const safeR = 40; // safe plotting disk radius (~81.6% of outer radius 49px), keeping dots cleanly inside rim and zero collision with outside labels

  // 1. Group songs by persona and compute category centroids
  const catMap = new Map();
  userSongs.forEach((s) => {
    const p = s.persona;
    if (!catMap.has(p)) catMap.set(p, []);
    catMap.get(p).push(s);
  });

  const categoryCentroids = [];
  catMap.forEach((songsInCat, persona) => {
    const avgX = songsInCat.reduce((sum, s) => sum + (s.pca_x || 0), 0) / songsInCat.length;
    const avgY = songsInCat.reduce((sum, s) => sum + (s.pca_y || 0), 0) / songsInCat.length;
    categoryCentroids.push({
      persona,
      count: songsInCat.length,
      pca_x: avgX,
      pca_y: avgY,
    });
  });

  // 2. Normalize and project all key points into the safe plotting disk
  const allPoints = [...categoryCentroids];
  if (anchorTrack) allPoints.push(anchorTrack);
  if (outlierTrack && (!anchorTrack || outlierTrack.track_id !== anchorTrack.track_id)) {
    allPoints.push(outlierTrack);
  }

  const maxMag = Math.max(1.5, ...allPoints.map((s) => Math.hypot(s.pca_x || 0, s.pca_y || 0)));

  const projectPoint = (x, y) => {
    const rawX = (x || 0) / maxMag;
    const rawY = (y || 0) / maxMag;
    const mag = Math.hypot(rawX, rawY);
    const clampedNorm = Math.min(1, mag);
    const angle = Math.atan2(rawY, rawX);
    const r = clampedNorm * safeR;
    return {
      px: mid + Math.cos(angle) * r,
      py: mid - Math.sin(angle) * r,
    };
  };

  // Category dots
  const catNodes = categoryCentroids.map((c) => {
    const pt = projectPoint(c.pca_x, c.pca_y);
    return {
      type: "category",
      persona: c.persona,
      count: c.count,
      px: pt.px,
      py: pt.py,
    };
  });

  // Anchor and Outlier nodes
  let anchorNode = null;
  if (anchorTrack) {
    const pt = projectPoint(anchorTrack.pca_x, anchorTrack.pca_y);
    anchorNode = {
      type: "anchor",
      persona: anchorTrack.persona,
      track_name: anchorTrack.track_name,
      artists: anchorTrack.artists,
      px: pt.px,
      py: pt.py,
    };
  }

  let outlierNode = null;
  if (outlierTrack) {
    const pt = projectPoint(outlierTrack.pca_x, outlierTrack.pca_y);
    outlierNode = {
      type: "outlier",
      persona: outlierTrack.persona,
      track_name: outlierTrack.track_name,
      artists: outlierTrack.artists,
      px: pt.px,
      py: pt.py,
    };
  }

  // Combined polygon nodes
  const hullNodes = [...catNodes];
  if (anchorNode && !hullNodes.some((n) => Math.hypot(n.px - anchorNode.px, n.py - anchorNode.py) < 3)) {
    hullNodes.push(anchorNode);
  }
  if (outlierNode && !hullNodes.some((n) => Math.hypot(n.px - outlierNode.px, n.py - outlierNode.py) < 3)) {
    hullNodes.push(outlierNode);
  }

  const hull = computeConvexHull(hullNodes);

  return { size, mid, safeR, catNodes, anchorNode, outlierNode, hull };
}

function openShareModal() {
  if (!state.profile || !state.profile.persona_breakdown) {
    alert("Please analyze your taste first to generate your Sonality card.");
    return;
  }
  const nameInput = document.getElementById("sc-user-name");
  const clearNameBtn = document.getElementById("sc-name-clear");
  let savedName = "";
  try { savedName = localStorage.getItem("sonality_user_name") || ""; } catch (_) {}
  if (nameInput) nameInput.value = savedName;
  if (clearNameBtn) clearNameBtn.hidden = !savedName;

  updateShareCardData();
  const modal = document.getElementById("share-modal");
  if (!modal) return;
  const dlText = document.getElementById("sc-dl-text");
  if (dlText) {
    dlText.textContent = isIOS() ? "Save to Photos" : "Download image";
  }
  const shareBtn = document.getElementById("sc-share");
  if (shareBtn) {
    shareBtn.hidden = !(typeof navigator !== "undefined" && Boolean(navigator.share));
  }
  if (typeof modal.showModal === "function") {
    modal.showModal();
  } else {
    modal.setAttribute("open", "");
  }
  document.body.style.overflow = "hidden";
  requestAnimationFrame(() => {
    fitCard(document.getElementById("share-card"));
  });
}

function closeShareModal() {
  const modal = document.getElementById("share-modal");
  if (!modal) return;
  if (typeof modal.close === "function") {
    modal.close();
  } else {
    modal.removeAttribute("open");
  }
  document.body.style.removeProperty("overflow");
}

function fitCard(c) {
  // Editorial Craft card scales organically with 400x565 aspect ratio and container query units (cqw)
}

function updateShareCardData() {
  const p = state.profile;
  if (!p || !p.persona_breakdown) return;
  const entries = breakdownEntries(p.persona_breakdown);
  if (!entries.length) return;
  const top = entries[0];
  const matched = (p.user_songs || []).length || (state.tracks || []).length;
  const card = document.getElementById("share-card");
  if (!card) return;

  const dominantColor = colorOf(top.persona);
  card.style.setProperty("--c", dominantColor);
  card.style.setProperty("--c-sheen", hexToRgba(dominantColor, 0.055));

  // 3. Dynamic Editorial Headline Copy
  const topPct = top.pct;
  const second = entries[1];
  const secondPct = second ? second.pct : 0;
  const isTied = Boolean(second && topPct === secondPct);

  let userName = "";
  try {
    const inputEl = document.getElementById("sc-user-name");
    userName = inputEl && inputEl.value !== undefined ? inputEl.value : (localStorage.getItem("sonality_user_name") || "");
  } catch (_) {}
  const cleanName = (userName || "").trim();
  const subject = cleanName
    ? (cleanName.toLowerCase().endsWith("s") ? `${cleanName}’` : `${cleanName}’s`) + " sound"
    : "My sound";

  let leadInText = "";
  if (isTied) {
    leadInText = `${subject} is split, led by`;
  } else if (topPct >= 50) {
    leadInText = `${subject} is mostly`;
  } else {
    leadInText = `${subject} leans`;
  }

  const eyebrowEl = document.getElementById("sc-eyebrow");
  if (eyebrowEl) {
    eyebrowEl.textContent = leadInText;
  }

  const tagEl = document.getElementById("sc-tag");
  if (tagEl) {
    tagEl.textContent = new Date().toLocaleDateString("en-AU", { month: "short", year: "numeric" });
  }

  const headEl = document.getElementById("sc-head");
  if (headEl) {
    headEl.textContent = `${nameOf(top.persona)}.`;
  }

  const subEl = document.getElementById("sc-sub");
  if (subEl) {
    if (entries.length === 1) {
      subEl.innerHTML = `Pure <strong>${esc(nameOf(top.persona))}</strong><span class="sc-sep">&middot;</span>1 of 7 sounds`;
    } else if (isTied) {
      subEl.innerHTML = `Tied with <strong>${esc(nameOf(second.persona))}</strong><span class="sc-sep">&middot;</span>${entries.length} of 7 sounds`;
    } else {
      subEl.innerHTML = `With a streak of <strong>${esc(nameOf(second.persona))}</strong><span class="sc-sep">&middot;</span>${entries.length} of 7 sounds`;
    }
  }

  // Anchor and Outlier track identification (n < 3 edge case: hide Outlier, display only Anchor)
  const userSongs = p.user_songs || [];
  let anchor = p.anchor;
  let outlier = userSongs.length < 3 ? null : p.outlier;
  if (outlier && anchor && outlier.track_id === anchor.track_id) {
    outlier = null;
  }
  if (!anchor && userSongs.length > 0) {
    if (userSongs.length < 3) {
      let anchorIdx = 0;
      if (userSongs.length === 2) {
        const p0Dom = userSongs[0].persona === top.persona;
        const p1Dom = userSongs[1].persona === top.persona;
        if (!p0Dom && p1Dom) anchorIdx = 1;
      }
      anchor = userSongs[anchorIdx];
      outlier = null;
    } else {
      const dists = userSongs.map((s1, i) => {
        let sum = 0;
        userSongs.forEach((s2, j) => {
          if (i !== j) sum += Math.hypot((s1.pca_x || 0) - (s2.pca_x || 0), (s1.pca_y || 0) - (s2.pca_y || 0));
        });
        return { track: s1, sum, isDominant: s1.persona === top.persona };
      });
      const forAnchor = [...dists].sort((a, b) => {
        if (Math.abs(a.sum - b.sum) > 1e-6) return a.sum - b.sum;
        if (a.isDominant !== b.isDominant) return a.isDominant ? -1 : 1;
        return 0;
      });
      anchor = forAnchor[0].track;

      const forOutlier = dists.filter((d) => d.track.track_id !== anchor.track_id).sort((a, b) => {
        if (Math.abs(a.sum - b.sum) > 1e-6) return b.sum - a.sum;
        if (a.isDominant !== b.isDominant) return a.isDominant ? 1 : -1;
        return 0;
      });
      outlier = forOutlier.length > 0 ? forOutlier[0].track : null;
    }
  }

  // 3. Centerpiece: Circular Radar SVG (Category Centroids + Anchor & Outlier)
  const radarSvg = document.getElementById("sc-radar-svg");
  if (radarSvg && userSongs.length > 0) {
    const mapData = computeRefinedRadarMap(userSongs, anchor, outlier);
    const hullPts = mapData.hull.map((pt) => `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`).join(" ");

    // Chords from anchor to category centroids and outlier
    let chords = "";
    if (mapData.anchorNode) {
      const anc = mapData.anchorNode;
      const otherPoints = [...mapData.catNodes];
      if (mapData.outlierNode && mapData.outlierNode !== anc) {
        otherPoints.push(mapData.outlierNode);
      }
      chords = otherPoints
        .filter((n) => Math.hypot(n.px - anc.px, n.py - anc.py) > 2)
        .map((n) => `<line x1="${anc.px.toFixed(1)}" y1="${anc.py.toFixed(1)}" x2="${n.px.toFixed(1)}" y2="${n.py.toFixed(1)}" stroke="${hexToRgba(dominantColor, 0.28)}" stroke-width="1" />`)
        .join("");
    }

    // Category Centroid dots
    const catDotsSVG = mapData.catNodes.map((n) => {
      const c = colorOf(n.persona);
      const px = n.px.toFixed(1);
      const py = n.py.toFixed(1);
      return `<circle cx="${px}" cy="${py}" r="4.5" fill="${c}" stroke="#141117" stroke-width="2" />`;
    }).join("");

    // Anchor node (solid halo ring)
    let anchorDotSVG = "";
    if (mapData.anchorNode) {
      const px = mapData.anchorNode.px.toFixed(1);
      const py = mapData.anchorNode.py.toFixed(1);
      const c = colorOf(mapData.anchorNode.persona);
      anchorDotSVG = `
        <circle cx="${px}" cy="${py}" r="8.5" fill="none" stroke="rgba(255, 255, 255, 0.45)" stroke-width="1.1" />
        <circle cx="${px}" cy="${py}" r="4" fill="${c}" stroke="#141117" stroke-width="1.8" />
      `;
    }

    // Outlier node (dashed colored ring)
    let outlierDotSVG = "";
    if (mapData.outlierNode && mapData.outlierNode !== mapData.anchorNode) {
      const px = mapData.outlierNode.px.toFixed(1);
      const py = mapData.outlierNode.py.toFixed(1);
      const c = colorOf(mapData.outlierNode.persona);
      outlierDotSVG = `
        <circle cx="${px}" cy="${py}" r="9" fill="none" stroke="${c}" stroke-width="1.15" stroke-dasharray="2.5 2" />
        <circle cx="${px}" cy="${py}" r="4" fill="${c}" stroke="#141117" stroke-width="1.8" />
      `;
    }

    radarSvg.innerHTML = `
      <circle cx="66" cy="66" r="49" fill="rgba(255, 255, 255, 0.016)" stroke="rgba(255, 255, 255, 0.12)" stroke-width="1" />
      <circle cx="66" cy="66" r="32.7" fill="none" stroke="rgba(255, 255, 255, 0.06)" stroke-width="1" stroke-dasharray="2 3" />
      <circle cx="66" cy="66" r="16.3" fill="none" stroke="rgba(255, 255, 255, 0.06)" stroke-width="1" stroke-dasharray="2 3" />
      <line x1="17" y1="66" x2="115" y2="66" stroke="rgba(255, 255, 255, 0.055)" stroke-width="1" />
      <line x1="66" y1="17" x2="66" y2="115" stroke="rgba(255, 255, 255, 0.055)" stroke-width="1" />

      <!-- High-contrast outside axis labels (~55% opacity, 9.5-10px) -->
      <text x="66" y="9" text-anchor="middle" dy="0.3em" font-size="7.6" font-family="'IBM Plex Mono', monospace" fill="#FFFFFF" letter-spacing="0.08em" opacity="0.55">UPBEAT</text>
      <text x="66" y="123" text-anchor="middle" dy="0.3em" font-size="7.6" font-family="'IBM Plex Mono', monospace" fill="#FFFFFF" letter-spacing="0.08em" opacity="0.55">MOODY</text>
      <text x="9" y="66" text-anchor="middle" dy="0.3em" transform="rotate(-90 9 66)" font-size="7.4" font-family="'IBM Plex Mono', monospace" fill="#FFFFFF" letter-spacing="0.08em" opacity="0.55">ACOUSTIC</text>
      <text x="123" y="66" text-anchor="middle" dy="0.3em" transform="rotate(90 123 66)" font-size="7.4" font-family="'IBM Plex Mono', monospace" fill="#FFFFFF" letter-spacing="0.08em" opacity="0.55">INTENSE</text>

      <polygon points="${hullPts}" fill="${hexToRgba(dominantColor, 0.14)}" stroke="${hexToRgba(dominantColor, 0.48)}" stroke-width="1.25" stroke-linejoin="round" />
      ${chords}
      ${catDotsSVG}
      ${outlierDotSVG}
      ${anchorDotSVG}
    `;
  }

  // Sound Mix Legend & Proportional Continuous Split Bar (1-4 named vs 5+ sounds rules)
  const legEl = document.getElementById("sc-leg");
  const splitBarEl = document.getElementById("sc-split-bar");
  if (entries.length <= 4) {
    // 1 to 4 sounds: List all clusters by name in legend, map 1:1 to mix bar
    if (legEl) {
      legEl.innerHTML = entries.map((e, idx) => `
        <li class="${idx === 0 ? 'is-primary' : ''}">
          <div class="sc-leg-left">
            <span class="sc-leg-dot" style="background-color: ${colorOf(e.persona)}"></span>
            <span class="sc-leg-name" title="${esc(nameOf(e.persona))}">${esc(nameOf(e.persona))}</span>
          </div>
          <span class="sc-leg-pct">${e.pct}%</span>
        </li>
      `).join("");
    }
    if (splitBarEl) {
      splitBarEl.innerHTML = entries.map((e) => `
        <div style="flex: ${e.pct} 0 0%; background-color: ${colorOf(e.persona)}; border-radius: 999px;"></div>
      `).join("");
    }
  } else {
    // 5 or more sounds: Top 3 by name + muted 4th row "+{remainingCount} other sounds"
    const top3 = entries.slice(0, 3);
    const remCount = entries.length - 3;
    const topSum = top3.reduce((sum, e) => sum + e.pct, 0);
    const remPct = Math.max(0, 100 - topSum);

    if (legEl) {
      const top3HTML = top3.map((e, idx) => `
        <li class="${idx === 0 ? 'is-primary' : ''}">
          <div class="sc-leg-left">
            <span class="sc-leg-dot" style="background-color: ${colorOf(e.persona)}"></span>
            <span class="sc-leg-name" title="${esc(nameOf(e.persona))}">${esc(nameOf(e.persona))}</span>
          </div>
          <span class="sc-leg-pct">${e.pct}%</span>
        </li>
      `).join("");
      const otherRowHTML = `
        <li class="is-other">
          <div class="sc-leg-left">
            <span class="sc-leg-dot" style="background-color: rgba(255, 255, 255, 0.22);"></span>
            <span class="sc-leg-name">+${remCount} other sound${remCount === 1 ? '' : 's'}</span>
          </div>
          <span class="sc-leg-pct">${remPct}%</span>
        </li>
      `;
      legEl.innerHTML = top3HTML + otherRowHTML;
    }

    if (splitBarEl) {
      const segs = top3.map((e) => `
        <div style="flex: ${e.pct} 0 0%; background-color: ${colorOf(e.persona)}; border-radius: 999px;"></div>
      `);
      segs.push(`
        <div style="flex: ${remPct} 0 0%; background-color: rgba(255, 255, 255, 0.14); border-radius: 999px;" title="+${remPct}% other sounds"></div>
      `);
      splitBarEl.innerHTML = segs.join("");
    }
  }

  // 4. Key Tracks: Flipped Spanning Layout with Scaled Up 46x46 Album Art (n < 3 edge case supported)
  const tracksEl = document.getElementById("sc-tracks");
  if (tracksEl && anchor) {
    const anchorColor = colorOf(anchor.persona);
    const hasOutlier = Boolean(outlier && outlier.track_id !== anchor.track_id);
    const outlierColor = hasOutlier ? colorOf(outlier.persona) : null;

    const anchorGlyph = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <circle cx="6" cy="6" r="5.1" stroke="rgba(255, 255, 255, 0.55)" stroke-width="1.05" />
      <circle cx="6" cy="6" r="2.6" fill="${anchorColor}" />
    </svg>`;

    const outlierGlyph = hasOutlier ? `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <circle cx="6" cy="6" r="5.1" stroke="${outlierColor}" stroke-width="1.1" stroke-dasharray="2.1 1.7" />
      <circle cx="6" cy="6" r="2.6" fill="${outlierColor}" />
    </svg>` : '';

    const renderThumb = (trackId, color) => {
      const known = artUrls.get(trackId);
      const imgTag = known ? `<img src="${esc(known)}" alt="" />` : "";
      const fallback = `<span class="sc-track-fallback" style="background:${hexToRgba(color, 0.22)}; color:${color}; ${known ? 'display:none;' : ''}">♪</span>`;
      return `<div class="sc-track-thumb" data-art="${esc(trackId)}">${imgTag}${fallback}</div>`;
    };

    const anchorRowHTML = `
      <div class="sc-track-row">
        <div class="sc-track-main">
          ${renderThumb(anchor.track_id, anchorColor)}
          <div class="sc-track-info">
            <span class="sc-track-title" title="${esc(anchor.track_name)}">${esc(anchor.track_name)}</span>
            <span class="sc-track-artist" title="${esc(anchor.artists)}">${esc(anchor.artists)}</span>
          </div>
        </div>
        <div class="sc-track-badge" data-tip="Anchor: The track closest on average to all your selected songs (medoid)" title="Anchor: The track closest on average to all your selected songs (medoid)">
          <div class="sc-track-badge-top">
            <span class="sc-track-glyph">${anchorGlyph}</span>
            <span class="sc-track-tag">Anchor</span>
          </div>
          <span class="sc-track-subtag">Core center</span>
        </div>
      </div>
    `;

    const outlierRowHTML = hasOutlier ? `
      <div class="sc-track-row">
        <div class="sc-track-main">
          ${renderThumb(outlier.track_id, outlierColor)}
          <div class="sc-track-info">
            <span class="sc-track-title" title="${esc(outlier.track_name)}">${esc(outlier.track_name)}</span>
            <span class="sc-track-artist" title="${esc(outlier.artists)}">${esc(outlier.artists)}</span>
          </div>
        </div>
        <div class="sc-track-badge" data-tip="Outlier: The track furthest from the rest of your taste" title="Outlier: The track furthest from the rest of your taste">
          <div class="sc-track-badge-top">
            <span class="sc-track-glyph">${outlierGlyph}</span>
            <span class="sc-track-tag">Outlier</span>
          </div>
          <span class="sc-track-subtag">Furthest pull</span>
        </div>
      </div>
    ` : '';

    tracksEl.innerHTML = anchorRowHTML + outlierRowHTML;

    // Hydrate album art for Anchor and Outlier (if present)
    [anchor.track_id, hasOutlier ? outlier.track_id : null].filter(Boolean).forEach((id) => {
      getArt(id).then((url) => {
        if (!url) return;
        const thumb = tracksEl.querySelector(`[data-art="${id}"]`);
        if (thumb && !thumb.querySelector("img")) {
          const img = new Image();
          img.alt = "";
          img.onload = () => {
            thumb.innerHTML = "";
            thumb.appendChild(img);
          };
          img.src = url;
        }
      });
    });
  }

  // 5. Audio Profile Diverging Rows
  const tblEl = document.getElementById("sc-tbl");
  if (tblEl && p.percentiles) {
    const rawMetrics = [
      ["Danceability", p.percentiles.danceability],
      ["Energy", p.percentiles.energy],
      ["Mood", p.percentiles.valence],
      ["Acousticness", p.percentiles.acousticness],
    ];
    const ord = (n) => n + (["th", "st", "nd", "rd"][(n % 100 > 10 && n % 100 < 14) ? 0 : Math.min(n % 10, 4) % 4] || "th");
    const sg = (d) => d === 0 ? "0" : (d < 0 ? "−" : "+") + Math.abs(d);

    tblEl.innerHTML = rawMetrics.map(([label, rawVal]) => {
      const val = Math.round(rawVal !== undefined ? rawVal : 50);
      const delta = val - 50;
      const isPos = delta >= 0;
      const widthPct = Math.min(50, Math.abs(delta));
      const leftPct = isPos ? 50 : 50 - widthPct;
      const barCol = isPos ? dominantColor : "#766E80";
      const tipLeft = isPos ? 50 + widthPct : 50 - widthPct;

      return `
        <span class="sc-prof-row-name">${esc(label)}</span>
        <div class="sc-prof-bar-box">
          <div class="sc-prof-bar-track"></div>
          <div class="sc-prof-bar-fill" style="left: ${leftPct}%; width: ${widthPct}%; background-color: ${barCol};"></div>
          <div class="sc-prof-bar-tip" style="left: ${tipLeft}%; background-color: ${barCol};"></div>
        </div>
        <span class="sc-prof-row-pctl">${ord(val)}</span>
        <span class="sc-prof-row-delta" style="color: ${isPos ? dominantColor : '#766E80'};">${sg(delta)}</span>
      `;
    }).join("");
  }

  // 6. Clean Footer
  const metaEl = document.getElementById("sc-meta");
  if (metaEl) {
    metaEl.textContent = `${matched} songs · vs. ${fmtInt(state.datasetSize || 127900)} tracks`;
  }

  const urlEl = document.getElementById("sc-url");
  if (urlEl) {
    const host = window.location.hostname;
    urlEl.textContent = (host && host !== "localhost" && host !== "127.0.0.1")
      ? host
      : "app.sonality.workers.dev";
  }

  fitCard(card);
}

async function exportShareCardBlob() {
  if (typeof htmlToImage === "undefined") {
    throw new Error("htmlToImage library is not loaded");
  }
  await document.fonts.ready;
  const card = document.getElementById("share-card");
  if (!card) throw new Error("Share card element not found");

  // 400x600 logical aspect ratio -> 800x1200 at 2x retina (full-bleed matte, no outer border)
  const stage = document.createElement("div");
  stage.className = "sc-frame";
  stage.style.cssText = "position:fixed;left:-99999px;top:0;width:800px;height:1200px;border-radius:0;box-shadow:none;border:none;background:#141117;z-index:-1;";
  const clone = card.cloneNode(true);
  clone.removeAttribute("id");
  clone.style.border = "none";
  clone.style.borderRadius = "0";
  clone.style.boxShadow = "none";
  stage.appendChild(clone);
  document.body.appendChild(stage);

  try {
    fitCard(clone);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const options = { width: 800, height: 1200, pixelRatio: 2, cacheBust: true };
    let blob;
    try {
      blob = await htmlToImage.toBlob(clone, options);
    } catch (_) {
      blob = await htmlToImage.toBlob(clone, { ...options, skipFonts: true });
    }
    return blob;
  } finally {
    stage.remove();
  }
}

async function downloadShareCardPng() {
  const btn = document.getElementById("sc-dl");
  const textEl = document.getElementById("sc-dl-text");
  const isApple = isIOS();
  const defaultText = isApple ? "Save to Photos" : "Download image";
  const oldText = textEl ? textEl.textContent : defaultText;
  if (textEl) textEl.textContent = isApple ? "Preparing…" : "Rendering 1200p…";
  if (btn) btn.disabled = true;

  try {
    const blob = await exportShareCardBlob();
    let savedName = "";
    try { savedName = (localStorage.getItem("sonality_user_name") || "").trim(); } catch (_) {}
    const prefix = savedName ? savedName.toLowerCase().replace(/[^a-z0-9]+/g, "-") : "my-sound";
    const dominantName = (state.profile && state.profile.persona_breakdown) ? Object.keys(state.profile.persona_breakdown)[0] : "card";
    const fileName = `sonality-${prefix}-${dominantName.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
    const file = new File([blob], fileName, { type: "image/png" });

    if (isApple && navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({
        files: [file],
      });
      if (textEl) textEl.textContent = "Saved!";
      setTimeout(() => { if (textEl) textEl.textContent = defaultText; }, 1800);
    } else {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = fileName;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      if (textEl) textEl.textContent = "Downloaded!";
      setTimeout(() => { if (textEl) textEl.textContent = defaultText; }, 1800);
    }
  } catch (err) {
    if (err && err.name === "AbortError") {
      if (textEl) textEl.textContent = defaultText;
      return;
    }
    try {
      const blob = await exportShareCardBlob();
      let savedName = "";
      try { savedName = (localStorage.getItem("sonality_user_name") || "").trim(); } catch (_) {}
      const prefix = savedName ? savedName.toLowerCase().replace(/[^a-z0-9]+/g, "-") : "my-sound";
      const dominantName = (state.profile && state.profile.persona_breakdown) ? Object.keys(state.profile.persona_breakdown)[0] : "card";
      const fileName = `sonality-${prefix}-${dominantName.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = fileName;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    } catch (_) {}
    if (textEl) textEl.textContent = defaultText;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function shareNativeOrCopy() {
  const btn = document.getElementById("sc-share");
  const textEl = document.getElementById("sc-share-text");
  const oldText = textEl ? textEl.textContent : "Share";
  if (textEl) textEl.textContent = "Preparing…";
  if (btn) btn.disabled = true;

  try {
    const blob = await exportShareCardBlob();
    const fileName = "sonality-card.png";
    const file = new File([blob], fileName, { type: "image/png" });

    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({
        title: "My Sonality Music Taste Profile",
        text: `My sound is mostly ${nameOf(currentEntries()[0]?.persona || "Sonality")}! Check out your sound persona on Sonality:`,
        files: [file],
      });
      if (textEl) textEl.textContent = "Shared!";
      setTimeout(() => { if (textEl) textEl.textContent = oldText; }, 1500);
    } else if (navigator.share) {
      await navigator.share({
        title: "My Sonality Music Taste Profile",
        text: `My sound is mostly ${nameOf(currentEntries()[0]?.persona || "Sonality")}!`,
        url: window.location.href,
      });
      if (textEl) textEl.textContent = "Shared!";
      setTimeout(() => { if (textEl) textEl.textContent = oldText; }, 1500);
    } else {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = fileName;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      copyShareLink();
      if (textEl) textEl.textContent = "Downloaded!";
      setTimeout(() => { if (textEl) textEl.textContent = oldText; }, 1800);
    }
  } catch (err) {
    if (err && err.name !== "AbortError") {
      copyShareLink();
    }
    if (textEl) textEl.textContent = oldText;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function copyShareLink() {
  const textEl = document.getElementById("sc-copy-text");
  const oldText = textEl ? textEl.textContent : "Copy link";
  let ok = true;
  try {
    await navigator.clipboard.writeText(window.location.href);
  } catch (_) {
    try {
      const t = document.createElement("textarea");
      t.value = window.location.href;
      document.body.appendChild(t);
      t.select();
      document.execCommand("copy");
      t.remove();
    } catch (_) {
      ok = false;
    }
  }
  if (textEl) textEl.textContent = ok ? "Copied!" : "Failed";
  announce(ok ? "Link copied to clipboard" : "Could not copy link");
  setTimeout(() => { if (textEl) textEl.textContent = oldText; }, 1800);
}

/* ---------- events ---------- */
function wireEvents() {
  const focusFrom = (e) => { const node = e.target.closest("[data-persona]"); setFocusPersona(node ? node.dataset.persona : null); };
  el.hero.addEventListener("pointerover", focusFrom);
  el.hero.addEventListener("pointerleave", () => setFocusPersona(null));
  el.hero.addEventListener("focusin", focusFrom);
  el.hero.addEventListener("focusout", (e) => { if (!el.hero.contains(e.relatedTarget)) setFocusPersona(null); });
  el.mapLabels.addEventListener("pointerover", focusFrom);
  el.mapLabels.addEventListener("pointerleave", () => setFocusPersona(null));

  el.mapCanvas.addEventListener("pointermove", onMapPointerMove);
  el.mapCanvas.addEventListener("pointerup", onMapTap);
  el.mapCanvas.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse" && state.hoverSource === "map") setHoverTrack(null); });
  document.addEventListener("pointerdown", (e) => {
    if (e.pointerType !== "mouse" && state.hoverSource === "map" && !e.target.closest("#plot")) setHoverTrack(null);
  });

  const listLink = (e) => {
    const row = e.target.closest(".song");
    if (state.status !== "results") return;
    const id = row && state.profile.user_songs.some((s) => s.track_id === row.dataset.id) ? row.dataset.id : null;
    setHoverTrack(id, id ? "list" : null);
  };
  el.songList.addEventListener("pointerover", listLink);
  el.songList.addEventListener("focusin", listLink);
  el.songList.addEventListener("pointerleave", () => { if (state.hoverSource === "list") setHoverTrack(null); });
  el.songList.addEventListener("focusout", (e) => { if (!el.songList.contains(e.relatedTarget) && state.hoverSource === "list") setHoverTrack(null); });

  // Search input typing & debounce
  el.searchInput.addEventListener("input", (e) => {
    const q = e.target.value;
    state.search.query = q;
    clearTimeout(searchTimer);
    if (!q.trim()) {
      closeSearch();
    } else {
      searchTimer = setTimeout(() => runSearch(q), 220);
    }
  });

  // Search input keyboard navigation
  el.searchInput.addEventListener("keydown", (e) => {
    const { results, active } = state.search;
    if (el.searchPop.hidden || !results.length) {
      if (e.key === "ArrowDown" && results.length) {
        el.searchPop.hidden = false;
        e.preventDefault();
      }
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      state.search.active = (active + 1) % results.length;
      renderSearchPop();
      const sel = el.searchOptions.querySelector(`[data-idx="${state.search.active}"]`);
      if (sel) sel.scrollIntoView({ block: "nearest" });
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      state.search.active = (active - 1 + results.length) % results.length;
      renderSearchPop();
      const sel = el.searchOptions.querySelector(`[data-idx="${state.search.active}"]`);
      if (sel) sel.scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0 && results[active]) {
        addTrack(results[active]);
      }
    } else if (e.key === "Escape") {
      closeSearch();
    }
  });

  // Close search popup when clicking outside
  document.addEventListener("pointerdown", (e) => {
    if (!e.target.closest("#search")) closeSearch();
  });

  // Click on search option
  el.searchOptions.addEventListener("click", (e) => {
    const opt = e.target.closest(".option");
    if (!opt || opt.getAttribute("aria-disabled") === "true") return;
    const idx = Number(opt.dataset.idx);
    const track = state.search.results[idx];
    if (track) addTrack(track);
  });

  // Click remove button on song list
  el.songList.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-remove]");
    if (btn) removeTrack(btn.dataset.remove);
  });

  // Clear button
  el.clearBtn.addEventListener("click", clearTracks);

  // Analyze button
  el.analyzeBtn.addEventListener("click", analyzeTaste);

  // Delegated clicks for embedded recommendation players and song list previews
  document.addEventListener("click", (e) => {
    // 1. Play/pause button in recommendation card
    const recPlayBtn = e.target.closest(".rec-play-btn");
    if (recPlayBtn) {
      const card = recPlayBtn.closest(".rec-card");
      if (card) {
        playPreview(card.dataset.trackId, card.dataset.name, card.dataset.artist);
      }
      return;
    }

    // 2. Click on recommendation progress bar to seek
    const recBar = e.target.closest(".rec-progress-bar");
    if (recBar) {
      const card = recBar.closest(".rec-card");
      if (!card) return;
      const trackId = card.dataset.trackId;
      if (player.track && player.track.track_id === trackId) {
        const rect = recBar.getBoundingClientRect();
        const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        player.audio.currentTime = pos * player.durationLimit;
        updatePlayerProgress();
      }
      return;
    }

    // 3. Song list row preview button
    const previewBtn = e.target.closest("[data-play-preview]");
    if (previewBtn) {
      playPreview(previewBtn.dataset.playPreview, previewBtn.dataset.name, previewBtn.dataset.artist);
      return;
    }

    // 4. Share Card Modal Triggers & Close Actions
    if (e.target.closest("#btn-hero-share")) {
      openShareModal();
      return;
    }
    if (e.target.closest("#sc-modal-close") || e.target.closest("#sc-modal-backdrop")) {
      closeShareModal();
      return;
    }
  });

  const dlBtn = document.getElementById("sc-dl");
  if (dlBtn) dlBtn.addEventListener("click", downloadShareCardPng);
  const nameInput = document.getElementById("sc-user-name");
  const clearNameBtn = document.getElementById("sc-name-clear");
  if (nameInput) {
    nameInput.addEventListener("input", (e) => {
      const val = e.target.value;
      try {
        if (val.trim()) localStorage.setItem("sonality_user_name", val.trim());
        else localStorage.removeItem("sonality_user_name");
      } catch (_) {}
      if (clearNameBtn) clearNameBtn.hidden = !val;
      const eyebrow = document.getElementById("sc-eyebrow");
      if (eyebrow) eyebrow.textContent = formatLeadIn(val);
    });
  }
  if (clearNameBtn) {
    clearNameBtn.addEventListener("click", () => {
      if (nameInput) {
        nameInput.value = "";
        nameInput.focus();
      }
      try { localStorage.removeItem("sonality_user_name"); } catch (_) {}
      clearNameBtn.hidden = true;
      const eyebrow = document.getElementById("sc-eyebrow");
      if (eyebrow) eyebrow.textContent = formatLeadIn("");
    });
  }


  const shareBtn = document.getElementById("sc-share");
  if (shareBtn) shareBtn.addEventListener("click", shareNativeOrCopy);

  const copyBtn = document.getElementById("sc-copy");
  if (copyBtn) copyBtn.addEventListener("click", copyShareLink);

  // Preload audio on mouse hover or touch before click completes
  document.addEventListener("pointerover", (e) => {
    const card = e.target.closest(".rec-card");
    if (card) {
      const id = card.dataset.trackId;
      if (id && !previewCache.has(id)) {
        preloadRecPreviews([{ track_id: id, track_name: card.dataset.name, artists: card.dataset.artist }]);
      }
    }
  }, { passive: true });

  document.addEventListener("pointerover", (e) => { const t = e.target.closest("[data-tip]"); if (t) onTipEnter(t, false); });
  document.addEventListener("pointerout", (e) => {
    const t = e.target.closest("[data-tip]");
    if (t && t === tipOwner && !t.contains(e.relatedTarget)) hideTip();
    else if (t) clearTimeout(tipTimer);
  });
  document.addEventListener("focusin", (e) => { const t = e.target.closest("[data-tip]"); if (t && t.matches(":focus-visible")) onTipEnter(t, true); });
  document.addEventListener("focusout", (e) => { if (e.target === tipOwner) hideTip(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      hideTip();
      const modal = document.getElementById("share-modal");
      if (modal && (modal.open || modal.hasAttribute("open"))) {
        closeShareModal();
      }
    }
  });
  window.addEventListener("scroll", hideTip, { passive: true });

  new ResizeObserver(() => { resizeMap(); drawBackground(); drawMap(); renderMapLabels(); }).observe(el.plot);
  let fitFrame = 0;
  window.addEventListener("resize", () => { cancelAnimationFrame(fitFrame); fitFrame = requestAnimationFrame(fitHeadline); });
}

function renderAll() {
  const results = state.status === "results";
  document.querySelector(".card-compare").hidden = !results;
  document.querySelector(".card-recs").hidden = !results;
  renderHeader();
  renderSongs();
  renderHero();
  if (results) { renderCompare(); renderRecs(); }
  refreshMap();
}

async function getJSON(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function loadSoundMap() {
  state.mapStatus = "loading";
  renderMapStatus();
  try {
    const data = await getJSON(`/api/sound-map?n=${SOUND_MAP_SAMPLE}`);
    state.map = { points: data.points || [], explained_variance: data.explained_variance || [] };
    state.mapStatus = "ready";
    prepareMap();
  } catch (_) {
    state.mapStatus = "error";
  }
  renderHow();
  refreshMap();
}

async function loadHealth() {
  try {
    const h = await getJSON("/api/health");
    if (h && h.loaded_tracks > 0) {
      state.datasetSize = h.loaded_tracks;
      const cmpHint = document.querySelector(".cmp-header-hint");
      if (cmpHint) {
        cmpHint.textContent = `Your listening profile compared to ${fmtInt(state.datasetSize)} tracks in the dataset.`;
      }
      if (state.status !== "results") renderHero();
      renderHow();
    }
  } catch (_) {}
}

function init() {
  loadPersonaColors();
  POINT_RING = cssVar("--text-1") || "#F4EDE8";
  el.xLow.textContent = `← ${AXIS_LABELS.x.low}`;
  el.xHigh.textContent = `${AXIS_LABELS.x.high} →`;
  el.yHigh.textContent = `↑ ${AXIS_LABELS.y.high}`;
  el.yLow.textContent = `↓ ${AXIS_LABELS.y.low}`;
  wireEvents();
  renderHow();
  renderAll();
  loadHealth();
  loadSoundMap();
  document.fonts && document.fonts.ready.then(() => { renderMapLabels(); fitHeadline(); });
}

init();
