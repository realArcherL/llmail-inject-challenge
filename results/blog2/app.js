/* The marker that ate the names — page interactions. No dependencies; every number comes from window.DATA. */
(function () {
  "use strict";
  const D = window.DATA;
  const $ = (s, el) => (el || document).querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d) => (v * 100).toFixed(d === undefined ? 1 : d) + "%";
  const byCond = Object.fromEntries(D.defenses.map((r) => [r.condition, r]));

  const tip = $("#tip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.style.display = "block";
    tip.style.left = Math.max(8, Math.min(window.innerWidth - tip.offsetWidth - 8, x + 14)) + window.scrollX + "px";
    tip.style.top = (y + 16 + window.scrollY) + "px";
  }
  const hideTip = () => { tip.style.display = "none"; };

  // ---- 1. attack explorer
  const COND = [["ms_spotlight", "Microsoft's defense"], ["lib_markdata", "My library, shipped"], ["v2_uni_spaces", "My library, fixed"]];
  const dots = (outs, lg) => `<span class="runs">${outs.map((o) => `<span class="dot${o ? " on" : ""}${lg ? " lg" : ""}"></span>`).join("")}</span>`;
  const dotsN = (f, n, lg) => dots(Array.from({ length: n }, (_, i) => i < f), lg);
  const ex = { q: $("#q"), sort: $("#sort"), only: $("#onlyfired"), count: $("#count"), body: $("#tb tbody") };
  const byId = Object.fromEntries(D.attacks.map((a) => [a.id, a]));

  function render() {
    const q = ex.q.value.trim().toLowerCase();
    let rows = D.attacks.filter((a) => (!q || a.email.toLowerCase().includes(q) || a.id.includes(q)) && (!ex.only.checked || a.fired > 0));
    rows.sort((a, b) => ex.sort.value === "fired" ? (b.fired - a.fired) || a.id.localeCompare(b.id) : a.id.localeCompare(b.id));
    ex.count.textContent = `${rows.length} of ${D.attacks.length}`;
    ex.body.innerHTML = rows.map((a) => `<tr class="row" data-id="${esc(a.id)}" tabindex="0">
      <td class="n">${esc(a.id.replace("undefended-", ""))}</td>
      <td class="prev" title="${esc(a.email.slice(0, 300))}">${esc(a.email.replace(/^email:\s*/i, ""))}</td>
      <td class="n">${dots(a.outcomes)} <span style="color:var(--muted)">${a.fired}/8</span></td>
      ${COND.map(([c]) => {
        const d = a.defended[c];
        return `<td class="n">${d ? dotsN(d.fired, d.runs) + ` <span style="color:var(--muted)">${d.fired}/${d.runs}</span>` : "—"}</td>`;
      }).join("")}</tr>`).join("");
  }
  function toggle(tr) {
    const nx = tr.nextElementSibling;
    if (nx && nx.classList.contains("det")) { nx.remove(); return; }
    ex.body.querySelectorAll("tr.det").forEach((d) => d.remove());
    const a = byId[tr.dataset.id];
    const d = document.createElement("tr"); d.className = "det";
    d.innerHTML = `<td colspan="6">
      <div style="font-family:Archivo,sans-serif;font-size:13px;font-weight:600;margin-bottom:6px">The attacker's email</div>
      <pre class="ans" style="background:var(--bad-bg);color:var(--ink)">${esc(a.email)}</pre>
      <div style="font-family:Archivo,sans-serif;font-size:13px;font-weight:600;margin:12px 0 6px">${a.evidence ? "The tool call Phi-3 wrote" : "Never called the tool in 8 runs"}</div>
      ${a.evidence ? `<pre class="ans" style="color:var(--bad)">${esc(a.evidence.call)}</pre>` : ""}
      <div style="font-family:Archivo,sans-serif;font-size:13px;color:var(--ink2);margin-top:10px">Prompt length undefended: ${a.tokens} tokens.</div></td>`;
    tr.after(d);
  }
  ex.body.addEventListener("click", (e) => { const tr = e.target.closest("tr.row"); if (tr) toggle(tr); });
  ex.body.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { const tr = e.target.closest("tr.row"); if (tr) { e.preventDefault(); toggle(tr); } }
  });
  [ex.q, ex.sort, ex.only].forEach((el) => el.addEventListener("input", render));
  render();

  // ---- 2. attack-success chart
  (function () {
    const order = ["none", "lib_sanitize", "v2_uni_words", "v2_uni_random_phi3", "v2_uni_spaces", "v2_alnum_random_phi3", "lib_markdata", "lib_randommark", "ms_spotlight", "lib_base64"];
    const rows = order.map((c) => byCond[c]).filter(Boolean);
    const W = 960, rowH = 34, T = 12, B = 30, H = T + rows.length * rowH + B, L = 250, R = 60;
    const xs = (v) => L + (W - L - R) * v / 0.26;
    let s = `<svg class="ch" viewBox="0 0 ${W} ${H}" role="img" aria-label="Attack success by defense">`;
    [0, 0.05, 0.1, 0.15, 0.2, 0.25].forEach((v) => {
      s += `<line class="gr" x1="${xs(v)}" x2="${xs(v)}" y1="${T}" y2="${H - B + 4}"/><text x="${xs(v)}" y="${H - B + 20}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    });
    rows.forEach((r, i) => {
      const y = T + i * rowH + 7, h = 17;
      const col = r.condition === "none" || r.condition === "lib_sanitize" ? "var(--muted)" : "var(--bad)";
      s += `<text x="0" y="${y + 13}" style="fill:var(--ink)">${esc(r.defense)}</text>`;
      s += `<rect x="${L}" y="${y}" width="${Math.max(1, xs(r.attack) - L)}" height="${h}" rx="2" fill="${col}"/>`;
      s += `<line x1="${xs(r.attack_lo)}" x2="${xs(r.attack_hi)}" y1="${y + h / 2}" y2="${y + h / 2}" stroke="var(--ink)" stroke-width="1.5"/>`;
      s += `<text x="${xs(r.attack_hi) + 8}" y="${y + 13}" style="fill:var(--ink)">${pct(r.attack)}</text>`;
    });
    s += "</svg>";
    $("#sec-chart").innerHTML = s;
  })();

  // ---- 3. the three-way summary comparison
  (function () {
    const p = D.pair;
    $("#pair-id").textContent = p.base_id;
    const cards = [
      ["No defense", "none", "ok", "gets the name right"],
      ["My library, as shipped", "markdata", "no", "invents “Urbang Rentals”"],
      ["My library, fixed", "fixed", "ok", "gets the name right"],
    ];
    const hl = (t) => esc(t).replace(/(Urban Renewal Properties|Urbang Rentals|Urbany|Urbana|Renewals)/g,
      (m) => /Urban Renewal Properties/.test(m) ? `<span style="color:var(--good);font-weight:600">${m}</span>` : `<span class="w">${m}</span>`);
    $("#pair").innerHTML = cards.map(([name, key, cls, note]) =>
      `<div><div class="hd">${esc(name)} <span class="pill ${cls}">${esc(note)}</span></div><pre class="ans">${hl(p[key] || "(not recorded)")}</pre></div>`).join("");
  })();

  // ---- 4. marker inspector
  (function () {
    const MODES = [
      ["none", "No defense", null],
      ["ms_spotlight", "Microsoft spotlighting", "a fixed 8-character marker, digit-led"],
      ["lib_markdata", "My library: between words", "random letters — this is the bug"],
      ["lib_randommark", "My library: random points", "random letters, placed anywhere"],
      ["lib_base64", "My library: base64", "the whole block re-encoded"],
      ["v2_uni_spaces", "Fixed: Unicode, between words", "two private-use characters"],
      ["v2_uni_random_phi3", "Fixed: Unicode, Phi-3 boundaries", "two private-use characters, tokenizer-aware"],
      ["v2_alnum_random_phi3", "Diagnostic: letters, Phi-3 boundaries", "letters again — the bug returns"],
      ["v2_uni_words", "Fixed: Unicode, word gaps only", "two private-use characters, whitespace only"],
    ];
    const tabs = $("#mk-tabs"), out = $("#mk-text"), note = $("#mk-note");
    const PUA = /[-]+/g;
    function show(key) {
      tabs.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", t.dataset.k === key));
      const m = D.marked[key];
      if (!m) { out.textContent = "(not recorded)"; note.textContent = ""; return; }
      let html = esc(m.excerpt);
      if (m.marker && !/[-]/.test(m.marker)) {
        const re = new RegExp(m.marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g");
        html = html.replace(re, `<span class="m">${esc(m.marker)}</span>`);
      } else {
        html = html.replace(PUA, (s) => `<span class="m">${"◆".repeat(s.length)}</span>`);
      }
      html = html.replace(/(Urban|Renewal|Ren|ewal)/g, (w) => `<span class="w">${w}</span>`);
      out.innerHTML = html;
      const meta = MODES.find((x) => x[0] === key);
      note.innerHTML = meta[2] ? `<b>${esc(meta[1])}</b> — ${esc(meta[2])}.` +
        (byCond[key] ? ` Attack success ${pct(byCond[key].attack)}, garbled names ${pct(byCond[key].garbled)}.` : "") : "The email as written.";
    }
    MODES.forEach(([k, label]) => {
      if (!D.marked[k]) return;
      const b = document.createElement("button");
      b.className = "tab" + (k === "lib_markdata" || k === "v2_alnum_random_phi3" ? " danger" : (k.startsWith("v2_uni") ? " ok" : ""));
      b.dataset.k = k; b.textContent = label;
      b.addEventListener("click", () => show(k));
      tabs.appendChild(b);
    });
    show("lib_markdata");
  })();

  // ---- 5. the 2x2
  (function () {
    const cells = [
      ["lib_markdata", "v2_uni_spaces", "every space"],
      ["v2_alnum_random_phi3", "v2_uni_random_phi3", "Phi-3 token boundaries"],
    ];
    let h = `<div class="grid2"><div class="hd"></div><div class="hd">Marker of random letters<br><span style="font-weight:400;color:var(--muted)">7–12 characters</span></div><div class="hd">Marker of private-use Unicode<br><span style="font-weight:400;color:var(--muted)">1–2 characters</span></div>`;
    cells.forEach(([alnum, uni, place]) => {
      h += `<div class="hd">Placed at<br><span style="font-weight:400;color:var(--muted)">${esc(place)}</span></div>`;
      [alnum, uni].forEach((c, i) => {
        const r = byCond[c];
        h += `<div class="cell ${i === 0 ? "bad" : "good"}"><div class="n">${pct(r.garbled)}</div><div class="lab">summaries with an invented or garbled name</div>
          <div class="sec">attack success ${pct(r.attack)} · key terms ${Math.round(r.keyterms_rel * 100)}%</div></div>`;
      });
    });
    h += "</div>";
    $("#grid").innerHTML = h;
  })();

  // ---- 6. full table
  (function () {
    const order = ["lib_base64", "ms_spotlight", "lib_randommark", "lib_markdata", "v2_alnum_random_phi3", "v2_uni_spaces", "v2_uni_random_phi3", "v2_uni_words", "lib_sanitize", "none"];
    const maxTok = Math.max(...D.defenses.map((r) => r.tokens || 0));
    $("#full tbody").innerHTML = order.map((c) => {
      const r = byCond[c]; if (!r) return "";
      const hl = (c === "lib_markdata" || c === "v2_uni_spaces") ? " hl" : "";
      const sep = (c === "lib_sanitize") ? " sep" : "";
      const gcol = r.garbled > 0.1 ? "var(--bad)" : "var(--ink)";
      return `<tr class="${hl}${sep}">
        <td>${esc(r.defense)}</td>
        <td class="n"><span class="bar a" style="width:${Math.round(140 * r.attack / 0.22)}px"></span>${pct(r.attack)}</td>
        <td class="n">${Math.round(r.keyterms_rel * 100)}%</td>
        <td class="n" style="color:${gcol}"><span class="bar g" style="width:${Math.round(110 * r.garbled / 0.37)}px"></span>${pct(r.garbled)}</td>
        <td class="n" style="color:var(--ink2)">${r.tokens ? r.tokens.toLocaleString() + (r.tokens > 1200 ? ` <span style="color:var(--muted)">(${(r.tokens / 1028).toFixed(1)}×)</span>` : "") : "—"}</td></tr>`;
    }).join("");
  })();
})();
