/**
 * NLA at Home — Browser Demo
 *
 * Gallery mode: Pre-computed thought traces for Phi-4 Mini, instant display.
 * Layers animate top-to-bottom showing how the model processes the prompt.
 */

const GALLERY_URL = 'gallery.json';
const N_LAYERS = 32;
let galleryData = null;
let animationSpeed = 120;

async function init() {
  try {
    const resp = await fetch(GALLERY_URL);
    galleryData = await resp.json();
    populateGallery();
  } catch (e) {
    console.warn('Gallery not loaded:', e);
  }

  document.getElementById('speed')?.addEventListener('input', (e) => {
    animationSpeed = parseInt(e.target.value);
    document.getElementById('speed-display').textContent = animationSpeed + 'ms';
  });
}

function populateGallery() {
  const select = document.getElementById('gallery');

  const categories = {};
  for (const item of galleryData.prompts) {
    const cat = item.category || 'other';
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(item);
  }

  for (const [cat, items] of Object.entries(categories)) {
    const group = document.createElement('optgroup');
    group.label = cat;
    for (const item of items) {
      const opt = document.createElement('option');
      opt.value = item.id;
      opt.textContent = item.text.slice(0, 70) + (item.text.length > 70 ? '...' : '');
      group.appendChild(opt);
    }
    select.appendChild(group);
  }

  select.addEventListener('change', onGallerySelect);
}

function onGallerySelect(e) {
  const id = e.target.value;
  if (!id) return;
  const item = galleryData.prompts.find(p => p.id === id);
  if (item) showTrace(item.text, item.layers);
}

function getPhase(depthPct) {
  if (depthPct < 25) return 'early';
  if (depthPct < 60) return 'mid';
  if (depthPct < 88) return 'late';
  return 'final';
}

function formatDescription(desc) {
  if (!desc) return '';
  const lines = desc.split('\n').filter(l => l.trim());
  return lines.map(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith('- ')) {
      return `<span class="bullet">${trimmed}</span>`;
    }
    return `<span class="bullet">${trimmed}</span>`;
  }).join('');
}

function showTrace(promptText, layers) {
  const section = document.getElementById('brain');
  const display = document.getElementById('prompt-display');
  const container = document.getElementById('trace-layers');

  display.textContent = `"${promptText}"`;
  container.innerHTML = '';
  section.hidden = false;

  for (const entry of layers) {
    const row = document.createElement('div');
    const depthPct = entry.depth_pct !== undefined
      ? entry.depth_pct
      : Math.round(entry.layer * 100 / N_LAYERS);
    const phase = getPhase(depthPct);

    row.className = `layer-row ${phase}`;
    row.innerHTML = `
      <div class="layer-label">
        L${entry.layer}
        <span class="depth">${depthPct}%</span>
      </div>
      <div class="layer-desc">${formatDescription(entry.description)}</div>
    `;
    container.appendChild(row);
  }

  animateTrace();
}

function animateTrace() {
  const rows = document.querySelectorAll('.layer-row');
  rows.forEach(r => {
    r.classList.remove('visible');
    r.classList.remove('active');
  });

  rows.forEach((row, i) => {
    setTimeout(() => {
      row.classList.add('visible');
      row.classList.add('active');
      if (i > 0) rows[i - 1].classList.remove('active');
      if (i === rows.length - 1) {
        setTimeout(() => row.classList.remove('active'), animationSpeed * 2);
      }
    }, i * animationSpeed);
  });
}

document.getElementById('animate-btn')?.addEventListener('click', animateTrace);

init();
