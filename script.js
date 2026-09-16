'use strict';

// Values transcribed from the supplied manuscript, Table 1 and Table 4.
const benchmarks = {
  "clean": {
    "title": "RoboTwin 2.0 · Clean scenes",
    "description": "RDT-Bridge3D achieves the highest average success rate of 84.8% across 10 selected tasks, surpassing 3D Diffusion Policy (82.8%), Spatial Forcing (77.1%), and PointVLA (76.0%).",
    "rows": [
      [
        "GR00T-N1.5",
        47.5
      ],
      [
        "GR00T-N1.5-Bridge3D",
        54.4
      ],
      [
        "RDT",
        68.7
      ],
      [
        "π₀",
        71.7
      ],
      [
        "Spatial Forcing",
        77.1
      ],
      [
        "RDT-Bridge3D",
        84.8
      ]
    ],
    "note": "RoboTwin 2.0 results."
  },
  "randomized": {
    "title": "RoboTwin 2.0 · Randomized scenes",
    "description": "Bridge3D improves the average success rates of RDT and GR00T-N1.5 by 28.7% and 54.7%, respectively. These relative gains substantially exceed the 9.5% improvement of Spatial Forcing.",
    "rows": [
      [
        "GR00T-N1.5",
        13.7
      ],
      [
        "GR00T-N1.5-Bridge3D",
        21.2
      ],
      [
        "RDT",
        26.1
      ],
      [
        "π₀",
        32.7
      ],
      [
        "Spatial Forcing",
        35.8
      ],
      [
        "RDT-Bridge3D",
        33.6
      ]
    ],
    "note": "RoboTwin 2.0 results."
  },
  "real": {
    "title": "Real world experiments",
    "description": "Success rate (%) across six real-world XMAN-R1 tasks with 100 training demonstrations per task. Bridge3D reaches 49.0%, exceeding Spatial Forcing by 11.7 percentage points.",
    "rows": [
      [
        "GR00T-N1.5",
        26.3
      ],
      [
        "RDT",
        34.3
      ],
      [
        "Spatial Forcing",
        37.3
      ],
      [
        "RDT-Bridge3D",
        49.0
      ]
    ],
    "note": "Real world experiments."
  }
};

const tabs = [...document.querySelectorAll('[data-setting]')];
function setBenchmark(setting) {
  const data = benchmarks[setting];
  if (!data) return;
  tabs.forEach(tab => { const selected = tab.dataset.setting === setting; tab.setAttribute('aria-selected', String(selected)); tab.tabIndex = selected ? 0 : -1; });
  document.getElementById('results-panel').setAttribute('aria-labelledby', `tab-${setting}`);
  document.getElementById('chart-title').textContent = data.title;
  document.getElementById('chart-description').textContent = data.description;
  document.getElementById('chart-note').textContent = data.note;
  const chart = document.getElementById('bar-chart');
  const fragment = document.createDocumentFragment();
  data.rows.forEach(([name, value]) => {
    const row = document.createElement('div'); row.className = 'bar-row' + (name.endsWith('-Bridge3D') ? ' ours' : '');
    const label = document.createElement('span'); label.className = 'bar-label'; label.textContent = name;
    const track = document.createElement('div'); track.className = 'bar-track'; track.setAttribute('aria-hidden','true');
    const fill = document.createElement('span'); fill.className = 'bar-fill'; fill.style.setProperty('--value', `${value}%`); const benchmarkColors = {"GR00T-N1.5": "#d0dbe2", "GR00T-N1.5-Bridge3D": "#48a79c", "RDT": "#9eafbd", "π₀": "#b7c6d2", "Spatial Forcing": "#527c9b", "RDT-Bridge3D": "#087b76"}; fill.style.backgroundColor = benchmarkColors[name]; track.append(fill);
    const number = document.createElement('span'); number.className = 'bar-value'; number.textContent = value.toFixed(1);
    row.append(label,track,number); fragment.append(row);
  });
  const axis = document.createElement('div'); axis.className = 'chart-axis'; axis.setAttribute('aria-hidden','true');
  ['0','25','50','75','100%'].forEach(value => { const label = document.createElement('span'); label.textContent = value; axis.append(label); });
  fragment.append(axis); chart.replaceChildren(fragment);
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => setBenchmark(tab.dataset.setting));
  tab.addEventListener('keydown', event => {
    let target;
    if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') target = 0;
    else if (event.key === 'End') target = tabs.length - 1;
    else return;
    event.preventDefault(); tabs[target].focus(); setBenchmark(tabs[target].dataset.setting);
  });
});

const video = document.getElementById('demo-video');
// Local file previews use the same chapter buttons without a cross-origin VTT request.
if (window.location.protocol === 'file:') video.querySelector('track')?.remove();
const chapters = [...document.querySelectorAll('.chapter')];
const videoStatus = document.getElementById('video-status');
let pendingSeek = null;
function updateChapter(time) {
  let active = 0;
  chapters.forEach((button, i) => { if (time >= Number(button.dataset.time)) active = i; });
  chapters.forEach((button, i) => { button.classList.toggle('active', i === active); button.setAttribute('aria-pressed', String(i === active)); });
}
async function seekAndPlay(time) {
  if (video.readyState < 1) { pendingSeek = time; videoStatus.textContent = 'Loading demonstration…'; video.load(); return; }
  pendingSeek = null;
  video.currentTime = Math.min(time, Number.isFinite(video.duration) ? Math.max(0,video.duration - .1) : time);
  updateChapter(time);
  try { await video.play(); videoStatus.textContent = ''; }
  catch { videoStatus.textContent = 'Use the play button on the video to start this task.'; }
}
chapters.forEach(button => button.addEventListener('click', () => {
  seekAndPlay(Number(button.dataset.time));
  if (window.matchMedia('(max-width: 850px)').matches) {
    video.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  }
}));
video.addEventListener('loadedmetadata', () => { if (pendingSeek !== null) seekAndPlay(pendingSeek); });
video.addEventListener('timeupdate', () => updateChapter(video.currentTime));
video.addEventListener('error', () => { videoStatus.textContent = 'The video could not load. Download the demonstration from the research links below.'; });

const dialog = document.getElementById('image-dialog');
const dialogImage = document.getElementById('dialog-image');
const dialogCaption = document.getElementById('dialog-caption');
document.querySelectorAll('[data-lightbox]').forEach(link => {
  link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || typeof dialog.showModal !== 'function') return;
    event.preventDefault();
    const image = link.querySelector('img');
    dialogImage.src = link.href;
    dialogImage.alt = image?.alt || 'Research figure';
    dialogCaption.textContent = 'Bridge3D · Research figure';
    dialog.showModal(); dialog.scrollTop = 0; dialog.scrollLeft = 0;
    document.body.classList.add('dialog-open');
  });
});
document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));
dialog.addEventListener('click', event => { if (event.target !== dialog) return; const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); });

const navLinks = [...document.querySelectorAll('.nav-links a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      navLinks.forEach(link => { const active = link.hash === `#${entry.target.id}`; link.classList.toggle('active',active); if (active) link.setAttribute('aria-current','location'); else link.removeAttribute('aria-current'); });
    });
  }, { rootMargin: '-15% 0px -55% 0px', threshold: 0 });
  navLinks.forEach(link => { const section = document.querySelector(link.hash); if (section) observer.observe(section); });
}
