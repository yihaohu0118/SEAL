function copyBibTeX() {
  const bibtexElement = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const copyText = button ? button.querySelector(".copy-text") : null;

  if (!bibtexElement || !button || !copyText) return;

  const markCopied = () => {
    button.classList.add("copied");
    copyText.textContent = "Copied";
    setTimeout(() => {
      button.classList.remove("copied");
      copyText.textContent = "Copy";
    }, 1800);
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(bibtexElement.textContent).then(markCopied).catch(() => {
      fallbackCopy(bibtexElement.textContent);
      markCopied();
    });
    return;
  }

  fallbackCopy(bibtexElement.textContent);
  markCopied();
}

function fallbackCopy(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

window.addEventListener("scroll", () => {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (!scrollButton) return;

  if (window.pageYOffset > 300) {
    scrollButton.classList.add("visible");
  } else {
    scrollButton.classList.remove("visible");
  }
});

function initScrollSpy() {
  const navLinks = Array.from(document.querySelectorAll(".topbar a[href^='#']"));
  if (!navLinks.length) return;

  const linkById = new Map();
  navLinks.forEach((link) => {
    const id = link.getAttribute("href").slice(1);
    if (id) linkById.set(id, link);
  });

  function setActive(id) {
    navLinks.forEach((link) => {
      link.classList.toggle("active", linkById.get(id) === link);
    });
  }

  setActive("abstract");

  const sections = Array.from(linkById.keys())
    .map((id) => document.getElementById(id))
    .filter(Boolean);

  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

    if (visible) setActive(visible.target.id);
  }, {
    rootMargin: "-28% 0px -55% 0px",
    threshold: [0.08, 0.2, 0.45, 0.7]
  });

  sections.forEach((section) => observer.observe(section));

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const id = link.getAttribute("href").slice(1);
      if (id) setActive(id);
    });
  });
}

function initHeroParticleField() {
  const canvas = document.getElementById("heroParticleField");
  const hero = document.querySelector(".publication-hero");
  if (!canvas || !hero) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ctx = canvas.getContext("2d", { alpha: true });
  if (!ctx) return;

  const colors = [
    [8, 127, 140],
    [178, 58, 47],
    [47, 125, 87],
    [138, 109, 29]
  ];
  const pointer = { x: 0, y: 0, active: false };
  const particles = [];
  let width = 0;
  let height = 0;
  let dpr = 1;
  let running = true;
  let frameId = 0;

  function resize() {
    const rect = hero.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const targetCount = reducedMotion ? 34 : Math.min(120, Math.max(64, Math.floor(width * height / 12500)));
    particles.length = 0;
    for (let i = 0; i < targetCount; i += 1) {
      const color = colors[i % colors.length];
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        px: Math.random() * width,
        py: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.28,
        vy: (Math.random() - 0.5) * 0.28,
        phase: Math.random() * Math.PI * 2,
        speed: 0.45 + Math.random() * 0.85,
        size: 0.8 + Math.random() * 1.9,
        color
      });
    }
  }

  function setPointer(event) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = event.clientX - rect.left;
    pointer.y = event.clientY - rect.top;
    pointer.active = true;
  }

  function clearPointer() {
    pointer.active = false;
  }

  function draw() {
    if (!running) return;

    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = reducedMotion ? "rgba(255,255,255,0.22)" : "rgba(255,255,255,0.105)";
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = "multiply";

    const time = performance.now() * 0.001;
    const centerPullX = width * 0.62;
    const centerPullY = height * 0.35;

    for (const p of particles) {
      p.px = p.x;
      p.py = p.y;

      const curl = Math.sin(time * p.speed + p.phase + p.y * 0.006) * 0.018;
      const drift = Math.cos(time * 0.65 + p.phase + p.x * 0.004) * 0.014;
      p.vx += curl + (centerPullX - p.x) * 0.000003;
      p.vy += drift + (centerPullY - p.y) * 0.000002;

      if (pointer.active && !reducedMotion) {
        const dx = p.x - pointer.x;
        const dy = p.y - pointer.y;
        const distSq = dx * dx + dy * dy;
        const radius = 210;
        if (distSq < radius * radius && distSq > 0.01) {
          const dist = Math.sqrt(distSq);
          const force = (1 - dist / radius) * 0.085;
          const tangent = Math.atan2(dy, dx) + Math.PI / 2;
          p.vx += Math.cos(tangent) * force + (dx / dist) * force * 0.38;
          p.vy += Math.sin(tangent) * force + (dy / dist) * force * 0.38;
        }
      }

      p.vx *= 0.968;
      p.vy *= 0.968;
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < -20) p.x = width + 20;
      if (p.x > width + 20) p.x = -20;
      if (p.y < -20) p.y = height + 20;
      if (p.y > height + 20) p.y = -20;

      const [r, g, b] = p.color;
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.34)`;
      ctx.lineWidth = p.size;
      ctx.beginPath();
      ctx.moveTo(p.px, p.py);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.42)`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * 0.78, 0, Math.PI * 2);
      ctx.fill();
    }

    for (let i = 0; i < particles.length; i += 1) {
      const a = particles[i];
      for (let j = i + 1; j < particles.length; j += 1) {
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const distSq = dx * dx + dy * dy;
        if (distSq > 7200) continue;
        const alpha = (1 - distSq / 7200) * 0.095;
        ctx.strokeStyle = `rgba(8, 127, 140, ${alpha})`;
        ctx.lineWidth = 0.75;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    frameId = window.requestAnimationFrame(draw);
  }

  const observer = new IntersectionObserver((entries) => {
    running = entries[0].isIntersecting;
    if (running && !frameId) frameId = window.requestAnimationFrame(draw);
    if (!running && frameId) {
      window.cancelAnimationFrame(frameId);
      frameId = 0;
    }
  }, { threshold: 0.05 });

  resize();
  ctx.fillStyle = "rgba(255,255,255,1)";
  ctx.fillRect(0, 0, width, height);
  draw();
  observer.observe(hero);

  window.addEventListener("resize", resize);
  hero.addEventListener("pointermove", setPointer);
  hero.addEventListener("pointerleave", clearPointer);
}

window.addEventListener("DOMContentLoaded", initHeroParticleField);
window.addEventListener("DOMContentLoaded", initScrollSpy);
