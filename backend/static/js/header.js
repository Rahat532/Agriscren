// /static/js/header.js
(function (global) {
  function mount() {
    const el = document.querySelector("#site-header");
    if (!el) return; // quietly do nothing if target isn't present

    const title = "AgriScan";
    const logo  = "/static/img/logo.png";

    el.innerHTML = `
      <header class="bg-green-700 text-white py-4 px-6 md:flex md:items-center md:justify-between">
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <a href="/" class="flex items-center space-x-2">
                <img
                  src="${logo}"
                  alt="Logo"
                  class="w-10 h-10 rounded-full bg-white p-1"
                />
                <span class="text-2xl font-bold">${title}</span>
              </a>
          </div>
          <button class="md:hidden text-white text-2xl" id="nav-toggle" aria-label="Toggle Navigation">
            <i class="fas fa-bars"></i>
          </button>
        </div>
        <nav id="mobile-nav" class="hidden md:flex space-x-6 mt-4 md:mt-0">
          <a href="#tomato" class="hover:text-green-300" data-nav="tomato">🍅 Tomato</a>
          <a href="#onion"  class="hover:text-green-300" data-nav="onion">🧅 Onion</a>
          <a href="#maize"  class="hover:text-green-300" data-nav="maize">🌽 Maize</a>
        </nav>
      </header>
    `;

    // Mobile toggle
    const toggleBtn = el.querySelector("#nav-toggle");
    const nav = el.querySelector("#mobile-nav");
    if (toggleBtn && nav) {
      toggleBtn.addEventListener("click", () => {
        nav.classList.toggle("hidden");
      });
    }

    // Auto-highlight active link from URL hash
    const active = (location.hash || "").replace("#", "");
    if (active) {
      const a = el.querySelector(`[data-nav="${active}"]`);
      if (a) a.classList.add("text-green-300", "font-semibold");
    }
  }

  // Expose (no params) & auto-mount
  global.AgriHeader = { mount };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mount);
  } else {
    mount();
  }
})(window);
