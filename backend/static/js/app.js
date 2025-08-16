// /static/js/app.js — UI helpers for AgriScan
(() => {
  "use strict";

  // ---------- tiny utils ----------
  const ready = (fn) =>
    document.readyState !== "loading"
      ? fn()
      : document.addEventListener("DOMContentLoaded", fn);

  const prefersReducedMotion = () =>
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- smooth internal links ----------
  function smoothInternalLinks() {
    const isModified = (e) =>
      e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0;

    document.querySelectorAll('a[href^="/"], a[href^="#"]').forEach((link) => {
      const href = link.getAttribute("href");

      // Same-page hash links
      if (href === "#" || href.startsWith("#")) {
        link.addEventListener("click", (e) => {
          if (isModified(e)) return;
          e.preventDefault();
          const target = document.querySelector(href);
          if (!target) return;
          target.scrollIntoView({
            behavior: prefersReducedMotion() ? "auto" : "smooth",
          });
        });
        return;
      }

      // Fade transition for same-origin navigations (no new tab)
      try {
        const url = new URL(link.href);
        const sameOrigin = url.origin === window.location.origin;
        const isAnchor = url.hash && url.pathname === window.location.pathname;
        const newTab = link.target === "_blank";
        if (sameOrigin && !isAnchor && !newTab) {
          link.addEventListener("click", (e) => {
            if (isModified(e)) return;
            e.preventDefault();
            document.body.classList.add("opacity-0");
            setTimeout(() => (window.location.href = url.href), 160);
          });
        }
      } catch {
        /* ignore invalid URLs */
      }
    });
  }

  // ---------- predict forms UX ----------
  function setupForms() {
    const waitBanner = document.getElementById("wait-banner");

    document
      .querySelectorAll('form.predict-form, form[action="/predict"]')
      .forEach((form) => {
        const fileInput = form.querySelector('input[type="file"]');
        const submitBtn =
          form.querySelector('button[type="submit"]') ||
          form.querySelector("button");

        // Disable submit until a file is chosen
        if (fileInput && submitBtn) {
          submitBtn.setAttribute("disabled", "disabled");
          submitBtn.classList.add("opacity-50", "cursor-not-allowed");
          fileInput.addEventListener("change", () => {
            if (fileInput.files && fileInput.files.length > 0) {
              submitBtn.removeAttribute("disabled");
              submitBtn.classList.remove("opacity-50", "cursor-not-allowed");
            }
          });
        }

        // Show wait banner on submit and remember last crop
        form.addEventListener("submit", () => {
          if (waitBanner) {
            waitBanner.classList.remove("hidden");
            waitBanner.setAttribute("aria-busy", "true");
          }
          if (submitBtn) {
            submitBtn.setAttribute("disabled", "disabled");
            submitBtn.classList.add("opacity-50", "cursor-not-allowed");
          }
          const cropHidden = form.querySelector('input[name="crop_type"]');
          if (cropHidden)
            sessionStorage.setItem("selectedCrop", cropHidden.value || "");
        });
      });
  }

  // ---------- back-to-top (dedup + float above footer) ----------
  function ensureBackToTopButton() {
    // Prefer an existing #toTop; if both exist, remove legacy #backToTop
    const pageBtn = document.getElementById("toTop");
    const legacy = document.getElementById("backToTop");
    if (pageBtn && legacy) legacy.remove();

    let btn = pageBtn || legacy;
    if (!btn) {
      // create one if page didn't include it
      btn = document.createElement("button");
      btn.id = "toTop";
      btn.type = "button";
      btn.setAttribute("aria-label", "Back to top");
      btn.title = "Back to top";
      btn.style.display = "none"; // hidden until scrolled
      btn.className =
        "fixed right-4 bottom-4 bg-green-700 text-white p-3 rounded-full shadow-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-white";
      btn.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fill-rule="evenodd" d="M10 3a1 1 0 01.832.445l5 7a1 1 0 11-1.664 1.11L11 6.882V17a1 1 0 11-2 0V6.882L5.832 11.555a1 1 0 11-1.664-1.11l5-7A1 1 0 0110 3z" clip-rule="evenodd"/></svg>';
      document.body.appendChild(btn);
    }
    return btn;
  }

  function backToTopController() {
    const btn = ensureBackToTopButton();
    const footer = document.querySelector("footer");

    const scrollTop = () =>
      window.scrollTo({
        top: 0,
        behavior: prefersReducedMotion() ? "auto" : "smooth",
      });

    btn.addEventListener("click", scrollTop);
    btn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        scrollTop();
      }
    });

    let ticking = false;
    const adjust = () => {
      // show/hide after some scroll
      const shouldShow = window.scrollY > 240;
      btn.style.display = shouldShow ? "inline-flex" : "none";

      // keep above footer when it enters viewport
      if (footer) {
        const rect = footer.getBoundingClientRect();
        const overlap = window.innerHeight - rect.top; // >0 when footer rises
        btn.style.bottom = overlap > 0 ? overlap + 16 + "px" : "1rem";
      }
      ticking = false;
    };

    const onScrollOrResize = () => {
      if (!ticking) {
        requestAnimationFrame(adjust);
        ticking = true;
      }
    };

    window.addEventListener("scroll", onScrollOrResize, { passive: true });
    window.addEventListener("resize", onScrollOrResize);
    adjust();
  }

  // ---------- last crop flash on home ----------
  function highlightLastCrop() {
    const last = sessionStorage.getItem("selectedCrop");
    if (last && window.location.pathname === "/") {
      const section = document.getElementById(last.toLowerCase());
      if (section) {
        section.classList.add("ring", "ring-4", "ring-green-300");
        setTimeout(
          () => section.classList.remove("ring", "ring-4", "ring-green-300"),
          2000
        );
      }
    }
  }

  // ---------- make header logo/brand go home ----------
  function brandHomeLink() {
    const header = document.querySelector("header");
    if (!header) return;

    const goHome = () => {
      if (window.location.pathname !== "/") {
        document.body.classList.add("opacity-0");
        setTimeout(() => (window.location.href = "/"), 160);
      } else {
        window.scrollTo({
          top: 0,
          behavior: prefersReducedMotion() ? "auto" : "smooth",
        });
      }
    };

    // try common candidates in your markup
    const candidates = [
      header.querySelector('img[alt="Logo"]'),
      header.querySelector('img[src*="logo"]'),
      header.querySelector(".site-name"),
      header.querySelector(".site-title"),
      header.querySelector(".brand"),
      header.querySelector(".logo"),
      header.querySelector("span.text-2xl.font-bold"),
    ].filter(Boolean);

    // de-dup and attach
    const seen = new Set();
    candidates.forEach((el) => {
      if (!el || seen.has(el)) return;
      seen.add(el);
      el.style.cursor = "pointer";
      el.setAttribute("role", "link");
      el.setAttribute("tabindex", "0");
      el.addEventListener("click", goHome);
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          goHome();
        }
      });
    });
  }

  // ---------- boot ----------
  ready(() => {
    // remove initial fade
    document.body.classList.remove("opacity-0");

    // mobile nav toggle
    const navToggle = document.getElementById("nav-toggle");
    const mobileNav = document.getElementById("mobile-nav");
    if (navToggle && mobileNav) {
      navToggle.addEventListener("click", () =>
        mobileNav.classList.toggle("hidden")
      );
    }

    smoothInternalLinks();
    setupForms();
    backToTopController();
    highlightLastCrop();
    brandHomeLink();
  });
})();
