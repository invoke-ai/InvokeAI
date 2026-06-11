/**
 * Invoke AI — Main JavaScript
 * Handles theme toggle and basic interactivity
 */

(function () {
  'use strict';

  // ============================================
  // Theme Toggle
  // ============================================
  const STORAGE_KEY = 'invoke-theme';

  function getPreferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }

  // Apply theme immediately
  setTheme(getPreferredTheme());

  // Theme toggle buttons
  document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const current = document.documentElement.getAttribute('data-theme');
      setTheme(current === 'dark' ? 'light' : 'dark');
    });
  });

  // ============================================
  // Hero Prompt Input — Enter to Generate
  // ============================================
  var heroInput = document.getElementById('hero-prompt-input');
  if (heroInput) {
    heroInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        window.location.href = 'pages/dashboard.html?prompt=' + encodeURIComponent(heroInput.value);
      }
    });
  }

  // ============================================
  // Smooth Scroll for Anchor Links
  // ============================================
  document.querySelectorAll('a[href^="#"]').forEach(function (link) {
    link.addEventListener('click', function (e) {
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

})();
