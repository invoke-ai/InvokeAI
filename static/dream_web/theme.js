function initThemeSelector() {
    const themeSelect = document.getElementById("theme-selector");
    const themeStylesheetLink = document.getElementById("themeStylesheetLink");
    const currentTheme = localStorage.getItem("theme") || "theme_1";

    function activateTheme(themeName) {
        themeStylesheetLink.setAttribute("href", `static/dream_web/themes/${themeName}.css`);
    }

    themeSelect.addEventListener("change", () => {
        activateTheme(themeSelect.value);
        localStorage.setItem("theme", themeSelect.value);
    });


    themeSelect.value = currentTheme;
    activateTheme(currentTheme);
    }

    initThemeSelector();