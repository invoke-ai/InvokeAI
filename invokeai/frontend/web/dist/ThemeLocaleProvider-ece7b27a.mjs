import { j as y, e4 as Ie, n as mt, e5 as gt, O as qr, e6 as o, e7 as b, e8 as v, e9 as S, ea as Nr, eb as ht, ec as pt, e3 as yt, ed as St, y as xt, ee as $t, ef as kt, a0 as _t, $ as wt } from "./exports-21bec2df.mjs";
import { useState as sr, useMemo as Ae, useCallback as dr, useEffect as Xe, createContext as Gr, useContext as Ct, useRef as zt } from "react";
import { T as At, a as Pt } from "./chunk-ZP4IHIHZ-8f3a3ec4.mjs";
import "react-dom";
var Ur = `
  :root {
    --chakra-vh: 100vh;
  }

  @supports (height: -webkit-fill-available) {
    :root {
      --chakra-vh: -webkit-fill-available;
    }
  }

  @supports (height: -moz-fill-available) {
    :root {
      --chakra-vh: -moz-fill-available;
    }
  }

  @supports (height: 100dvh) {
    :root {
      --chakra-vh: 100dvh;
    }
  }
`, Tt = () => /* @__PURE__ */ y.jsx(Ie, { styles: Ur }), Ft = () => /* @__PURE__ */ y.jsx(
  Ie,
  {
    styles: `
      html {
        line-height: 1.5;
        -webkit-text-size-adjust: 100%;
        font-family: system-ui, sans-serif;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
        -moz-osx-font-smoothing: grayscale;
        touch-action: manipulation;
      }

      body {
        position: relative;
        min-height: 100%;
        font-feature-settings: 'kern';
      }

      *,
      *::before,
      *::after {
        border-width: 0;
        border-style: solid;
        box-sizing: border-box;
      }

      main {
        display: block;
      }

      hr {
        border-top-width: 1px;
        box-sizing: content-box;
        height: 0;
        overflow: visible;
      }

      pre,
      code,
      kbd,
      samp {
        font-family: SFMono-Regular,  Menlo, Monaco, Consolas, monospace;
        font-size: 1em;
      }

      a {
        background-color: transparent;
        color: inherit;
        text-decoration: inherit;
      }

      abbr[title] {
        border-bottom: none;
        text-decoration: underline;
        -webkit-text-decoration: underline dotted;
        text-decoration: underline dotted;
      }

      b,
      strong {
        font-weight: bold;
      }

      small {
        font-size: 80%;
      }

      sub,
      sup {
        font-size: 75%;
        line-height: 0;
        position: relative;
        vertical-align: baseline;
      }

      sub {
        bottom: -0.25em;
      }

      sup {
        top: -0.5em;
      }

      img {
        border-style: none;
      }

      button,
      input,
      optgroup,
      select,
      textarea {
        font-family: inherit;
        font-size: 100%;
        line-height: 1.15;
        margin: 0;
      }

      button,
      input {
        overflow: visible;
      }

      button,
      select {
        text-transform: none;
      }

      button::-moz-focus-inner,
      [type="button"]::-moz-focus-inner,
      [type="reset"]::-moz-focus-inner,
      [type="submit"]::-moz-focus-inner {
        border-style: none;
        padding: 0;
      }

      fieldset {
        padding: 0.35em 0.75em 0.625em;
      }

      legend {
        box-sizing: border-box;
        color: inherit;
        display: table;
        max-width: 100%;
        padding: 0;
        white-space: normal;
      }

      progress {
        vertical-align: baseline;
      }

      textarea {
        overflow: auto;
      }

      [type="checkbox"],
      [type="radio"] {
        box-sizing: border-box;
        padding: 0;
      }

      [type="number"]::-webkit-inner-spin-button,
      [type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
      }

      input[type="number"] {
        -moz-appearance: textfield;
      }

      [type="search"] {
        -webkit-appearance: textfield;
        outline-offset: -2px;
      }

      [type="search"]::-webkit-search-decoration {
        -webkit-appearance: none !important;
      }

      ::-webkit-file-upload-button {
        -webkit-appearance: button;
        font: inherit;
      }

      details {
        display: block;
      }

      summary {
        display: list-item;
      }

      template {
        display: none;
      }

      [hidden] {
        display: none !important;
      }

      body,
      blockquote,
      dl,
      dd,
      h1,
      h2,
      h3,
      h4,
      h5,
      h6,
      hr,
      figure,
      p,
      pre {
        margin: 0;
      }

      button {
        background: transparent;
        padding: 0;
      }

      fieldset {
        margin: 0;
        padding: 0;
      }

      ol,
      ul {
        margin: 0;
        padding: 0;
      }

      textarea {
        resize: vertical;
      }

      button,
      [role="button"] {
        cursor: pointer;
      }

      button::-moz-focus-inner {
        border: 0 !important;
      }

      table {
        border-collapse: collapse;
      }

      h1,
      h2,
      h3,
      h4,
      h5,
      h6 {
        font-size: inherit;
        font-weight: inherit;
      }

      button,
      input,
      optgroup,
      select,
      textarea {
        padding: 0;
        line-height: inherit;
        color: inherit;
      }

      img,
      svg,
      video,
      canvas,
      audio,
      iframe,
      embed,
      object {
        display: block;
      }

      img,
      video {
        max-width: 100%;
        height: auto;
      }

      [data-js-focus-visible] :focus:not([data-focus-visible-added]):not([data-focus-visible-disabled]) {
        outline: none;
        box-shadow: none;
      }

      select::-ms-expand {
        display: none;
      }

      ${Ur}
    `
  }
), he = {
  light: "chakra-ui-light",
  dark: "chakra-ui-dark"
};
function It(e = {}) {
  const { preventTransition: r = !0 } = e, t = {
    setDataset: (a) => {
      const n = r ? t.preventTransition() : void 0;
      document.documentElement.dataset.theme = a, document.documentElement.style.colorScheme = a, n == null || n();
    },
    setClassName(a) {
      document.body.classList.add(a ? he.dark : he.light), document.body.classList.remove(a ? he.light : he.dark);
    },
    query() {
      return window.matchMedia("(prefers-color-scheme: dark)");
    },
    getSystemTheme(a) {
      var n;
      return ((n = t.query().matches) != null ? n : a === "dark") ? "dark" : "light";
    },
    addListener(a) {
      const n = t.query(), i = (l) => {
        a(l.matches ? "dark" : "light");
      };
      return typeof n.addListener == "function" ? n.addListener(i) : n.addEventListener("change", i), () => {
        typeof n.removeListener == "function" ? n.removeListener(i) : n.removeEventListener("change", i);
      };
    },
    preventTransition() {
      const a = document.createElement("style");
      return a.appendChild(
        document.createTextNode(
          "*{-webkit-transition:none!important;-moz-transition:none!important;-o-transition:none!important;-ms-transition:none!important;transition:none!important}"
        )
      ), document.head.appendChild(a), () => {
        window.getComputedStyle(document.body), requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            document.head.removeChild(a);
          });
        });
      };
    }
  };
  return t;
}
var Bt = "chakra-ui-color-mode";
function Mt(e) {
  return {
    ssr: !1,
    type: "localStorage",
    get(r) {
      if (!(globalThis != null && globalThis.document))
        return r;
      let t;
      try {
        t = localStorage.getItem(e) || r;
      } catch {
      }
      return t || r;
    },
    set(r) {
      try {
        localStorage.setItem(e, r);
      } catch {
      }
    }
  };
}
var Et = Mt(Bt), cr = () => {
};
function br(e, r) {
  return e.type === "cookie" && e.ssr ? e.get(r) : r;
}
function Yr(e) {
  const {
    value: r,
    children: t,
    options: {
      useSystemColorMode: a,
      initialColorMode: n,
      disableTransitionOnChange: i
    } = {},
    colorModeManager: l = Et
  } = e, d = n === "dark" ? "dark" : "light", [c, $] = sr(
    () => br(l, d)
  ), [g, h] = sr(
    () => br(l)
  ), { getSystemTheme: k, setClassName: T, setDataset: fe, addListener: ir } = Ae(
    () => It({ preventTransition: i }),
    [i]
  ), me = n === "system" && !c ? g : c, F = dr(
    (ee) => {
      const ge = ee === "system" ? k() : ee;
      $(ge), T(ge === "dark"), fe(ge), l.set(ge);
    },
    [l, k, T, fe]
  );
  mt(() => {
    n === "system" && h(k());
  }, []), Xe(() => {
    const ee = l.get();
    if (ee) {
      F(ee);
      return;
    }
    if (n === "system") {
      F("system");
      return;
    }
    F(d);
  }, [l, d, n, F]);
  const lr = dr(() => {
    F(me === "dark" ? "light" : "dark");
  }, [me, F]);
  Xe(() => {
    if (a)
      return ir(F);
  }, [a, ir, F]);
  const ft = Ae(
    () => ({
      colorMode: r ?? me,
      toggleColorMode: r ? cr : lr,
      setColorMode: r ? cr : F,
      forced: r !== void 0
    }),
    [me, lr, F, r]
  );
  return /* @__PURE__ */ y.jsx(gt.Provider, { value: ft, children: t });
}
Yr.displayName = "ColorModeProvider";
var Dt = [
  "borders",
  "breakpoints",
  "colors",
  "components",
  "config",
  "direction",
  "fonts",
  "fontSizes",
  "fontWeights",
  "letterSpacings",
  "lineHeights",
  "radii",
  "shadows",
  "sizes",
  "space",
  "styles",
  "transition",
  "zIndices"
];
function jt(e) {
  return qr(e) ? Dt.every(
    (r) => Object.prototype.hasOwnProperty.call(e, r)
  ) : !1;
}
var Rt = {
  hide: -1,
  auto: "auto",
  base: 0,
  docked: 10,
  dropdown: 1e3,
  sticky: 1100,
  banner: 1200,
  overlay: 1300,
  modal: 1400,
  popover: 1500,
  skipLink: 1600,
  toast: 1700,
  tooltip: 1800
}, Wt = Rt, Ht = {
  base: "0em",
  sm: "30em",
  md: "48em",
  lg: "62em",
  xl: "80em",
  "2xl": "96em"
}, Lt = Ht, Vt = {
  transparent: "transparent",
  current: "currentColor",
  black: "#000000",
  white: "#FFFFFF",
  whiteAlpha: {
    50: "rgba(255, 255, 255, 0.04)",
    100: "rgba(255, 255, 255, 0.06)",
    200: "rgba(255, 255, 255, 0.08)",
    300: "rgba(255, 255, 255, 0.16)",
    400: "rgba(255, 255, 255, 0.24)",
    500: "rgba(255, 255, 255, 0.36)",
    600: "rgba(255, 255, 255, 0.48)",
    700: "rgba(255, 255, 255, 0.64)",
    800: "rgba(255, 255, 255, 0.80)",
    900: "rgba(255, 255, 255, 0.92)"
  },
  blackAlpha: {
    50: "rgba(0, 0, 0, 0.04)",
    100: "rgba(0, 0, 0, 0.06)",
    200: "rgba(0, 0, 0, 0.08)",
    300: "rgba(0, 0, 0, 0.16)",
    400: "rgba(0, 0, 0, 0.24)",
    500: "rgba(0, 0, 0, 0.36)",
    600: "rgba(0, 0, 0, 0.48)",
    700: "rgba(0, 0, 0, 0.64)",
    800: "rgba(0, 0, 0, 0.80)",
    900: "rgba(0, 0, 0, 0.92)"
  },
  gray: {
    50: "#F7FAFC",
    100: "#EDF2F7",
    200: "#E2E8F0",
    300: "#CBD5E0",
    400: "#A0AEC0",
    500: "#718096",
    600: "#4A5568",
    700: "#2D3748",
    800: "#1A202C",
    900: "#171923"
  },
  red: {
    50: "#FFF5F5",
    100: "#FED7D7",
    200: "#FEB2B2",
    300: "#FC8181",
    400: "#F56565",
    500: "#E53E3E",
    600: "#C53030",
    700: "#9B2C2C",
    800: "#822727",
    900: "#63171B"
  },
  orange: {
    50: "#FFFAF0",
    100: "#FEEBC8",
    200: "#FBD38D",
    300: "#F6AD55",
    400: "#ED8936",
    500: "#DD6B20",
    600: "#C05621",
    700: "#9C4221",
    800: "#7B341E",
    900: "#652B19"
  },
  yellow: {
    50: "#FFFFF0",
    100: "#FEFCBF",
    200: "#FAF089",
    300: "#F6E05E",
    400: "#ECC94B",
    500: "#D69E2E",
    600: "#B7791F",
    700: "#975A16",
    800: "#744210",
    900: "#5F370E"
  },
  green: {
    50: "#F0FFF4",
    100: "#C6F6D5",
    200: "#9AE6B4",
    300: "#68D391",
    400: "#48BB78",
    500: "#38A169",
    600: "#2F855A",
    700: "#276749",
    800: "#22543D",
    900: "#1C4532"
  },
  teal: {
    50: "#E6FFFA",
    100: "#B2F5EA",
    200: "#81E6D9",
    300: "#4FD1C5",
    400: "#38B2AC",
    500: "#319795",
    600: "#2C7A7B",
    700: "#285E61",
    800: "#234E52",
    900: "#1D4044"
  },
  blue: {
    50: "#ebf8ff",
    100: "#bee3f8",
    200: "#90cdf4",
    300: "#63b3ed",
    400: "#4299e1",
    500: "#3182ce",
    600: "#2b6cb0",
    700: "#2c5282",
    800: "#2a4365",
    900: "#1A365D"
  },
  cyan: {
    50: "#EDFDFD",
    100: "#C4F1F9",
    200: "#9DECF9",
    300: "#76E4F7",
    400: "#0BC5EA",
    500: "#00B5D8",
    600: "#00A3C4",
    700: "#0987A0",
    800: "#086F83",
    900: "#065666"
  },
  purple: {
    50: "#FAF5FF",
    100: "#E9D8FD",
    200: "#D6BCFA",
    300: "#B794F4",
    400: "#9F7AEA",
    500: "#805AD5",
    600: "#6B46C1",
    700: "#553C9A",
    800: "#44337A",
    900: "#322659"
  },
  pink: {
    50: "#FFF5F7",
    100: "#FED7E2",
    200: "#FBB6CE",
    300: "#F687B3",
    400: "#ED64A6",
    500: "#D53F8C",
    600: "#B83280",
    700: "#97266D",
    800: "#702459",
    900: "#521B41"
  },
  linkedin: {
    50: "#E8F4F9",
    100: "#CFEDFB",
    200: "#9BDAF3",
    300: "#68C7EC",
    400: "#34B3E4",
    500: "#00A0DC",
    600: "#008CC9",
    700: "#0077B5",
    800: "#005E93",
    900: "#004471"
  },
  facebook: {
    50: "#E8F4F9",
    100: "#D9DEE9",
    200: "#B7C2DA",
    300: "#6482C0",
    400: "#4267B2",
    500: "#385898",
    600: "#314E89",
    700: "#29487D",
    800: "#223B67",
    900: "#1E355B"
  },
  messenger: {
    50: "#D0E6FF",
    100: "#B9DAFF",
    200: "#A2CDFF",
    300: "#7AB8FF",
    400: "#2E90FF",
    500: "#0078FF",
    600: "#0063D1",
    700: "#0052AC",
    800: "#003C7E",
    900: "#002C5C"
  },
  whatsapp: {
    50: "#dffeec",
    100: "#b9f5d0",
    200: "#90edb3",
    300: "#65e495",
    400: "#3cdd78",
    500: "#22c35e",
    600: "#179848",
    700: "#0c6c33",
    800: "#01421c",
    900: "#001803"
  },
  twitter: {
    50: "#E5F4FD",
    100: "#C8E9FB",
    200: "#A8DCFA",
    300: "#83CDF7",
    400: "#57BBF5",
    500: "#1DA1F2",
    600: "#1A94DA",
    700: "#1681BF",
    800: "#136B9E",
    900: "#0D4D71"
  },
  telegram: {
    50: "#E3F2F9",
    100: "#C5E4F3",
    200: "#A2D4EC",
    300: "#7AC1E4",
    400: "#47A9DA",
    500: "#0088CC",
    600: "#007AB8",
    700: "#006BA1",
    800: "#005885",
    900: "#003F5E"
  }
}, Ot = Vt, qt = {
  none: "0",
  sm: "0.125rem",
  base: "0.25rem",
  md: "0.375rem",
  lg: "0.5rem",
  xl: "0.75rem",
  "2xl": "1rem",
  "3xl": "1.5rem",
  full: "9999px"
}, Nt = qt, Gt = {
  xs: "0 0 0 1px rgba(0, 0, 0, 0.05)",
  sm: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
  base: "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
  md: "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
  lg: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
  xl: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
  "2xl": "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
  outline: "0 0 0 3px rgba(66, 153, 225, 0.6)",
  inner: "inset 0 2px 4px 0 rgba(0,0,0,0.06)",
  none: "none",
  "dark-lg": "rgba(0, 0, 0, 0.1) 0px 0px 0px 1px, rgba(0, 0, 0, 0.2) 0px 5px 10px, rgba(0, 0, 0, 0.4) 0px 15px 40px"
}, Ut = Gt, Yt = {
  common: "background-color, border-color, color, fill, stroke, opacity, box-shadow, transform",
  colors: "background-color, border-color, color, fill, stroke",
  dimensions: "width, height",
  position: "left, right, top, bottom",
  background: "background-color, background-image, background-position"
}, Xt = {
  "ease-in": "cubic-bezier(0.4, 0, 1, 1)",
  "ease-out": "cubic-bezier(0, 0, 0.2, 1)",
  "ease-in-out": "cubic-bezier(0.4, 0, 0.2, 1)"
}, Kt = {
  "ultra-fast": "50ms",
  faster: "100ms",
  fast: "150ms",
  normal: "200ms",
  slow: "300ms",
  slower: "400ms",
  "ultra-slow": "500ms"
}, Zt = {
  property: Yt,
  easing: Xt,
  duration: Kt
}, Jt = Zt, Qt = {
  none: 0,
  sm: "4px",
  base: "8px",
  md: "12px",
  lg: "16px",
  xl: "24px",
  "2xl": "40px",
  "3xl": "64px"
}, eo = Qt, ro = {
  none: 0,
  "1px": "1px solid",
  "2px": "2px solid",
  "4px": "4px solid",
  "8px": "8px solid"
}, to = ro, oo = {
  letterSpacings: {
    tighter: "-0.05em",
    tight: "-0.025em",
    normal: "0",
    wide: "0.025em",
    wider: "0.05em",
    widest: "0.1em"
  },
  lineHeights: {
    normal: "normal",
    none: 1,
    shorter: 1.25,
    short: 1.375,
    base: 1.5,
    tall: 1.625,
    taller: "2",
    3: ".75rem",
    4: "1rem",
    5: "1.25rem",
    6: "1.5rem",
    7: "1.75rem",
    8: "2rem",
    9: "2.25rem",
    10: "2.5rem"
  },
  fontWeights: {
    hairline: 100,
    thin: 200,
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
    extrabold: 800,
    black: 900
  },
  fonts: {
    heading: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"',
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"',
    mono: 'SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace'
  },
  fontSizes: {
    "3xs": "0.45rem",
    "2xs": "0.625rem",
    xs: "0.75rem",
    sm: "0.875rem",
    md: "1rem",
    lg: "1.125rem",
    xl: "1.25rem",
    "2xl": "1.5rem",
    "3xl": "1.875rem",
    "4xl": "2.25rem",
    "5xl": "3rem",
    "6xl": "3.75rem",
    "7xl": "4.5rem",
    "8xl": "6rem",
    "9xl": "8rem"
  }
}, Xr = oo, Kr = {
  px: "1px",
  0.5: "0.125rem",
  1: "0.25rem",
  1.5: "0.375rem",
  2: "0.5rem",
  2.5: "0.625rem",
  3: "0.75rem",
  3.5: "0.875rem",
  4: "1rem",
  5: "1.25rem",
  6: "1.5rem",
  7: "1.75rem",
  8: "2rem",
  9: "2.25rem",
  10: "2.5rem",
  12: "3rem",
  14: "3.5rem",
  16: "4rem",
  20: "5rem",
  24: "6rem",
  28: "7rem",
  32: "8rem",
  36: "9rem",
  40: "10rem",
  44: "11rem",
  48: "12rem",
  52: "13rem",
  56: "14rem",
  60: "15rem",
  64: "16rem",
  72: "18rem",
  80: "20rem",
  96: "24rem"
}, ao = {
  max: "max-content",
  min: "min-content",
  full: "100%",
  "3xs": "14rem",
  "2xs": "16rem",
  xs: "20rem",
  sm: "24rem",
  md: "28rem",
  lg: "32rem",
  xl: "36rem",
  "2xl": "42rem",
  "3xl": "48rem",
  "4xl": "56rem",
  "5xl": "64rem",
  "6xl": "72rem",
  "7xl": "80rem",
  "8xl": "90rem",
  prose: "60ch"
}, no = {
  sm: "640px",
  md: "768px",
  lg: "1024px",
  xl: "1280px"
}, io = {
  ...Kr,
  ...ao,
  container: no
}, Zr = io, lo = {
  breakpoints: Lt,
  zIndices: Wt,
  radii: Nt,
  blur: eo,
  colors: Ot,
  ...Xr,
  sizes: Zr,
  shadows: Ut,
  space: Kr,
  borders: to,
  transition: Jt
};
function f(e, r = {}) {
  let t = !1;
  function a() {
    if (!t) {
      t = !0;
      return;
    }
    throw new Error(
      "[anatomy] .part(...) should only be called once. Did you mean to use .extend(...) ?"
    );
  }
  function n(...g) {
    a();
    for (const h of g)
      r[h] = c(h);
    return f(e, r);
  }
  function i(...g) {
    for (const h of g)
      h in r || (r[h] = c(h));
    return f(e, r);
  }
  function l() {
    return Object.fromEntries(
      Object.entries(r).map(([h, k]) => [h, k.selector])
    );
  }
  function d() {
    return Object.fromEntries(
      Object.entries(r).map(([h, k]) => [h, k.className])
    );
  }
  function c(g) {
    const T = `chakra-${(["container", "root"].includes(g ?? "") ? [e] : [e, g]).filter(Boolean).join("__")}`;
    return {
      className: T,
      selector: `.${T}`,
      toString: () => g
    };
  }
  return {
    parts: n,
    toPart: c,
    extend: i,
    selectors: l,
    classnames: d,
    get keys() {
      return Object.keys(r);
    },
    __type: {}
  };
}
var so = f("accordion").parts("root", "container", "button", "panel").extend("icon"), co = f("alert").parts("title", "description", "container").extend("icon", "spinner"), bo = f("avatar").parts("label", "badge", "container").extend("excessLabel", "group"), uo = f("breadcrumb").parts("link", "item", "container").extend("separator");
f("button").parts();
var vo = f("checkbox").parts("control", "icon", "container").extend("label");
f("progress").parts("track", "filledTrack").extend("label");
var fo = f("drawer").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer"), mo = f("editable").parts(
  "preview",
  "input",
  "textarea"
), go = f("form").parts(
  "container",
  "requiredIndicator",
  "helperText"
), ho = f("formError").parts("text", "icon"), po = f("input").parts("addon", "field", "element"), yo = f("list").parts("container", "item", "icon"), So = f("menu").parts("button", "list", "item").extend("groupTitle", "command", "divider"), xo = f("modal").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer"), $o = f("numberinput").parts(
  "root",
  "field",
  "stepperGroup",
  "stepper"
);
f("pininput").parts("field");
var ko = f("popover").parts("content", "header", "body", "footer").extend("popper", "arrow", "closeButton"), _o = f("progress").parts(
  "label",
  "filledTrack",
  "track"
), wo = f("radio").parts(
  "container",
  "control",
  "label"
), Co = f("select").parts("field", "icon"), zo = f("slider").parts(
  "container",
  "track",
  "thumb",
  "filledTrack",
  "mark"
), Ao = f("stat").parts(
  "container",
  "label",
  "helpText",
  "number",
  "icon"
), Po = f("switch").parts(
  "container",
  "track",
  "thumb"
), To = f("table").parts(
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "tfoot",
  "caption"
), Fo = f("tabs").parts(
  "root",
  "tab",
  "tablist",
  "tabpanel",
  "tabpanels",
  "indicator"
), Io = f("tag").parts(
  "container",
  "label",
  "closeButton"
), Bo = f("card").parts(
  "container",
  "header",
  "body",
  "footer"
);
function O(e, r, t) {
  return Math.min(Math.max(e, t), r);
}
class Mo extends Error {
  constructor(r) {
    super(`Failed to parse color: "${r}"`);
  }
}
var oe = Mo;
function er(e) {
  if (typeof e != "string")
    throw new oe(e);
  if (e.trim().toLowerCase() === "transparent")
    return [0, 0, 0, 0];
  let r = e.trim();
  r = Vo.test(e) ? jo(e) : e;
  const t = Ro.exec(r);
  if (t) {
    const l = Array.from(t).slice(1);
    return [...l.slice(0, 3).map((d) => parseInt(ce(d, 2), 16)), parseInt(ce(l[3] || "f", 2), 16) / 255];
  }
  const a = Wo.exec(r);
  if (a) {
    const l = Array.from(a).slice(1);
    return [...l.slice(0, 3).map((d) => parseInt(d, 16)), parseInt(l[3] || "ff", 16) / 255];
  }
  const n = Ho.exec(r);
  if (n) {
    const l = Array.from(n).slice(1);
    return [...l.slice(0, 3).map((d) => parseInt(d, 10)), parseFloat(l[3] || "1")];
  }
  const i = Lo.exec(r);
  if (i) {
    const [l, d, c, $] = Array.from(i).slice(1).map(parseFloat);
    if (O(0, 100, d) !== d)
      throw new oe(e);
    if (O(0, 100, c) !== c)
      throw new oe(e);
    return [...Oo(l, d, c), Number.isNaN($) ? 1 : $];
  }
  throw new oe(e);
}
function Eo(e) {
  let r = 5381, t = e.length;
  for (; t; )
    r = r * 33 ^ e.charCodeAt(--t);
  return (r >>> 0) % 2341;
}
const ur = (e) => parseInt(e.replace(/_/g, ""), 36), Do = "1q29ehhb 1n09sgk7 1kl1ekf_ _yl4zsno 16z9eiv3 1p29lhp8 _bd9zg04 17u0____ _iw9zhe5 _to73___ _r45e31e _7l6g016 _jh8ouiv _zn3qba8 1jy4zshs 11u87k0u 1ro9yvyo 1aj3xael 1gz9zjz0 _3w8l4xo 1bf1ekf_ _ke3v___ _4rrkb__ 13j776yz _646mbhl _nrjr4__ _le6mbhl 1n37ehkb _m75f91n _qj3bzfz 1939yygw 11i5z6x8 _1k5f8xs 1509441m 15t5lwgf _ae2th1n _tg1ugcv 1lp1ugcv 16e14up_ _h55rw7n _ny9yavn _7a11xb_ 1ih442g9 _pv442g9 1mv16xof 14e6y7tu 1oo9zkds 17d1cisi _4v9y70f _y98m8kc 1019pq0v 12o9zda8 _348j4f4 1et50i2o _8epa8__ _ts6senj 1o350i2o 1mi9eiuo 1259yrp0 1ln80gnw _632xcoy 1cn9zldc _f29edu4 1n490c8q _9f9ziet 1b94vk74 _m49zkct 1kz6s73a 1eu9dtog _q58s1rz 1dy9sjiq __u89jo3 _aj5nkwg _ld89jo3 13h9z6wx _qa9z2ii _l119xgq _bs5arju 1hj4nwk9 1qt4nwk9 1ge6wau6 14j9zlcw 11p1edc_ _ms1zcxe _439shk6 _jt9y70f _754zsow 1la40eju _oq5p___ _x279qkz 1fa5r3rv _yd2d9ip _424tcku _8y1di2_ _zi2uabw _yy7rn9h 12yz980_ __39ljp6 1b59zg0x _n39zfzp 1fy9zest _b33k___ _hp9wq92 1il50hz4 _io472ub _lj9z3eo 19z9ykg0 _8t8iu3a 12b9bl4a 1ak5yw0o _896v4ku _tb8k8lv _s59zi6t _c09ze0p 1lg80oqn 1id9z8wb _238nba5 1kq6wgdi _154zssg _tn3zk49 _da9y6tc 1sg7cv4f _r12jvtt 1gq5fmkz 1cs9rvci _lp9jn1c _xw1tdnb 13f9zje6 16f6973h _vo7ir40 _bt5arjf _rc45e4t _hr4e100 10v4e100 _hc9zke2 _w91egv_ _sj2r1kk 13c87yx8 _vqpds__ _ni8ggk8 _tj9yqfb 1ia2j4r4 _7x9b10u 1fc9ld4j 1eq9zldr _5j9lhpx _ez9zl6o _md61fzm".split(" ").reduce((e, r) => {
  const t = ur(r.substring(0, 3)), a = ur(r.substring(3)).toString(16);
  let n = "";
  for (let i = 0; i < 6 - a.length; i++)
    n += "0";
  return e[t] = `${n}${a}`, e;
}, {});
function jo(e) {
  const r = e.toLowerCase().trim(), t = Do[Eo(r)];
  if (!t)
    throw new oe(e);
  return `#${t}`;
}
const ce = (e, r) => Array.from(Array(r)).map(() => e).join(""), Ro = new RegExp(`^#${ce("([a-f0-9])", 3)}([a-f0-9])?$`, "i"), Wo = new RegExp(`^#${ce("([a-f0-9]{2})", 3)}([a-f0-9]{2})?$`, "i"), Ho = new RegExp(`^rgba?\\(\\s*(\\d+)\\s*${ce(",\\s*(\\d+)\\s*", 2)}(?:,\\s*([\\d.]+))?\\s*\\)$`, "i"), Lo = /^hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%(?:\s*,\s*([\d.]+))?\s*\)$/i, Vo = /^[a-z]+$/i, vr = (e) => Math.round(e * 255), Oo = (e, r, t) => {
  let a = t / 100;
  if (r === 0)
    return [a, a, a].map(vr);
  const n = (e % 360 + 360) % 360 / 60, i = (1 - Math.abs(2 * a - 1)) * (r / 100), l = i * (1 - Math.abs(n % 2 - 1));
  let d = 0, c = 0, $ = 0;
  n >= 0 && n < 1 ? (d = i, c = l) : n >= 1 && n < 2 ? (d = l, c = i) : n >= 2 && n < 3 ? (c = i, $ = l) : n >= 3 && n < 4 ? (c = l, $ = i) : n >= 4 && n < 5 ? (d = l, $ = i) : n >= 5 && n < 6 && (d = i, $ = l);
  const g = a - i / 2, h = d + g, k = c + g, T = $ + g;
  return [h, k, T].map(vr);
};
function qo(e, r, t, a) {
  return `rgba(${O(0, 255, e).toFixed()}, ${O(0, 255, r).toFixed()}, ${O(0, 255, t).toFixed()}, ${parseFloat(O(0, 1, a).toFixed(3))})`;
}
function No(e, r) {
  const [t, a, n, i] = er(e);
  return qo(t, a, n, i - r);
}
function Go(e) {
  const [r, t, a, n] = er(e);
  let i = (l) => {
    const d = O(0, 255, l).toString(16);
    return d.length === 1 ? `0${d}` : d;
  };
  return `#${i(r)}${i(t)}${i(a)}${n < 1 ? i(Math.round(n * 255)) : ""}`;
}
function Uo(e, r, t, a, n) {
  for (r = r.split ? r.split(".") : r, a = 0; a < r.length; a++)
    e = e ? e[r[a]] : n;
  return e === n ? t : e;
}
var Yo = (e) => Object.keys(e).length === 0, _ = (e, r, t) => {
  const a = Uo(e, `colors.${r}`, r);
  try {
    return Go(a), a;
  } catch {
    return t ?? "#000000";
  }
}, Xo = (e) => {
  const [r, t, a] = er(e);
  return (r * 299 + t * 587 + a * 114) / 1e3;
}, Ko = (e) => (r) => {
  const t = _(r, e);
  return Xo(t) < 128 ? "dark" : "light";
}, Zo = (e) => (r) => Ko(e)(r) === "dark", Q = (e, r) => (t) => {
  const a = _(t, e);
  return No(a, 1 - r);
};
function fr(e = "1rem", r = "rgba(255, 255, 255, 0.15)") {
  return {
    backgroundImage: `linear-gradient(
    45deg,
    ${r} 25%,
    transparent 25%,
    transparent 50%,
    ${r} 50%,
    ${r} 75%,
    transparent 75%,
    transparent
  )`,
    backgroundSize: `${e} ${e}`
  };
}
var Jo = () => `#${Math.floor(Math.random() * 16777215).toString(16).padEnd(6, "0")}`;
function Qo(e) {
  const r = Jo();
  return !e || Yo(e) ? r : e.string && e.colors ? ra(e.string, e.colors) : e.string && !e.colors ? ea(e.string) : e.colors && !e.string ? ta(e.colors) : r;
}
function ea(e) {
  let r = 0;
  if (e.length === 0)
    return r.toString();
  for (let a = 0; a < e.length; a += 1)
    r = e.charCodeAt(a) + ((r << 5) - r), r = r & r;
  let t = "#";
  for (let a = 0; a < 3; a += 1) {
    const n = r >> a * 8 & 255;
    t += `00${n.toString(16)}`.substr(-2);
  }
  return t;
}
function ra(e, r) {
  let t = 0;
  if (e.length === 0)
    return r[0];
  for (let a = 0; a < e.length; a += 1)
    t = e.charCodeAt(a) + ((t << 5) - t), t = t & t;
  return t = (t % r.length + r.length) % r.length, r[t];
}
function ta(e) {
  return e[Math.floor(Math.random() * e.length)];
}
function s(e, r) {
  return (t) => t.colorMode === "dark" ? r : e;
}
function rr(e) {
  const { orientation: r, vertical: t, horizontal: a } = e;
  return r ? r === "vertical" ? t : a : {};
}
function Jr(e) {
  return qr(e) && e.reference ? e.reference : String(e);
}
var Be = (e, ...r) => r.map(Jr).join(` ${e} `).replace(/calc/g, ""), mr = (...e) => `calc(${Be("+", ...e)})`, gr = (...e) => `calc(${Be("-", ...e)})`, Ke = (...e) => `calc(${Be("*", ...e)})`, hr = (...e) => `calc(${Be("/", ...e)})`, pr = (e) => {
  const r = Jr(e);
  return r != null && !Number.isNaN(parseFloat(r)) ? String(r).startsWith("-") ? String(r).slice(1) : `-${r}` : Ke(r, -1);
}, E = Object.assign(
  (e) => ({
    add: (...r) => E(mr(e, ...r)),
    subtract: (...r) => E(gr(e, ...r)),
    multiply: (...r) => E(Ke(e, ...r)),
    divide: (...r) => E(hr(e, ...r)),
    negate: () => E(pr(e)),
    toString: () => e.toString()
  }),
  {
    add: mr,
    subtract: gr,
    multiply: Ke,
    divide: hr,
    negate: pr
  }
);
function oa(e) {
  return !Number.isInteger(parseFloat(e.toString()));
}
function aa(e, r = "-") {
  return e.replace(/\s+/g, r);
}
function Qr(e) {
  const r = aa(e.toString());
  return r.includes("\\.") ? e : oa(e) ? r.replace(".", "\\.") : e;
}
function na(e, r = "") {
  return [r, Qr(e)].filter(Boolean).join("-");
}
function ia(e, r) {
  return `var(${Qr(e)}${r ? `, ${r}` : ""})`;
}
function la(e, r = "") {
  return `--${na(e, r)}`;
}
function x(e, r) {
  const t = la(e, r == null ? void 0 : r.prefix);
  return {
    variable: t,
    reference: ia(t, sa(r == null ? void 0 : r.fallback))
  };
}
function sa(e) {
  return typeof e == "string" ? e : e == null ? void 0 : e.reference;
}
var { defineMultiStyleConfig: da, definePartsStyle: _e } = b(Po.keys), ie = x("switch-track-width"), q = x("switch-track-height"), De = x("switch-track-diff"), ca = E.subtract(ie, q), Ze = x("switch-thumb-x"), re = x("switch-bg"), ba = o((e) => {
  const { colorScheme: r } = e;
  return {
    borderRadius: "full",
    p: "0.5",
    width: [ie.reference],
    height: [q.reference],
    transitionProperty: "common",
    transitionDuration: "fast",
    [re.variable]: "colors.gray.300",
    _dark: {
      [re.variable]: "colors.whiteAlpha.400"
    },
    _focusVisible: {
      boxShadow: "outline"
    },
    _disabled: {
      opacity: 0.4,
      cursor: "not-allowed"
    },
    _checked: {
      [re.variable]: `colors.${r}.500`,
      _dark: {
        [re.variable]: `colors.${r}.200`
      }
    },
    bg: re.reference
  };
}), ua = o({
  bg: "white",
  transitionProperty: "transform",
  transitionDuration: "normal",
  borderRadius: "inherit",
  width: [q.reference],
  height: [q.reference],
  _checked: {
    transform: `translateX(${Ze.reference})`
  }
}), va = _e((e) => ({
  container: {
    [De.variable]: ca,
    [Ze.variable]: De.reference,
    _rtl: {
      [Ze.variable]: E(De).negate().toString()
    }
  },
  track: ba(e),
  thumb: ua
})), fa = {
  sm: _e({
    container: {
      [ie.variable]: "1.375rem",
      [q.variable]: "sizes.3"
    }
  }),
  md: _e({
    container: {
      [ie.variable]: "1.875rem",
      [q.variable]: "sizes.4"
    }
  }),
  lg: _e({
    container: {
      [ie.variable]: "2.875rem",
      [q.variable]: "sizes.6"
    }
  })
}, ma = da({
  baseStyle: va,
  sizes: fa,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), { defineMultiStyleConfig: ga, definePartsStyle: Y } = b(To.keys), ha = Y({
  table: {
    fontVariantNumeric: "lining-nums tabular-nums",
    borderCollapse: "collapse",
    width: "full"
  },
  th: {
    fontFamily: "heading",
    fontWeight: "bold",
    textTransform: "uppercase",
    letterSpacing: "wider",
    textAlign: "start"
  },
  td: {
    textAlign: "start"
  },
  caption: {
    mt: 4,
    fontFamily: "heading",
    textAlign: "center",
    fontWeight: "medium"
  }
}), Pe = o({
  "&[data-is-numeric=true]": {
    textAlign: "end"
  }
}), pa = Y((e) => {
  const { colorScheme: r } = e;
  return {
    th: {
      color: s("gray.600", "gray.400")(e),
      borderBottom: "1px",
      borderColor: s(`${r}.100`, `${r}.700`)(e),
      ...Pe
    },
    td: {
      borderBottom: "1px",
      borderColor: s(`${r}.100`, `${r}.700`)(e),
      ...Pe
    },
    caption: {
      color: s("gray.600", "gray.100")(e)
    },
    tfoot: {
      tr: {
        "&:last-of-type": {
          th: { borderBottomWidth: 0 }
        }
      }
    }
  };
}), ya = Y((e) => {
  const { colorScheme: r } = e;
  return {
    th: {
      color: s("gray.600", "gray.400")(e),
      borderBottom: "1px",
      borderColor: s(`${r}.100`, `${r}.700`)(e),
      ...Pe
    },
    td: {
      borderBottom: "1px",
      borderColor: s(`${r}.100`, `${r}.700`)(e),
      ...Pe
    },
    caption: {
      color: s("gray.600", "gray.100")(e)
    },
    tbody: {
      tr: {
        "&:nth-of-type(odd)": {
          "th, td": {
            borderBottomWidth: "1px",
            borderColor: s(`${r}.100`, `${r}.700`)(e)
          },
          td: {
            background: s(`${r}.100`, `${r}.700`)(e)
          }
        }
      }
    },
    tfoot: {
      tr: {
        "&:last-of-type": {
          th: { borderBottomWidth: 0 }
        }
      }
    }
  };
}), Sa = {
  simple: pa,
  striped: ya,
  unstyled: o({})
}, xa = {
  sm: Y({
    th: {
      px: "4",
      py: "1",
      lineHeight: "4",
      fontSize: "xs"
    },
    td: {
      px: "4",
      py: "2",
      fontSize: "sm",
      lineHeight: "4"
    },
    caption: {
      px: "4",
      py: "2",
      fontSize: "xs"
    }
  }),
  md: Y({
    th: {
      px: "6",
      py: "3",
      lineHeight: "4",
      fontSize: "xs"
    },
    td: {
      px: "6",
      py: "4",
      lineHeight: "5"
    },
    caption: {
      px: "6",
      py: "2",
      fontSize: "sm"
    }
  }),
  lg: Y({
    th: {
      px: "8",
      py: "4",
      lineHeight: "5",
      fontSize: "sm"
    },
    td: {
      px: "8",
      py: "5",
      lineHeight: "6"
    },
    caption: {
      px: "6",
      py: "2",
      fontSize: "md"
    }
  })
}, $a = ga({
  baseStyle: ha,
  variants: Sa,
  sizes: xa,
  defaultProps: {
    variant: "simple",
    size: "md",
    colorScheme: "gray"
  }
}), C = v("tabs-color"), P = v("tabs-bg"), pe = v("tabs-border-color"), { defineMultiStyleConfig: ka, definePartsStyle: M } = b(Fo.keys), _a = o((e) => {
  const { orientation: r } = e;
  return {
    display: r === "vertical" ? "flex" : "block"
  };
}), wa = o((e) => {
  const { isFitted: r } = e;
  return {
    flex: r ? 1 : void 0,
    transitionProperty: "common",
    transitionDuration: "normal",
    _focusVisible: {
      zIndex: 1,
      boxShadow: "outline"
    },
    _disabled: {
      cursor: "not-allowed",
      opacity: 0.4
    }
  };
}), Ca = o((e) => {
  const { align: r = "start", orientation: t } = e;
  return {
    justifyContent: {
      end: "flex-end",
      center: "center",
      start: "flex-start"
    }[r],
    flexDirection: t === "vertical" ? "column" : "row"
  };
}), za = o({
  p: 4
}), Aa = M((e) => ({
  root: _a(e),
  tab: wa(e),
  tablist: Ca(e),
  tabpanel: za
})), Pa = {
  sm: M({
    tab: {
      py: 1,
      px: 4,
      fontSize: "sm"
    }
  }),
  md: M({
    tab: {
      fontSize: "md",
      py: 2,
      px: 4
    }
  }),
  lg: M({
    tab: {
      fontSize: "lg",
      py: 3,
      px: 4
    }
  })
}, Ta = M((e) => {
  const { colorScheme: r, orientation: t } = e, a = t === "vertical", n = t === "vertical" ? "borderStart" : "borderBottom", i = a ? "marginStart" : "marginBottom";
  return {
    tablist: {
      [n]: "2px solid",
      borderColor: "inherit"
    },
    tab: {
      [n]: "2px solid",
      borderColor: "transparent",
      [i]: "-2px",
      _selected: {
        [C.variable]: `colors.${r}.600`,
        _dark: {
          [C.variable]: `colors.${r}.300`
        },
        borderColor: "currentColor"
      },
      _active: {
        [P.variable]: "colors.gray.200",
        _dark: {
          [P.variable]: "colors.whiteAlpha.300"
        }
      },
      _disabled: {
        _active: { bg: "none" }
      },
      color: C.reference,
      bg: P.reference
    }
  };
}), Fa = M((e) => {
  const { colorScheme: r } = e;
  return {
    tab: {
      borderTopRadius: "md",
      border: "1px solid",
      borderColor: "transparent",
      mb: "-1px",
      [pe.variable]: "transparent",
      _selected: {
        [C.variable]: `colors.${r}.600`,
        [pe.variable]: "colors.white",
        _dark: {
          [C.variable]: `colors.${r}.300`,
          [pe.variable]: "colors.gray.800"
        },
        borderColor: "inherit",
        borderBottomColor: pe.reference
      },
      color: C.reference
    },
    tablist: {
      mb: "-1px",
      borderBottom: "1px solid",
      borderColor: "inherit"
    }
  };
}), Ia = M((e) => {
  const { colorScheme: r } = e;
  return {
    tab: {
      border: "1px solid",
      borderColor: "inherit",
      [P.variable]: "colors.gray.50",
      _dark: {
        [P.variable]: "colors.whiteAlpha.50"
      },
      mb: "-1px",
      _notLast: {
        marginEnd: "-1px"
      },
      _selected: {
        [P.variable]: "colors.white",
        [C.variable]: `colors.${r}.600`,
        _dark: {
          [P.variable]: "colors.gray.800",
          [C.variable]: `colors.${r}.300`
        },
        borderColor: "inherit",
        borderTopColor: "currentColor",
        borderBottomColor: "transparent"
      },
      color: C.reference,
      bg: P.reference
    },
    tablist: {
      mb: "-1px",
      borderBottom: "1px solid",
      borderColor: "inherit"
    }
  };
}), Ba = M((e) => {
  const { colorScheme: r, theme: t } = e;
  return {
    tab: {
      borderRadius: "full",
      fontWeight: "semibold",
      color: "gray.600",
      _selected: {
        color: _(t, `${r}.700`),
        bg: _(t, `${r}.100`)
      }
    }
  };
}), Ma = M((e) => {
  const { colorScheme: r } = e;
  return {
    tab: {
      borderRadius: "full",
      fontWeight: "semibold",
      [C.variable]: "colors.gray.600",
      _dark: {
        [C.variable]: "inherit"
      },
      _selected: {
        [C.variable]: "colors.white",
        [P.variable]: `colors.${r}.600`,
        _dark: {
          [C.variable]: "colors.gray.800",
          [P.variable]: `colors.${r}.300`
        }
      },
      color: C.reference,
      bg: P.reference
    }
  };
}), Ea = M({}), Da = {
  line: Ta,
  enclosed: Fa,
  "enclosed-colored": Ia,
  "soft-rounded": Ba,
  "solid-rounded": Ma,
  unstyled: Ea
}, ja = ka({
  baseStyle: Aa,
  sizes: Pa,
  variants: Da,
  defaultProps: {
    size: "md",
    variant: "line",
    colorScheme: "blue"
  }
}), Ra = o({
  px: 1,
  textTransform: "uppercase",
  fontSize: "xs",
  borderRadius: "sm",
  fontWeight: "bold"
}), X = v("badge-bg"), B = v("badge-color"), Wa = o((e) => {
  const { colorScheme: r, theme: t } = e, a = Q(`${r}.500`, 0.6)(t);
  return {
    [X.variable]: `colors.${r}.500`,
    [B.variable]: "colors.white",
    _dark: {
      [X.variable]: a,
      [B.variable]: "colors.whiteAlpha.800"
    },
    bg: X.reference,
    color: B.reference
  };
}), Ha = o((e) => {
  const { colorScheme: r, theme: t } = e, a = Q(`${r}.200`, 0.16)(t);
  return {
    [X.variable]: `colors.${r}.100`,
    [B.variable]: `colors.${r}.800`,
    _dark: {
      [X.variable]: a,
      [B.variable]: `colors.${r}.200`
    },
    bg: X.reference,
    color: B.reference
  };
}), La = o((e) => {
  const { colorScheme: r, theme: t } = e, a = Q(`${r}.200`, 0.8)(t);
  return {
    [B.variable]: `colors.${r}.500`,
    _dark: {
      [B.variable]: a
    },
    color: B.reference,
    boxShadow: `inset 0 0 0px 1px ${B.reference}`
  };
}), Va = {
  solid: Wa,
  subtle: Ha,
  outline: La
}, le = S({
  baseStyle: Ra,
  variants: Va,
  defaultProps: {
    variant: "subtle",
    colorScheme: "gray"
  }
}), { defineMultiStyleConfig: Oa, definePartsStyle: N } = b(Io.keys), qa = o({
  fontWeight: "medium",
  lineHeight: 1.2,
  outline: 0,
  borderRadius: "md",
  _focusVisible: {
    boxShadow: "outline"
  }
}), Na = o({
  lineHeight: 1.2,
  overflow: "visible"
}), Ga = o({
  fontSize: "lg",
  w: "5",
  h: "5",
  transitionProperty: "common",
  transitionDuration: "normal",
  borderRadius: "full",
  marginStart: "1.5",
  marginEnd: "-1",
  opacity: 0.5,
  _disabled: {
    opacity: 0.4
  },
  _focusVisible: {
    boxShadow: "outline",
    bg: "rgba(0, 0, 0, 0.14)"
  },
  _hover: {
    opacity: 0.8
  },
  _active: {
    opacity: 1
  }
}), Ua = N({
  container: qa,
  label: Na,
  closeButton: Ga
}), Ya = {
  sm: N({
    container: {
      minH: "5",
      minW: "5",
      fontSize: "xs",
      px: "2"
    },
    closeButton: {
      marginEnd: "-2px",
      marginStart: "0.35rem"
    }
  }),
  md: N({
    container: {
      minH: "6",
      minW: "6",
      fontSize: "sm",
      px: "2"
    }
  }),
  lg: N({
    container: {
      minH: "8",
      minW: "8",
      fontSize: "md",
      px: "3"
    }
  })
}, Xa = {
  subtle: N((e) => {
    var r;
    return {
      container: (r = le.variants) == null ? void 0 : r.subtle(e)
    };
  }),
  solid: N((e) => {
    var r;
    return {
      container: (r = le.variants) == null ? void 0 : r.solid(e)
    };
  }),
  outline: N((e) => {
    var r;
    return {
      container: (r = le.variants) == null ? void 0 : r.outline(e)
    };
  })
}, Ka = Oa({
  variants: Xa,
  baseStyle: Ua,
  sizes: Ya,
  defaultProps: {
    size: "md",
    variant: "subtle",
    colorScheme: "gray"
  }
}), { definePartsStyle: D, defineMultiStyleConfig: Za } = b(po.keys), Ja = D({
  field: {
    width: "100%",
    minWidth: 0,
    outline: 0,
    position: "relative",
    appearance: "none",
    transitionProperty: "common",
    transitionDuration: "normal",
    _disabled: {
      opacity: 0.4,
      cursor: "not-allowed"
    }
  }
}), W = {
  lg: o({
    fontSize: "lg",
    px: "4",
    h: "12",
    borderRadius: "md"
  }),
  md: o({
    fontSize: "md",
    px: "4",
    h: "10",
    borderRadius: "md"
  }),
  sm: o({
    fontSize: "sm",
    px: "3",
    h: "8",
    borderRadius: "sm"
  }),
  xs: o({
    fontSize: "xs",
    px: "2",
    h: "6",
    borderRadius: "sm"
  })
}, Qa = {
  lg: D({
    field: W.lg,
    addon: W.lg
  }),
  md: D({
    field: W.md,
    addon: W.md
  }),
  sm: D({
    field: W.sm,
    addon: W.sm
  }),
  xs: D({
    field: W.xs,
    addon: W.xs
  })
};
function tr(e) {
  const { focusBorderColor: r, errorBorderColor: t } = e;
  return {
    focusBorderColor: r || s("blue.500", "blue.300")(e),
    errorBorderColor: t || s("red.500", "red.300")(e)
  };
}
var en = D((e) => {
  const { theme: r } = e, { focusBorderColor: t, errorBorderColor: a } = tr(e);
  return {
    field: {
      border: "1px solid",
      borderColor: "inherit",
      bg: "inherit",
      _hover: {
        borderColor: s("gray.300", "whiteAlpha.400")(e)
      },
      _readOnly: {
        boxShadow: "none !important",
        userSelect: "all"
      },
      _invalid: {
        borderColor: _(r, a),
        boxShadow: `0 0 0 1px ${_(r, a)}`
      },
      _focusVisible: {
        zIndex: 1,
        borderColor: _(r, t),
        boxShadow: `0 0 0 1px ${_(r, t)}`
      }
    },
    addon: {
      border: "1px solid",
      borderColor: s("inherit", "whiteAlpha.50")(e),
      bg: s("gray.100", "whiteAlpha.300")(e)
    }
  };
}), rn = D((e) => {
  const { theme: r } = e, { focusBorderColor: t, errorBorderColor: a } = tr(e);
  return {
    field: {
      border: "2px solid",
      borderColor: "transparent",
      bg: s("gray.100", "whiteAlpha.50")(e),
      _hover: {
        bg: s("gray.200", "whiteAlpha.100")(e)
      },
      _readOnly: {
        boxShadow: "none !important",
        userSelect: "all"
      },
      _invalid: {
        borderColor: _(r, a)
      },
      _focusVisible: {
        bg: "transparent",
        borderColor: _(r, t)
      }
    },
    addon: {
      border: "2px solid",
      borderColor: "transparent",
      bg: s("gray.100", "whiteAlpha.50")(e)
    }
  };
}), tn = D((e) => {
  const { theme: r } = e, { focusBorderColor: t, errorBorderColor: a } = tr(e);
  return {
    field: {
      borderBottom: "1px solid",
      borderColor: "inherit",
      borderRadius: "0",
      px: "0",
      bg: "transparent",
      _readOnly: {
        boxShadow: "none !important",
        userSelect: "all"
      },
      _invalid: {
        borderColor: _(r, a),
        boxShadow: `0px 1px 0px 0px ${_(r, a)}`
      },
      _focusVisible: {
        borderColor: _(r, t),
        boxShadow: `0px 1px 0px 0px ${_(r, t)}`
      }
    },
    addon: {
      borderBottom: "2px solid",
      borderColor: "inherit",
      borderRadius: "0",
      px: "0",
      bg: "transparent"
    }
  };
}), on = D({
  field: {
    bg: "transparent",
    px: "0",
    height: "auto"
  },
  addon: {
    bg: "transparent",
    px: "0",
    height: "auto"
  }
}), an = {
  outline: en,
  filled: rn,
  flushed: tn,
  unstyled: on
}, u = Za({
  baseStyle: Ja,
  sizes: Qa,
  variants: an,
  defaultProps: {
    size: "md",
    variant: "outline"
  }
}), yr, nn = o({
  ...(yr = u.baseStyle) == null ? void 0 : yr.field,
  paddingY: "2",
  minHeight: "20",
  lineHeight: "short",
  verticalAlign: "top"
}), Sr, xr, ln = {
  outline: o(
    (e) => {
      var r, t;
      return (t = (r = u.variants) == null ? void 0 : r.outline(e).field) != null ? t : {};
    }
  ),
  flushed: o(
    (e) => {
      var r, t;
      return (t = (r = u.variants) == null ? void 0 : r.flushed(e).field) != null ? t : {};
    }
  ),
  filled: o(
    (e) => {
      var r, t;
      return (t = (r = u.variants) == null ? void 0 : r.filled(e).field) != null ? t : {};
    }
  ),
  unstyled: (xr = (Sr = u.variants) == null ? void 0 : Sr.unstyled.field) != null ? xr : {}
}, $r, kr, _r, wr, Cr, zr, Ar, Pr, sn = {
  xs: (kr = ($r = u.sizes) == null ? void 0 : $r.xs.field) != null ? kr : {},
  sm: (wr = (_r = u.sizes) == null ? void 0 : _r.sm.field) != null ? wr : {},
  md: (zr = (Cr = u.sizes) == null ? void 0 : Cr.md.field) != null ? zr : {},
  lg: (Pr = (Ar = u.sizes) == null ? void 0 : Ar.lg.field) != null ? Pr : {}
}, dn = S({
  baseStyle: nn,
  sizes: sn,
  variants: ln,
  defaultProps: {
    size: "md",
    variant: "outline"
  }
}), ye = x("tooltip-bg"), je = x("tooltip-fg"), cn = x("popper-arrow-bg"), bn = o({
  bg: ye.reference,
  color: je.reference,
  [ye.variable]: "colors.gray.700",
  [je.variable]: "colors.whiteAlpha.900",
  _dark: {
    [ye.variable]: "colors.gray.300",
    [je.variable]: "colors.gray.900"
  },
  [cn.variable]: ye.reference,
  px: "2",
  py: "0.5",
  borderRadius: "sm",
  fontWeight: "medium",
  fontSize: "sm",
  boxShadow: "md",
  maxW: "xs",
  zIndex: "tooltip"
}), un = S({
  baseStyle: bn
}), { defineMultiStyleConfig: vn, definePartsStyle: ae } = b(_o.keys), fn = o((e) => {
  const { colorScheme: r, theme: t, isIndeterminate: a, hasStripe: n } = e, i = s(
    fr(),
    fr("1rem", "rgba(0,0,0,0.1)")
  )(e), l = s(`${r}.500`, `${r}.200`)(e), d = `linear-gradient(
    to right,
    transparent 0%,
    ${_(t, l)} 50%,
    transparent 100%
  )`;
  return {
    ...!a && n && i,
    ...a ? { bgImage: d } : { bgColor: l }
  };
}), mn = o({
  lineHeight: "1",
  fontSize: "0.25em",
  fontWeight: "bold",
  color: "white"
}), gn = o((e) => ({
  bg: s("gray.100", "whiteAlpha.300")(e)
})), hn = o((e) => ({
  transitionProperty: "common",
  transitionDuration: "slow",
  ...fn(e)
})), pn = ae((e) => ({
  label: mn,
  filledTrack: hn(e),
  track: gn(e)
})), yn = {
  xs: ae({
    track: { h: "1" }
  }),
  sm: ae({
    track: { h: "2" }
  }),
  md: ae({
    track: { h: "3" }
  }),
  lg: ae({
    track: { h: "4" }
  })
}, Sn = vn({
  sizes: yn,
  baseStyle: pn,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), xn = (e) => typeof e == "function";
function w(e, ...r) {
  return xn(e) ? e(...r) : e;
}
var { definePartsStyle: we, defineMultiStyleConfig: $n } = b(vo.keys), se = v("checkbox-size"), kn = o((e) => {
  const { colorScheme: r } = e;
  return {
    w: se.reference,
    h: se.reference,
    transitionProperty: "box-shadow",
    transitionDuration: "normal",
    border: "2px solid",
    borderRadius: "sm",
    borderColor: "inherit",
    color: "white",
    _checked: {
      bg: s(`${r}.500`, `${r}.200`)(e),
      borderColor: s(`${r}.500`, `${r}.200`)(e),
      color: s("white", "gray.900")(e),
      _hover: {
        bg: s(`${r}.600`, `${r}.300`)(e),
        borderColor: s(`${r}.600`, `${r}.300`)(e)
      },
      _disabled: {
        borderColor: s("gray.200", "transparent")(e),
        bg: s("gray.200", "whiteAlpha.300")(e),
        color: s("gray.500", "whiteAlpha.500")(e)
      }
    },
    _indeterminate: {
      bg: s(`${r}.500`, `${r}.200`)(e),
      borderColor: s(`${r}.500`, `${r}.200`)(e),
      color: s("white", "gray.900")(e)
    },
    _disabled: {
      bg: s("gray.100", "whiteAlpha.100")(e),
      borderColor: s("gray.100", "transparent")(e)
    },
    _focusVisible: {
      boxShadow: "outline"
    },
    _invalid: {
      borderColor: s("red.500", "red.300")(e)
    }
  };
}), _n = o({
  _disabled: { cursor: "not-allowed" }
}), wn = o({
  userSelect: "none",
  _disabled: { opacity: 0.4 }
}), Cn = o({
  transitionProperty: "transform",
  transitionDuration: "normal"
}), zn = we((e) => ({
  icon: Cn,
  container: _n,
  control: w(kn, e),
  label: wn
})), An = {
  sm: we({
    control: { [se.variable]: "sizes.3" },
    label: { fontSize: "sm" },
    icon: { fontSize: "3xs" }
  }),
  md: we({
    control: { [se.variable]: "sizes.4" },
    label: { fontSize: "md" },
    icon: { fontSize: "2xs" }
  }),
  lg: we({
    control: { [se.variable]: "sizes.5" },
    label: { fontSize: "lg" },
    icon: { fontSize: "2xs" }
  })
}, Te = $n({
  baseStyle: zn,
  sizes: An,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), { defineMultiStyleConfig: Pn, definePartsStyle: Ce } = b(wo.keys), Tn = o((e) => {
  var r;
  const t = (r = w(Te.baseStyle, e)) == null ? void 0 : r.control;
  return {
    ...t,
    borderRadius: "full",
    _checked: {
      ...t == null ? void 0 : t._checked,
      _before: {
        content: '""',
        display: "inline-block",
        pos: "relative",
        w: "50%",
        h: "50%",
        borderRadius: "50%",
        bg: "currentColor"
      }
    }
  };
}), Fn = Ce((e) => {
  var r, t, a, n;
  return {
    label: (t = (r = Te).baseStyle) == null ? void 0 : t.call(r, e).label,
    container: (n = (a = Te).baseStyle) == null ? void 0 : n.call(a, e).container,
    control: Tn(e)
  };
}), In = {
  md: Ce({
    control: { w: "4", h: "4" },
    label: { fontSize: "md" }
  }),
  lg: Ce({
    control: { w: "5", h: "5" },
    label: { fontSize: "lg" }
  }),
  sm: Ce({
    control: { width: "3", height: "3" },
    label: { fontSize: "sm" }
  })
}, Bn = Pn({
  baseStyle: Fn,
  sizes: In,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), { defineMultiStyleConfig: Mn, definePartsStyle: En } = b(Co.keys), Se = v("select-bg"), Tr, Dn = o({
  ...(Tr = u.baseStyle) == null ? void 0 : Tr.field,
  appearance: "none",
  paddingBottom: "1px",
  lineHeight: "normal",
  bg: Se.reference,
  [Se.variable]: "colors.white",
  _dark: {
    [Se.variable]: "colors.gray.700"
  },
  "> option, > optgroup": {
    bg: Se.reference
  }
}), jn = o({
  width: "6",
  height: "100%",
  insetEnd: "2",
  position: "relative",
  color: "currentColor",
  fontSize: "xl",
  _disabled: {
    opacity: 0.5
  }
}), Rn = En({
  field: Dn,
  icon: jn
}), xe = o({
  paddingInlineEnd: "8"
}), Fr, Ir, Br, Mr, Er, Dr, jr, Rr, Wn = {
  lg: {
    ...(Fr = u.sizes) == null ? void 0 : Fr.lg,
    field: {
      ...(Ir = u.sizes) == null ? void 0 : Ir.lg.field,
      ...xe
    }
  },
  md: {
    ...(Br = u.sizes) == null ? void 0 : Br.md,
    field: {
      ...(Mr = u.sizes) == null ? void 0 : Mr.md.field,
      ...xe
    }
  },
  sm: {
    ...(Er = u.sizes) == null ? void 0 : Er.sm,
    field: {
      ...(Dr = u.sizes) == null ? void 0 : Dr.sm.field,
      ...xe
    }
  },
  xs: {
    ...(jr = u.sizes) == null ? void 0 : jr.xs,
    field: {
      ...(Rr = u.sizes) == null ? void 0 : Rr.xs.field,
      ...xe
    },
    icon: {
      insetEnd: "1"
    }
  }
}, Hn = Mn({
  baseStyle: Rn,
  sizes: Wn,
  variants: u.variants,
  defaultProps: u.defaultProps
}), Re = v("skeleton-start-color"), We = v("skeleton-end-color"), Ln = o({
  [Re.variable]: "colors.gray.100",
  [We.variable]: "colors.gray.400",
  _dark: {
    [Re.variable]: "colors.gray.800",
    [We.variable]: "colors.gray.600"
  },
  background: Re.reference,
  borderColor: We.reference,
  opacity: 0.7,
  borderRadius: "sm"
}), Vn = S({
  baseStyle: Ln
}), He = v("skip-link-bg"), On = o({
  borderRadius: "md",
  fontWeight: "semibold",
  _focusVisible: {
    boxShadow: "outline",
    padding: "4",
    position: "fixed",
    top: "6",
    insetStart: "6",
    [He.variable]: "colors.white",
    _dark: {
      [He.variable]: "colors.gray.700"
    },
    bg: He.reference
  }
}), qn = S({
  baseStyle: On
}), { defineMultiStyleConfig: Nn, definePartsStyle: Me } = b(zo.keys), be = v("slider-thumb-size"), ue = v("slider-track-size"), L = v("slider-bg"), Gn = o((e) => {
  const { orientation: r } = e;
  return {
    display: "inline-block",
    position: "relative",
    cursor: "pointer",
    _disabled: {
      opacity: 0.6,
      cursor: "default",
      pointerEvents: "none"
    },
    ...rr({
      orientation: r,
      vertical: { h: "100%" },
      horizontal: { w: "100%" }
    })
  };
}), Un = o((e) => ({
  ...rr({
    orientation: e.orientation,
    horizontal: { h: ue.reference },
    vertical: { w: ue.reference }
  }),
  overflow: "hidden",
  borderRadius: "sm",
  [L.variable]: "colors.gray.200",
  _dark: {
    [L.variable]: "colors.whiteAlpha.200"
  },
  _disabled: {
    [L.variable]: "colors.gray.300",
    _dark: {
      [L.variable]: "colors.whiteAlpha.300"
    }
  },
  bg: L.reference
})), Yn = o((e) => {
  const { orientation: r } = e;
  return {
    ...rr({
      orientation: r,
      vertical: {
        left: "50%",
        transform: "translateX(-50%)",
        _active: {
          transform: "translateX(-50%) scale(1.15)"
        }
      },
      horizontal: {
        top: "50%",
        transform: "translateY(-50%)",
        _active: {
          transform: "translateY(-50%) scale(1.15)"
        }
      }
    }),
    w: be.reference,
    h: be.reference,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "absolute",
    outline: 0,
    zIndex: 1,
    borderRadius: "full",
    bg: "white",
    boxShadow: "base",
    border: "1px solid",
    borderColor: "transparent",
    transitionProperty: "transform",
    transitionDuration: "normal",
    _focusVisible: {
      boxShadow: "outline"
    },
    _disabled: {
      bg: "gray.300"
    }
  };
}), Xn = o((e) => {
  const { colorScheme: r } = e;
  return {
    width: "inherit",
    height: "inherit",
    [L.variable]: `colors.${r}.500`,
    _dark: {
      [L.variable]: `colors.${r}.200`
    },
    bg: L.reference
  };
}), Kn = Me((e) => ({
  container: Gn(e),
  track: Un(e),
  thumb: Yn(e),
  filledTrack: Xn(e)
})), Zn = Me({
  container: {
    [be.variable]: "sizes.4",
    [ue.variable]: "sizes.1"
  }
}), Jn = Me({
  container: {
    [be.variable]: "sizes.3.5",
    [ue.variable]: "sizes.1"
  }
}), Qn = Me({
  container: {
    [be.variable]: "sizes.2.5",
    [ue.variable]: "sizes.0.5"
  }
}), ei = {
  lg: Zn,
  md: Jn,
  sm: Qn
}, ri = Nn({
  baseStyle: Kn,
  sizes: ei,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), V = x("spinner-size"), ti = o({
  width: [V.reference],
  height: [V.reference]
}), oi = {
  xs: o({
    [V.variable]: "sizes.3"
  }),
  sm: o({
    [V.variable]: "sizes.4"
  }),
  md: o({
    [V.variable]: "sizes.6"
  }),
  lg: o({
    [V.variable]: "sizes.8"
  }),
  xl: o({
    [V.variable]: "sizes.12"
  })
}, ai = S({
  baseStyle: ti,
  sizes: oi,
  defaultProps: {
    size: "md"
  }
}), { defineMultiStyleConfig: ni, definePartsStyle: et } = b(Ao.keys), ii = o({
  fontWeight: "medium"
}), li = o({
  opacity: 0.8,
  marginBottom: "2"
}), si = o({
  verticalAlign: "baseline",
  fontWeight: "semibold"
}), di = o({
  marginEnd: 1,
  w: "3.5",
  h: "3.5",
  verticalAlign: "middle"
}), ci = et({
  container: {},
  label: ii,
  helpText: li,
  number: si,
  icon: di
}), bi = {
  md: et({
    label: { fontSize: "sm" },
    helpText: { fontSize: "sm" },
    number: { fontSize: "2xl" }
  })
}, ui = ni({
  baseStyle: ci,
  sizes: bi,
  defaultProps: {
    size: "md"
  }
}), Le = v("kbd-bg"), vi = o({
  [Le.variable]: "colors.gray.100",
  _dark: {
    [Le.variable]: "colors.whiteAlpha.100"
  },
  bg: Le.reference,
  borderRadius: "md",
  borderWidth: "1px",
  borderBottomWidth: "3px",
  fontSize: "0.8em",
  fontWeight: "bold",
  lineHeight: "normal",
  px: "0.4em",
  whiteSpace: "nowrap"
}), fi = S({
  baseStyle: vi
}), mi = o({
  transitionProperty: "common",
  transitionDuration: "fast",
  transitionTimingFunction: "ease-out",
  cursor: "pointer",
  textDecoration: "none",
  outline: "none",
  color: "inherit",
  _hover: {
    textDecoration: "underline"
  },
  _focusVisible: {
    boxShadow: "outline"
  }
}), gi = S({
  baseStyle: mi
}), { defineMultiStyleConfig: hi, definePartsStyle: pi } = b(yo.keys), yi = o({
  marginEnd: "2",
  display: "inline",
  verticalAlign: "text-bottom"
}), Si = pi({
  icon: yi
}), xi = hi({
  baseStyle: Si
}), { defineMultiStyleConfig: $i, definePartsStyle: ki } = b(So.keys), I = v("menu-bg"), Ve = v("menu-shadow"), _i = o({
  [I.variable]: "#fff",
  [Ve.variable]: "shadows.sm",
  _dark: {
    [I.variable]: "colors.gray.700",
    [Ve.variable]: "shadows.dark-lg"
  },
  color: "inherit",
  minW: "3xs",
  py: "2",
  zIndex: 1,
  borderRadius: "md",
  borderWidth: "1px",
  bg: I.reference,
  boxShadow: Ve.reference
}), wi = o({
  py: "1.5",
  px: "3",
  transitionProperty: "background",
  transitionDuration: "ultra-fast",
  transitionTimingFunction: "ease-in",
  _focus: {
    [I.variable]: "colors.gray.100",
    _dark: {
      [I.variable]: "colors.whiteAlpha.100"
    }
  },
  _active: {
    [I.variable]: "colors.gray.200",
    _dark: {
      [I.variable]: "colors.whiteAlpha.200"
    }
  },
  _expanded: {
    [I.variable]: "colors.gray.100",
    _dark: {
      [I.variable]: "colors.whiteAlpha.100"
    }
  },
  _disabled: {
    opacity: 0.4,
    cursor: "not-allowed"
  },
  bg: I.reference
}), Ci = o({
  mx: 4,
  my: 2,
  fontWeight: "semibold",
  fontSize: "sm"
}), zi = o({
  opacity: 0.6
}), Ai = o({
  border: 0,
  borderBottom: "1px solid",
  borderColor: "inherit",
  my: "2",
  opacity: 0.6
}), Pi = o({
  transitionProperty: "common",
  transitionDuration: "normal"
}), Ti = ki({
  button: Pi,
  list: _i,
  item: wi,
  groupTitle: Ci,
  command: zi,
  divider: Ai
}), Fi = $i({
  baseStyle: Ti
}), { defineMultiStyleConfig: Ii, definePartsStyle: Je } = b(xo.keys), Bi = o({
  bg: "blackAlpha.600",
  zIndex: "modal"
}), Mi = o((e) => {
  const { isCentered: r, scrollBehavior: t } = e;
  return {
    display: "flex",
    zIndex: "modal",
    justifyContent: "center",
    alignItems: r ? "center" : "flex-start",
    overflow: t === "inside" ? "hidden" : "auto",
    overscrollBehaviorY: "none"
  };
}), Ei = o((e) => {
  const { scrollBehavior: r } = e;
  return {
    borderRadius: "md",
    bg: s("white", "gray.700")(e),
    color: "inherit",
    my: "16",
    zIndex: "modal",
    maxH: r === "inside" ? "calc(100% - 7.5rem)" : void 0,
    boxShadow: s("lg", "dark-lg")(e)
  };
}), Di = o({
  px: "6",
  py: "4",
  fontSize: "xl",
  fontWeight: "semibold"
}), ji = o({
  position: "absolute",
  top: "2",
  insetEnd: "3"
}), Ri = o((e) => {
  const { scrollBehavior: r } = e;
  return {
    px: "6",
    py: "2",
    flex: "1",
    overflow: r === "inside" ? "auto" : void 0
  };
}), Wi = o({
  px: "6",
  py: "4"
}), Hi = Je((e) => ({
  overlay: Bi,
  dialogContainer: w(Mi, e),
  dialog: w(Ei, e),
  header: Di,
  closeButton: ji,
  body: w(Ri, e),
  footer: Wi
}));
function A(e) {
  return Je(e === "full" ? {
    dialog: {
      maxW: "100vw",
      minH: "$100vh",
      my: "0",
      borderRadius: "0"
    }
  } : {
    dialog: { maxW: e }
  });
}
var Li = {
  xs: A("xs"),
  sm: A("sm"),
  md: A("md"),
  lg: A("lg"),
  xl: A("xl"),
  "2xl": A("2xl"),
  "3xl": A("3xl"),
  "4xl": A("4xl"),
  "5xl": A("5xl"),
  "6xl": A("6xl"),
  full: A("full")
}, Vi = Ii({
  baseStyle: Hi,
  sizes: Li,
  defaultProps: { size: "md" }
}), { defineMultiStyleConfig: Oi, definePartsStyle: rt } = b($o.keys), or = x("number-input-stepper-width"), tt = x("number-input-input-padding"), qi = E(or).add("0.5rem").toString(), Oe = x("number-input-bg"), qe = x("number-input-color"), Ne = x("number-input-border-color"), Ni = o({
  [or.variable]: "sizes.6",
  [tt.variable]: qi
}), Gi = o(
  (e) => {
    var r, t;
    return (t = (r = w(u.baseStyle, e)) == null ? void 0 : r.field) != null ? t : {};
  }
), Ui = o({
  width: or.reference
}), Yi = o({
  borderStart: "1px solid",
  borderStartColor: Ne.reference,
  color: qe.reference,
  bg: Oe.reference,
  [qe.variable]: "colors.chakra-body-text",
  [Ne.variable]: "colors.chakra-border-color",
  _dark: {
    [qe.variable]: "colors.whiteAlpha.800",
    [Ne.variable]: "colors.whiteAlpha.300"
  },
  _active: {
    [Oe.variable]: "colors.gray.200",
    _dark: {
      [Oe.variable]: "colors.whiteAlpha.300"
    }
  },
  _disabled: {
    opacity: 0.4,
    cursor: "not-allowed"
  }
}), Xi = rt((e) => {
  var r;
  return {
    root: Ni,
    field: (r = w(Gi, e)) != null ? r : {},
    stepperGroup: Ui,
    stepper: Yi
  };
});
function $e(e) {
  var r, t, a;
  const n = (r = u.sizes) == null ? void 0 : r[e], i = {
    lg: "md",
    md: "md",
    sm: "sm",
    xs: "sm"
  }, l = (a = (t = n.field) == null ? void 0 : t.fontSize) != null ? a : "md", d = Xr.fontSizes[l];
  return rt({
    field: {
      ...n.field,
      paddingInlineEnd: tt.reference,
      verticalAlign: "top"
    },
    stepper: {
      fontSize: E(d).multiply(0.75).toString(),
      _first: {
        borderTopEndRadius: i[e]
      },
      _last: {
        borderBottomEndRadius: i[e],
        mt: "-1px",
        borderTopWidth: 1
      }
    }
  });
}
var Ki = {
  xs: $e("xs"),
  sm: $e("sm"),
  md: $e("md"),
  lg: $e("lg")
}, Zi = Oi({
  baseStyle: Xi,
  sizes: Ki,
  variants: u.variants,
  defaultProps: u.defaultProps
}), Wr, Ji = o({
  ...(Wr = u.baseStyle) == null ? void 0 : Wr.field,
  textAlign: "center"
}), Qi = {
  lg: o({
    fontSize: "lg",
    w: 12,
    h: 12,
    borderRadius: "md"
  }),
  md: o({
    fontSize: "md",
    w: 10,
    h: 10,
    borderRadius: "md"
  }),
  sm: o({
    fontSize: "sm",
    w: 8,
    h: 8,
    borderRadius: "sm"
  }),
  xs: o({
    fontSize: "xs",
    w: 6,
    h: 6,
    borderRadius: "sm"
  })
}, Hr, Lr, el = {
  outline: o(
    (e) => {
      var r, t, a;
      return (a = (t = w((r = u.variants) == null ? void 0 : r.outline, e)) == null ? void 0 : t.field) != null ? a : {};
    }
  ),
  flushed: o(
    (e) => {
      var r, t, a;
      return (a = (t = w((r = u.variants) == null ? void 0 : r.flushed, e)) == null ? void 0 : t.field) != null ? a : {};
    }
  ),
  filled: o(
    (e) => {
      var r, t, a;
      return (a = (t = w((r = u.variants) == null ? void 0 : r.filled, e)) == null ? void 0 : t.field) != null ? a : {};
    }
  ),
  unstyled: (Lr = (Hr = u.variants) == null ? void 0 : Hr.unstyled.field) != null ? Lr : {}
}, rl = S({
  baseStyle: Ji,
  sizes: Qi,
  variants: el,
  defaultProps: u.defaultProps
}), { defineMultiStyleConfig: tl, definePartsStyle: ol } = b(ko.keys), ke = x("popper-bg"), al = x("popper-arrow-bg"), Vr = x("popper-arrow-shadow-color"), nl = o({ zIndex: 10 }), il = o({
  [ke.variable]: "colors.white",
  bg: ke.reference,
  [al.variable]: ke.reference,
  [Vr.variable]: "colors.gray.200",
  _dark: {
    [ke.variable]: "colors.gray.700",
    [Vr.variable]: "colors.whiteAlpha.300"
  },
  width: "xs",
  border: "1px solid",
  borderColor: "inherit",
  borderRadius: "md",
  boxShadow: "sm",
  zIndex: "inherit",
  _focusVisible: {
    outline: 0,
    boxShadow: "outline"
  }
}), ll = o({
  px: 3,
  py: 2,
  borderBottomWidth: "1px"
}), sl = o({
  px: 3,
  py: 2
}), dl = o({
  px: 3,
  py: 2,
  borderTopWidth: "1px"
}), cl = o({
  position: "absolute",
  borderRadius: "md",
  top: 1,
  insetEnd: 2,
  padding: 2
}), bl = ol({
  popper: nl,
  content: il,
  header: ll,
  body: sl,
  footer: dl,
  closeButton: cl
}), ul = tl({
  baseStyle: bl
}), { definePartsStyle: Qe, defineMultiStyleConfig: vl } = b(fo.keys), Ge = v("drawer-bg"), Ue = v("drawer-box-shadow");
function U(e) {
  return Qe(e === "full" ? {
    dialog: { maxW: "100vw", h: "100vh" }
  } : {
    dialog: { maxW: e }
  });
}
var fl = o({
  bg: "blackAlpha.600",
  zIndex: "overlay"
}), ml = o({
  display: "flex",
  zIndex: "modal",
  justifyContent: "center"
}), gl = o((e) => {
  const { isFullHeight: r } = e;
  return {
    ...r && { height: "100vh" },
    zIndex: "modal",
    maxH: "100vh",
    color: "inherit",
    [Ge.variable]: "colors.white",
    [Ue.variable]: "shadows.lg",
    _dark: {
      [Ge.variable]: "colors.gray.700",
      [Ue.variable]: "shadows.dark-lg"
    },
    bg: Ge.reference,
    boxShadow: Ue.reference
  };
}), hl = o({
  px: "6",
  py: "4",
  fontSize: "xl",
  fontWeight: "semibold"
}), pl = o({
  position: "absolute",
  top: "2",
  insetEnd: "3"
}), yl = o({
  px: "6",
  py: "2",
  flex: "1",
  overflow: "auto"
}), Sl = o({
  px: "6",
  py: "4"
}), xl = Qe((e) => ({
  overlay: fl,
  dialogContainer: ml,
  dialog: w(gl, e),
  header: hl,
  closeButton: pl,
  body: yl,
  footer: Sl
})), $l = {
  xs: U("xs"),
  sm: U("md"),
  md: U("lg"),
  lg: U("2xl"),
  xl: U("4xl"),
  full: U("full")
}, kl = vl({
  baseStyle: xl,
  sizes: $l,
  defaultProps: {
    size: "xs"
  }
}), { definePartsStyle: _l, defineMultiStyleConfig: wl } = b(mo.keys), Cl = o({
  borderRadius: "md",
  py: "1",
  transitionProperty: "common",
  transitionDuration: "normal"
}), zl = o({
  borderRadius: "md",
  py: "1",
  transitionProperty: "common",
  transitionDuration: "normal",
  width: "full",
  _focusVisible: { boxShadow: "outline" },
  _placeholder: { opacity: 0.6 }
}), Al = o({
  borderRadius: "md",
  py: "1",
  transitionProperty: "common",
  transitionDuration: "normal",
  width: "full",
  _focusVisible: { boxShadow: "outline" },
  _placeholder: { opacity: 0.6 }
}), Pl = _l({
  preview: Cl,
  input: zl,
  textarea: Al
}), Tl = wl({
  baseStyle: Pl
}), { definePartsStyle: Fl, defineMultiStyleConfig: Il } = b(go.keys), K = v("form-control-color"), Bl = o({
  marginStart: "1",
  [K.variable]: "colors.red.500",
  _dark: {
    [K.variable]: "colors.red.300"
  },
  color: K.reference
}), Ml = o({
  mt: "2",
  [K.variable]: "colors.gray.600",
  _dark: {
    [K.variable]: "colors.whiteAlpha.600"
  },
  color: K.reference,
  lineHeight: "normal",
  fontSize: "sm"
}), El = Fl({
  container: {
    width: "100%",
    position: "relative"
  },
  requiredIndicator: Bl,
  helperText: Ml
}), Dl = Il({
  baseStyle: El
}), { definePartsStyle: jl, defineMultiStyleConfig: Rl } = b(ho.keys), Z = v("form-error-color"), Wl = o({
  [Z.variable]: "colors.red.500",
  _dark: {
    [Z.variable]: "colors.red.300"
  },
  color: Z.reference,
  mt: "2",
  fontSize: "sm",
  lineHeight: "normal"
}), Hl = o({
  marginEnd: "0.5em",
  [Z.variable]: "colors.red.500",
  _dark: {
    [Z.variable]: "colors.red.300"
  },
  color: Z.reference
}), Ll = jl({
  text: Wl,
  icon: Hl
}), Vl = Rl({
  baseStyle: Ll
}), Ol = o({
  fontSize: "md",
  marginEnd: "3",
  mb: "2",
  fontWeight: "medium",
  transitionProperty: "common",
  transitionDuration: "normal",
  opacity: 1,
  _disabled: {
    opacity: 0.4
  }
}), ql = S({
  baseStyle: Ol
}), Nl = o({
  fontFamily: "heading",
  fontWeight: "bold"
}), Gl = {
  "4xl": o({
    fontSize: ["6xl", null, "7xl"],
    lineHeight: 1
  }),
  "3xl": o({
    fontSize: ["5xl", null, "6xl"],
    lineHeight: 1
  }),
  "2xl": o({
    fontSize: ["4xl", null, "5xl"],
    lineHeight: [1.2, null, 1]
  }),
  xl: o({
    fontSize: ["3xl", null, "4xl"],
    lineHeight: [1.33, null, 1.2]
  }),
  lg: o({
    fontSize: ["2xl", null, "3xl"],
    lineHeight: [1.33, null, 1.2]
  }),
  md: o({
    fontSize: "xl",
    lineHeight: 1.2
  }),
  sm: o({
    fontSize: "md",
    lineHeight: 1.2
  }),
  xs: o({
    fontSize: "sm",
    lineHeight: 1.2
  })
}, Ul = S({
  baseStyle: Nl,
  sizes: Gl,
  defaultProps: {
    size: "xl"
  }
}), { defineMultiStyleConfig: Yl, definePartsStyle: Xl } = b(uo.keys), Kl = o({
  transitionProperty: "common",
  transitionDuration: "fast",
  transitionTimingFunction: "ease-out",
  cursor: "pointer",
  textDecoration: "none",
  outline: "none",
  color: "inherit",
  _hover: {
    textDecoration: "underline"
  },
  _focusVisible: {
    boxShadow: "outline"
  }
}), Zl = Xl({
  link: Kl
}), Jl = Yl({
  baseStyle: Zl
}), Ql = o({
  lineHeight: "1.2",
  borderRadius: "md",
  fontWeight: "semibold",
  transitionProperty: "common",
  transitionDuration: "normal",
  _focusVisible: {
    boxShadow: "outline"
  },
  _disabled: {
    opacity: 0.4,
    cursor: "not-allowed",
    boxShadow: "none"
  },
  _hover: {
    _disabled: {
      bg: "initial"
    }
  }
}), ot = o((e) => {
  const { colorScheme: r, theme: t } = e;
  if (r === "gray")
    return {
      color: s("inherit", "whiteAlpha.900")(e),
      _hover: {
        bg: s("gray.100", "whiteAlpha.200")(e)
      },
      _active: { bg: s("gray.200", "whiteAlpha.300")(e) }
    };
  const a = Q(`${r}.200`, 0.12)(t), n = Q(`${r}.200`, 0.24)(t);
  return {
    color: s(`${r}.600`, `${r}.200`)(e),
    bg: "transparent",
    _hover: {
      bg: s(`${r}.50`, a)(e)
    },
    _active: {
      bg: s(`${r}.100`, n)(e)
    }
  };
}), es = o((e) => {
  const { colorScheme: r } = e, t = s("gray.200", "whiteAlpha.300")(e);
  return {
    border: "1px solid",
    borderColor: r === "gray" ? t : "currentColor",
    ".chakra-button__group[data-attached][data-orientation=horizontal] > &:not(:last-of-type)": { marginEnd: "-1px" },
    ".chakra-button__group[data-attached][data-orientation=vertical] > &:not(:last-of-type)": { marginBottom: "-1px" },
    ...w(ot, e)
  };
}), rs = {
  yellow: {
    bg: "yellow.400",
    color: "black",
    hoverBg: "yellow.500",
    activeBg: "yellow.600"
  },
  cyan: {
    bg: "cyan.400",
    color: "black",
    hoverBg: "cyan.500",
    activeBg: "cyan.600"
  }
}, ts = o((e) => {
  var r;
  const { colorScheme: t } = e;
  if (t === "gray") {
    const c = s("gray.100", "whiteAlpha.200")(e);
    return {
      bg: c,
      _hover: {
        bg: s("gray.200", "whiteAlpha.300")(e),
        _disabled: {
          bg: c
        }
      },
      _active: { bg: s("gray.300", "whiteAlpha.400")(e) }
    };
  }
  const {
    bg: a = `${t}.500`,
    color: n = "white",
    hoverBg: i = `${t}.600`,
    activeBg: l = `${t}.700`
  } = (r = rs[t]) != null ? r : {}, d = s(a, `${t}.200`)(e);
  return {
    bg: d,
    color: s(n, "gray.800")(e),
    _hover: {
      bg: s(i, `${t}.300`)(e),
      _disabled: {
        bg: d
      }
    },
    _active: { bg: s(l, `${t}.400`)(e) }
  };
}), os = o((e) => {
  const { colorScheme: r } = e;
  return {
    padding: 0,
    height: "auto",
    lineHeight: "normal",
    verticalAlign: "baseline",
    color: s(`${r}.500`, `${r}.200`)(e),
    _hover: {
      textDecoration: "underline",
      _disabled: {
        textDecoration: "none"
      }
    },
    _active: {
      color: s(`${r}.700`, `${r}.500`)(e)
    }
  };
}), as = o({
  bg: "none",
  color: "inherit",
  display: "inline",
  lineHeight: "inherit",
  m: "0",
  p: "0"
}), ns = {
  ghost: ot,
  outline: es,
  solid: ts,
  link: os,
  unstyled: as
}, is = {
  lg: o({
    h: "12",
    minW: "12",
    fontSize: "lg",
    px: "6"
  }),
  md: o({
    h: "10",
    minW: "10",
    fontSize: "md",
    px: "4"
  }),
  sm: o({
    h: "8",
    minW: "8",
    fontSize: "sm",
    px: "3"
  }),
  xs: o({
    h: "6",
    minW: "6",
    fontSize: "xs",
    px: "2"
  })
}, ls = S({
  baseStyle: Ql,
  variants: ns,
  sizes: is,
  defaultProps: {
    variant: "solid",
    size: "md",
    colorScheme: "gray"
  }
}), { definePartsStyle: G, defineMultiStyleConfig: ss } = b(Bo.keys), Fe = v("card-bg"), j = v("card-padding"), at = v("card-shadow"), ze = v("card-radius"), nt = v("card-border-width", "0"), it = v("card-border-color"), ds = G({
  container: {
    [Fe.variable]: "colors.chakra-body-bg",
    backgroundColor: Fe.reference,
    boxShadow: at.reference,
    borderRadius: ze.reference,
    color: "chakra-body-text",
    borderWidth: nt.reference,
    borderColor: it.reference
  },
  body: {
    padding: j.reference,
    flex: "1 1 0%"
  },
  header: {
    padding: j.reference
  },
  footer: {
    padding: j.reference
  }
}), cs = {
  sm: G({
    container: {
      [ze.variable]: "radii.base",
      [j.variable]: "space.3"
    }
  }),
  md: G({
    container: {
      [ze.variable]: "radii.md",
      [j.variable]: "space.5"
    }
  }),
  lg: G({
    container: {
      [ze.variable]: "radii.xl",
      [j.variable]: "space.7"
    }
  })
}, bs = {
  elevated: G({
    container: {
      [at.variable]: "shadows.base",
      _dark: {
        [Fe.variable]: "colors.gray.700"
      }
    }
  }),
  outline: G({
    container: {
      [nt.variable]: "1px",
      [it.variable]: "colors.chakra-border-color"
    }
  }),
  filled: G({
    container: {
      [Fe.variable]: "colors.chakra-subtle-bg"
    }
  }),
  unstyled: {
    body: {
      [j.variable]: 0
    },
    header: {
      [j.variable]: 0
    },
    footer: {
      [j.variable]: 0
    }
  }
}, us = ss({
  baseStyle: ds,
  variants: bs,
  sizes: cs,
  defaultProps: {
    variant: "elevated",
    size: "md"
  }
}), de = x("close-button-size"), te = x("close-button-bg"), vs = o({
  w: [de.reference],
  h: [de.reference],
  borderRadius: "md",
  transitionProperty: "common",
  transitionDuration: "normal",
  _disabled: {
    opacity: 0.4,
    cursor: "not-allowed",
    boxShadow: "none"
  },
  _hover: {
    [te.variable]: "colors.blackAlpha.100",
    _dark: {
      [te.variable]: "colors.whiteAlpha.100"
    }
  },
  _active: {
    [te.variable]: "colors.blackAlpha.200",
    _dark: {
      [te.variable]: "colors.whiteAlpha.200"
    }
  },
  _focusVisible: {
    boxShadow: "outline"
  },
  bg: te.reference
}), fs = {
  lg: o({
    [de.variable]: "sizes.10",
    fontSize: "md"
  }),
  md: o({
    [de.variable]: "sizes.8",
    fontSize: "xs"
  }),
  sm: o({
    [de.variable]: "sizes.6",
    fontSize: "2xs"
  })
}, ms = S({
  baseStyle: vs,
  sizes: fs,
  defaultProps: {
    size: "md"
  }
}), { variants: gs, defaultProps: hs } = le, ps = o({
  fontFamily: "mono",
  fontSize: "sm",
  px: "0.2em",
  borderRadius: "sm"
}), ys = S({
  baseStyle: ps,
  variants: gs,
  defaultProps: hs
}), Ss = o({
  w: "100%",
  mx: "auto",
  maxW: "prose",
  px: "4"
}), xs = S({
  baseStyle: Ss
}), $s = o({
  opacity: 0.6,
  borderColor: "inherit"
}), ks = o({
  borderStyle: "solid"
}), _s = o({
  borderStyle: "dashed"
}), ws = {
  solid: ks,
  dashed: _s
}, Cs = S({
  baseStyle: $s,
  variants: ws,
  defaultProps: {
    variant: "solid"
  }
}), { definePartsStyle: zs, defineMultiStyleConfig: As } = b(so.keys), Ps = o({
  borderTopWidth: "1px",
  borderColor: "inherit",
  _last: {
    borderBottomWidth: "1px"
  }
}), Ts = o({
  transitionProperty: "common",
  transitionDuration: "normal",
  fontSize: "md",
  _focusVisible: {
    boxShadow: "outline"
  },
  _hover: {
    bg: "blackAlpha.50"
  },
  _disabled: {
    opacity: 0.4,
    cursor: "not-allowed"
  },
  px: "4",
  py: "2"
}), Fs = o({
  pt: "2",
  px: "4",
  pb: "5"
}), Is = o({
  fontSize: "1.25em"
}), Bs = zs({
  container: Ps,
  button: Ts,
  panel: Fs,
  icon: Is
}), Ms = As({ baseStyle: Bs }), { definePartsStyle: ve, defineMultiStyleConfig: Es } = b(co.keys), z = v("alert-fg"), R = v("alert-bg"), Ds = ve({
  container: {
    bg: R.reference,
    px: "4",
    py: "3"
  },
  title: {
    fontWeight: "bold",
    lineHeight: "6",
    marginEnd: "2"
  },
  description: {
    lineHeight: "6"
  },
  icon: {
    color: z.reference,
    flexShrink: 0,
    marginEnd: "3",
    w: "5",
    h: "6"
  },
  spinner: {
    color: z.reference,
    flexShrink: 0,
    marginEnd: "3",
    w: "5",
    h: "5"
  }
});
function ar(e) {
  const { theme: r, colorScheme: t } = e, a = Q(`${t}.200`, 0.16)(r);
  return {
    light: `colors.${t}.100`,
    dark: a
  };
}
var js = ve((e) => {
  const { colorScheme: r } = e, t = ar(e);
  return {
    container: {
      [z.variable]: `colors.${r}.500`,
      [R.variable]: t.light,
      _dark: {
        [z.variable]: `colors.${r}.200`,
        [R.variable]: t.dark
      }
    }
  };
}), Rs = ve((e) => {
  const { colorScheme: r } = e, t = ar(e);
  return {
    container: {
      [z.variable]: `colors.${r}.500`,
      [R.variable]: t.light,
      _dark: {
        [z.variable]: `colors.${r}.200`,
        [R.variable]: t.dark
      },
      paddingStart: "3",
      borderStartWidth: "4px",
      borderStartColor: z.reference
    }
  };
}), Ws = ve((e) => {
  const { colorScheme: r } = e, t = ar(e);
  return {
    container: {
      [z.variable]: `colors.${r}.500`,
      [R.variable]: t.light,
      _dark: {
        [z.variable]: `colors.${r}.200`,
        [R.variable]: t.dark
      },
      pt: "2",
      borderTopWidth: "4px",
      borderTopColor: z.reference
    }
  };
}), Hs = ve((e) => {
  const { colorScheme: r } = e;
  return {
    container: {
      [z.variable]: "colors.white",
      [R.variable]: `colors.${r}.500`,
      _dark: {
        [z.variable]: "colors.gray.900",
        [R.variable]: `colors.${r}.200`
      },
      color: z.reference
    }
  };
}), Ls = {
  subtle: js,
  "left-accent": Rs,
  "top-accent": Ws,
  solid: Hs
}, Vs = Es({
  baseStyle: Ds,
  variants: Ls,
  defaultProps: {
    variant: "subtle",
    colorScheme: "blue"
  }
}), { definePartsStyle: lt, defineMultiStyleConfig: Os } = b(bo.keys), J = v("avatar-border-color"), Ye = v("avatar-bg"), qs = o({
  borderRadius: "full",
  border: "0.2em solid",
  [J.variable]: "white",
  _dark: {
    [J.variable]: "colors.gray.800"
  },
  borderColor: J.reference
}), Ns = o({
  [Ye.variable]: "colors.gray.200",
  _dark: {
    [Ye.variable]: "colors.whiteAlpha.400"
  },
  bgColor: Ye.reference
}), Or = v("avatar-background"), Gs = o((e) => {
  const { name: r, theme: t } = e, a = r ? Qo({ string: r }) : "colors.gray.400", n = Zo(a)(t);
  let i = "white";
  return n || (i = "gray.800"), {
    bg: Or.reference,
    "&:not([data-loaded])": {
      [Or.variable]: a
    },
    color: i,
    [J.variable]: "colors.white",
    _dark: {
      [J.variable]: "colors.gray.800"
    },
    borderColor: J.reference,
    verticalAlign: "top"
  };
}), Us = lt((e) => ({
  badge: w(qs, e),
  excessLabel: w(Ns, e),
  container: w(Gs, e)
}));
function H(e) {
  const r = e !== "100%" ? Zr[e] : void 0;
  return lt({
    container: {
      width: e,
      height: e,
      fontSize: `calc(${r ?? e} / 2.5)`
    },
    excessLabel: {
      width: e,
      height: e
    },
    label: {
      fontSize: `calc(${r ?? e} / 2.5)`,
      lineHeight: e !== "100%" ? r ?? e : void 0
    }
  });
}
var Ys = {
  "2xs": H(4),
  xs: H(6),
  sm: H(8),
  md: H(12),
  lg: H(16),
  xl: H(24),
  "2xl": H(32),
  full: H("100%")
}, Xs = Os({
  baseStyle: Us,
  sizes: Ys,
  defaultProps: { size: "md" }
}), Ks = {
  Accordion: Ms,
  Alert: Vs,
  Avatar: Xs,
  Badge: le,
  Breadcrumb: Jl,
  Button: ls,
  Checkbox: Te,
  CloseButton: ms,
  Code: ys,
  Container: xs,
  Divider: Cs,
  Drawer: kl,
  Editable: Tl,
  Form: Dl,
  FormError: Vl,
  FormLabel: ql,
  Heading: Ul,
  Input: u,
  Kbd: fi,
  Link: gi,
  List: xi,
  Menu: Fi,
  Modal: Vi,
  NumberInput: Zi,
  PinInput: rl,
  Popover: ul,
  Progress: Sn,
  Radio: Bn,
  Select: Hn,
  Skeleton: Vn,
  SkipLink: qn,
  Slider: ri,
  Spinner: ai,
  Stat: ui,
  Switch: ma,
  Table: $a,
  Tabs: ja,
  Tag: Ka,
  Textarea: dn,
  Tooltip: un,
  Card: us
}, Zs = {
  colors: {
    "chakra-body-text": { _light: "gray.800", _dark: "whiteAlpha.900" },
    "chakra-body-bg": { _light: "white", _dark: "gray.800" },
    "chakra-border-color": { _light: "gray.200", _dark: "whiteAlpha.300" },
    "chakra-subtle-bg": { _light: "gray.100", _dark: "gray.700" },
    "chakra-placeholder-color": { _light: "gray.500", _dark: "whiteAlpha.400" }
  }
}, Js = {
  global: {
    body: {
      fontFamily: "body",
      color: "chakra-body-text",
      bg: "chakra-body-bg",
      transitionProperty: "background-color",
      transitionDuration: "normal",
      lineHeight: "base"
    },
    "*::placeholder": {
      color: "chakra-placeholder-color"
    },
    "*, *::before, &::after": {
      borderColor: "chakra-border-color",
      wordWrap: "break-word"
    }
  }
}, Qs = "ltr", ed = {
  useSystemColorMode: !1,
  initialColorMode: "light",
  cssVarPrefix: "chakra"
}, st = {
  semanticTokens: Zs,
  direction: Qs,
  ...lo,
  components: Ks,
  styles: Js,
  config: ed
};
function ne(e) {
  return typeof e == "function";
}
function rd(...e) {
  return (r) => e.reduce((t, a) => a(t), r);
}
var td = (e) => function(...t) {
  let a = [...t], n = t[t.length - 1];
  return jt(n) && a.length > 1 ? a = a.slice(0, a.length - 1) : n = e, rd(
    ...a.map(
      (i) => (l) => ne(i) ? i(l) : ad(l, i)
    )
  )(n);
}, od = td(st);
function ad(...e) {
  return Nr({}, ...e, dt);
}
function dt(e, r, t, a) {
  if ((ne(e) || ne(r)) && Object.prototype.hasOwnProperty.call(a, t))
    return (...n) => {
      const i = ne(e) ? e(...n) : e, l = ne(r) ? r(...n) : r;
      return Nr({}, i, l, dt);
    };
}
function nd(e = {}) {
  const {
    strict: r = !0,
    errorMessage: t = "useContext: `context` is undefined. Seems you forgot to wrap component within the Provider",
    name: a
  } = e, n = Gr(void 0);
  n.displayName = a;
  function i() {
    var l;
    const d = Ct(n);
    if (!d && r) {
      const c = new Error(t);
      throw c.name = "ContextError", (l = Error.captureStackTrace) == null || l.call(Error, c, i), c;
    }
    return d;
  }
  return [
    n.Provider,
    i,
    n
  ];
}
function id(e) {
  const { cssVarsRoot: r, theme: t, children: a } = e, n = Ae(() => ht(t), [t]);
  return /* @__PURE__ */ y.jsxs(pt, { theme: n, children: [
    /* @__PURE__ */ y.jsx(ld, { root: r }),
    a
  ] });
}
function ld({ root: e = ":host, :root" }) {
  const r = [e, "[data-theme]"].join(",");
  return /* @__PURE__ */ y.jsx(Ie, { styles: (t) => ({ [r]: t.__cssVars }) });
}
nd({
  name: "StylesContext",
  errorMessage: "useStyles: `styles` is undefined. Seems you forgot to wrap the components in `<StylesProvider />` "
});
function sd() {
  const { colorMode: e } = yt();
  return /* @__PURE__ */ y.jsx(
    Ie,
    {
      styles: (r) => {
        const t = St(r, "styles.global"), a = xt(t, { theme: r, colorMode: e });
        return a ? $t(a)(r) : void 0;
      }
    }
  );
}
var ct = Gr({
  getDocument() {
    return document;
  },
  getWindow() {
    return window;
  }
});
ct.displayName = "EnvironmentContext";
function bt(e) {
  const { children: r, environment: t, disabled: a } = e, n = zt(null), i = Ae(() => t || {
    getDocument: () => {
      var d, c;
      return (c = (d = n.current) == null ? void 0 : d.ownerDocument) != null ? c : document;
    },
    getWindow: () => {
      var d, c;
      return (c = (d = n.current) == null ? void 0 : d.ownerDocument.defaultView) != null ? c : window;
    }
  }, [t]), l = !a || !t;
  return /* @__PURE__ */ y.jsxs(ct.Provider, { value: i, children: [
    r,
    l && /* @__PURE__ */ y.jsx("span", { id: "__chakra_env", hidden: !0, ref: n })
  ] });
}
bt.displayName = "EnvironmentProvider";
var dd = (e) => {
  const {
    children: r,
    colorModeManager: t,
    portalZIndex: a,
    resetCSS: n = !0,
    theme: i = {},
    environment: l,
    cssVarsRoot: d,
    disableEnvironment: c
  } = e, $ = /* @__PURE__ */ y.jsx(
    bt,
    {
      environment: l,
      disabled: c,
      children: r
    }
  );
  return /* @__PURE__ */ y.jsx(id, { theme: i, cssVarsRoot: d, children: /* @__PURE__ */ y.jsxs(
    Yr,
    {
      colorModeManager: t,
      options: i.config,
      children: [
        n ? /* @__PURE__ */ y.jsx(Ft, {}) : /* @__PURE__ */ y.jsx(Tt, {}),
        /* @__PURE__ */ y.jsx(sd, {}),
        a ? /* @__PURE__ */ y.jsx(kt, { zIndex: a, children: $ }) : $
      ]
    }
  ) });
}, cd = (e) => function({
  children: t,
  theme: a = e,
  toastOptions: n,
  ...i
}) {
  return /* @__PURE__ */ y.jsxs(dd, { theme: a, ...i, children: [
    /* @__PURE__ */ y.jsx(At, { value: n == null ? void 0 : n.defaultOptions, children: t }),
    /* @__PURE__ */ y.jsx(Pt, { ...n })
  ] });
}, bd = cd(st);
function p(e, r, t = !1) {
  e = String(e), r = String(r);
  const a = Array.from({ length: 21 }, (d, c) => c * 50), n = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "59",
    "64",
    "68",
    "73",
    "77",
    "82",
    "86",
    "95",
    "100"
  ], i = {}, l = {};
  return a.forEach((d, c) => {
    i[d] = `hsl(${e}, ${r}%, ${n[a.length - 1 - c]}%)`, l[d] = `hsl(${e}, ${r}%, ${n[c]}%)`;
  }), t ? l : i;
}
const ut = {
  base: p(225, 15),
  accent: p(250, 50),
  working: p(47, 67),
  warning: p(28, 75),
  ok: p(113, 70),
  error: p(0, 76),
  gridLineColor: "rgba(255, 255, 255, 0.2)"
};
function m(e, r = {}) {
  let t = !1;
  function a() {
    if (!t) {
      t = !0;
      return;
    }
    throw new Error(
      "[anatomy] .part(...) should only be called once. Did you mean to use .extend(...) ?"
    );
  }
  function n(...g) {
    a();
    for (const h of g)
      r[h] = c(h);
    return m(e, r);
  }
  function i(...g) {
    for (const h of g)
      h in r || (r[h] = c(h));
    return m(e, r);
  }
  function l() {
    return Object.fromEntries(
      Object.entries(r).map(([h, k]) => [h, k.selector])
    );
  }
  function d() {
    return Object.fromEntries(
      Object.entries(r).map(([h, k]) => [h, k.className])
    );
  }
  function c(g) {
    const T = `chakra-${(["container", "root"].includes(g ?? "") ? [e] : [e, g]).filter(Boolean).join("__")}`;
    return {
      className: T,
      selector: `.${T}`,
      toString: () => g
    };
  }
  return {
    parts: n,
    toPart: c,
    extend: i,
    selectors: l,
    classnames: d,
    get keys() {
      return Object.keys(r);
    },
    __type: {}
  };
}
var ud = m("accordion").parts("root", "container", "button", "panel").extend("icon");
m("alert").parts("title", "description", "container").extend("icon", "spinner");
m("avatar").parts("label", "badge", "container").extend("excessLabel", "group");
m("breadcrumb").parts("link", "item", "container").extend("separator");
m("button").parts();
var vd = m("checkbox").parts("control", "icon", "container").extend("label");
m("progress").parts("track", "filledTrack").extend("label");
m("drawer").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer");
m("editable").parts(
  "preview",
  "input",
  "textarea"
);
m("form").parts(
  "container",
  "requiredIndicator",
  "helperText"
);
m("formError").parts("text", "icon");
var fd = m("input").parts("addon", "field", "element");
m("list").parts("container", "item", "icon");
var md = m("menu").parts("button", "list", "item").extend("groupTitle", "command", "divider"), gd = m("modal").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer"), hd = m("numberinput").parts(
  "root",
  "field",
  "stepperGroup",
  "stepper"
);
m("pininput").parts("field");
var pd = m("popover").parts("content", "header", "body", "footer").extend("popper", "arrow", "closeButton"), yd = m("progress").parts(
  "label",
  "filledTrack",
  "track"
);
m("radio").parts(
  "container",
  "control",
  "label"
);
var Sd = m("select").parts("field", "icon"), xd = m("slider").parts(
  "container",
  "track",
  "thumb",
  "filledTrack",
  "mark"
);
m("stat").parts(
  "container",
  "label",
  "helpText",
  "number",
  "icon"
);
var $d = m("switch").parts(
  "container",
  "track",
  "thumb"
);
m("table").parts(
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "tfoot",
  "caption"
);
var kd = m("tabs").parts(
  "root",
  "tab",
  "tablist",
  "tabpanel",
  "tabpanels",
  "indicator"
);
m("tag").parts(
  "container",
  "label",
  "closeButton"
);
m("card").parts(
  "container",
  "header",
  "body",
  "footer"
);
const { definePartsStyle: _d, defineMultiStyleConfig: wd } = b(ud.keys), Cd = o({
  border: "none"
}), zd = o((e) => {
  const { colorScheme: r } = e;
  return {
    fontWeight: "600",
    fontSize: "sm",
    border: "none",
    borderRadius: "base",
    bg: `${r}.800`,
    color: "base.100",
    _hover: {
      bg: `${r}.700`
    },
    _expanded: {
      bg: `${r}.750`,
      borderBottomRadius: "none",
      _hover: {
        bg: `${r}.700`
      }
    }
  };
}), Ad = o((e) => {
  const { colorScheme: r } = e;
  return {
    bg: `${r}.800`,
    borderRadius: "base",
    borderTopRadius: "none"
  };
}), Pd = o({}), Td = _d((e) => ({
  container: Cd,
  button: zd(e),
  panel: Ad(e),
  icon: Pd
})), Fd = wd({
  variants: { invokeAI: Td },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "base"
  }
}), Id = o((e) => {
  const { colorScheme: r } = e, t = {
    bg: `${r}.600`,
    color: `${r}.100`,
    svg: {
      fill: `${r}.100`
    }
  };
  return {
    bg: `${r}.700`,
    color: `${r}.100`,
    borderRadius: "base",
    svg: {
      fill: `${r}.100`
    },
    _disabled: t,
    _hover: {
      bg: `${r}.650`,
      color: `${r}.50`,
      svg: {
        fill: `${r}.50`
      },
      _disabled: t
    },
    _checked: {
      bg: "accent.700",
      color: "accent.100",
      svg: {
        fill: "accent.100"
      },
      _disabled: t,
      _hover: {
        bg: "accent.600",
        color: "accent.50",
        svg: {
          fill: "accent.50"
        },
        _disabled: t
      }
    }
  };
}), Bd = S({
  variants: {
    invokeAI: Id
  },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "base"
  }
}), { definePartsStyle: Md, defineMultiStyleConfig: Ed } = b(vd.keys), Dd = o((e) => {
  const { colorScheme: r } = e;
  return {
    _checked: {
      bg: `${r}.200`,
      borderColor: `${r}.200`,
      color: "base.900",
      _hover: {
        bg: `${r}.300`,
        borderColor: `${r}.300`
      },
      _disabled: {
        borderColor: "transparent",
        bg: "whiteAlpha.300",
        color: "whiteAlpha.500"
      }
    },
    _indeterminate: {
      bg: `${r}.200`,
      borderColor: `${r}.200`,
      color: "base.900"
    },
    _disabled: {
      bg: "whiteAlpha.100",
      borderColor: "transparent"
    },
    _focusVisible: {
      boxShadow: "outline"
    },
    _invalid: {
      borderColor: "red.300"
    }
  };
}), jd = Md((e) => ({
  control: Dd(e)
})), Rd = Ed({
  variants: {
    invokeAI: jd
  },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "accent"
  }
}), Wd = o((e) => ({
  fontSize: "sm",
  marginEnd: 0,
  mb: 1,
  fontWeight: "400",
  transitionProperty: "common",
  transitionDuration: "normal",
  whiteSpace: "nowrap",
  _disabled: {
    opacity: 0.4
  },
  color: "base.300"
})), Hd = S({
  variants: {
    invokeAI: Wd
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), Ee = (e) => ({
  outline: "none",
  borderWidth: 2,
  borderStyle: "solid",
  borderColor: "base.800",
  bg: "base.900",
  borderRadius: "base",
  color: "base.100",
  boxShadow: "none",
  _hover: {
    borderColor: "base.600"
  },
  _focus: {
    borderColor: "accent.700",
    boxShadow: "none",
    _hover: {
      borderColor: "accent.600"
    }
  },
  _invalid: {
    borderColor: "error.700",
    boxShadow: "none",
    _hover: {
      borderColor: "error.600"
    }
  },
  _disabled: {
    borderColor: "base.700",
    bg: "base.700",
    color: "base.400",
    _hover: {
      borderColor: "base.700"
    }
  },
  _placeholder: {
    color: "base.400"
  }
}), { definePartsStyle: Ld, defineMultiStyleConfig: Vd } = b(fd.keys), Od = Ld((e) => ({
  field: Ee()
})), qd = Vd({
  variants: {
    invokeAI: Od
  },
  defaultProps: {
    size: "sm",
    variant: "invokeAI"
  }
}), { definePartsStyle: Nd, defineMultiStyleConfig: Gd } = b(md.keys), Ud = Nd({
  // define the part you're going to style
  button: {
    // this will style the MenuButton component
    fontWeight: "600",
    bg: "base.500",
    color: "base.200",
    _hover: {
      bg: "base.600",
      color: "white"
    }
  },
  list: {
    zIndex: 9999,
    bg: "base.800"
  },
  item: {
    // this will style the MenuItem and MenuItemOption components
    fontSize: "sm",
    bg: "base.800",
    _hover: {
      bg: "base.750"
    },
    _focus: {
      bg: "base.700"
    }
  }
}), Yd = Gd({
  variants: {
    invokeAI: Ud
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), { defineMultiStyleConfig: Xd, definePartsStyle: Kd } = b(gd.keys), Zd = o({
  bg: "blackAlpha.600"
}), Jd = o({}), Qd = o((e) => ({
  bg: "base.850",
  maxH: "80vh"
})), ec = o((e) => ({
  fontWeight: "600",
  fontSize: "lg",
  color: "base.200"
})), rc = o({}), tc = o({
  overflowY: "scroll"
}), oc = o({}), ac = Kd((e) => ({
  overlay: Zd,
  dialogContainer: Jd,
  dialog: Qd(e),
  header: ec(e),
  closeButton: rc,
  body: tc,
  footer: oc
})), nc = Xd({
  variants: {
    invokeAI: ac
  },
  defaultProps: { variant: "invokeAI", size: "lg" }
}), { defineMultiStyleConfig: ic, definePartsStyle: lc } = b(hd.keys), sc = o((e) => ({
  height: 8
})), dc = o((e) => ({
  border: "none",
  fontWeight: "600",
  height: "auto",
  py: 1,
  ps: 2,
  pe: 6,
  ...Ee()
})), cc = o((e) => ({
  display: "flex"
})), bc = o((e) => ({
  border: "none",
  // expand arrow hitbox
  px: 2,
  py: 0,
  mx: -2,
  my: 0,
  svg: {
    color: "base.300",
    width: 2.5,
    height: 2.5,
    _hover: {
      color: "base.50"
    }
  }
})), uc = lc((e) => ({
  root: sc(e),
  field: dc(e),
  stepperGroup: cc(e),
  stepper: bc(e)
})), vc = ic({
  variants: {
    invokeAI: uc
  },
  defaultProps: {
    size: "sm",
    variant: "invokeAI"
  }
});
function fc(e) {
  return !Number.isInteger(parseFloat(e.toString()));
}
function mc(e, r = "-") {
  return e.replace(/\s+/g, r);
}
function vt(e) {
  const r = mc(e.toString());
  return r.includes("\\.") ? e : fc(e) ? r.replace(".", "\\.") : e;
}
function gc(e, r = "") {
  return [r, vt(e)].filter(Boolean).join("-");
}
function hc(e, r) {
  return `var(${vt(e)}${r ? `, ${r}` : ""})`;
}
function pc(e, r = "") {
  return `--${gc(e, r)}`;
}
function nr(e, r) {
  const t = pc(e, r == null ? void 0 : r.prefix);
  return {
    variable: t,
    reference: hc(t, yc(r == null ? void 0 : r.fallback))
  };
}
function yc(e) {
  return typeof e == "string" ? e : e == null ? void 0 : e.reference;
}
const { defineMultiStyleConfig: Sc, definePartsStyle: xc } = b(pd.keys), $c = nr("popper-bg"), kc = nr("popper-arrow-bg"), _c = nr("popper-arrow-shadow-color"), wc = o((e) => ({
  [kc.variable]: "colors.base.800",
  [$c.variable]: "colors.base.800",
  [_c.variable]: "colors.base.600",
  minW: "unset",
  width: "unset",
  p: 4,
  borderWidth: "2px",
  borderStyle: "solid",
  borderColor: "base.600",
  bg: "base.800"
})), Cc = xc((e) => ({
  content: wc(e)
})), zc = Sc({
  variants: {
    invokeAI: Cc
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), { defineMultiStyleConfig: Ac, definePartsStyle: Pc } = b(yd.keys), Tc = o((e) => ({
  bg: "accent.600",
  transition: "width 0.2s ease-in-out",
  _indeterminate: {
    bgGradient: "linear(to-r, transparent 0%, accent.600 50%, transparent 100%);"
  }
})), Fc = o((e) => ({
  bg: "base.800"
})), Ic = Pc((e) => ({
  filledTrack: Tc(e),
  track: Fc(e)
})), Bc = Ac({
  variants: {
    invokeAI: Ic
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), Mc = {
  "::-webkit-scrollbar": {
    display: "none"
  },
  scrollbarWidth: "none"
}, { definePartsStyle: Ec, defineMultiStyleConfig: Dc } = b(Sd.keys), jc = o((e) => ({
  color: "base.300"
})), Rc = o((e) => ({
  fontWeight: "600",
  ...Ee()
})), Wc = Ec((e) => ({
  field: Rc(e),
  icon: jc(e)
})), Hc = Dc({
  variants: {
    invokeAI: Wc
  },
  defaultProps: {
    size: "sm",
    variant: "invokeAI"
  }
}), { definePartsStyle: Lc, defineMultiStyleConfig: Vc } = b(xd.keys), Oc = o((e) => ({
  bg: "base.400",
  h: 1.5
})), qc = o((e) => {
  const { colorScheme: r } = e;
  return {
    bg: `${r}.600`,
    h: 1.5
  };
}), Nc = o((e) => ({
  w: 2,
  h: 4
})), Gc = o((e) => ({
  fontSize: "xs",
  fontWeight: "500",
  color: "base.200",
  mt: 2,
  insetInlineStart: "unset"
})), Uc = Lc((e) => ({
  track: Oc(e),
  filledTrack: qc(e),
  thumb: Nc(e),
  mark: Gc(e)
})), Yc = Vc({
  variants: { invokeAI: Uc },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "accent"
  }
}), { defineMultiStyleConfig: Xc, definePartsStyle: Kc } = b($d.keys), Zc = o((e) => {
  const { colorScheme: r } = e;
  return {
    bg: "base.600",
    _focusVisible: {
      boxShadow: "none"
    },
    _checked: {
      bg: `${r}.600`
    }
  };
}), Jc = o((e) => {
  const { colorScheme: r } = e;
  return {
    bg: `${r}.50`
  };
}), Qc = Kc((e) => ({
  container: {},
  track: Zc(e),
  thumb: Jc(e)
})), e0 = Xc({
  variants: { invokeAI: Qc },
  defaultProps: {
    size: "md",
    variant: "invokeAI",
    colorScheme: "accent"
  }
}), { defineMultiStyleConfig: r0, definePartsStyle: t0 } = b(kd.keys), o0 = o((e) => ({
  display: "flex",
  columnGap: 4
})), a0 = o((e) => ({})), n0 = o((e) => ({
  display: "flex",
  flexDirection: "column",
  gap: 1,
  color: "base.700",
  button: {
    fontSize: "sm",
    padding: 2,
    borderRadius: "base",
    _selected: {
      bg: "accent.700",
      color: "accent.100",
      _hover: {
        bg: "accent.600",
        color: "accent.50"
      }
    },
    _hover: {
      bg: "base.600",
      color: "base.50"
    }
  }
})), i0 = o((e) => ({
  padding: 0,
  height: "100%"
})), l0 = t0((e) => ({
  root: o0(e),
  tab: a0(e),
  tablist: n0(e),
  tabpanel: i0(e)
})), s0 = r0({
  variants: {
    invokeAI: l0
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), d0 = o((e) => ({
  color: "base.400"
})), c0 = S({
  variants: {
    subtext: d0
  }
}), b0 = o((e) => Ee()), u0 = S({
  variants: {
    invokeAI: b0
  },
  defaultProps: {
    size: "md",
    variant: "invokeAI"
  }
}), v0 = {
  config: {
    cssVarPrefix: "invokeai"
  },
  styles: {
    global: (e) => ({
      body: {
        bg: "base.900",
        color: "base.50",
        overflow: "hidden"
      },
      "*": { ...Mc }
    })
  },
  direction: "ltr",
  fonts: {
    body: "'Inter', sans-serif"
  },
  shadows: {
    light: {
      accent: "0 0 10px 0 var(--invokeai-colors-accent-300)",
      accentHover: "0 0 10px 0 var(--invokeai-colors-accent-400)",
      ok: "0 0 7px var(--invokeai-colors-ok-600)",
      working: "0 0 7px var(--invokeai-colors-working-600)",
      error: "0 0 7px var(--invokeai-colors-error-600)"
    },
    dark: {
      accent: "0 0 10px 0 var(--invokeai-colors-accent-600)",
      accentHover: "0 0 10px 0 var(--invokeai-colors-accent-500)",
      ok: "0 0 7px var(--invokeai-colors-ok-400)",
      working: "0 0 7px var(--invokeai-colors-working-400)",
      error: "0 0 7px var(--invokeai-colors-error-400)"
    }
  },
  colors: {
    ...ut
  },
  components: {
    Button: Bd,
    // Button and IconButton
    Input: qd,
    Textarea: u0,
    Tabs: s0,
    Progress: Bc,
    Accordion: Fd,
    FormLabel: Hd,
    Switch: e0,
    NumberInput: vc,
    Select: Hc,
    Slider: Yc,
    Popover: zc,
    Modal: nc,
    Checkbox: Rd,
    Menu: Yd,
    Text: c0
  }
}, f0 = {
  base: p(223, 10),
  accent: p(155, 80),
  working: p(47, 68),
  warning: p(28, 75),
  ok: p(122, 49),
  error: p(0, 50),
  gridLineColor: "rgba(255, 255, 255, 0.2)"
}, m0 = {
  base: p(223, 10, !0),
  accent: p(40, 80, !0),
  working: p(47, 68, !0),
  warning: p(28, 75, !0),
  ok: p(122, 49, !0),
  error: p(0, 50, !0),
  gridLineColor: "rgba(0, 0, 0, 0.2)"
}, g0 = {
  base: p(220, 30),
  accent: p(210, 80),
  working: p(47, 68),
  warning: p(28, 75),
  ok: p(122, 49),
  error: p(0, 100),
  gridLineColor: "rgba(136, 148, 184, 0.2)"
}, h0 = {
  dark: ut,
  light: m0,
  green: f0,
  ocean: g0
};
function $0({ children: e }) {
  const { i18n: r } = _t(), t = wt(
    (i) => i.ui.currentTheme
  ), a = r.dir(), n = od({
    ...v0,
    colors: h0[t],
    direction: a
  });
  return Xe(() => {
    document.body.dir = a;
  }, [a]), /* @__PURE__ */ y.jsx(bd, { theme: n, children: e });
}
export {
  $0 as default
};
