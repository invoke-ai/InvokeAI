import { j as y, cI as Ie, cJ as mt, q as qr, cK as o, cL as b, cM as v, cN as S, cO as Nr, cP as ht, cQ as gt, cR as pt, cS as yt, n as St, cT as xt, E as $t } from "./component-6919b100.mjs";
import { d as kt, i as _t, T as wt, j as Ct, s as zt, g as At } from "./scrollbar-d87b3f9c.mjs";
import { useState as sr, useMemo as Ae, useCallback as dr, useEffect as Ke, createContext as Gr, useContext as Pt, useRef as Tt } from "react";
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
`, Ft = () => /* @__PURE__ */ y.jsx(Ie, { styles: Ur }), It = () => /* @__PURE__ */ y.jsx(
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
), ge = {
  light: "chakra-ui-light",
  dark: "chakra-ui-dark"
};
function Bt(e = {}) {
  const { preventTransition: r = !0 } = e, t = {
    setDataset: (a) => {
      const n = r ? t.preventTransition() : void 0;
      document.documentElement.dataset.theme = a, document.documentElement.style.colorScheme = a, n == null || n();
    },
    setClassName(a) {
      document.body.classList.add(a ? ge.dark : ge.light), document.body.classList.remove(a ? ge.light : ge.dark);
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
var Mt = "chakra-ui-color-mode";
function Et(e) {
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
var Dt = Et(Mt), cr = () => {
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
    colorModeManager: l = Dt
  } = e, d = n === "dark" ? "dark" : "light", [c, $] = sr(
    () => br(l, d)
  ), [h, g] = sr(
    () => br(l)
  ), { getSystemTheme: k, setClassName: T, setDataset: fe, addListener: ir } = Ae(
    () => Bt({ preventTransition: i }),
    [i]
  ), me = n === "system" && !c ? h : c, F = dr(
    (ee) => {
      const he = ee === "system" ? k() : ee;
      $(he), T(he === "dark"), fe(he), l.set(he);
    },
    [l, k, T, fe]
  );
  kt(() => {
    n === "system" && g(k());
  }, []), Ke(() => {
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
  Ke(() => {
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
  return /* @__PURE__ */ y.jsx(mt.Provider, { value: ft, children: t });
}
Yr.displayName = "ColorModeProvider";
var jt = [
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
function Rt(e) {
  return qr(e) ? jt.every(
    (r) => Object.prototype.hasOwnProperty.call(e, r)
  ) : !1;
}
var Wt = {
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
}, Ht = Wt, Lt = {
  base: "0em",
  sm: "30em",
  md: "48em",
  lg: "62em",
  xl: "80em",
  "2xl": "96em"
}, Vt = Lt, Ot = {
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
}, qt = Ot, Nt = {
  none: "0",
  sm: "0.125rem",
  base: "0.25rem",
  md: "0.375rem",
  lg: "0.5rem",
  xl: "0.75rem",
  "2xl": "1rem",
  "3xl": "1.5rem",
  full: "9999px"
}, Gt = Nt, Ut = {
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
}, Yt = Ut, Kt = {
  common: "background-color, border-color, color, fill, stroke, opacity, box-shadow, transform",
  colors: "background-color, border-color, color, fill, stroke",
  dimensions: "width, height",
  position: "left, right, top, bottom",
  background: "background-color, background-image, background-position"
}, Xt = {
  "ease-in": "cubic-bezier(0.4, 0, 1, 1)",
  "ease-out": "cubic-bezier(0, 0, 0.2, 1)",
  "ease-in-out": "cubic-bezier(0.4, 0, 0.2, 1)"
}, Jt = {
  "ultra-fast": "50ms",
  faster: "100ms",
  fast: "150ms",
  normal: "200ms",
  slow: "300ms",
  slower: "400ms",
  "ultra-slow": "500ms"
}, Qt = {
  property: Kt,
  easing: Xt,
  duration: Jt
}, Zt = Qt, eo = {
  none: 0,
  sm: "4px",
  base: "8px",
  md: "12px",
  lg: "16px",
  xl: "24px",
  "2xl": "40px",
  "3xl": "64px"
}, ro = eo, to = {
  none: 0,
  "1px": "1px solid",
  "2px": "2px solid",
  "4px": "4px solid",
  "8px": "8px solid"
}, oo = to, ao = {
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
}, Kr = ao, Xr = {
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
}, no = {
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
}, io = {
  sm: "640px",
  md: "768px",
  lg: "1024px",
  xl: "1280px"
}, lo = {
  ...Xr,
  ...no,
  container: io
}, Jr = lo, so = {
  breakpoints: Vt,
  zIndices: Ht,
  radii: Gt,
  blur: ro,
  colors: qt,
  ...Kr,
  sizes: Jr,
  shadows: Yt,
  space: Xr,
  borders: oo,
  transition: Zt
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
  function n(...h) {
    a();
    for (const g of h)
      r[g] = c(g);
    return f(e, r);
  }
  function i(...h) {
    for (const g of h)
      g in r || (r[g] = c(g));
    return f(e, r);
  }
  function l() {
    return Object.fromEntries(
      Object.entries(r).map(([g, k]) => [g, k.selector])
    );
  }
  function d() {
    return Object.fromEntries(
      Object.entries(r).map(([g, k]) => [g, k.className])
    );
  }
  function c(h) {
    const T = `chakra-${(["container", "root"].includes(h ?? "") ? [e] : [e, h]).filter(Boolean).join("__")}`;
    return {
      className: T,
      selector: `.${T}`,
      toString: () => h
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
var co = f("accordion").parts("root", "container", "button", "panel").extend("icon"), bo = f("alert").parts("title", "description", "container").extend("icon", "spinner"), uo = f("avatar").parts("label", "badge", "container").extend("excessLabel", "group"), vo = f("breadcrumb").parts("link", "item", "container").extend("separator");
f("button").parts();
var fo = f("checkbox").parts("control", "icon", "container").extend("label");
f("progress").parts("track", "filledTrack").extend("label");
var mo = f("drawer").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer"), ho = f("editable").parts(
  "preview",
  "input",
  "textarea"
), go = f("form").parts(
  "container",
  "requiredIndicator",
  "helperText"
), po = f("formError").parts("text", "icon"), yo = f("input").parts("addon", "field", "element"), So = f("list").parts("container", "item", "icon"), xo = f("menu").parts("button", "list", "item").extend("groupTitle", "command", "divider"), $o = f("modal").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer"), ko = f("numberinput").parts(
  "root",
  "field",
  "stepperGroup",
  "stepper"
);
f("pininput").parts("field");
var _o = f("popover").parts("content", "header", "body", "footer").extend("popper", "arrow", "closeButton"), wo = f("progress").parts(
  "label",
  "filledTrack",
  "track"
), Co = f("radio").parts(
  "container",
  "control",
  "label"
), zo = f("select").parts("field", "icon"), Ao = f("slider").parts(
  "container",
  "track",
  "thumb",
  "filledTrack",
  "mark"
), Po = f("stat").parts(
  "container",
  "label",
  "helpText",
  "number",
  "icon"
), To = f("switch").parts(
  "container",
  "track",
  "thumb"
), Fo = f("table").parts(
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
  "tfoot",
  "caption"
), Io = f("tabs").parts(
  "root",
  "tab",
  "tablist",
  "tabpanel",
  "tabpanels",
  "indicator"
), Bo = f("tag").parts(
  "container",
  "label",
  "closeButton"
), Mo = f("card").parts(
  "container",
  "header",
  "body",
  "footer"
);
function O(e, r, t) {
  return Math.min(Math.max(e, t), r);
}
class Eo extends Error {
  constructor(r) {
    super(`Failed to parse color: "${r}"`);
  }
}
var oe = Eo;
function er(e) {
  if (typeof e != "string")
    throw new oe(e);
  if (e.trim().toLowerCase() === "transparent")
    return [0, 0, 0, 0];
  let r = e.trim();
  r = Oo.test(e) ? Ro(e) : e;
  const t = Wo.exec(r);
  if (t) {
    const l = Array.from(t).slice(1);
    return [...l.slice(0, 3).map((d) => parseInt(ce(d, 2), 16)), parseInt(ce(l[3] || "f", 2), 16) / 255];
  }
  const a = Ho.exec(r);
  if (a) {
    const l = Array.from(a).slice(1);
    return [...l.slice(0, 3).map((d) => parseInt(d, 16)), parseInt(l[3] || "ff", 16) / 255];
  }
  const n = Lo.exec(r);
  if (n) {
    const l = Array.from(n).slice(1);
    return [...l.slice(0, 3).map((d) => parseInt(d, 10)), parseFloat(l[3] || "1")];
  }
  const i = Vo.exec(r);
  if (i) {
    const [l, d, c, $] = Array.from(i).slice(1).map(parseFloat);
    if (O(0, 100, d) !== d)
      throw new oe(e);
    if (O(0, 100, c) !== c)
      throw new oe(e);
    return [...qo(l, d, c), Number.isNaN($) ? 1 : $];
  }
  throw new oe(e);
}
function Do(e) {
  let r = 5381, t = e.length;
  for (; t; )
    r = r * 33 ^ e.charCodeAt(--t);
  return (r >>> 0) % 2341;
}
const ur = (e) => parseInt(e.replace(/_/g, ""), 36), jo = "1q29ehhb 1n09sgk7 1kl1ekf_ _yl4zsno 16z9eiv3 1p29lhp8 _bd9zg04 17u0____ _iw9zhe5 _to73___ _r45e31e _7l6g016 _jh8ouiv _zn3qba8 1jy4zshs 11u87k0u 1ro9yvyo 1aj3xael 1gz9zjz0 _3w8l4xo 1bf1ekf_ _ke3v___ _4rrkb__ 13j776yz _646mbhl _nrjr4__ _le6mbhl 1n37ehkb _m75f91n _qj3bzfz 1939yygw 11i5z6x8 _1k5f8xs 1509441m 15t5lwgf _ae2th1n _tg1ugcv 1lp1ugcv 16e14up_ _h55rw7n _ny9yavn _7a11xb_ 1ih442g9 _pv442g9 1mv16xof 14e6y7tu 1oo9zkds 17d1cisi _4v9y70f _y98m8kc 1019pq0v 12o9zda8 _348j4f4 1et50i2o _8epa8__ _ts6senj 1o350i2o 1mi9eiuo 1259yrp0 1ln80gnw _632xcoy 1cn9zldc _f29edu4 1n490c8q _9f9ziet 1b94vk74 _m49zkct 1kz6s73a 1eu9dtog _q58s1rz 1dy9sjiq __u89jo3 _aj5nkwg _ld89jo3 13h9z6wx _qa9z2ii _l119xgq _bs5arju 1hj4nwk9 1qt4nwk9 1ge6wau6 14j9zlcw 11p1edc_ _ms1zcxe _439shk6 _jt9y70f _754zsow 1la40eju _oq5p___ _x279qkz 1fa5r3rv _yd2d9ip _424tcku _8y1di2_ _zi2uabw _yy7rn9h 12yz980_ __39ljp6 1b59zg0x _n39zfzp 1fy9zest _b33k___ _hp9wq92 1il50hz4 _io472ub _lj9z3eo 19z9ykg0 _8t8iu3a 12b9bl4a 1ak5yw0o _896v4ku _tb8k8lv _s59zi6t _c09ze0p 1lg80oqn 1id9z8wb _238nba5 1kq6wgdi _154zssg _tn3zk49 _da9y6tc 1sg7cv4f _r12jvtt 1gq5fmkz 1cs9rvci _lp9jn1c _xw1tdnb 13f9zje6 16f6973h _vo7ir40 _bt5arjf _rc45e4t _hr4e100 10v4e100 _hc9zke2 _w91egv_ _sj2r1kk 13c87yx8 _vqpds__ _ni8ggk8 _tj9yqfb 1ia2j4r4 _7x9b10u 1fc9ld4j 1eq9zldr _5j9lhpx _ez9zl6o _md61fzm".split(" ").reduce((e, r) => {
  const t = ur(r.substring(0, 3)), a = ur(r.substring(3)).toString(16);
  let n = "";
  for (let i = 0; i < 6 - a.length; i++)
    n += "0";
  return e[t] = `${n}${a}`, e;
}, {});
function Ro(e) {
  const r = e.toLowerCase().trim(), t = jo[Do(r)];
  if (!t)
    throw new oe(e);
  return `#${t}`;
}
const ce = (e, r) => Array.from(Array(r)).map(() => e).join(""), Wo = new RegExp(`^#${ce("([a-f0-9])", 3)}([a-f0-9])?$`, "i"), Ho = new RegExp(`^#${ce("([a-f0-9]{2})", 3)}([a-f0-9]{2})?$`, "i"), Lo = new RegExp(`^rgba?\\(\\s*(\\d+)\\s*${ce(",\\s*(\\d+)\\s*", 2)}(?:,\\s*([\\d.]+))?\\s*\\)$`, "i"), Vo = /^hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%(?:\s*,\s*([\d.]+))?\s*\)$/i, Oo = /^[a-z]+$/i, vr = (e) => Math.round(e * 255), qo = (e, r, t) => {
  let a = t / 100;
  if (r === 0)
    return [a, a, a].map(vr);
  const n = (e % 360 + 360) % 360 / 60, i = (1 - Math.abs(2 * a - 1)) * (r / 100), l = i * (1 - Math.abs(n % 2 - 1));
  let d = 0, c = 0, $ = 0;
  n >= 0 && n < 1 ? (d = i, c = l) : n >= 1 && n < 2 ? (d = l, c = i) : n >= 2 && n < 3 ? (c = i, $ = l) : n >= 3 && n < 4 ? (c = l, $ = i) : n >= 4 && n < 5 ? (d = l, $ = i) : n >= 5 && n < 6 && (d = i, $ = l);
  const h = a - i / 2, g = d + h, k = c + h, T = $ + h;
  return [g, k, T].map(vr);
};
function No(e, r, t, a) {
  return `rgba(${O(0, 255, e).toFixed()}, ${O(0, 255, r).toFixed()}, ${O(0, 255, t).toFixed()}, ${parseFloat(O(0, 1, a).toFixed(3))})`;
}
function Go(e, r) {
  const [t, a, n, i] = er(e);
  return No(t, a, n, i - r);
}
function Uo(e) {
  const [r, t, a, n] = er(e);
  let i = (l) => {
    const d = O(0, 255, l).toString(16);
    return d.length === 1 ? `0${d}` : d;
  };
  return `#${i(r)}${i(t)}${i(a)}${n < 1 ? i(Math.round(n * 255)) : ""}`;
}
function Yo(e, r, t, a, n) {
  for (r = r.split ? r.split(".") : r, a = 0; a < r.length; a++)
    e = e ? e[r[a]] : n;
  return e === n ? t : e;
}
var Ko = (e) => Object.keys(e).length === 0, _ = (e, r, t) => {
  const a = Yo(e, `colors.${r}`, r);
  try {
    return Uo(a), a;
  } catch {
    return t ?? "#000000";
  }
}, Xo = (e) => {
  const [r, t, a] = er(e);
  return (r * 299 + t * 587 + a * 114) / 1e3;
}, Jo = (e) => (r) => {
  const t = _(r, e);
  return Xo(t) < 128 ? "dark" : "light";
}, Qo = (e) => (r) => Jo(e)(r) === "dark", Z = (e, r) => (t) => {
  const a = _(t, e);
  return Go(a, 1 - r);
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
var Zo = () => `#${Math.floor(Math.random() * 16777215).toString(16).padEnd(6, "0")}`;
function ea(e) {
  const r = Zo();
  return !e || Ko(e) ? r : e.string && e.colors ? ta(e.string, e.colors) : e.string && !e.colors ? ra(e.string) : e.colors && !e.string ? oa(e.colors) : r;
}
function ra(e) {
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
function ta(e, r) {
  let t = 0;
  if (e.length === 0)
    return r[0];
  for (let a = 0; a < e.length; a += 1)
    t = e.charCodeAt(a) + ((t << 5) - t), t = t & t;
  return t = (t % r.length + r.length) % r.length, r[t];
}
function oa(e) {
  return e[Math.floor(Math.random() * e.length)];
}
function s(e, r) {
  return (t) => t.colorMode === "dark" ? r : e;
}
function rr(e) {
  const { orientation: r, vertical: t, horizontal: a } = e;
  return r ? r === "vertical" ? t : a : {};
}
function Qr(e) {
  return qr(e) && e.reference ? e.reference : String(e);
}
var Be = (e, ...r) => r.map(Qr).join(` ${e} `).replace(/calc/g, ""), mr = (...e) => `calc(${Be("+", ...e)})`, hr = (...e) => `calc(${Be("-", ...e)})`, Xe = (...e) => `calc(${Be("*", ...e)})`, gr = (...e) => `calc(${Be("/", ...e)})`, pr = (e) => {
  const r = Qr(e);
  return r != null && !Number.isNaN(parseFloat(r)) ? String(r).startsWith("-") ? String(r).slice(1) : `-${r}` : Xe(r, -1);
}, E = Object.assign(
  (e) => ({
    add: (...r) => E(mr(e, ...r)),
    subtract: (...r) => E(hr(e, ...r)),
    multiply: (...r) => E(Xe(e, ...r)),
    divide: (...r) => E(gr(e, ...r)),
    negate: () => E(pr(e)),
    toString: () => e.toString()
  }),
  {
    add: mr,
    subtract: hr,
    multiply: Xe,
    divide: gr,
    negate: pr
  }
);
function aa(e) {
  return !Number.isInteger(parseFloat(e.toString()));
}
function na(e, r = "-") {
  return e.replace(/\s+/g, r);
}
function Zr(e) {
  const r = na(e.toString());
  return r.includes("\\.") ? e : aa(e) ? r.replace(".", "\\.") : e;
}
function ia(e, r = "") {
  return [r, Zr(e)].filter(Boolean).join("-");
}
function la(e, r) {
  return `var(${Zr(e)}${r ? `, ${r}` : ""})`;
}
function sa(e, r = "") {
  return `--${ia(e, r)}`;
}
function x(e, r) {
  const t = sa(e, r == null ? void 0 : r.prefix);
  return {
    variable: t,
    reference: la(t, da(r == null ? void 0 : r.fallback))
  };
}
function da(e) {
  return typeof e == "string" ? e : e == null ? void 0 : e.reference;
}
var { defineMultiStyleConfig: ca, definePartsStyle: _e } = b(To.keys), ie = x("switch-track-width"), q = x("switch-track-height"), De = x("switch-track-diff"), ba = E.subtract(ie, q), Je = x("switch-thumb-x"), re = x("switch-bg"), ua = o((e) => {
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
}), va = o({
  bg: "white",
  transitionProperty: "transform",
  transitionDuration: "normal",
  borderRadius: "inherit",
  width: [q.reference],
  height: [q.reference],
  _checked: {
    transform: `translateX(${Je.reference})`
  }
}), fa = _e((e) => ({
  container: {
    [De.variable]: ba,
    [Je.variable]: De.reference,
    _rtl: {
      [Je.variable]: E(De).negate().toString()
    }
  },
  track: ua(e),
  thumb: va
})), ma = {
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
}, ha = ca({
  baseStyle: fa,
  sizes: ma,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), { defineMultiStyleConfig: ga, definePartsStyle: Y } = b(Fo.keys), pa = Y({
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
    tfoot: {
      tr: {
        "&:last-of-type": {
          th: { borderBottomWidth: 0 }
        }
      }
    }
  };
}), Sa = Y((e) => {
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
}), xa = {
  simple: ya,
  striped: Sa,
  unstyled: o({})
}, $a = {
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
}, ka = ga({
  baseStyle: pa,
  variants: xa,
  sizes: $a,
  defaultProps: {
    variant: "simple",
    size: "md",
    colorScheme: "gray"
  }
}), C = v("tabs-color"), P = v("tabs-bg"), pe = v("tabs-border-color"), { defineMultiStyleConfig: _a, definePartsStyle: M } = b(Io.keys), wa = o((e) => {
  const { orientation: r } = e;
  return {
    display: r === "vertical" ? "flex" : "block"
  };
}), Ca = o((e) => {
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
}), za = o((e) => {
  const { align: r = "start", orientation: t } = e;
  return {
    justifyContent: {
      end: "flex-end",
      center: "center",
      start: "flex-start"
    }[r],
    flexDirection: t === "vertical" ? "column" : "row"
  };
}), Aa = o({
  p: 4
}), Pa = M((e) => ({
  root: wa(e),
  tab: Ca(e),
  tablist: za(e),
  tabpanel: Aa
})), Ta = {
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
}, Fa = M((e) => {
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
}), Ia = M((e) => {
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
}), Ba = M((e) => {
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
}), Ma = M((e) => {
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
}), Ea = M((e) => {
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
}), Da = M({}), ja = {
  line: Fa,
  enclosed: Ia,
  "enclosed-colored": Ba,
  "soft-rounded": Ma,
  "solid-rounded": Ea,
  unstyled: Da
}, Ra = _a({
  baseStyle: Pa,
  sizes: Ta,
  variants: ja,
  defaultProps: {
    size: "md",
    variant: "line",
    colorScheme: "blue"
  }
}), Wa = o({
  px: 1,
  textTransform: "uppercase",
  fontSize: "xs",
  borderRadius: "sm",
  fontWeight: "bold"
}), K = v("badge-bg"), B = v("badge-color"), Ha = o((e) => {
  const { colorScheme: r, theme: t } = e, a = Z(`${r}.500`, 0.6)(t);
  return {
    [K.variable]: `colors.${r}.500`,
    [B.variable]: "colors.white",
    _dark: {
      [K.variable]: a,
      [B.variable]: "colors.whiteAlpha.800"
    },
    bg: K.reference,
    color: B.reference
  };
}), La = o((e) => {
  const { colorScheme: r, theme: t } = e, a = Z(`${r}.200`, 0.16)(t);
  return {
    [K.variable]: `colors.${r}.100`,
    [B.variable]: `colors.${r}.800`,
    _dark: {
      [K.variable]: a,
      [B.variable]: `colors.${r}.200`
    },
    bg: K.reference,
    color: B.reference
  };
}), Va = o((e) => {
  const { colorScheme: r, theme: t } = e, a = Z(`${r}.200`, 0.8)(t);
  return {
    [B.variable]: `colors.${r}.500`,
    _dark: {
      [B.variable]: a
    },
    color: B.reference,
    boxShadow: `inset 0 0 0px 1px ${B.reference}`
  };
}), Oa = {
  solid: Ha,
  subtle: La,
  outline: Va
}, le = S({
  baseStyle: Wa,
  variants: Oa,
  defaultProps: {
    variant: "subtle",
    colorScheme: "gray"
  }
}), { defineMultiStyleConfig: qa, definePartsStyle: N } = b(Bo.keys), Na = o({
  fontWeight: "medium",
  lineHeight: 1.2,
  outline: 0,
  borderRadius: "md",
  _focusVisible: {
    boxShadow: "outline"
  }
}), Ga = o({
  lineHeight: 1.2,
  overflow: "visible"
}), Ua = o({
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
}), Ya = N({
  container: Na,
  label: Ga,
  closeButton: Ua
}), Ka = {
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
}, Ja = qa({
  variants: Xa,
  baseStyle: Ya,
  sizes: Ka,
  defaultProps: {
    size: "md",
    variant: "subtle",
    colorScheme: "gray"
  }
}), { definePartsStyle: D, defineMultiStyleConfig: Qa } = b(yo.keys), Za = D({
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
}, en = {
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
var rn = D((e) => {
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
}), tn = D((e) => {
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
}), on = D((e) => {
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
}), an = D({
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
}), nn = {
  outline: rn,
  filled: tn,
  flushed: on,
  unstyled: an
}, u = Qa({
  baseStyle: Za,
  sizes: en,
  variants: nn,
  defaultProps: {
    size: "md",
    variant: "outline"
  }
}), yr, ln = o({
  ...(yr = u.baseStyle) == null ? void 0 : yr.field,
  paddingY: "2",
  minHeight: "20",
  lineHeight: "short",
  verticalAlign: "top"
}), Sr, xr, sn = {
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
}, $r, kr, _r, wr, Cr, zr, Ar, Pr, dn = {
  xs: (kr = ($r = u.sizes) == null ? void 0 : $r.xs.field) != null ? kr : {},
  sm: (wr = (_r = u.sizes) == null ? void 0 : _r.sm.field) != null ? wr : {},
  md: (zr = (Cr = u.sizes) == null ? void 0 : Cr.md.field) != null ? zr : {},
  lg: (Pr = (Ar = u.sizes) == null ? void 0 : Ar.lg.field) != null ? Pr : {}
}, cn = S({
  baseStyle: ln,
  sizes: dn,
  variants: sn,
  defaultProps: {
    size: "md",
    variant: "outline"
  }
}), ye = x("tooltip-bg"), je = x("tooltip-fg"), bn = x("popper-arrow-bg"), un = o({
  bg: ye.reference,
  color: je.reference,
  [ye.variable]: "colors.gray.700",
  [je.variable]: "colors.whiteAlpha.900",
  _dark: {
    [ye.variable]: "colors.gray.300",
    [je.variable]: "colors.gray.900"
  },
  [bn.variable]: ye.reference,
  px: "2",
  py: "0.5",
  borderRadius: "sm",
  fontWeight: "medium",
  fontSize: "sm",
  boxShadow: "md",
  maxW: "xs",
  zIndex: "tooltip"
}), vn = S({
  baseStyle: un
}), { defineMultiStyleConfig: fn, definePartsStyle: ae } = b(wo.keys), mn = o((e) => {
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
}), hn = o({
  lineHeight: "1",
  fontSize: "0.25em",
  fontWeight: "bold",
  color: "white"
}), gn = o((e) => ({
  bg: s("gray.100", "whiteAlpha.300")(e)
})), pn = o((e) => ({
  transitionProperty: "common",
  transitionDuration: "slow",
  ...mn(e)
})), yn = ae((e) => ({
  label: hn,
  filledTrack: pn(e),
  track: gn(e)
})), Sn = {
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
}, xn = fn({
  sizes: Sn,
  baseStyle: yn,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), $n = (e) => typeof e == "function";
function w(e, ...r) {
  return $n(e) ? e(...r) : e;
}
var { definePartsStyle: we, defineMultiStyleConfig: kn } = b(fo.keys), se = v("checkbox-size"), _n = o((e) => {
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
}), wn = o({
  _disabled: { cursor: "not-allowed" }
}), Cn = o({
  userSelect: "none",
  _disabled: { opacity: 0.4 }
}), zn = o({
  transitionProperty: "transform",
  transitionDuration: "normal"
}), An = we((e) => ({
  icon: zn,
  container: wn,
  control: w(_n, e),
  label: Cn
})), Pn = {
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
}, Te = kn({
  baseStyle: An,
  sizes: Pn,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), { defineMultiStyleConfig: Tn, definePartsStyle: Ce } = b(Co.keys), Fn = o((e) => {
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
}), In = Ce((e) => {
  var r, t, a, n;
  return {
    label: (t = (r = Te).baseStyle) == null ? void 0 : t.call(r, e).label,
    container: (n = (a = Te).baseStyle) == null ? void 0 : n.call(a, e).container,
    control: Fn(e)
  };
}), Bn = {
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
}, Mn = Tn({
  baseStyle: In,
  sizes: Bn,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), { defineMultiStyleConfig: En, definePartsStyle: Dn } = b(zo.keys), Se = v("select-bg"), Tr, jn = o({
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
}), Rn = o({
  width: "6",
  height: "100%",
  insetEnd: "2",
  position: "relative",
  color: "currentColor",
  fontSize: "xl",
  _disabled: {
    opacity: 0.5
  }
}), Wn = Dn({
  field: jn,
  icon: Rn
}), xe = o({
  paddingInlineEnd: "8"
}), Fr, Ir, Br, Mr, Er, Dr, jr, Rr, Hn = {
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
}, Ln = En({
  baseStyle: Wn,
  sizes: Hn,
  variants: u.variants,
  defaultProps: u.defaultProps
}), Re = v("skeleton-start-color"), We = v("skeleton-end-color"), Vn = o({
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
}), On = S({
  baseStyle: Vn
}), He = v("skip-link-bg"), qn = o({
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
}), Nn = S({
  baseStyle: qn
}), { defineMultiStyleConfig: Gn, definePartsStyle: Me } = b(Ao.keys), be = v("slider-thumb-size"), ue = v("slider-track-size"), L = v("slider-bg"), Un = o((e) => {
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
}), Yn = o((e) => ({
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
})), Kn = o((e) => {
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
}), Jn = Me((e) => ({
  container: Un(e),
  track: Yn(e),
  thumb: Kn(e),
  filledTrack: Xn(e)
})), Qn = Me({
  container: {
    [be.variable]: "sizes.4",
    [ue.variable]: "sizes.1"
  }
}), Zn = Me({
  container: {
    [be.variable]: "sizes.3.5",
    [ue.variable]: "sizes.1"
  }
}), ei = Me({
  container: {
    [be.variable]: "sizes.2.5",
    [ue.variable]: "sizes.0.5"
  }
}), ri = {
  lg: Qn,
  md: Zn,
  sm: ei
}, ti = Gn({
  baseStyle: Jn,
  sizes: ri,
  defaultProps: {
    size: "md",
    colorScheme: "blue"
  }
}), V = x("spinner-size"), oi = o({
  width: [V.reference],
  height: [V.reference]
}), ai = {
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
}, ni = S({
  baseStyle: oi,
  sizes: ai,
  defaultProps: {
    size: "md"
  }
}), { defineMultiStyleConfig: ii, definePartsStyle: et } = b(Po.keys), li = o({
  fontWeight: "medium"
}), si = o({
  opacity: 0.8,
  marginBottom: "2"
}), di = o({
  verticalAlign: "baseline",
  fontWeight: "semibold"
}), ci = o({
  marginEnd: 1,
  w: "3.5",
  h: "3.5",
  verticalAlign: "middle"
}), bi = et({
  container: {},
  label: li,
  helpText: si,
  number: di,
  icon: ci
}), ui = {
  md: et({
    label: { fontSize: "sm" },
    helpText: { fontSize: "sm" },
    number: { fontSize: "2xl" }
  })
}, vi = ii({
  baseStyle: bi,
  sizes: ui,
  defaultProps: {
    size: "md"
  }
}), Le = v("kbd-bg"), fi = o({
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
}), mi = S({
  baseStyle: fi
}), hi = o({
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
  baseStyle: hi
}), { defineMultiStyleConfig: pi, definePartsStyle: yi } = b(So.keys), Si = o({
  marginEnd: "2",
  display: "inline",
  verticalAlign: "text-bottom"
}), xi = yi({
  icon: Si
}), $i = pi({
  baseStyle: xi
}), { defineMultiStyleConfig: ki, definePartsStyle: _i } = b(xo.keys), I = v("menu-bg"), Ve = v("menu-shadow"), wi = o({
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
}), Ci = o({
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
}), zi = o({
  mx: 4,
  my: 2,
  fontWeight: "semibold",
  fontSize: "sm"
}), Ai = o({
  opacity: 0.6
}), Pi = o({
  border: 0,
  borderBottom: "1px solid",
  borderColor: "inherit",
  my: "2",
  opacity: 0.6
}), Ti = o({
  transitionProperty: "common",
  transitionDuration: "normal"
}), Fi = _i({
  button: Ti,
  list: wi,
  item: Ci,
  groupTitle: zi,
  command: Ai,
  divider: Pi
}), Ii = ki({
  baseStyle: Fi
}), { defineMultiStyleConfig: Bi, definePartsStyle: Qe } = b($o.keys), Mi = o({
  bg: "blackAlpha.600",
  zIndex: "modal"
}), Ei = o((e) => {
  const { isCentered: r, scrollBehavior: t } = e;
  return {
    display: "flex",
    zIndex: "modal",
    justifyContent: "center",
    alignItems: r ? "center" : "flex-start",
    overflow: t === "inside" ? "hidden" : "auto",
    overscrollBehaviorY: "none"
  };
}), Di = o((e) => {
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
}), ji = o({
  px: "6",
  py: "4",
  fontSize: "xl",
  fontWeight: "semibold"
}), Ri = o({
  position: "absolute",
  top: "2",
  insetEnd: "3"
}), Wi = o((e) => {
  const { scrollBehavior: r } = e;
  return {
    px: "6",
    py: "2",
    flex: "1",
    overflow: r === "inside" ? "auto" : void 0
  };
}), Hi = o({
  px: "6",
  py: "4"
}), Li = Qe((e) => ({
  overlay: Mi,
  dialogContainer: w(Ei, e),
  dialog: w(Di, e),
  header: ji,
  closeButton: Ri,
  body: w(Wi, e),
  footer: Hi
}));
function A(e) {
  return Qe(e === "full" ? {
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
var Vi = {
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
}, Oi = Bi({
  baseStyle: Li,
  sizes: Vi,
  defaultProps: { size: "md" }
}), { defineMultiStyleConfig: qi, definePartsStyle: rt } = b(ko.keys), or = x("number-input-stepper-width"), tt = x("number-input-input-padding"), Ni = E(or).add("0.5rem").toString(), Oe = x("number-input-bg"), qe = x("number-input-color"), Ne = x("number-input-border-color"), Gi = o({
  [or.variable]: "sizes.6",
  [tt.variable]: Ni
}), Ui = o(
  (e) => {
    var r, t;
    return (t = (r = w(u.baseStyle, e)) == null ? void 0 : r.field) != null ? t : {};
  }
), Yi = o({
  width: or.reference
}), Ki = o({
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
    root: Gi,
    field: (r = w(Ui, e)) != null ? r : {},
    stepperGroup: Yi,
    stepper: Ki
  };
});
function $e(e) {
  var r, t, a;
  const n = (r = u.sizes) == null ? void 0 : r[e], i = {
    lg: "md",
    md: "md",
    sm: "sm",
    xs: "sm"
  }, l = (a = (t = n.field) == null ? void 0 : t.fontSize) != null ? a : "md", d = Kr.fontSizes[l];
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
var Ji = {
  xs: $e("xs"),
  sm: $e("sm"),
  md: $e("md"),
  lg: $e("lg")
}, Qi = qi({
  baseStyle: Xi,
  sizes: Ji,
  variants: u.variants,
  defaultProps: u.defaultProps
}), Wr, Zi = o({
  ...(Wr = u.baseStyle) == null ? void 0 : Wr.field,
  textAlign: "center"
}), el = {
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
}, Hr, Lr, rl = {
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
}, tl = S({
  baseStyle: Zi,
  sizes: el,
  variants: rl,
  defaultProps: u.defaultProps
}), { defineMultiStyleConfig: ol, definePartsStyle: al } = b(_o.keys), ke = x("popper-bg"), nl = x("popper-arrow-bg"), Vr = x("popper-arrow-shadow-color"), il = o({ zIndex: 10 }), ll = o({
  [ke.variable]: "colors.white",
  bg: ke.reference,
  [nl.variable]: ke.reference,
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
}), sl = o({
  px: 3,
  py: 2,
  borderBottomWidth: "1px"
}), dl = o({
  px: 3,
  py: 2
}), cl = o({
  px: 3,
  py: 2,
  borderTopWidth: "1px"
}), bl = o({
  position: "absolute",
  borderRadius: "md",
  top: 1,
  insetEnd: 2,
  padding: 2
}), ul = al({
  popper: il,
  content: ll,
  header: sl,
  body: dl,
  footer: cl,
  closeButton: bl
}), vl = ol({
  baseStyle: ul
}), { definePartsStyle: Ze, defineMultiStyleConfig: fl } = b(mo.keys), Ge = v("drawer-bg"), Ue = v("drawer-box-shadow");
function U(e) {
  return Ze(e === "full" ? {
    dialog: { maxW: "100vw", h: "100vh" }
  } : {
    dialog: { maxW: e }
  });
}
var ml = o({
  bg: "blackAlpha.600",
  zIndex: "overlay"
}), hl = o({
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
}), pl = o({
  px: "6",
  py: "4",
  fontSize: "xl",
  fontWeight: "semibold"
}), yl = o({
  position: "absolute",
  top: "2",
  insetEnd: "3"
}), Sl = o({
  px: "6",
  py: "2",
  flex: "1",
  overflow: "auto"
}), xl = o({
  px: "6",
  py: "4"
}), $l = Ze((e) => ({
  overlay: ml,
  dialogContainer: hl,
  dialog: w(gl, e),
  header: pl,
  closeButton: yl,
  body: Sl,
  footer: xl
})), kl = {
  xs: U("xs"),
  sm: U("md"),
  md: U("lg"),
  lg: U("2xl"),
  xl: U("4xl"),
  full: U("full")
}, _l = fl({
  baseStyle: $l,
  sizes: kl,
  defaultProps: {
    size: "xs"
  }
}), { definePartsStyle: wl, defineMultiStyleConfig: Cl } = b(ho.keys), zl = o({
  borderRadius: "md",
  py: "1",
  transitionProperty: "common",
  transitionDuration: "normal"
}), Al = o({
  borderRadius: "md",
  py: "1",
  transitionProperty: "common",
  transitionDuration: "normal",
  width: "full",
  _focusVisible: { boxShadow: "outline" },
  _placeholder: { opacity: 0.6 }
}), Pl = o({
  borderRadius: "md",
  py: "1",
  transitionProperty: "common",
  transitionDuration: "normal",
  width: "full",
  _focusVisible: { boxShadow: "outline" },
  _placeholder: { opacity: 0.6 }
}), Tl = wl({
  preview: zl,
  input: Al,
  textarea: Pl
}), Fl = Cl({
  baseStyle: Tl
}), { definePartsStyle: Il, defineMultiStyleConfig: Bl } = b(go.keys), X = v("form-control-color"), Ml = o({
  marginStart: "1",
  [X.variable]: "colors.red.500",
  _dark: {
    [X.variable]: "colors.red.300"
  },
  color: X.reference
}), El = o({
  mt: "2",
  [X.variable]: "colors.gray.600",
  _dark: {
    [X.variable]: "colors.whiteAlpha.600"
  },
  color: X.reference,
  lineHeight: "normal",
  fontSize: "sm"
}), Dl = Il({
  container: {
    width: "100%",
    position: "relative"
  },
  requiredIndicator: Ml,
  helperText: El
}), jl = Bl({
  baseStyle: Dl
}), { definePartsStyle: Rl, defineMultiStyleConfig: Wl } = b(po.keys), J = v("form-error-color"), Hl = o({
  [J.variable]: "colors.red.500",
  _dark: {
    [J.variable]: "colors.red.300"
  },
  color: J.reference,
  mt: "2",
  fontSize: "sm",
  lineHeight: "normal"
}), Ll = o({
  marginEnd: "0.5em",
  [J.variable]: "colors.red.500",
  _dark: {
    [J.variable]: "colors.red.300"
  },
  color: J.reference
}), Vl = Rl({
  text: Hl,
  icon: Ll
}), Ol = Wl({
  baseStyle: Vl
}), ql = o({
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
}), Nl = S({
  baseStyle: ql
}), Gl = o({
  fontFamily: "heading",
  fontWeight: "bold"
}), Ul = {
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
}, Yl = S({
  baseStyle: Gl,
  sizes: Ul,
  defaultProps: {
    size: "xl"
  }
}), { defineMultiStyleConfig: Kl, definePartsStyle: Xl } = b(vo.keys), Jl = o({
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
}), Ql = Xl({
  link: Jl
}), Zl = Kl({
  baseStyle: Ql
}), es = o({
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
  const a = Z(`${r}.200`, 0.12)(t), n = Z(`${r}.200`, 0.24)(t);
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
}), rs = o((e) => {
  const { colorScheme: r } = e, t = s("gray.200", "whiteAlpha.300")(e);
  return {
    border: "1px solid",
    borderColor: r === "gray" ? t : "currentColor",
    ".chakra-button__group[data-attached][data-orientation=horizontal] > &:not(:last-of-type)": { marginEnd: "-1px" },
    ".chakra-button__group[data-attached][data-orientation=vertical] > &:not(:last-of-type)": { marginBottom: "-1px" },
    ...w(ot, e)
  };
}), ts = {
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
}, os = o((e) => {
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
  } = (r = ts[t]) != null ? r : {}, d = s(a, `${t}.200`)(e);
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
}), as = o((e) => {
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
}), ns = o({
  bg: "none",
  color: "inherit",
  display: "inline",
  lineHeight: "inherit",
  m: "0",
  p: "0"
}), is = {
  ghost: ot,
  outline: rs,
  solid: os,
  link: as,
  unstyled: ns
}, ls = {
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
}, ss = S({
  baseStyle: es,
  variants: is,
  sizes: ls,
  defaultProps: {
    variant: "solid",
    size: "md",
    colorScheme: "gray"
  }
}), { definePartsStyle: G, defineMultiStyleConfig: ds } = b(Mo.keys), Fe = v("card-bg"), j = v("card-padding"), at = v("card-shadow"), ze = v("card-radius"), nt = v("card-border-width", "0"), it = v("card-border-color"), cs = G({
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
}), bs = {
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
}, us = {
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
}, vs = ds({
  baseStyle: cs,
  variants: us,
  sizes: bs,
  defaultProps: {
    variant: "elevated",
    size: "md"
  }
}), de = x("close-button-size"), te = x("close-button-bg"), fs = o({
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
}), ms = {
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
}, hs = S({
  baseStyle: fs,
  sizes: ms,
  defaultProps: {
    size: "md"
  }
}), { variants: gs, defaultProps: ps } = le, ys = o({
  fontFamily: "mono",
  fontSize: "sm",
  px: "0.2em",
  borderRadius: "sm"
}), Ss = S({
  baseStyle: ys,
  variants: gs,
  defaultProps: ps
}), xs = o({
  w: "100%",
  mx: "auto",
  maxW: "prose",
  px: "4"
}), $s = S({
  baseStyle: xs
}), ks = o({
  opacity: 0.6,
  borderColor: "inherit"
}), _s = o({
  borderStyle: "solid"
}), ws = o({
  borderStyle: "dashed"
}), Cs = {
  solid: _s,
  dashed: ws
}, zs = S({
  baseStyle: ks,
  variants: Cs,
  defaultProps: {
    variant: "solid"
  }
}), { definePartsStyle: As, defineMultiStyleConfig: Ps } = b(co.keys), Ts = o({
  borderTopWidth: "1px",
  borderColor: "inherit",
  _last: {
    borderBottomWidth: "1px"
  }
}), Fs = o({
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
}), Is = o({
  pt: "2",
  px: "4",
  pb: "5"
}), Bs = o({
  fontSize: "1.25em"
}), Ms = As({
  container: Ts,
  button: Fs,
  panel: Is,
  icon: Bs
}), Es = Ps({ baseStyle: Ms }), { definePartsStyle: ve, defineMultiStyleConfig: Ds } = b(bo.keys), z = v("alert-fg"), R = v("alert-bg"), js = ve({
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
  const { theme: r, colorScheme: t } = e, a = Z(`${t}.200`, 0.16)(r);
  return {
    light: `colors.${t}.100`,
    dark: a
  };
}
var Rs = ve((e) => {
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
      paddingStart: "3",
      borderStartWidth: "4px",
      borderStartColor: z.reference
    }
  };
}), Hs = ve((e) => {
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
}), Ls = ve((e) => {
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
}), Vs = {
  subtle: Rs,
  "left-accent": Ws,
  "top-accent": Hs,
  solid: Ls
}, Os = Ds({
  baseStyle: js,
  variants: Vs,
  defaultProps: {
    variant: "subtle",
    colorScheme: "blue"
  }
}), { definePartsStyle: lt, defineMultiStyleConfig: qs } = b(uo.keys), Q = v("avatar-border-color"), Ye = v("avatar-bg"), Ns = o({
  borderRadius: "full",
  border: "0.2em solid",
  [Q.variable]: "white",
  _dark: {
    [Q.variable]: "colors.gray.800"
  },
  borderColor: Q.reference
}), Gs = o({
  [Ye.variable]: "colors.gray.200",
  _dark: {
    [Ye.variable]: "colors.whiteAlpha.400"
  },
  bgColor: Ye.reference
}), Or = v("avatar-background"), Us = o((e) => {
  const { name: r, theme: t } = e, a = r ? ea({ string: r }) : "colors.gray.400", n = Qo(a)(t);
  let i = "white";
  return n || (i = "gray.800"), {
    bg: Or.reference,
    "&:not([data-loaded])": {
      [Or.variable]: a
    },
    color: i,
    [Q.variable]: "colors.white",
    _dark: {
      [Q.variable]: "colors.gray.800"
    },
    borderColor: Q.reference,
    verticalAlign: "top"
  };
}), Ys = lt((e) => ({
  badge: w(Ns, e),
  excessLabel: w(Gs, e),
  container: w(Us, e)
}));
function H(e) {
  const r = e !== "100%" ? Jr[e] : void 0;
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
var Ks = {
  "2xs": H(4),
  xs: H(6),
  sm: H(8),
  md: H(12),
  lg: H(16),
  xl: H(24),
  "2xl": H(32),
  full: H("100%")
}, Xs = qs({
  baseStyle: Ys,
  sizes: Ks,
  defaultProps: { size: "md" }
}), Js = {
  Accordion: Es,
  Alert: Os,
  Avatar: Xs,
  Badge: le,
  Breadcrumb: Zl,
  Button: ss,
  Checkbox: Te,
  CloseButton: hs,
  Code: Ss,
  Container: $s,
  Divider: zs,
  Drawer: _l,
  Editable: Fl,
  Form: jl,
  FormError: Ol,
  FormLabel: Nl,
  Heading: Yl,
  Input: u,
  Kbd: mi,
  Link: gi,
  List: $i,
  Menu: Ii,
  Modal: Oi,
  NumberInput: Qi,
  PinInput: tl,
  Popover: vl,
  Progress: xn,
  Radio: Mn,
  Select: Ln,
  Skeleton: On,
  SkipLink: Nn,
  Slider: ti,
  Spinner: ni,
  Stat: vi,
  Switch: ha,
  Table: ka,
  Tabs: Ra,
  Tag: Ja,
  Textarea: cn,
  Tooltip: vn,
  Card: vs
}, Qs = {
  colors: {
    "chakra-body-text": { _light: "gray.800", _dark: "whiteAlpha.900" },
    "chakra-body-bg": { _light: "white", _dark: "gray.800" },
    "chakra-border-color": { _light: "gray.200", _dark: "whiteAlpha.300" },
    "chakra-subtle-bg": { _light: "gray.100", _dark: "gray.700" },
    "chakra-placeholder-color": { _light: "gray.500", _dark: "whiteAlpha.400" }
  }
}, Zs = {
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
}, ed = "ltr", rd = {
  useSystemColorMode: !1,
  initialColorMode: "light",
  cssVarPrefix: "chakra"
}, st = {
  semanticTokens: Qs,
  direction: ed,
  ...so,
  components: Js,
  styles: Zs,
  config: rd
};
function ne(e) {
  return typeof e == "function";
}
function td(...e) {
  return (r) => e.reduce((t, a) => a(t), r);
}
var od = (e) => function(...t) {
  let a = [...t], n = t[t.length - 1];
  return Rt(n) && a.length > 1 ? a = a.slice(0, a.length - 1) : n = e, td(
    ...a.map(
      (i) => (l) => ne(i) ? i(l) : nd(l, i)
    )
  )(n);
}, ad = od(st);
function nd(...e) {
  return Nr({}, ...e, dt);
}
function dt(e, r, t, a) {
  if ((ne(e) || ne(r)) && Object.prototype.hasOwnProperty.call(a, t))
    return (...n) => {
      const i = ne(e) ? e(...n) : e, l = ne(r) ? r(...n) : r;
      return Nr({}, i, l, dt);
    };
}
function id(e = {}) {
  const {
    strict: r = !0,
    errorMessage: t = "useContext: `context` is undefined. Seems you forgot to wrap component within the Provider",
    name: a
  } = e, n = Gr(void 0);
  n.displayName = a;
  function i() {
    var l;
    const d = Pt(n);
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
function ld(e) {
  const { cssVarsRoot: r, theme: t, children: a } = e, n = Ae(() => ht(t), [t]);
  return /* @__PURE__ */ y.jsxs(gt, { theme: n, children: [
    /* @__PURE__ */ y.jsx(sd, { root: r }),
    a
  ] });
}
function sd({ root: e = ":host, :root" }) {
  const r = [e, "[data-theme]"].join(",");
  return /* @__PURE__ */ y.jsx(Ie, { styles: (t) => ({ [r]: t.__cssVars }) });
}
id({
  name: "StylesContext",
  errorMessage: "useStyles: `styles` is undefined. Seems you forgot to wrap the components in `<StylesProvider />` "
});
function dd() {
  const { colorMode: e } = pt();
  return /* @__PURE__ */ y.jsx(
    Ie,
    {
      styles: (r) => {
        const t = yt(r, "styles.global"), a = St(t, { theme: r, colorMode: e });
        return a ? xt(a)(r) : void 0;
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
  const { children: r, environment: t, disabled: a } = e, n = Tt(null), i = Ae(() => t || {
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
var cd = (e) => {
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
  return /* @__PURE__ */ y.jsx(ld, { theme: i, cssVarsRoot: d, children: /* @__PURE__ */ y.jsxs(
    Yr,
    {
      colorModeManager: t,
      options: i.config,
      children: [
        n ? /* @__PURE__ */ y.jsx(It, {}) : /* @__PURE__ */ y.jsx(Ft, {}),
        /* @__PURE__ */ y.jsx(dd, {}),
        a ? /* @__PURE__ */ y.jsx(_t, { zIndex: a, children: $ }) : $
      ]
    }
  ) });
}, bd = (e) => function({
  children: t,
  theme: a = e,
  toastOptions: n,
  ...i
}) {
  return /* @__PURE__ */ y.jsxs(cd, { theme: a, ...i, children: [
    /* @__PURE__ */ y.jsx(wt, { value: n == null ? void 0 : n.defaultOptions, children: t }),
    /* @__PURE__ */ y.jsx(Ct, { ...n })
  ] });
}, ud = bd(st);
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
  error: p(0, 76)
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
  function n(...h) {
    a();
    for (const g of h)
      r[g] = c(g);
    return m(e, r);
  }
  function i(...h) {
    for (const g of h)
      g in r || (r[g] = c(g));
    return m(e, r);
  }
  function l() {
    return Object.fromEntries(
      Object.entries(r).map(([g, k]) => [g, k.selector])
    );
  }
  function d() {
    return Object.fromEntries(
      Object.entries(r).map(([g, k]) => [g, k.className])
    );
  }
  function c(h) {
    const T = `chakra-${(["container", "root"].includes(h ?? "") ? [e] : [e, h]).filter(Boolean).join("__")}`;
    return {
      className: T,
      selector: `.${T}`,
      toString: () => h
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
var vd = m("accordion").parts("root", "container", "button", "panel").extend("icon");
m("alert").parts("title", "description", "container").extend("icon", "spinner");
m("avatar").parts("label", "badge", "container").extend("excessLabel", "group");
m("breadcrumb").parts("link", "item", "container").extend("separator");
m("button").parts();
var fd = m("checkbox").parts("control", "icon", "container").extend("label");
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
var md = m("input").parts("addon", "field", "element");
m("list").parts("container", "item", "icon");
var hd = m("menu").parts("button", "list", "item").extend("groupTitle", "command", "divider"), gd = m("modal").parts("overlay", "dialogContainer", "dialog").extend("header", "closeButton", "body", "footer"), pd = m("numberinput").parts(
  "root",
  "field",
  "stepperGroup",
  "stepper"
);
m("pininput").parts("field");
var yd = m("popover").parts("content", "header", "body", "footer").extend("popper", "arrow", "closeButton"), Sd = m("progress").parts(
  "label",
  "filledTrack",
  "track"
);
m("radio").parts(
  "container",
  "control",
  "label"
);
var xd = m("select").parts("field", "icon"), $d = m("slider").parts(
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
var kd = m("switch").parts(
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
var _d = m("tabs").parts(
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
const { definePartsStyle: wd, defineMultiStyleConfig: Cd } = b(vd.keys), zd = o({
  border: "none",
  pt: 2
}), Ad = o((e) => {
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
}), Pd = o((e) => {
  const { colorScheme: r } = e;
  return {
    bg: `${r}.800`,
    borderRadius: "base",
    borderTopRadius: "none",
    p: 4
  };
}), Td = o({}), Fd = wd((e) => ({
  container: zd,
  button: Ad(e),
  panel: Pd(e),
  icon: Td
})), Id = Cd({
  variants: { invokeAI: Fd },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "base"
  }
}), Bd = o((e) => {
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
}), Md = S({
  variants: {
    invokeAI: Bd
  },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "base"
  }
}), { definePartsStyle: Ed, defineMultiStyleConfig: Dd } = b(fd.keys), jd = o((e) => {
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
}), Rd = Ed((e) => ({
  control: jd(e)
})), Wd = Dd({
  variants: {
    invokeAI: Rd
  },
  defaultProps: {
    variant: "invokeAI",
    colorScheme: "accent"
  }
}), Hd = o((e) => ({
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
})), Ld = S({
  variants: {
    invokeAI: Hd
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
}), { definePartsStyle: Vd, defineMultiStyleConfig: Od } = b(md.keys), qd = Vd((e) => ({
  field: Ee()
})), Nd = Od({
  variants: {
    invokeAI: qd
  },
  defaultProps: {
    size: "sm",
    variant: "invokeAI"
  }
}), { definePartsStyle: Gd, defineMultiStyleConfig: Ud } = b(hd.keys), Yd = Gd({
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
}), Kd = Ud({
  variants: {
    invokeAI: Yd
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), { defineMultiStyleConfig: Xd, definePartsStyle: Jd } = b(gd.keys), Qd = o({
  bg: "blackAlpha.600"
}), Zd = o({}), ec = o((e) => ({
  bg: "base.850",
  maxH: "80vh"
})), rc = o((e) => ({
  fontWeight: "600",
  fontSize: "lg",
  color: "base.200"
})), tc = o({}), oc = o({
  overflowY: "scroll"
}), ac = o({}), nc = Jd((e) => ({
  overlay: Qd,
  dialogContainer: Zd,
  dialog: ec(e),
  header: rc(e),
  closeButton: tc,
  body: oc,
  footer: ac
})), ic = Xd({
  variants: {
    invokeAI: nc
  },
  defaultProps: { variant: "invokeAI", size: "lg" }
}), { defineMultiStyleConfig: lc, definePartsStyle: sc } = b(pd.keys), dc = o((e) => ({
  height: 8
})), cc = o((e) => ({
  border: "none",
  fontWeight: "600",
  height: "auto",
  py: 1,
  ps: 2,
  pe: 6,
  ...Ee()
})), bc = o((e) => ({
  display: "flex"
})), uc = o((e) => ({
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
})), vc = sc((e) => ({
  root: dc(e),
  field: cc(e),
  stepperGroup: bc(e),
  stepper: uc(e)
})), fc = lc({
  variants: {
    invokeAI: vc
  },
  defaultProps: {
    size: "sm",
    variant: "invokeAI"
  }
});
function mc(e) {
  return !Number.isInteger(parseFloat(e.toString()));
}
function hc(e, r = "-") {
  return e.replace(/\s+/g, r);
}
function vt(e) {
  const r = hc(e.toString());
  return r.includes("\\.") ? e : mc(e) ? r.replace(".", "\\.") : e;
}
function gc(e, r = "") {
  return [r, vt(e)].filter(Boolean).join("-");
}
function pc(e, r) {
  return `var(${vt(e)}${r ? `, ${r}` : ""})`;
}
function yc(e, r = "") {
  return `--${gc(e, r)}`;
}
function nr(e, r) {
  const t = yc(e, r == null ? void 0 : r.prefix);
  return {
    variable: t,
    reference: pc(t, Sc(r == null ? void 0 : r.fallback))
  };
}
function Sc(e) {
  return typeof e == "string" ? e : e == null ? void 0 : e.reference;
}
const { defineMultiStyleConfig: xc, definePartsStyle: $c } = b(yd.keys), kc = nr("popper-bg"), _c = nr("popper-arrow-bg"), wc = nr("popper-arrow-shadow-color"), Cc = o((e) => ({
  [_c.variable]: "colors.base.800",
  [kc.variable]: "colors.base.800",
  [wc.variable]: "colors.base.600",
  minW: "unset",
  width: "unset",
  p: 4,
  borderWidth: "2px",
  borderStyle: "solid",
  borderColor: "base.600",
  bg: "base.800"
})), zc = $c((e) => ({
  content: Cc(e)
})), Ac = xc({
  variants: {
    invokeAI: zc
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), { defineMultiStyleConfig: Pc, definePartsStyle: Tc } = b(Sd.keys), Fc = o((e) => ({
  bg: "accent.600",
  transition: "width 0.2s ease-in-out",
  _indeterminate: {
    bgGradient: "linear(to-r, transparent 0%, accent.600 50%, transparent 100%);"
  }
})), Ic = o((e) => ({
  bg: "base.800"
})), Bc = Tc((e) => ({
  filledTrack: Fc(e),
  track: Ic(e)
})), Mc = Pc({
  variants: {
    invokeAI: Bc
  },
  defaultProps: {
    variant: "invokeAI"
  }
}), { definePartsStyle: Ec, defineMultiStyleConfig: Dc } = b(xd.keys), jc = o((e) => ({
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
}), { definePartsStyle: Lc, defineMultiStyleConfig: Vc } = b($d.keys), Oc = o((e) => ({
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
}), { defineMultiStyleConfig: Kc, definePartsStyle: Xc } = b(kd.keys), Jc = o((e) => {
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
}), Qc = o((e) => {
  const { colorScheme: r } = e;
  return {
    bg: `${r}.50`
  };
}), Zc = Xc((e) => ({
  container: {},
  track: Jc(e),
  thumb: Qc(e)
})), e0 = Kc({
  variants: { invokeAI: Zc },
  defaultProps: {
    size: "md",
    variant: "invokeAI",
    colorScheme: "accent"
  }
}), { defineMultiStyleConfig: r0, definePartsStyle: t0 } = b(_d.keys), o0 = o((e) => ({
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
      ...zt
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
    Button: Md,
    // Button and IconButton
    Input: Nd,
    Textarea: u0,
    Tabs: s0,
    Progress: Mc,
    Accordion: Id,
    FormLabel: Ld,
    Switch: e0,
    NumberInput: fc,
    Select: Hc,
    Slider: Yc,
    Popover: Ac,
    Modal: ic,
    Checkbox: Wd,
    Menu: Kd,
    Text: c0
  }
}, f0 = {
  base: p(223, 10),
  accent: p(155, 80),
  working: p(47, 68),
  warning: p(28, 75),
  ok: p(122, 49),
  error: p(0, 50)
}, m0 = {
  base: p(223, 10, !0),
  accent: p(40, 80, !0),
  working: p(47, 68, !0),
  warning: p(28, 75, !0),
  ok: p(122, 49, !0),
  error: p(0, 50, !0)
}, h0 = {
  base: p(220, 30),
  accent: p(210, 80),
  working: p(47, 68),
  warning: p(28, 75),
  ok: p(122, 49),
  error: p(0, 100)
}, g0 = {
  dark: ut,
  light: m0,
  green: f0,
  ocean: h0
};
function $0({ children: e }) {
  const { i18n: r } = $t(), t = At(
    (i) => i.ui.currentTheme
  ), a = r.dir(), n = ad({
    ...v0,
    colors: g0[t],
    direction: a
  });
  return Ke(() => {
    document.body.dir = a;
  }, [a]), /* @__PURE__ */ y.jsx(ud, { theme: n, children: e });
}
export {
  $0 as default
};
