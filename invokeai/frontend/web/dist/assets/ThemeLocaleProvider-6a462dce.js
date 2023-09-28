import{w as s,hQ as T,v as l,a2 as V,hR as A,ae as I,hS as z,hT as j,hU as D,hV as F,hW as W,hX as G,hY as K,hZ as U,h_ as Y}from"./index-375621ca.js";import{u as Z,M as H}from"./MantineProvider-f12d896d.js";var P=String.raw,E=P`
  :root,
  :host {
    --chakra-vh: 100vh;
  }

  @supports (height: -webkit-fill-available) {
    :root,
    :host {
      --chakra-vh: -webkit-fill-available;
    }
  }

  @supports (height: -moz-fill-available) {
    :root,
    :host {
      --chakra-vh: -moz-fill-available;
    }
  }

  @supports (height: 100dvh) {
    :root,
    :host {
      --chakra-vh: 100dvh;
    }
  }
`,Q=()=>s.jsx(T,{styles:E}),X=({scope:e=""})=>s.jsx(T,{styles:P`
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
        margin: 0;
        font-feature-settings: "kern";
      }

      ${e} :where(*, *::before, *::after) {
        border-width: 0;
        border-style: solid;
        box-sizing: border-box;
        word-wrap: break-word;
      }

      main {
        display: block;
      }

      ${e} hr {
        border-top-width: 1px;
        box-sizing: content-box;
        height: 0;
        overflow: visible;
      }

      ${e} :where(pre, code, kbd,samp) {
        font-family: SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1em;
      }

      ${e} a {
        background-color: transparent;
        color: inherit;
        text-decoration: inherit;
      }

      ${e} abbr[title] {
        border-bottom: none;
        text-decoration: underline;
        -webkit-text-decoration: underline dotted;
        text-decoration: underline dotted;
      }

      ${e} :where(b, strong) {
        font-weight: bold;
      }

      ${e} small {
        font-size: 80%;
      }

      ${e} :where(sub,sup) {
        font-size: 75%;
        line-height: 0;
        position: relative;
        vertical-align: baseline;
      }

      ${e} sub {
        bottom: -0.25em;
      }

      ${e} sup {
        top: -0.5em;
      }

      ${e} img {
        border-style: none;
      }

      ${e} :where(button, input, optgroup, select, textarea) {
        font-family: inherit;
        font-size: 100%;
        line-height: 1.15;
        margin: 0;
      }

      ${e} :where(button, input) {
        overflow: visible;
      }

      ${e} :where(button, select) {
        text-transform: none;
      }

      ${e} :where(
          button::-moz-focus-inner,
          [type="button"]::-moz-focus-inner,
          [type="reset"]::-moz-focus-inner,
          [type="submit"]::-moz-focus-inner
        ) {
        border-style: none;
        padding: 0;
      }

      ${e} fieldset {
        padding: 0.35em 0.75em 0.625em;
      }

      ${e} legend {
        box-sizing: border-box;
        color: inherit;
        display: table;
        max-width: 100%;
        padding: 0;
        white-space: normal;
      }

      ${e} progress {
        vertical-align: baseline;
      }

      ${e} textarea {
        overflow: auto;
      }

      ${e} :where([type="checkbox"], [type="radio"]) {
        box-sizing: border-box;
        padding: 0;
      }

      ${e} input[type="number"]::-webkit-inner-spin-button,
      ${e} input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none !important;
      }

      ${e} input[type="number"] {
        -moz-appearance: textfield;
      }

      ${e} input[type="search"] {
        -webkit-appearance: textfield;
        outline-offset: -2px;
      }

      ${e} input[type="search"]::-webkit-search-decoration {
        -webkit-appearance: none !important;
      }

      ${e} ::-webkit-file-upload-button {
        -webkit-appearance: button;
        font: inherit;
      }

      ${e} details {
        display: block;
      }

      ${e} summary {
        display: list-item;
      }

      template {
        display: none;
      }

      [hidden] {
        display: none !important;
      }

      ${e} :where(
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
          pre
        ) {
        margin: 0;
      }

      ${e} button {
        background: transparent;
        padding: 0;
      }

      ${e} fieldset {
        margin: 0;
        padding: 0;
      }

      ${e} :where(ol, ul) {
        margin: 0;
        padding: 0;
      }

      ${e} textarea {
        resize: vertical;
      }

      ${e} :where(button, [role="button"]) {
        cursor: pointer;
      }

      ${e} button::-moz-focus-inner {
        border: 0 !important;
      }

      ${e} table {
        border-collapse: collapse;
      }

      ${e} :where(h1, h2, h3, h4, h5, h6) {
        font-size: inherit;
        font-weight: inherit;
      }

      ${e} :where(button, input, optgroup, select, textarea) {
        padding: 0;
        line-height: inherit;
        color: inherit;
      }

      ${e} :where(img, svg, video, canvas, audio, iframe, embed, object) {
        display: block;
      }

      ${e} :where(img, video) {
        max-width: 100%;
        height: auto;
      }

      [data-js-focus-visible]
        :focus:not([data-focus-visible-added]):not(
          [data-focus-visible-disabled]
        ) {
        outline: none;
        box-shadow: none;
      }

      ${e} select::-ms-expand {
        display: none;
      }

      ${E}
    `}),g={light:"chakra-ui-light",dark:"chakra-ui-dark"};function B(e={}){const{preventTransition:o=!0}=e,n={setDataset:r=>{const t=o?n.preventTransition():void 0;document.documentElement.dataset.theme=r,document.documentElement.style.colorScheme=r,t==null||t()},setClassName(r){document.body.classList.add(r?g.dark:g.light),document.body.classList.remove(r?g.light:g.dark)},query(){return window.matchMedia("(prefers-color-scheme: dark)")},getSystemTheme(r){var t;return((t=n.query().matches)!=null?t:r==="dark")?"dark":"light"},addListener(r){const t=n.query(),a=i=>{r(i.matches?"dark":"light")};return typeof t.addListener=="function"?t.addListener(a):t.addEventListener("change",a),()=>{typeof t.removeListener=="function"?t.removeListener(a):t.removeEventListener("change",a)}},preventTransition(){const r=document.createElement("style");return r.appendChild(document.createTextNode("*{-webkit-transition:none!important;-moz-transition:none!important;-o-transition:none!important;-ms-transition:none!important;transition:none!important}")),document.head.appendChild(r),()=>{window.getComputedStyle(document.body),requestAnimationFrame(()=>{requestAnimationFrame(()=>{document.head.removeChild(r)})})}}};return n}var J="chakra-ui-color-mode";function L(e){return{ssr:!1,type:"localStorage",get(o){if(!(globalThis!=null&&globalThis.document))return o;let n;try{n=localStorage.getItem(e)||o}catch{}return n||o},set(o){try{localStorage.setItem(e,o)}catch{}}}}var ee=L(J),M=()=>{};function S(e,o){return e.type==="cookie"&&e.ssr?e.get(o):o}function O(e){const{value:o,children:n,options:{useSystemColorMode:r,initialColorMode:t,disableTransitionOnChange:a}={},colorModeManager:i=ee}=e,d=t==="dark"?"dark":"light",[u,p]=l.useState(()=>S(i,d)),[y,b]=l.useState(()=>S(i)),{getSystemTheme:w,setClassName:k,setDataset:x,addListener:$}=l.useMemo(()=>B({preventTransition:a}),[a]),v=t==="system"&&!u?y:u,h=l.useCallback(c=>{const f=c==="system"?w():c;p(f),k(f==="dark"),x(f),i.set(f)},[i,w,k,x]);V(()=>{t==="system"&&b(w())},[]),l.useEffect(()=>{const c=i.get();if(c){h(c);return}if(t==="system"){h("system");return}h(d)},[i,d,t,h]);const C=l.useCallback(()=>{h(v==="dark"?"light":"dark")},[v,h]);l.useEffect(()=>{if(r)return $(h)},[r,$,h]);const R=l.useMemo(()=>({colorMode:o??v,toggleColorMode:o?M:C,setColorMode:o?M:h,forced:o!==void 0}),[v,C,h,o]);return s.jsx(A.Provider,{value:R,children:n})}O.displayName="ColorModeProvider";var te=["borders","breakpoints","colors","components","config","direction","fonts","fontSizes","fontWeights","letterSpacings","lineHeights","radii","shadows","sizes","space","styles","transition","zIndices"];function re(e){return I(e)?te.every(o=>Object.prototype.hasOwnProperty.call(e,o)):!1}function m(e){return typeof e=="function"}function oe(...e){return o=>e.reduce((n,r)=>r(n),o)}var ne=e=>function(...n){let r=[...n],t=n[n.length-1];return re(t)&&r.length>1?r=r.slice(0,r.length-1):t=e,oe(...r.map(a=>i=>m(a)?a(i):ie(i,a)))(t)},ae=ne(j);function ie(...e){return z({},...e,_)}function _(e,o,n,r){if((m(e)||m(o))&&Object.prototype.hasOwnProperty.call(r,n))return(...t)=>{const a=m(e)?e(...t):e,i=m(o)?o(...t):o;return z({},a,i,_)}}var q=l.createContext({getDocument(){return document},getWindow(){return window}});q.displayName="EnvironmentContext";function N(e){const{children:o,environment:n,disabled:r}=e,t=l.useRef(null),a=l.useMemo(()=>n||{getDocument:()=>{var d,u;return(u=(d=t.current)==null?void 0:d.ownerDocument)!=null?u:document},getWindow:()=>{var d,u;return(u=(d=t.current)==null?void 0:d.ownerDocument.defaultView)!=null?u:window}},[n]),i=!r||!n;return s.jsxs(q.Provider,{value:a,children:[o,i&&s.jsx("span",{id:"__chakra_env",hidden:!0,ref:t})]})}N.displayName="EnvironmentProvider";var se=e=>{const{children:o,colorModeManager:n,portalZIndex:r,resetScope:t,resetCSS:a=!0,theme:i={},environment:d,cssVarsRoot:u,disableEnvironment:p,disableGlobalStyle:y}=e,b=s.jsx(N,{environment:d,disabled:p,children:o});return s.jsx(D,{theme:i,cssVarsRoot:u,children:s.jsxs(O,{colorModeManager:n,options:i.config,children:[a?s.jsx(X,{scope:t}):s.jsx(Q,{}),!y&&s.jsx(F,{}),r?s.jsx(W,{zIndex:r,children:b}):b]})})},le=e=>function({children:n,theme:r=e,toastOptions:t,...a}){return s.jsxs(se,{theme:r,...a,children:[s.jsx(G,{value:t==null?void 0:t.defaultOptions,children:n}),s.jsx(K,{...t})]})},de=le(j);const ue=()=>l.useMemo(()=>({colorScheme:"dark",fontFamily:"'Inter Variable', sans-serif",components:{ScrollArea:{defaultProps:{scrollbarSize:10},styles:{scrollbar:{"&:hover":{backgroundColor:"var(--invokeai-colors-baseAlpha-300)"}},thumb:{backgroundColor:"var(--invokeai-colors-baseAlpha-300)"}}}}}),[]),he=L("@@invokeai-color-mode");function ce({children:e}){const{i18n:o}=Z(),n=o.dir(),r=l.useMemo(()=>ae({...U,direction:n}),[n]);l.useEffect(()=>{document.body.dir=n},[n]);const t=ue();return s.jsx(H,{theme:t,children:s.jsx(de,{theme:r,colorModeManager:he,toastOptions:Y,children:e})})}const ve=l.memo(ce);export{ve as default};
