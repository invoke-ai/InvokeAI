import{D as s,iE as T,r as l,Z as A,iF as D,a7 as R,iG as z,iH as j,iI as V,iJ as F,iK as G,iL as W,iM as K,at as H,iN as Z,iO as J}from"./index-fbe0e055.js";import{M as U}from"./MantineProvider-44862fff.js";var P=String.raw,E=P`
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
`,Y=()=>s.jsx(T,{styles:E}),B=({scope:e=""})=>s.jsx(T,{styles:P`
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
    `}),g={light:"chakra-ui-light",dark:"chakra-ui-dark"};function Q(e={}){const{preventTransition:o=!0}=e,n={setDataset:r=>{const t=o?n.preventTransition():void 0;document.documentElement.dataset.theme=r,document.documentElement.style.colorScheme=r,t==null||t()},setClassName(r){document.body.classList.add(r?g.dark:g.light),document.body.classList.remove(r?g.light:g.dark)},query(){return window.matchMedia("(prefers-color-scheme: dark)")},getSystemTheme(r){var t;return((t=n.query().matches)!=null?t:r==="dark")?"dark":"light"},addListener(r){const t=n.query(),i=a=>{r(a.matches?"dark":"light")};return typeof t.addListener=="function"?t.addListener(i):t.addEventListener("change",i),()=>{typeof t.removeListener=="function"?t.removeListener(i):t.removeEventListener("change",i)}},preventTransition(){const r=document.createElement("style");return r.appendChild(document.createTextNode("*{-webkit-transition:none!important;-moz-transition:none!important;-o-transition:none!important;-ms-transition:none!important;transition:none!important}")),document.head.appendChild(r),()=>{window.getComputedStyle(document.body),requestAnimationFrame(()=>{requestAnimationFrame(()=>{document.head.removeChild(r)})})}}};return n}var X="chakra-ui-color-mode";function L(e){return{ssr:!1,type:"localStorage",get(o){if(!(globalThis!=null&&globalThis.document))return o;let n;try{n=localStorage.getItem(e)||o}catch{}return n||o},set(o){try{localStorage.setItem(e,o)}catch{}}}}var ee=L(X),M=()=>{};function S(e,o){return e.type==="cookie"&&e.ssr?e.get(o):o}function O(e){const{value:o,children:n,options:{useSystemColorMode:r,initialColorMode:t,disableTransitionOnChange:i}={},colorModeManager:a=ee}=e,d=t==="dark"?"dark":"light",[u,p]=l.useState(()=>S(a,d)),[y,b]=l.useState(()=>S(a)),{getSystemTheme:w,setClassName:k,setDataset:x,addListener:$}=l.useMemo(()=>Q({preventTransition:i}),[i]),f=t==="system"&&!u?y:u,c=l.useCallback(m=>{const v=m==="system"?w():m;p(v),k(v==="dark"),x(v),a.set(v)},[a,w,k,x]);A(()=>{t==="system"&&b(w())},[]),l.useEffect(()=>{const m=a.get();if(m){c(m);return}if(t==="system"){c("system");return}c(d)},[a,d,t,c]);const C=l.useCallback(()=>{c(f==="dark"?"light":"dark")},[f,c]);l.useEffect(()=>{if(r)return $(c)},[r,$,c]);const I=l.useMemo(()=>({colorMode:o??f,toggleColorMode:o?M:C,setColorMode:o?M:c,forced:o!==void 0}),[f,C,c,o]);return s.jsx(D.Provider,{value:I,children:n})}O.displayName="ColorModeProvider";var te=["borders","breakpoints","colors","components","config","direction","fonts","fontSizes","fontWeights","letterSpacings","lineHeights","radii","shadows","sizes","space","styles","transition","zIndices"];function re(e){return R(e)?te.every(o=>Object.prototype.hasOwnProperty.call(e,o)):!1}function h(e){return typeof e=="function"}function oe(...e){return o=>e.reduce((n,r)=>r(n),o)}var ne=e=>function(...n){let r=[...n],t=n[n.length-1];return re(t)&&r.length>1?r=r.slice(0,r.length-1):t=e,oe(...r.map(i=>a=>h(i)?i(a):ae(a,i)))(t)},ie=ne(j);function ae(...e){return z({},...e,_)}function _(e,o,n,r){if((h(e)||h(o))&&Object.prototype.hasOwnProperty.call(r,n))return(...t)=>{const i=h(e)?e(...t):e,a=h(o)?o(...t):o;return z({},i,a,_)}}var N=l.createContext({getDocument(){return document},getWindow(){return window}});N.displayName="EnvironmentContext";function q(e){const{children:o,environment:n,disabled:r}=e,t=l.useRef(null),i=l.useMemo(()=>n||{getDocument:()=>{var d,u;return(u=(d=t.current)==null?void 0:d.ownerDocument)!=null?u:document},getWindow:()=>{var d,u;return(u=(d=t.current)==null?void 0:d.ownerDocument.defaultView)!=null?u:window}},[n]),a=!r||!n;return s.jsxs(N.Provider,{value:i,children:[o,a&&s.jsx("span",{id:"__chakra_env",hidden:!0,ref:t})]})}q.displayName="EnvironmentProvider";var se=e=>{const{children:o,colorModeManager:n,portalZIndex:r,resetScope:t,resetCSS:i=!0,theme:a={},environment:d,cssVarsRoot:u,disableEnvironment:p,disableGlobalStyle:y}=e,b=s.jsx(q,{environment:d,disabled:p,children:o});return s.jsx(V,{theme:a,cssVarsRoot:u,children:s.jsxs(O,{colorModeManager:n,options:a.config,children:[i?s.jsx(B,{scope:t}):s.jsx(Y,{}),!y&&s.jsx(F,{}),r?s.jsx(G,{zIndex:r,children:b}):b]})})},le=e=>function({children:n,theme:r=e,toastOptions:t,...i}){return s.jsxs(se,{theme:r,...i,children:[s.jsx(W,{value:t==null?void 0:t.defaultOptions,children:n}),s.jsx(K,{...t})]})},de=le(j);const ue=()=>l.useMemo(()=>({colorScheme:"dark",fontFamily:"'Inter Variable', sans-serif",components:{ScrollArea:{defaultProps:{scrollbarSize:10},styles:{scrollbar:{"&:hover":{backgroundColor:"var(--invokeai-colors-baseAlpha-300)"}},thumb:{backgroundColor:"var(--invokeai-colors-baseAlpha-300)"}}}}}),[]);const ce=L("@@invokeai-color-mode");function me({children:e}){const{i18n:o}=H(),n=o.dir(),r=l.useMemo(()=>ie({...Z,direction:n}),[n]);l.useEffect(()=>{document.body.dir=n},[n]);const t=ue();return s.jsx(U,{theme:t,children:s.jsx(de,{theme:r,colorModeManager:ce,toastOptions:J,children:e})})}const fe=l.memo(me);export{fe as default};
