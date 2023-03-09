import * as L from "react";
import Bs, { useContext as w, createContext as ut, useLayoutEffect as Fs, useEffect as U, useState as Dt, useRef as O, useMemo as q, useCallback as Xt, useInsertionEffect as Is, forwardRef as ho, createElement as po, useId as Xe, cloneElement as mo, Children as go, isValidElement as yo, memo as vo, useSyncExternalStore as xo } from "react";
import { cU as Jt, cV as bo, j as T, c as X, t as Ee, f as Bt, a as Ft, e as Os, S as Po, o as Ns, d as To, cW as So } from "./component-6919b100.mjs";
import { createPortal as Us } from "react-dom";
function _s(t = Jt) {
  const e = (
    // @ts-ignore
    t === Jt ? bo : () => w(t)
  );
  return function() {
    const {
      store: s
    } = e();
    return s;
  };
}
const Co = /* @__PURE__ */ _s();
function Ao(t = Jt) {
  const e = (
    // @ts-ignore
    t === Jt ? Co : _s(t)
  );
  return function() {
    return e().dispatch;
  };
}
const Vo = /* @__PURE__ */ Ao();
function wo(t, e) {
  return `${t} returned \`undefined\`. Seems you forgot to wrap component within ${e}`;
}
function It(t = {}) {
  const {
    name: e,
    strict: n = !0,
    hookName: s = "useContext",
    providerName: i = "Provider",
    errorMessage: r
  } = t, o = ut(void 0);
  o.displayName = e;
  function a() {
    var c;
    const l = w(o);
    if (!l && n) {
      const u = new Error(
        r ?? wo(s, i)
      );
      throw u.name = "ContextError", (c = Error.captureStackTrace) == null || c.call(Error, u, a), u;
    }
    return l;
  }
  return [o.Provider, a, o];
}
var [Mo, Do] = It({
  strict: !1,
  name: "PortalManagerContext"
});
function Ro(t) {
  const { children: e, zIndex: n } = t;
  return /* @__PURE__ */ T.jsx(Mo, { value: { zIndex: n }, children: e });
}
Ro.displayName = "PortalManager";
var Le = Boolean(globalThis == null ? void 0 : globalThis.document) ? Fs : U, [$s, Eo] = It({
  strict: !1,
  name: "PortalContext"
}), Je = "chakra-portal", Lo = ".chakra-portal", ko = (t) => /* @__PURE__ */ T.jsx(
  "div",
  {
    className: "chakra-portal-zIndex",
    style: {
      position: "absolute",
      zIndex: t.zIndex,
      top: 0,
      left: 0,
      right: 0
    },
    children: t.children
  }
), jo = (t) => {
  const { appendToParentPortal: e, children: n } = t, [s, i] = Dt(null), r = O(null), [, o] = Dt({});
  U(() => o({}), []);
  const a = Eo(), c = Do();
  Le(() => {
    if (!s)
      return;
    const u = s.ownerDocument, f = e ? a ?? u.body : u.body;
    if (!f)
      return;
    r.current = u.createElement("div"), r.current.className = Je, f.appendChild(r.current), o({});
    const h = r.current;
    return () => {
      f.contains(h) && f.removeChild(h);
    };
  }, [s]);
  const l = c != null && c.zIndex ? /* @__PURE__ */ T.jsx(ko, { zIndex: c == null ? void 0 : c.zIndex, children: n }) : n;
  return r.current ? Us(
    /* @__PURE__ */ T.jsx($s, { value: r.current, children: l }),
    r.current
  ) : /* @__PURE__ */ T.jsx(
    "span",
    {
      ref: (u) => {
        u && i(u);
      }
    }
  );
}, Bo = (t) => {
  const { children: e, containerRef: n, appendToParentPortal: s } = t, i = n.current, r = i ?? (typeof window < "u" ? document.body : void 0), o = q(() => {
    const c = i == null ? void 0 : i.ownerDocument.createElement("div");
    return c && (c.className = Je), c;
  }, [i]), [, a] = Dt({});
  return Le(() => a({}), []), Le(() => {
    if (!(!o || !r))
      return r.appendChild(o), () => {
        r.removeChild(o);
      };
  }, [o, r]), r && o ? Us(
    /* @__PURE__ */ T.jsx($s, { value: s ? o : null, children: e }),
    o
  ) : null;
};
function ae(t) {
  const e = {
    appendToParentPortal: !0,
    ...t
  }, { containerRef: n, ...s } = e;
  return n ? /* @__PURE__ */ T.jsx(Bo, { containerRef: n, ...s }) : /* @__PURE__ */ T.jsx(jo, { ...s });
}
ae.className = Je;
ae.selector = Lo;
ae.displayName = "Portal";
var Fo = (t, e) => t.find((n) => n.id === e);
function Mn(t, e) {
  const n = zs(t, e), s = n ? t[n].findIndex((i) => i.id === e) : -1;
  return {
    position: n,
    index: s
  };
}
function zs(t, e) {
  for (const [n, s] of Object.entries(t))
    if (Fo(s, e))
      return n;
}
function Io(t) {
  const e = t.includes("right"), n = t.includes("left");
  let s = "center";
  return e && (s = "flex-end"), n && (s = "flex-start"), {
    display: "flex",
    flexDirection: "column",
    alignItems: s
  };
}
function Oo(t) {
  const n = t === "top" || t === "bottom" ? "0 auto" : void 0, s = t.includes("top") ? "env(safe-area-inset-top, 0px)" : void 0, i = t.includes("bottom") ? "env(safe-area-inset-bottom, 0px)" : void 0, r = t.includes("left") ? void 0 : "env(safe-area-inset-right, 0px)", o = t.includes("right") ? void 0 : "env(safe-area-inset-left, 0px)";
  return {
    position: "fixed",
    zIndex: 5500,
    pointerEvents: "none",
    display: "flex",
    flexDirection: "column",
    margin: n,
    top: s,
    bottom: i,
    right: r,
    left: o
  };
}
function No(t, e = []) {
  const n = O(t);
  return U(() => {
    n.current = t;
  }), Xt((...s) => {
    var i;
    return (i = n.current) == null ? void 0 : i.call(n, ...s);
  }, e);
}
function Uo(t, e) {
  const n = No(t);
  U(() => {
    if (e == null)
      return;
    let s = null;
    return s = window.setTimeout(() => {
      n();
    }, e), () => {
      s && window.clearTimeout(s);
    };
  }, [e, n]);
}
function Dn(t, e) {
  const n = O(!1), s = O(!1);
  U(() => {
    if (n.current && s.current)
      return t();
    s.current = !0;
  }, e), U(() => (n.current = !0, () => {
    n.current = !1;
  }), []);
}
const Ws = ut({
  transformPagePoint: (t) => t,
  isStatic: !1,
  reducedMotion: "never"
}), ce = ut({});
function _o() {
  return w(ce).visualElement;
}
const Ot = ut(null), le = typeof document < "u", Qt = le ? Fs : U, Hs = ut({ strict: !1 });
function $o(t, e, n, s) {
  const i = _o(), r = w(Hs), o = w(Ot), a = w(Ws).reducedMotion, c = O();
  s = s || r.renderer, !c.current && s && (c.current = s(t, {
    visualState: e,
    parent: i,
    props: n,
    presenceContext: o,
    blockInitialAnimation: o ? o.initial === !1 : !1,
    reducedMotionConfig: a
  }));
  const l = c.current;
  return Is(() => {
    l && l.update(n, o);
  }), Qt(() => {
    l && l.render();
  }), U(() => {
    l && l.updateFeatures();
  }), (window.HandoffAppearAnimations ? Qt : U)(() => {
    l && l.animationState && l.animationState.animateChanges();
  }), l;
}
function mt(t) {
  return typeof t == "object" && Object.prototype.hasOwnProperty.call(t, "current");
}
function zo(t, e, n) {
  return Xt(
    (s) => {
      s && t.mount && t.mount(s), e && (s ? e.mount(s) : e.unmount()), n && (typeof n == "function" ? n(s) : mt(n) && (n.current = s));
    },
    /**
     * Only pass a new ref callback to React if we've received a visual element
     * factory. Otherwise we'll be mounting/remounting every time externalRef
     * or other dependencies change.
     */
    [e]
  );
}
function Rt(t) {
  return typeof t == "string" || Array.isArray(t);
}
function ue(t) {
  return typeof t == "object" && typeof t.start == "function";
}
const Qe = [
  "animate",
  "whileInView",
  "whileFocus",
  "whileHover",
  "whileTap",
  "whileDrag",
  "exit"
], tn = ["initial", ...Qe];
function fe(t) {
  return ue(t.animate) || tn.some((e) => Rt(t[e]));
}
function Gs(t) {
  return Boolean(fe(t) || t.variants);
}
function Wo(t, e) {
  if (fe(t)) {
    const { initial: n, animate: s } = t;
    return {
      initial: n === !1 || Rt(n) ? n : void 0,
      animate: Rt(s) ? s : void 0
    };
  }
  return t.inherit !== !1 ? e : {};
}
function Ho(t) {
  const { initial: e, animate: n } = Wo(t, w(ce));
  return q(() => ({ initial: e, animate: n }), [Rn(e), Rn(n)]);
}
function Rn(t) {
  return Array.isArray(t) ? t.join(" ") : t;
}
const En = {
  animation: [
    "animate",
    "exit",
    "variants",
    "whileHover",
    "whileTap",
    "whileFocus",
    "whileDrag",
    "whileInView"
  ],
  exit: ["exit"],
  drag: ["drag", "dragControls"],
  focus: ["whileFocus"],
  hover: ["whileHover", "onHoverStart", "onHoverEnd"],
  tap: ["whileTap", "onTap", "onTapStart", "onTapCancel"],
  pan: ["onPan", "onPanStart", "onPanSessionStart", "onPanEnd"],
  inView: ["whileInView", "onViewportEnter", "onViewportLeave"],
  layout: ["layout", "layoutId"]
}, Et = {};
for (const t in En)
  Et[t] = {
    isEnabled: (e) => En[t].some((n) => !!e[n])
  };
function Go(t) {
  for (const e in t)
    Et[e] = {
      ...Et[e],
      ...t[e]
    };
}
function en(t) {
  const e = O(null);
  return e.current === null && (e.current = t()), e.current;
}
const St = {
  /**
   * Global flag as to whether the tree has animated since the last time
   * we resized the window
   */
  hasAnimatedSinceResize: !0,
  /**
   * We set this to true once, on the first update. Any nodes added to the tree beyond that
   * update will be given a `data-projection-id` attribute.
   */
  hasEverUpdated: !1
};
let Ko = 1;
function Zo() {
  return en(() => {
    if (St.hasEverUpdated)
      return Ko++;
  });
}
const nn = ut({}), Ks = ut({}), Yo = Symbol.for("motionComponentSymbol");
function qo({ preloadedFeatures: t, createVisualElement: e, useRender: n, useVisualState: s, Component: i }) {
  t && Go(t);
  function r(a, c) {
    let l;
    const u = {
      ...w(Ws),
      ...a,
      layoutId: Xo(a)
    }, { isStatic: f } = u, h = Ho(a), d = f ? void 0 : Zo(), p = s(a, f);
    if (!f && le) {
      h.visualElement = $o(i, p, u, e);
      const m = w(Ks), P = w(Hs).strict;
      h.visualElement && (l = h.visualElement.loadFeatures(
        // Note: Pass the full new combined props to correctly re-render dynamic feature components.
        u,
        P,
        t,
        d,
        m
      ));
    }
    return L.createElement(
      ce.Provider,
      { value: h },
      l && h.visualElement ? L.createElement(l, { visualElement: h.visualElement, ...u }) : null,
      n(i, a, d, zo(p, h.visualElement, c), p, f, h.visualElement)
    );
  }
  const o = ho(r);
  return o[Yo] = i, o;
}
function Xo({ layoutId: t }) {
  const e = w(nn).id;
  return e && t !== void 0 ? e + "-" + t : t;
}
function Jo(t) {
  function e(s, i = {}) {
    return qo(t(s, i));
  }
  if (typeof Proxy > "u")
    return e;
  const n = /* @__PURE__ */ new Map();
  return new Proxy(e, {
    /**
     * Called when `motion` is referenced with a prop: `motion.div`, `motion.input` etc.
     * The prop name is passed through as `key` and we can use that to generate a `motion`
     * DOM component with that name.
     */
    get: (s, i) => (n.has(i) || n.set(i, e(i)), n.get(i))
  });
}
const Qo = [
  "animate",
  "circle",
  "defs",
  "desc",
  "ellipse",
  "g",
  "image",
  "line",
  "filter",
  "marker",
  "mask",
  "metadata",
  "path",
  "pattern",
  "polygon",
  "polyline",
  "rect",
  "stop",
  "switch",
  "symbol",
  "svg",
  "text",
  "tspan",
  "use",
  "view"
];
function sn(t) {
  return (
    /**
     * If it's not a string, it's a custom React component. Currently we only support
     * HTML custom React components.
     */
    typeof t != "string" || /**
     * If it contains a dash, the element is a custom HTML webcomponent.
     */
    t.includes("-") ? !1 : (
      /**
       * If it's in our list of lowercase SVG tags, it's an SVG component
       */
      !!(Qo.indexOf(t) > -1 || /**
       * If it contains a capital letter, it's an SVG component
       */
      /[A-Z]/.test(t))
    )
  );
}
const te = {};
function tr(t) {
  Object.assign(te, t);
}
const he = [
  "transformPerspective",
  "x",
  "y",
  "z",
  "translateX",
  "translateY",
  "translateZ",
  "scale",
  "scaleX",
  "scaleY",
  "rotate",
  "rotateX",
  "rotateY",
  "rotateZ",
  "skew",
  "skewX",
  "skewY"
], ft = new Set(he);
function Zs(t, { layout: e, layoutId: n }) {
  return ft.has(t) || t.startsWith("origin") || (e || n !== void 0) && (!!te[t] || t === "opacity");
}
const j = (t) => Boolean(t && t.getVelocity), er = {
  x: "translateX",
  y: "translateY",
  z: "translateZ",
  transformPerspective: "perspective"
}, nr = he.length;
function sr(t, { enableHardwareAcceleration: e = !0, allowTransformNone: n = !0 }, s, i) {
  let r = "";
  for (let o = 0; o < nr; o++) {
    const a = he[o];
    if (t[a] !== void 0) {
      const c = er[a] || a;
      r += `${c}(${t[a]}) `;
    }
  }
  return e && !t.z && (r += "translateZ(0)"), r = r.trim(), i ? r = i(t, s ? "" : r) : n && s && (r = "none"), r;
}
const Ys = (t) => (e) => typeof e == "string" && e.startsWith(t), qs = Ys("--"), ke = Ys("var(--"), ir = (t, e) => e && typeof t == "number" ? e.transform(t) : t, vt = (t, e, n) => Math.min(Math.max(n, t), e), ht = {
  test: (t) => typeof t == "number",
  parse: parseFloat,
  transform: (t) => t
}, Ct = {
  ...ht,
  transform: (t) => vt(0, 1, t)
}, Ht = {
  ...ht,
  default: 1
}, At = (t) => Math.round(t * 1e5) / 1e5, Lt = /(-)?([\d]*\.?[\d])+/g, je = /(#[0-9a-f]{3,8}|(rgb|hsl)a?\((-?[\d\.]+%?[,\s]+){2}(-?[\d\.]+%?)\s*[\,\/]?\s*[\d\.]*%?\))/gi, or = /^(#[0-9a-f]{3,8}|(rgb|hsl)a?\((-?[\d\.]+%?[,\s]+){2}(-?[\d\.]+%?)\s*[\,\/]?\s*[\d\.]*%?\))$/i;
function Nt(t) {
  return typeof t == "string";
}
const Ut = (t) => ({
  test: (e) => Nt(e) && e.endsWith(t) && e.split(" ").length === 1,
  parse: parseFloat,
  transform: (e) => `${e}${t}`
}), tt = Ut("deg"), G = Ut("%"), b = Ut("px"), rr = Ut("vh"), ar = Ut("vw"), Ln = {
  ...G,
  parse: (t) => G.parse(t) / 100,
  transform: (t) => G.transform(t * 100)
}, kn = {
  ...ht,
  transform: Math.round
}, Xs = {
  // Border props
  borderWidth: b,
  borderTopWidth: b,
  borderRightWidth: b,
  borderBottomWidth: b,
  borderLeftWidth: b,
  borderRadius: b,
  radius: b,
  borderTopLeftRadius: b,
  borderTopRightRadius: b,
  borderBottomRightRadius: b,
  borderBottomLeftRadius: b,
  // Positioning props
  width: b,
  maxWidth: b,
  height: b,
  maxHeight: b,
  size: b,
  top: b,
  right: b,
  bottom: b,
  left: b,
  // Spacing props
  padding: b,
  paddingTop: b,
  paddingRight: b,
  paddingBottom: b,
  paddingLeft: b,
  margin: b,
  marginTop: b,
  marginRight: b,
  marginBottom: b,
  marginLeft: b,
  // Transform props
  rotate: tt,
  rotateX: tt,
  rotateY: tt,
  rotateZ: tt,
  scale: Ht,
  scaleX: Ht,
  scaleY: Ht,
  scaleZ: Ht,
  skew: tt,
  skewX: tt,
  skewY: tt,
  distance: b,
  translateX: b,
  translateY: b,
  translateZ: b,
  x: b,
  y: b,
  z: b,
  perspective: b,
  transformPerspective: b,
  opacity: Ct,
  originX: Ln,
  originY: Ln,
  originZ: b,
  // Misc
  zIndex: kn,
  // SVG
  fillOpacity: Ct,
  strokeOpacity: Ct,
  numOctaves: kn
};
function on(t, e, n, s) {
  const { style: i, vars: r, transform: o, transformOrigin: a } = t;
  let c = !1, l = !1, u = !0;
  for (const f in e) {
    const h = e[f];
    if (qs(f)) {
      r[f] = h;
      continue;
    }
    const d = Xs[f], p = ir(h, d);
    if (ft.has(f)) {
      if (c = !0, o[f] = p, !u)
        continue;
      h !== (d.default || 0) && (u = !1);
    } else
      f.startsWith("origin") ? (l = !0, a[f] = p) : i[f] = p;
  }
  if (e.transform || (c || s ? i.transform = sr(t.transform, n, u, s) : i.transform && (i.transform = "none")), l) {
    const { originX: f = "50%", originY: h = "50%", originZ: d = 0 } = a;
    i.transformOrigin = `${f} ${h} ${d}`;
  }
}
const rn = () => ({
  style: {},
  transform: {},
  transformOrigin: {},
  vars: {}
});
function Js(t, e, n) {
  for (const s in e)
    !j(e[s]) && !Zs(s, n) && (t[s] = e[s]);
}
function cr({ transformTemplate: t }, e, n) {
  return q(() => {
    const s = rn();
    return on(s, e, { enableHardwareAcceleration: !n }, t), Object.assign({}, s.vars, s.style);
  }, [e]);
}
function lr(t, e, n) {
  const s = t.style || {}, i = {};
  return Js(i, s, t), Object.assign(i, cr(t, e, n)), t.transformValues ? t.transformValues(i) : i;
}
function ur(t, e, n) {
  const s = {}, i = lr(t, e, n);
  return t.drag && t.dragListener !== !1 && (s.draggable = !1, i.userSelect = i.WebkitUserSelect = i.WebkitTouchCallout = "none", i.touchAction = t.drag === !0 ? "none" : `pan-${t.drag === "x" ? "y" : "x"}`), t.tabIndex === void 0 && (t.onTap || t.onTapStart || t.whileTap) && (s.tabIndex = 0), s.style = i, s;
}
const fr = /* @__PURE__ */ new Set([
  "animate",
  "exit",
  "variants",
  "initial",
  "style",
  "values",
  "variants",
  "transition",
  "transformTemplate",
  "transformValues",
  "custom",
  "inherit",
  "onLayoutAnimationStart",
  "onLayoutAnimationComplete",
  "onLayoutMeasure",
  "onBeforeLayoutMeasure",
  "onAnimationStart",
  "onAnimationComplete",
  "onUpdate",
  "onDragStart",
  "onDrag",
  "onDragEnd",
  "onMeasureDragConstraints",
  "onDirectionLock",
  "onDragTransitionEnd",
  "_dragX",
  "_dragY",
  "onHoverStart",
  "onHoverEnd",
  "onViewportEnter",
  "onViewportLeave",
  "ignoreStrict",
  "viewport"
]);
function ee(t) {
  return t.startsWith("while") || t.startsWith("drag") && t !== "draggable" || t.startsWith("layout") || t.startsWith("onTap") || t.startsWith("onPan") || fr.has(t);
}
let Qs = (t) => !ee(t);
function hr(t) {
  t && (Qs = (e) => e.startsWith("on") ? !ee(e) : t(e));
}
try {
  hr(require("@emotion/is-prop-valid").default);
} catch {
}
function dr(t, e, n) {
  const s = {};
  for (const i in t)
    i === "values" && typeof t.values == "object" || (Qs(i) || n === !0 && ee(i) || !e && !ee(i) || // If trying to use native HTML drag events, forward drag listeners
    t.draggable && i.startsWith("onDrag")) && (s[i] = t[i]);
  return s;
}
function jn(t, e, n) {
  return typeof t == "string" ? t : b.transform(e + n * t);
}
function pr(t, e, n) {
  const s = jn(e, t.x, t.width), i = jn(n, t.y, t.height);
  return `${s} ${i}`;
}
const mr = {
  offset: "stroke-dashoffset",
  array: "stroke-dasharray"
}, gr = {
  offset: "strokeDashoffset",
  array: "strokeDasharray"
};
function yr(t, e, n = 1, s = 0, i = !0) {
  t.pathLength = 1;
  const r = i ? mr : gr;
  t[r.offset] = b.transform(-s);
  const o = b.transform(e), a = b.transform(n);
  t[r.array] = `${o} ${a}`;
}
function an(t, {
  attrX: e,
  attrY: n,
  originX: s,
  originY: i,
  pathLength: r,
  pathSpacing: o = 1,
  pathOffset: a = 0,
  // This is object creation, which we try to avoid per-frame.
  ...c
}, l, u, f) {
  if (on(t, c, l, f), u) {
    t.style.viewBox && (t.attrs.viewBox = t.style.viewBox);
    return;
  }
  t.attrs = t.style, t.style = {};
  const { attrs: h, style: d, dimensions: p } = t;
  h.transform && (p && (d.transform = h.transform), delete h.transform), p && (s !== void 0 || i !== void 0 || d.transform) && (d.transformOrigin = pr(p, s !== void 0 ? s : 0.5, i !== void 0 ? i : 0.5)), e !== void 0 && (h.x = e), n !== void 0 && (h.y = n), r !== void 0 && yr(h, r, o, a, !1);
}
const ti = () => ({
  ...rn(),
  attrs: {}
}), cn = (t) => typeof t == "string" && t.toLowerCase() === "svg";
function vr(t, e, n, s) {
  const i = q(() => {
    const r = ti();
    return an(r, e, { enableHardwareAcceleration: !1 }, cn(s), t.transformTemplate), {
      ...r.attrs,
      style: { ...r.style }
    };
  }, [e]);
  if (t.style) {
    const r = {};
    Js(r, t.style, t), i.style = { ...r, ...i.style };
  }
  return i;
}
function xr(t = !1) {
  return (n, s, i, r, { latestValues: o }, a) => {
    const l = (sn(n) ? vr : ur)(s, o, a, n), f = {
      ...dr(s, typeof n == "string", t),
      ...l,
      ref: r
    }, { children: h } = s, d = q(() => j(h) ? h.get() : h, [h]);
    return i && (f["data-projection-id"] = i), po(n, {
      ...f,
      children: d
    });
  };
}
const ln = (t) => t.replace(/([a-z])([A-Z])/g, "$1-$2").toLowerCase();
function ei(t, { style: e, vars: n }, s, i) {
  Object.assign(t.style, e, i && i.getProjectionStyles(s));
  for (const r in n)
    t.style.setProperty(r, n[r]);
}
const ni = /* @__PURE__ */ new Set([
  "baseFrequency",
  "diffuseConstant",
  "kernelMatrix",
  "kernelUnitLength",
  "keySplines",
  "keyTimes",
  "limitingConeAngle",
  "markerHeight",
  "markerWidth",
  "numOctaves",
  "targetX",
  "targetY",
  "surfaceScale",
  "specularConstant",
  "specularExponent",
  "stdDeviation",
  "tableValues",
  "viewBox",
  "gradientTransform",
  "pathLength",
  "startOffset",
  "textLength",
  "lengthAdjust"
]);
function si(t, e, n, s) {
  ei(t, e, void 0, s);
  for (const i in e.attrs)
    t.setAttribute(ni.has(i) ? i : ln(i), e.attrs[i]);
}
function un(t, e) {
  const { style: n } = t, s = {};
  for (const i in n)
    (j(n[i]) || e.style && j(e.style[i]) || Zs(i, t)) && (s[i] = n[i]);
  return s;
}
function ii(t, e) {
  const n = un(t, e);
  for (const s in t)
    if (j(t[s]) || j(e[s])) {
      const i = s === "x" || s === "y" ? "attr" + s.toUpperCase() : s;
      n[i] = t[s];
    }
  return n;
}
function fn(t, e, n, s = {}, i = {}) {
  return typeof e == "function" && (e = e(n !== void 0 ? n : t.custom, s, i)), typeof e == "string" && (e = t.variants && t.variants[e]), typeof e == "function" && (e = e(n !== void 0 ? n : t.custom, s, i)), e;
}
const ne = (t) => Array.isArray(t), br = (t) => Boolean(t && typeof t == "object" && t.mix && t.toValue), Pr = (t) => ne(t) ? t[t.length - 1] || 0 : t;
function Yt(t) {
  const e = j(t) ? t.get() : t;
  return br(e) ? e.toValue() : e;
}
function Tr({ scrapeMotionValuesFromProps: t, createRenderState: e, onMount: n }, s, i, r) {
  const o = {
    latestValues: Sr(s, i, r, t),
    renderState: e()
  };
  return n && (o.mount = (a) => n(s, a, o)), o;
}
const oi = (t) => (e, n) => {
  const s = w(ce), i = w(Ot), r = () => Tr(t, e, s, i);
  return n ? r() : en(r);
};
function Sr(t, e, n, s) {
  const i = {}, r = s(t, {});
  for (const h in r)
    i[h] = Yt(r[h]);
  let { initial: o, animate: a } = t;
  const c = fe(t), l = Gs(t);
  e && l && !c && t.inherit !== !1 && (o === void 0 && (o = e.initial), a === void 0 && (a = e.animate));
  let u = n ? n.initial === !1 : !1;
  u = u || o === !1;
  const f = u ? a : o;
  return f && typeof f != "boolean" && !ue(f) && (Array.isArray(f) ? f : [f]).forEach((d) => {
    const p = fn(t, d);
    if (!p)
      return;
    const { transitionEnd: m, transition: P, ...x } = p;
    for (const v in x) {
      let g = x[v];
      if (Array.isArray(g)) {
        const y = u ? g.length - 1 : 0;
        g = g[y];
      }
      g !== null && (i[v] = g);
    }
    for (const v in m)
      i[v] = m[v];
  }), i;
}
const Cr = {
  useVisualState: oi({
    scrapeMotionValuesFromProps: ii,
    createRenderState: ti,
    onMount: (t, e, { renderState: n, latestValues: s }) => {
      try {
        n.dimensions = typeof e.getBBox == "function" ? e.getBBox() : e.getBoundingClientRect();
      } catch {
        n.dimensions = {
          x: 0,
          y: 0,
          width: 0,
          height: 0
        };
      }
      an(n, s, { enableHardwareAcceleration: !1 }, cn(e.tagName), t.transformTemplate), si(e, n);
    }
  })
}, Ar = {
  useVisualState: oi({
    scrapeMotionValuesFromProps: un,
    createRenderState: rn
  })
};
function Vr(t, { forwardMotionProps: e = !1 }, n, s) {
  return {
    ...sn(t) ? Cr : Ar,
    preloadedFeatures: n,
    useRender: xr(e),
    createVisualElement: s,
    Component: t
  };
}
function K(t, e, n, s = { passive: !0 }) {
  return t.addEventListener(e, n, s), () => t.removeEventListener(e, n);
}
const ri = (t) => t.pointerType === "mouse" ? typeof t.button != "number" || t.button <= 0 : t.isPrimary !== !1;
function de(t, e = "page") {
  return {
    point: {
      x: t[e + "X"],
      y: t[e + "Y"]
    }
  };
}
const wr = (t) => (e) => ri(e) && t(e, de(e));
function Z(t, e, n, s) {
  return K(t, e, wr(n), s);
}
const Mr = (t, e) => (n) => e(t(n)), nt = (...t) => t.reduce(Mr);
function ai(t) {
  let e = null;
  return () => {
    const n = () => {
      e = null;
    };
    return e === null ? (e = t, n) : !1;
  };
}
const Bn = ai("dragHorizontal"), Fn = ai("dragVertical");
function ci(t) {
  let e = !1;
  if (t === "y")
    e = Fn();
  else if (t === "x")
    e = Bn();
  else {
    const n = Bn(), s = Fn();
    n && s ? e = () => {
      n(), s();
    } : (n && n(), s && s());
  }
  return e;
}
function li() {
  const t = ci(!0);
  return t ? (t(), !1) : !0;
}
class ot {
  constructor(e) {
    this.isMounted = !1, this.node = e;
  }
  update() {
  }
}
function In(t, e) {
  const n = "pointer" + (e ? "enter" : "leave"), s = "onHover" + (e ? "Start" : "End"), i = (r, o) => {
    if (r.type === "touch" || li())
      return;
    const a = t.getProps();
    t.animationState && a.whileHover && t.animationState.setActive("whileHover", e), a[s] && a[s](r, o);
  };
  return Z(t.current, n, i, {
    passive: !t.getProps()[s]
  });
}
class Dr extends ot {
  mount() {
    this.unmount = nt(In(this.node, !0), In(this.node, !1));
  }
  unmount() {
  }
}
class Rr extends ot {
  constructor() {
    super(...arguments), this.isActive = !1;
  }
  onFocus() {
    let e = !1;
    try {
      e = this.node.current.matches(":focus-visible");
    } catch {
      e = !0;
    }
    !e || !this.node.animationState || (this.node.animationState.setActive("whileFocus", !0), this.isActive = !0);
  }
  onBlur() {
    !this.isActive || !this.node.animationState || (this.node.animationState.setActive("whileFocus", !1), this.isActive = !1);
  }
  mount() {
    this.unmount = nt(K(this.node.current, "focus", () => this.onFocus()), K(this.node.current, "blur", () => this.onBlur()));
  }
  unmount() {
  }
}
const ui = (t, e) => e ? t === e ? !0 : ui(t, e.parentElement) : !1, N = (t) => t;
function ve(t, e) {
  if (!e)
    return;
  const n = new PointerEvent("pointer" + t);
  e(n, de(n));
}
class Er extends ot {
  constructor() {
    super(...arguments), this.removeStartListeners = N, this.removeEndListeners = N, this.removeAccessibleListeners = N, this.startPointerPress = (e, n) => {
      if (this.removeEndListeners(), this.isPressing)
        return;
      const s = this.node.getProps(), r = Z(window, "pointerup", (a, c) => {
        if (!this.checkPressEnd())
          return;
        const { onTap: l, onTapCancel: u } = this.node.getProps();
        ui(this.node.current, a.target) ? l && l(a, c) : u && u(a, c);
      }, { passive: !(s.onTap || s.onPointerUp) }), o = Z(window, "pointercancel", (a, c) => this.cancelPress(a, c), { passive: !(s.onTapCancel || s.onPointerCancel) });
      this.removeEndListeners = nt(r, o), this.startPress(e, n);
    }, this.startAccessiblePress = () => {
      const e = (r) => {
        if (r.key !== "Enter" || this.isPressing)
          return;
        const o = (a) => {
          a.key !== "Enter" || !this.checkPressEnd() || ve("up", this.node.getProps().onTap);
        };
        this.removeEndListeners(), this.removeEndListeners = K(this.node.current, "keyup", o), ve("down", (a, c) => {
          this.startPress(a, c);
        });
      }, n = K(this.node.current, "keydown", e), s = () => {
        this.isPressing && ve("cancel", (r, o) => this.cancelPress(r, o));
      }, i = K(this.node.current, "blur", s);
      this.removeAccessibleListeners = nt(n, i);
    };
  }
  startPress(e, n) {
    this.isPressing = !0;
    const { onTapStart: s, whileTap: i } = this.node.getProps();
    i && this.node.animationState && this.node.animationState.setActive("whileTap", !0), s && s(e, n);
  }
  checkPressEnd() {
    return this.removeEndListeners(), this.isPressing = !1, this.node.getProps().whileTap && this.node.animationState && this.node.animationState.setActive("whileTap", !1), !li();
  }
  cancelPress(e, n) {
    if (!this.checkPressEnd())
      return;
    const { onTapCancel: s } = this.node.getProps();
    s && s(e, n);
  }
  mount() {
    const e = this.node.getProps(), n = Z(this.node.current, "pointerdown", this.startPointerPress, { passive: !(e.onTapStart || e.onPointerStart) }), s = K(this.node.current, "focus", this.startAccessiblePress);
    this.removeStartListeners = nt(n, s);
  }
  unmount() {
    this.removeStartListeners(), this.removeEndListeners(), this.removeAccessibleListeners();
  }
}
const Be = /* @__PURE__ */ new WeakMap(), xe = /* @__PURE__ */ new WeakMap(), Lr = (t) => {
  const e = Be.get(t.target);
  e && e(t);
}, kr = (t) => {
  t.forEach(Lr);
};
function jr({ root: t, ...e }) {
  const n = t || document;
  xe.has(n) || xe.set(n, {});
  const s = xe.get(n), i = JSON.stringify(e);
  return s[i] || (s[i] = new IntersectionObserver(kr, { root: t, ...e })), s[i];
}
function Br(t, e, n) {
  const s = jr(e);
  return Be.set(t, n), s.observe(t), () => {
    Be.delete(t), s.unobserve(t);
  };
}
const Fr = {
  some: 0,
  all: 1
};
class Ir extends ot {
  constructor() {
    super(...arguments), this.hasEnteredView = !1, this.isInView = !1;
  }
  /**
   * TODO: Remove this in 10.0
   */
  viewportFallback() {
    requestAnimationFrame(() => {
      this.hasEnteredView = !0;
      const { onViewportEnter: e } = this.node.getProps();
      e && e(null), this.node.animationState && this.node.animationState.setActive("whileInView", !0);
    });
  }
  startObserver() {
    this.unmount();
    const { viewport: e = {} } = this.node.getProps(), { root: n, margin: s, amount: i = "some", once: r, fallback: o = !0 } = e;
    if (typeof IntersectionObserver > "u") {
      o && this.viewportFallback();
      return;
    }
    const a = {
      root: n ? n.current : void 0,
      rootMargin: s,
      threshold: typeof i == "number" ? i : Fr[i]
    }, c = (l) => {
      const { isIntersecting: u } = l;
      if (this.isInView === u || (this.isInView = u, r && !u && this.hasEnteredView))
        return;
      u && (this.hasEnteredView = !0), this.node.animationState && this.node.animationState.setActive("whileInView", u);
      const { onViewportEnter: f, onViewportLeave: h } = this.node.getProps(), d = u ? f : h;
      d && d(l);
    };
    return Br(this.node.current, a, c);
  }
  mount() {
    this.startObserver();
  }
  update() {
    if (typeof IntersectionObserver > "u")
      return;
    const { props: e, prevProps: n } = this.node;
    ["amount", "margin", "root"].some(Or(e, n)) && this.startObserver();
  }
  unmount() {
  }
}
function Or({ viewport: t = {} }, { viewport: e = {} } = {}) {
  return (n) => t[n] !== e[n];
}
const Nr = {
  inView: {
    Feature: Ir
  },
  tap: {
    Feature: Er
  },
  focus: {
    Feature: Rr
  },
  hover: {
    Feature: Dr
  }
};
function fi(t, e) {
  if (!Array.isArray(e))
    return !1;
  const n = e.length;
  if (n !== t.length)
    return !1;
  for (let s = 0; s < n; s++)
    if (e[s] !== t[s])
      return !1;
  return !0;
}
const Ur = (t) => /^\-?\d*\.?\d+$/.test(t), _r = (t) => /^0[^.\s]+$/.test(t), Y = {
  delta: 0,
  timestamp: 0
}, hi = 1 / 60 * 1e3, $r = typeof performance < "u" ? () => performance.now() : () => Date.now(), di = typeof window < "u" ? (t) => window.requestAnimationFrame(t) : (t) => setTimeout(() => t($r()), hi);
function zr(t) {
  let e = [], n = [], s = 0, i = !1, r = !1;
  const o = /* @__PURE__ */ new WeakSet(), a = {
    /**
     * Schedule a process to run on the next frame.
     */
    schedule: (c, l = !1, u = !1) => {
      const f = u && i, h = f ? e : n;
      return l && o.add(c), h.indexOf(c) === -1 && (h.push(c), f && i && (s = e.length)), c;
    },
    /**
     * Cancel the provided callback from running on the next frame.
     */
    cancel: (c) => {
      const l = n.indexOf(c);
      l !== -1 && n.splice(l, 1), o.delete(c);
    },
    /**
     * Execute all schedule callbacks.
     */
    process: (c) => {
      if (i) {
        r = !0;
        return;
      }
      if (i = !0, [e, n] = [n, e], n.length = 0, s = e.length, s)
        for (let l = 0; l < s; l++) {
          const u = e[l];
          u(c), o.has(u) && (a.schedule(u), t());
        }
      i = !1, r && (r = !1, a.process(c));
    }
  };
  return a;
}
const Wr = 40;
let Fe = !0, kt = !1, Ie = !1;
const _t = [
  "read",
  "update",
  "preRender",
  "render",
  "postRender"
], pe = _t.reduce((t, e) => (t[e] = zr(() => kt = !0), t), {}), R = _t.reduce((t, e) => {
  const n = pe[e];
  return t[e] = (s, i = !1, r = !1) => (kt || Gr(), n.schedule(s, i, r)), t;
}, {}), st = _t.reduce((t, e) => (t[e] = pe[e].cancel, t), {}), be = _t.reduce((t, e) => (t[e] = () => pe[e].process(Y), t), {}), Hr = (t) => pe[t].process(Y), pi = (t) => {
  kt = !1, Y.delta = Fe ? hi : Math.max(Math.min(t - Y.timestamp, Wr), 1), Y.timestamp = t, Ie = !0, _t.forEach(Hr), Ie = !1, kt && (Fe = !1, di(pi));
}, Gr = () => {
  kt = !0, Fe = !0, Ie || di(pi);
};
function hn(t, e) {
  t.indexOf(e) === -1 && t.push(e);
}
function dn(t, e) {
  const n = t.indexOf(e);
  n > -1 && t.splice(n, 1);
}
class pn {
  constructor() {
    this.subscriptions = [];
  }
  add(e) {
    return hn(this.subscriptions, e), () => dn(this.subscriptions, e);
  }
  notify(e, n, s) {
    const i = this.subscriptions.length;
    if (i)
      if (i === 1)
        this.subscriptions[0](e, n, s);
      else
        for (let r = 0; r < i; r++) {
          const o = this.subscriptions[r];
          o && o(e, n, s);
        }
  }
  getSize() {
    return this.subscriptions.length;
  }
  clear() {
    this.subscriptions.length = 0;
  }
}
function mn(t, e) {
  return e ? t * (1e3 / e) : 0;
}
const Kr = (t) => !isNaN(parseFloat(t));
class Zr {
  /**
   * @param init - The initiating value
   * @param config - Optional configuration options
   *
   * -  `transformer`: A function to transform incoming values with.
   *
   * @internal
   */
  constructor(e, n = {}) {
    this.version = "9.1.7", this.timeDelta = 0, this.lastUpdated = 0, this.canTrackVelocity = !1, this.events = {}, this.updateAndNotify = (s, i = !0) => {
      this.prev = this.current, this.current = s;
      const { delta: r, timestamp: o } = Y;
      this.lastUpdated !== o && (this.timeDelta = r, this.lastUpdated = o, R.postRender(this.scheduleVelocityCheck)), this.prev !== this.current && this.events.change && this.events.change.notify(this.current), this.events.velocityChange && this.events.velocityChange.notify(this.getVelocity()), i && this.events.renderRequest && this.events.renderRequest.notify(this.current);
    }, this.scheduleVelocityCheck = () => R.postRender(this.velocityCheck), this.velocityCheck = ({ timestamp: s }) => {
      s !== this.lastUpdated && (this.prev = this.current, this.events.velocityChange && this.events.velocityChange.notify(this.getVelocity()));
    }, this.hasAnimated = !1, this.prev = this.current = e, this.canTrackVelocity = Kr(this.current), this.owner = n.owner;
  }
  /**
   * Adds a function that will be notified when the `MotionValue` is updated.
   *
   * It returns a function that, when called, will cancel the subscription.
   *
   * When calling `onChange` inside a React component, it should be wrapped with the
   * `useEffect` hook. As it returns an unsubscribe function, this should be returned
   * from the `useEffect` function to ensure you don't add duplicate subscribers..
   *
   * ```jsx
   * export const MyComponent = () => {
   *   const x = useMotionValue(0)
   *   const y = useMotionValue(0)
   *   const opacity = useMotionValue(1)
   *
   *   useEffect(() => {
   *     function updateOpacity() {
   *       const maxXY = Math.max(x.get(), y.get())
   *       const newOpacity = transform(maxXY, [0, 100], [1, 0])
   *       opacity.set(newOpacity)
   *     }
   *
   *     const unsubscribeX = x.on("change", updateOpacity)
   *     const unsubscribeY = y.on("change", updateOpacity)
   *
   *     return () => {
   *       unsubscribeX()
   *       unsubscribeY()
   *     }
   *   }, [])
   *
   *   return <motion.div style={{ x }} />
   * }
   * ```
   *
   * @param subscriber - A function that receives the latest value.
   * @returns A function that, when called, will cancel this subscription.
   *
   * @deprecated
   */
  onChange(e) {
    return this.on("change", e);
  }
  on(e, n) {
    this.events[e] || (this.events[e] = new pn());
    const s = this.events[e].add(n);
    return e === "change" ? () => {
      s(), R.read(() => {
        this.events.change.getSize() || this.stop();
      });
    } : s;
  }
  clearListeners() {
    for (const e in this.events)
      this.events[e].clear();
  }
  /**
   * Attaches a passive effect to the `MotionValue`.
   *
   * @internal
   */
  attach(e, n) {
    this.passiveEffect = e, this.stopPassiveEffect = n;
  }
  /**
   * Sets the state of the `MotionValue`.
   *
   * @remarks
   *
   * ```jsx
   * const x = useMotionValue(0)
   * x.set(10)
   * ```
   *
   * @param latest - Latest value to set.
   * @param render - Whether to notify render subscribers. Defaults to `true`
   *
   * @public
   */
  set(e, n = !0) {
    !n || !this.passiveEffect ? this.updateAndNotify(e, n) : this.passiveEffect(e, this.updateAndNotify);
  }
  setWithVelocity(e, n, s) {
    this.set(n), this.prev = e, this.timeDelta = s;
  }
  /**
   * Set the state of the `MotionValue`, stopping any active animations,
   * effects, and resets velocity to `0`.
   */
  jump(e) {
    this.updateAndNotify(e), this.prev = e, this.stop(), this.stopPassiveEffect && this.stopPassiveEffect();
  }
  /**
   * Returns the latest state of `MotionValue`
   *
   * @returns - The latest state of `MotionValue`
   *
   * @public
   */
  get() {
    return this.current;
  }
  /**
   * @public
   */
  getPrevious() {
    return this.prev;
  }
  /**
   * Returns the latest velocity of `MotionValue`
   *
   * @returns - The latest velocity of `MotionValue`. Returns `0` if the state is non-numerical.
   *
   * @public
   */
  getVelocity() {
    return this.canTrackVelocity ? (
      // These casts could be avoided if parseFloat would be typed better
      mn(parseFloat(this.current) - parseFloat(this.prev), this.timeDelta)
    ) : 0;
  }
  /**
   * Registers a new animation to control this `MotionValue`. Only one
   * animation can drive a `MotionValue` at one time.
   *
   * ```jsx
   * value.start()
   * ```
   *
   * @param animation - A function that starts the provided animation
   *
   * @internal
   */
  start(e) {
    return this.stop(), new Promise((n) => {
      this.hasAnimated = !0, this.animation = e(n) || null, this.events.animationStart && this.events.animationStart.notify();
    }).then(() => {
      this.events.animationComplete && this.events.animationComplete.notify(), this.clearAnimation();
    });
  }
  /**
   * Stop the currently active animation.
   *
   * @public
   */
  stop() {
    this.animation && (this.animation.stop(), this.events.animationCancel && this.events.animationCancel.notify()), this.clearAnimation();
  }
  /**
   * Returns `true` if this value is currently animating.
   *
   * @public
   */
  isAnimating() {
    return !!this.animation;
  }
  clearAnimation() {
    this.animation = null;
  }
  /**
   * Destroy and clean up subscribers to this `MotionValue`.
   *
   * The `MotionValue` hooks like `useMotionValue` and `useTransform` automatically
   * handle the lifecycle of the returned `MotionValue`, so this method is only necessary if you've manually
   * created a `MotionValue` via the `motionValue` function.
   *
   * @public
   */
  destroy() {
    this.clearListeners(), this.stop(), this.stopPassiveEffect && this.stopPassiveEffect();
  }
}
function xt(t, e) {
  return new Zr(t, e);
}
const gn = (t, e) => (n) => Boolean(Nt(n) && or.test(n) && n.startsWith(t) || e && Object.prototype.hasOwnProperty.call(n, e)), mi = (t, e, n) => (s) => {
  if (!Nt(s))
    return s;
  const [i, r, o, a] = s.match(Lt);
  return {
    [t]: parseFloat(i),
    [e]: parseFloat(r),
    [n]: parseFloat(o),
    alpha: a !== void 0 ? parseFloat(a) : 1
  };
}, Yr = (t) => vt(0, 255, t), Pe = {
  ...ht,
  transform: (t) => Math.round(Yr(t))
}, lt = {
  test: gn("rgb", "red"),
  parse: mi("red", "green", "blue"),
  transform: ({ red: t, green: e, blue: n, alpha: s = 1 }) => "rgba(" + Pe.transform(t) + ", " + Pe.transform(e) + ", " + Pe.transform(n) + ", " + At(Ct.transform(s)) + ")"
};
function qr(t) {
  let e = "", n = "", s = "", i = "";
  return t.length > 5 ? (e = t.substring(1, 3), n = t.substring(3, 5), s = t.substring(5, 7), i = t.substring(7, 9)) : (e = t.substring(1, 2), n = t.substring(2, 3), s = t.substring(3, 4), i = t.substring(4, 5), e += e, n += n, s += s, i += i), {
    red: parseInt(e, 16),
    green: parseInt(n, 16),
    blue: parseInt(s, 16),
    alpha: i ? parseInt(i, 16) / 255 : 1
  };
}
const Oe = {
  test: gn("#"),
  parse: qr,
  transform: lt.transform
}, gt = {
  test: gn("hsl", "hue"),
  parse: mi("hue", "saturation", "lightness"),
  transform: ({ hue: t, saturation: e, lightness: n, alpha: s = 1 }) => "hsla(" + Math.round(t) + ", " + G.transform(At(e)) + ", " + G.transform(At(n)) + ", " + At(Ct.transform(s)) + ")"
}, E = {
  test: (t) => lt.test(t) || Oe.test(t) || gt.test(t),
  parse: (t) => lt.test(t) ? lt.parse(t) : gt.test(t) ? gt.parse(t) : Oe.parse(t),
  transform: (t) => Nt(t) ? t : t.hasOwnProperty("red") ? lt.transform(t) : gt.transform(t)
}, gi = "${c}", yi = "${n}";
function Xr(t) {
  var e, n;
  return isNaN(t) && Nt(t) && (((e = t.match(Lt)) === null || e === void 0 ? void 0 : e.length) || 0) + (((n = t.match(je)) === null || n === void 0 ? void 0 : n.length) || 0) > 0;
}
function se(t) {
  typeof t == "number" && (t = `${t}`);
  const e = [];
  let n = 0, s = 0;
  const i = t.match(je);
  i && (n = i.length, t = t.replace(je, gi), e.push(...i.map(E.parse)));
  const r = t.match(Lt);
  return r && (s = r.length, t = t.replace(Lt, yi), e.push(...r.map(ht.parse))), { values: e, numColors: n, numNumbers: s, tokenised: t };
}
function vi(t) {
  return se(t).values;
}
function xi(t) {
  const { values: e, numColors: n, tokenised: s } = se(t), i = e.length;
  return (r) => {
    let o = s;
    for (let a = 0; a < i; a++)
      o = o.replace(a < n ? gi : yi, a < n ? E.transform(r[a]) : At(r[a]));
    return o;
  };
}
const Jr = (t) => typeof t == "number" ? 0 : t;
function Qr(t) {
  const e = vi(t);
  return xi(t)(e.map(Jr));
}
const it = { test: Xr, parse: vi, createTransformer: xi, getAnimatableNone: Qr }, ta = /* @__PURE__ */ new Set(["brightness", "contrast", "saturate", "opacity"]);
function ea(t) {
  const [e, n] = t.slice(0, -1).split("(");
  if (e === "drop-shadow")
    return t;
  const [s] = n.match(Lt) || [];
  if (!s)
    return t;
  const i = n.replace(s, "");
  let r = ta.has(e) ? 1 : 0;
  return s !== n && (r *= 100), e + "(" + r + i + ")";
}
const na = /([a-z-]*)\(.*?\)/g, Ne = {
  ...it,
  getAnimatableNone: (t) => {
    const e = t.match(na);
    return e ? e.map(ea).join(" ") : t;
  }
}, sa = {
  ...Xs,
  // Color props
  color: E,
  backgroundColor: E,
  outlineColor: E,
  fill: E,
  stroke: E,
  // Border props
  borderColor: E,
  borderTopColor: E,
  borderRightColor: E,
  borderBottomColor: E,
  borderLeftColor: E,
  filter: Ne,
  WebkitFilter: Ne
}, yn = (t) => sa[t];
function vn(t, e) {
  let n = yn(t);
  return n !== Ne && (n = it), n.getAnimatableNone ? n.getAnimatableNone(e) : void 0;
}
const bi = (t) => (e) => e.test(t), ia = {
  test: (t) => t === "auto",
  parse: (t) => t
}, Pi = [ht, b, G, tt, ar, rr, ia], bt = (t) => Pi.find(bi(t)), oa = [...Pi, E, it], ra = (t) => oa.find(bi(t));
function aa(t) {
  const e = {};
  return t.values.forEach((n, s) => e[s] = n.get()), e;
}
function ca(t) {
  const e = {};
  return t.values.forEach((n, s) => e[s] = n.getVelocity()), e;
}
function me(t, e, n) {
  const s = t.getProps();
  return fn(s, e, n !== void 0 ? n : s.custom, aa(t), ca(t));
}
function la(t, e, n) {
  t.hasValue(e) ? t.getValue(e).set(n) : t.addValue(e, xt(n));
}
function ua(t, e) {
  const n = me(t, e);
  let { transitionEnd: s = {}, transition: i = {}, ...r } = n ? t.makeTargetAnimatable(n, !1) : {};
  r = { ...r, ...s };
  for (const o in r) {
    const a = Pr(r[o]);
    la(t, o, a);
  }
}
function fa(t, e, n) {
  var s, i;
  const r = Object.keys(e).filter((a) => !t.hasValue(a)), o = r.length;
  if (o)
    for (let a = 0; a < o; a++) {
      const c = r[a], l = e[c];
      let u = null;
      Array.isArray(l) && (u = l[0]), u === null && (u = (i = (s = n[c]) !== null && s !== void 0 ? s : t.readValue(c)) !== null && i !== void 0 ? i : e[c]), u != null && (typeof u == "string" && (Ur(u) || _r(u)) ? u = parseFloat(u) : !ra(u) && it.test(l) && (u = vn(c, l)), t.addValue(c, xt(u, { owner: t })), n[c] === void 0 && (n[c] = u), u !== null && t.setBaseTarget(c, u));
    }
}
function ha(t, e) {
  return e ? (e[t] || e.default || e).from : void 0;
}
function da(t, e, n) {
  const s = {};
  for (const i in t) {
    const r = ha(i, e);
    if (r !== void 0)
      s[i] = r;
    else {
      const o = n.getValue(i);
      o && (s[i] = o.get());
    }
  }
  return s;
}
function ie(t) {
  return Boolean(j(t) && t.add);
}
const pa = "framerAppearId", ma = "data-" + ln(pa);
let $t = N, $ = N;
process.env.NODE_ENV !== "production" && ($t = (t, e) => {
  !t && typeof console < "u" && console.warn(e);
}, $ = (t, e) => {
  if (!t)
    throw new Error(e);
});
const qt = (t) => t * 1e3, ga = {
  current: !1
}, xn = (t) => (e) => e <= 0.5 ? t(2 * e) / 2 : (2 - t(2 * (1 - e))) / 2, bn = (t) => (e) => 1 - t(1 - e), Pn = (t) => t * t, ya = bn(Pn), Tn = xn(Pn), C = (t, e, n) => -n * t + n * e + t;
function Te(t, e, n) {
  return n < 0 && (n += 1), n > 1 && (n -= 1), n < 1 / 6 ? t + (e - t) * 6 * n : n < 1 / 2 ? e : n < 2 / 3 ? t + (e - t) * (2 / 3 - n) * 6 : t;
}
function va({ hue: t, saturation: e, lightness: n, alpha: s }) {
  t /= 360, e /= 100, n /= 100;
  let i = 0, r = 0, o = 0;
  if (!e)
    i = r = o = n;
  else {
    const a = n < 0.5 ? n * (1 + e) : n + e - n * e, c = 2 * n - a;
    i = Te(c, a, t + 1 / 3), r = Te(c, a, t), o = Te(c, a, t - 1 / 3);
  }
  return {
    red: Math.round(i * 255),
    green: Math.round(r * 255),
    blue: Math.round(o * 255),
    alpha: s
  };
}
const Se = (t, e, n) => {
  const s = t * t;
  return Math.sqrt(Math.max(0, n * (e * e - s) + s));
}, xa = [Oe, lt, gt], ba = (t) => xa.find((e) => e.test(t));
function On(t) {
  const e = ba(t);
  $(Boolean(e), `'${t}' is not an animatable color. Use the equivalent color code instead.`);
  let n = e.parse(t);
  return e === gt && (n = va(n)), n;
}
const Ti = (t, e) => {
  const n = On(t), s = On(e), i = { ...n };
  return (r) => (i.red = Se(n.red, s.red, r), i.green = Se(n.green, s.green, r), i.blue = Se(n.blue, s.blue, r), i.alpha = C(n.alpha, s.alpha, r), lt.transform(i));
};
function Si(t, e) {
  return typeof t == "number" ? (n) => C(t, e, n) : E.test(t) ? Ti(t, e) : Ai(t, e);
}
const Ci = (t, e) => {
  const n = [...t], s = n.length, i = t.map((r, o) => Si(r, e[o]));
  return (r) => {
    for (let o = 0; o < s; o++)
      n[o] = i[o](r);
    return n;
  };
}, Pa = (t, e) => {
  const n = { ...t, ...e }, s = {};
  for (const i in n)
    t[i] !== void 0 && e[i] !== void 0 && (s[i] = Si(t[i], e[i]));
  return (i) => {
    for (const r in s)
      n[r] = s[r](i);
    return n;
  };
}, Ai = (t, e) => {
  const n = it.createTransformer(e), s = se(t), i = se(e);
  return s.numColors === i.numColors && s.numNumbers >= i.numNumbers ? nt(Ci(s.values, i.values), n) : ($t(!0, `Complex values '${t}' and '${e}' too different to mix. Ensure all colors are of the same type, and that each contains the same quantity of number and color values. Falling back to instant transition.`), (o) => `${o > 0 ? e : t}`);
}, jt = (t, e, n) => {
  const s = e - t;
  return s === 0 ? 1 : (n - t) / s;
}, Nn = (t, e) => (n) => C(t, e, n);
function Ta(t) {
  return typeof t == "number" ? Nn : typeof t == "string" ? E.test(t) ? Ti : Ai : Array.isArray(t) ? Ci : typeof t == "object" ? Pa : Nn;
}
function Sa(t, e, n) {
  const s = [], i = n || Ta(t[0]), r = t.length - 1;
  for (let o = 0; o < r; o++) {
    let a = i(t[o], t[o + 1]);
    if (e) {
      const c = Array.isArray(e) ? e[o] : e;
      a = nt(c, a);
    }
    s.push(a);
  }
  return s;
}
function Vi(t, e, { clamp: n = !0, ease: s, mixer: i } = {}) {
  const r = t.length;
  $(r === e.length, "Both input and output ranges must be the same length"), $(!s || !Array.isArray(s) || s.length === r - 1, "Array of easing functions must be of length `input.length - 1`, as it applies to the transitions **between** the defined values."), t[0] > t[r - 1] && (t = [...t].reverse(), e = [...e].reverse());
  const o = Sa(e, s, i), a = o.length, c = (l) => {
    let u = 0;
    if (a > 1)
      for (; u < t.length - 2 && !(l < t[u + 1]); u++)
        ;
    const f = jt(t[u], t[u + 1], l);
    return o[u](f);
  };
  return n ? (l) => c(vt(t[0], t[r - 1], l)) : c;
}
function Ca(t, e) {
  const n = t[t.length - 1];
  for (let s = 1; s <= e; s++) {
    const i = jt(0, e, s);
    t.push(C(n, 1, i));
  }
}
function Aa(t) {
  const e = [0];
  return Ca(e, t.length - 1), e;
}
function Va(t, e) {
  return t.map((n) => n * e);
}
const wi = (t, e, n) => (((1 - 3 * n + 3 * e) * t + (3 * n - 6 * e)) * t + 3 * e) * t, wa = 1e-7, Ma = 12;
function Da(t, e, n, s, i) {
  let r, o, a = 0;
  do
    o = e + (n - e) / 2, r = wi(o, s, i) - t, r > 0 ? n = o : e = o;
  while (Math.abs(r) > wa && ++a < Ma);
  return o;
}
function Mi(t, e, n, s) {
  if (t === e && n === s)
    return N;
  const i = (r) => Da(r, 0, 1, t, n);
  return (r) => r === 0 || r === 1 ? r : wi(i(r), e, s);
}
const Di = (t) => 1 - Math.sin(Math.acos(t)), Sn = bn(Di), Ra = xn(Sn), Ri = Mi(0.33, 1.53, 0.69, 0.99), Cn = bn(Ri), Ea = xn(Cn), La = (t) => (t *= 2) < 1 ? 0.5 * Cn(t) : 0.5 * (2 - Math.pow(2, -10 * (t - 1))), Un = {
  linear: N,
  easeIn: Pn,
  easeInOut: Tn,
  easeOut: ya,
  circIn: Di,
  circInOut: Ra,
  circOut: Sn,
  backIn: Cn,
  backInOut: Ea,
  backOut: Ri,
  anticipate: La
}, _n = (t) => {
  if (Array.isArray(t)) {
    $(t.length === 4, "Cubic bezier arrays must contain four numerical values.");
    const [e, n, s, i] = t;
    return Mi(e, n, s, i);
  } else if (typeof t == "string")
    return $(Un[t] !== void 0, `Invalid easing type '${t}'`), Un[t];
  return t;
}, ka = (t) => Array.isArray(t) && typeof t[0] != "number";
function ja(t, e) {
  return t.map(() => e || Tn).splice(0, t.length - 1);
}
function Ue({ keyframes: t, ease: e = Tn, times: n, duration: s = 300 }) {
  t = [...t];
  const i = ka(e) ? e.map(_n) : _n(e), r = {
    done: !1,
    value: t[0]
  }, o = Va(
    // Only use the provided offsets if they're the correct length
    // TODO Maybe we should warn here if there's a length mismatch
    n && n.length === t.length ? n : Aa(t),
    s
  );
  function a() {
    return Vi(o, t, {
      ease: Array.isArray(i) ? i : ja(t, i)
    });
  }
  let c = a();
  return {
    next: (l) => (r.value = c(l), r.done = l >= s, r),
    flipTarget: () => {
      t.reverse(), c = a();
    }
  };
}
const Ce = 1e-3, Ba = 0.01, $n = 10, Fa = 0.05, Ia = 1;
function Oa({ duration: t = 800, bounce: e = 0.25, velocity: n = 0, mass: s = 1 }) {
  let i, r;
  $t(t <= $n * 1e3, "Spring duration must be 10 seconds or less");
  let o = 1 - e;
  o = vt(Fa, Ia, o), t = vt(Ba, $n, t / 1e3), o < 1 ? (i = (l) => {
    const u = l * o, f = u * t, h = u - n, d = _e(l, o), p = Math.exp(-f);
    return Ce - h / d * p;
  }, r = (l) => {
    const f = l * o * t, h = f * n + n, d = Math.pow(o, 2) * Math.pow(l, 2) * t, p = Math.exp(-f), m = _e(Math.pow(l, 2), o);
    return (-i(l) + Ce > 0 ? -1 : 1) * ((h - d) * p) / m;
  }) : (i = (l) => {
    const u = Math.exp(-l * t), f = (l - n) * t + 1;
    return -Ce + u * f;
  }, r = (l) => {
    const u = Math.exp(-l * t), f = (n - l) * (t * t);
    return u * f;
  });
  const a = 5 / t, c = Ua(i, r, a);
  if (t = t * 1e3, isNaN(c))
    return {
      stiffness: 100,
      damping: 10,
      duration: t
    };
  {
    const l = Math.pow(c, 2) * s;
    return {
      stiffness: l,
      damping: o * 2 * Math.sqrt(s * l),
      duration: t
    };
  }
}
const Na = 12;
function Ua(t, e, n) {
  let s = n;
  for (let i = 1; i < Na; i++)
    s = s - t(s) / e(s);
  return s;
}
function _e(t, e) {
  return t * Math.sqrt(1 - e * e);
}
const _a = ["duration", "bounce"], $a = ["stiffness", "damping", "mass"];
function zn(t, e) {
  return e.some((n) => t[n] !== void 0);
}
function za(t) {
  let e = {
    velocity: 0,
    stiffness: 100,
    damping: 10,
    mass: 1,
    isResolvedFromDuration: !1,
    ...t
  };
  if (!zn(t, $a) && zn(t, _a)) {
    const n = Oa(t);
    e = {
      ...e,
      ...n,
      velocity: 0,
      mass: 1
    }, e.isResolvedFromDuration = !0;
  }
  return e;
}
const Wa = 5;
function Ei({ keyframes: t, restDelta: e, restSpeed: n, ...s }) {
  let i = t[0], r = t[t.length - 1];
  const o = { done: !1, value: i }, { stiffness: a, damping: c, mass: l, velocity: u, duration: f, isResolvedFromDuration: h } = za(s);
  let d = Ha, p = u ? -(u / 1e3) : 0;
  const m = c / (2 * Math.sqrt(a * l));
  function P() {
    const x = r - i, v = Math.sqrt(a / l) / 1e3, g = Math.abs(x) < 5;
    if (n || (n = g ? 0.01 : 2), e || (e = g ? 5e-3 : 0.5), m < 1) {
      const y = _e(v, m);
      d = (S) => {
        const V = Math.exp(-m * v * S);
        return r - V * ((p + m * v * x) / y * Math.sin(y * S) + x * Math.cos(y * S));
      };
    } else if (m === 1)
      d = (y) => r - Math.exp(-v * y) * (x + (p + v * x) * y);
    else {
      const y = v * Math.sqrt(m * m - 1);
      d = (S) => {
        const V = Math.exp(-m * v * S), D = Math.min(y * S, 300);
        return r - V * ((p + m * v * x) * Math.sinh(D) + y * x * Math.cosh(D)) / y;
      };
    }
  }
  return P(), {
    next: (x) => {
      const v = d(x);
      if (h)
        o.done = x >= f;
      else {
        let g = p;
        if (x !== 0)
          if (m < 1) {
            const V = Math.max(0, x - Wa);
            g = mn(v - d(V), x - V);
          } else
            g = 0;
        const y = Math.abs(g) <= n, S = Math.abs(r - v) <= e;
        o.done = y && S;
      }
      return o.value = o.done ? r : v, o;
    },
    flipTarget: () => {
      p = -p, [i, r] = [r, i], P();
    }
  };
}
Ei.needsInterpolation = (t, e) => typeof t == "string" || typeof e == "string";
const Ha = (t) => 0;
function Ga({
  /**
   * The decay animation dynamically calculates an end of the animation
   * based on the initial keyframe, so we only need to define a single keyframe
   * as default.
   */
  keyframes: t = [0],
  velocity: e = 0,
  power: n = 0.8,
  timeConstant: s = 350,
  restDelta: i = 0.5,
  modifyTarget: r
}) {
  const o = t[0], a = { done: !1, value: o };
  let c = n * e;
  const l = o + c, u = r === void 0 ? l : r(l);
  return u !== l && (c = u - o), {
    next: (f) => {
      const h = -c * Math.exp(-f / s);
      return a.done = !(h > i || h < -i), a.value = a.done ? u : u + h, a;
    },
    flipTarget: () => {
    }
  };
}
const Ka = {
  decay: Ga,
  keyframes: Ue,
  tween: Ue,
  spring: Ei
};
function Li(t, e, n = 0) {
  return t - e - n;
}
function Za(t, e = 0, n = 0, s = !0) {
  return s ? Li(e + -t, e, n) : e - (t - e) + n;
}
function Ya(t, e, n, s) {
  return s ? t >= e + n : t <= -n;
}
const qa = (t) => {
  const e = ({ delta: n }) => t(n);
  return {
    start: () => R.update(e, !0),
    stop: () => st.update(e)
  };
};
function oe({ duration: t, driver: e = qa, elapsed: n = 0, repeat: s = 0, repeatType: i = "loop", repeatDelay: r = 0, keyframes: o, autoplay: a = !0, onPlay: c, onStop: l, onComplete: u, onRepeat: f, onUpdate: h, type: d = "keyframes", ...p }) {
  const m = n;
  let P, x = 0, v = t, g = !1, y = !0, S;
  const V = Ka[o.length > 2 ? "keyframes" : d] || Ue, D = o[0], z = o[o.length - 1];
  let F = { done: !1, value: D };
  const { needsInterpolation: _ } = V;
  _ && _(D, z) && (S = Vi([0, 100], [D, z], {
    clamp: !1
  }), o = [0, 100]);
  const J = V({
    ...p,
    duration: t,
    keyframes: o
  });
  function Wt() {
    x++, i === "reverse" ? (y = x % 2 === 0, n = Za(n, v, r, y)) : (n = Li(n, v, r), i === "mirror" && J.flipTarget()), g = !1, f && f();
  }
  function ge() {
    P && P.stop(), u && u();
  }
  function Q(k) {
    y || (k = -k), n += k, g || (F = J.next(Math.max(0, n)), S && (F.value = S(F.value)), g = y ? F.done : n <= 0), h && h(F.value), g && (x === 0 && (v = v !== void 0 ? v : n), x < s ? Ya(n, v, r, y) && Wt() : ge());
  }
  function M() {
    c && c(), P = e(Q), P.start();
  }
  return a && M(), {
    stop: () => {
      l && l(), P && P.stop();
    },
    /**
     * Set the current time of the animation. This is purposefully
     * mirroring the WAAPI animation API to make them interchanagable.
     * Going forward this file should be ported more towards
     * https://github.com/motiondivision/motionone/blob/main/packages/animation/src/Animation.ts
     * Which behaviourally adheres to WAAPI as far as possible.
     *
     * WARNING: This is not safe to use for most animations. We currently
     * only use it for handoff from WAAPI within Framer.
     *
     * This animation function consumes time every frame rather than being sampled for time.
     * So the sample() method performs some headless frames to ensure
     * repeats are handled correctly. Ideally in the future we will replace
     * that method with this, once repeat calculations are pure.
     */
    set currentTime(k) {
      n = m, Q(k);
    },
    /**
     * animate() can't yet be sampled for time, instead it
     * consumes time. So to sample it we have to run a low
     * temporal-resolution version.
     */
    sample: (k) => {
      n = m;
      const dt = t && typeof t == "number" ? Math.max(t * 0.5, 50) : 50;
      let ye = 0;
      for (Q(0); ye <= k; ) {
        const fo = k - ye;
        Q(Math.min(fo, dt)), ye += dt;
      }
      return F;
    }
  };
}
function Xa(t) {
  return !t || // Default easing
  Array.isArray(t) || // Bezier curve
  typeof t == "string" && ki[t];
}
const Tt = ([t, e, n, s]) => `cubic-bezier(${t}, ${e}, ${n}, ${s})`, ki = {
  linear: "linear",
  ease: "ease",
  easeIn: "ease-in",
  easeOut: "ease-out",
  easeInOut: "ease-in-out",
  circIn: Tt([0, 0.65, 0.55, 1]),
  circOut: Tt([0.55, 0, 1, 0.45]),
  backIn: Tt([0.31, 0.01, 0.66, -0.59]),
  backOut: Tt([0.33, 1.53, 0.69, 0.99])
};
function Ja(t) {
  if (t)
    return Array.isArray(t) ? Tt(t) : ki[t];
}
function Qa(t, e, n, { delay: s = 0, duration: i, repeat: r = 0, repeatType: o = "loop", ease: a, times: c } = {}) {
  return t.animate({ [e]: n, offset: c }, {
    delay: s,
    duration: i,
    easing: Ja(a),
    fill: "both",
    iterations: r + 1,
    direction: o === "reverse" ? "alternate" : "normal"
  });
}
const Wn = {
  waapi: () => Object.hasOwnProperty.call(Element.prototype, "animate")
}, Ae = {}, ji = {};
for (const t in Wn)
  ji[t] = () => (Ae[t] === void 0 && (Ae[t] = Wn[t]()), Ae[t]);
function tc(t, { repeat: e, repeatType: n = "loop" }) {
  const s = e && n !== "loop" && e % 2 === 1 ? 0 : t.length - 1;
  return t[s];
}
const ec = /* @__PURE__ */ new Set([
  "opacity",
  "clipPath",
  "filter",
  "transform"
]), Gt = 10;
function nc(t, e, { onUpdate: n, onComplete: s, ...i }) {
  if (!(ji.waapi() && ec.has(e) && !i.repeatDelay && i.repeatType !== "mirror" && i.damping !== 0))
    return !1;
  let { keyframes: o, duration: a = 300, elapsed: c = 0, ease: l } = i;
  if (i.type === "spring" || !Xa(i.ease)) {
    if (i.repeat === 1 / 0)
      return;
    const f = oe({ ...i, elapsed: 0 });
    let h = { done: !1, value: o[0] };
    const d = [];
    let p = 0;
    for (; !h.done && p < 2e4; )
      h = f.sample(p), d.push(h.value), p += Gt;
    o = d, a = p - Gt, l = "linear";
  }
  const u = Qa(t.owner.current, e, o, {
    ...i,
    delay: -c,
    duration: a,
    /**
     * This function is currently not called if ease is provided
     * as a function so the cast is safe.
     *
     * However it would be possible for a future refinement to port
     * in easing pregeneration from Motion One for browsers that
     * support the upcoming `linear()` easing function.
     */
    ease: l
  });
  return u.onfinish = () => {
    t.set(tc(o, i)), R.update(() => u.cancel()), s && s();
  }, {
    get currentTime() {
      return u.currentTime || 0;
    },
    set currentTime(f) {
      u.currentTime = f;
    },
    stop: () => {
      const { currentTime: f } = u;
      if (f) {
        const h = oe({
          ...i,
          autoplay: !1
        });
        t.setWithVelocity(h.sample(f - Gt).value, h.sample(f).value, Gt);
      }
      R.update(() => u.cancel());
    }
  };
}
function Bi(t, e) {
  const n = performance.now(), s = ({ timestamp: i }) => {
    const r = i - n;
    r >= e && (st.read(s), t(r - e));
  };
  return R.read(s, !0), () => st.read(s);
}
function sc({ keyframes: t, elapsed: e, onUpdate: n, onComplete: s }) {
  const i = () => {
    n && n(t[t.length - 1]), s && s();
  };
  return e ? { stop: Bi(i, -e) } : i();
}
function ic({ keyframes: t, velocity: e = 0, min: n, max: s, power: i = 0.8, timeConstant: r = 750, bounceStiffness: o = 500, bounceDamping: a = 10, restDelta: c = 1, modifyTarget: l, driver: u, onUpdate: f, onComplete: h, onStop: d }) {
  const p = t[0];
  let m;
  function P(y) {
    return n !== void 0 && y < n || s !== void 0 && y > s;
  }
  function x(y) {
    return n === void 0 ? s : s === void 0 || Math.abs(n - y) < Math.abs(s - y) ? n : s;
  }
  function v(y) {
    m && m.stop(), m = oe({
      keyframes: [0, 1],
      velocity: 0,
      ...y,
      driver: u,
      onUpdate: (S) => {
        f && f(S), y.onUpdate && y.onUpdate(S);
      },
      onComplete: h,
      onStop: d
    });
  }
  function g(y) {
    v({
      type: "spring",
      stiffness: o,
      damping: a,
      restDelta: c,
      ...y
    });
  }
  if (P(p))
    g({
      velocity: e,
      keyframes: [p, x(p)]
    });
  else {
    let y = i * e + p;
    typeof l < "u" && (y = l(y));
    const S = x(y), V = S === n ? -1 : 1;
    let D, z;
    const F = (_) => {
      D = z, z = _, e = mn(_ - D, Y.delta), (V === 1 && _ > S || V === -1 && _ < S) && g({ keyframes: [_, S], velocity: e });
    };
    v({
      type: "decay",
      keyframes: [p, 0],
      velocity: e,
      timeConstant: r,
      power: i,
      restDelta: c,
      modifyTarget: l,
      onUpdate: P(y) ? F : void 0
    });
  }
  return {
    stop: () => m && m.stop()
  };
}
const oc = {
  type: "spring",
  stiffness: 500,
  damping: 25,
  restSpeed: 10
}, rc = (t) => ({
  type: "spring",
  stiffness: 550,
  damping: t === 0 ? 2 * Math.sqrt(550) : 30,
  restSpeed: 10
}), ac = {
  type: "keyframes",
  duration: 0.8
}, cc = {
  type: "keyframes",
  ease: [0.25, 0.1, 0.35, 1],
  duration: 0.3
}, lc = (t, { keyframes: e }) => e.length > 2 ? ac : ft.has(t) ? t.startsWith("scale") ? rc(e[1]) : oc : cc, $e = (t, e) => t === "zIndex" ? !1 : !!(typeof e == "number" || Array.isArray(e) || typeof e == "string" && // It's animatable if we have a string
it.test(e) && // And it contains numbers and/or colors
!e.startsWith("url("));
function uc({ when: t, delay: e, delayChildren: n, staggerChildren: s, staggerDirection: i, repeat: r, repeatType: o, repeatDelay: a, from: c, elapsed: l, ...u }) {
  return !!Object.keys(u).length;
}
function Hn(t) {
  return t === 0 || typeof t == "string" && parseFloat(t) === 0 && t.indexOf(" ") === -1;
}
function Gn(t) {
  return typeof t == "number" ? 0 : vn("", t);
}
function Fi(t, e) {
  return t[e] || t.default || t;
}
function fc(t, e, n, s) {
  const i = $e(e, n);
  let r = s.from !== void 0 ? s.from : t.get();
  return r === "none" && i && typeof n == "string" ? r = vn(e, n) : Hn(r) && typeof n == "string" ? r = Gn(n) : !Array.isArray(n) && Hn(n) && typeof r == "string" && (n = Gn(r)), Array.isArray(n) ? (n[0] === null && (n[0] = r), n) : [r, n];
}
const An = (t, e, n, s = {}) => (i) => {
  const r = Fi(s, t) || {}, o = r.delay || s.delay || 0;
  let { elapsed: a = 0 } = s;
  a = a - qt(o);
  const c = fc(e, t, n, r), l = c[0], u = c[c.length - 1], f = $e(t, l), h = $e(t, u);
  $t(f === h, `You are trying to animate ${t} from "${l}" to "${u}". ${l} is not an animatable value - to enable this animation set ${l} to a value animatable to ${u} via the \`style\` property.`);
  let d = {
    keyframes: c,
    velocity: e.getVelocity(),
    ...r,
    elapsed: a,
    onUpdate: (p) => {
      e.set(p), r.onUpdate && r.onUpdate(p);
    },
    onComplete: () => {
      i(), r.onComplete && r.onComplete();
    }
  };
  if (!f || !h || ga.current || r.type === !1)
    return sc(d);
  if (r.type === "inertia")
    return ic(d);
  if (uc(r) || (d = {
    ...d,
    ...lc(t, d)
  }), d.duration && (d.duration = qt(d.duration)), d.repeatDelay && (d.repeatDelay = qt(d.repeatDelay)), e.owner && e.owner.current instanceof HTMLElement && !e.owner.getProps().onUpdate) {
    const p = nc(e, t, d);
    if (p)
      return p;
  }
  return oe(d);
};
function hc(t, e, n = {}) {
  t.notify("AnimationStart", e);
  let s;
  if (Array.isArray(e)) {
    const i = e.map((r) => ze(t, r, n));
    s = Promise.all(i);
  } else if (typeof e == "string")
    s = ze(t, e, n);
  else {
    const i = typeof e == "function" ? me(t, e, n.custom) : e;
    s = Ii(t, i, n);
  }
  return s.then(() => t.notify("AnimationComplete", e));
}
function ze(t, e, n = {}) {
  const s = me(t, e, n.custom);
  let { transition: i = t.getDefaultTransition() || {} } = s || {};
  n.transitionOverride && (i = n.transitionOverride);
  const r = s ? () => Ii(t, s, n) : () => Promise.resolve(), o = t.variantChildren && t.variantChildren.size ? (c = 0) => {
    const { delayChildren: l = 0, staggerChildren: u, staggerDirection: f } = i;
    return dc(t, e, l + c, u, f, n);
  } : () => Promise.resolve(), { when: a } = i;
  if (a) {
    const [c, l] = a === "beforeChildren" ? [r, o] : [o, r];
    return c().then(l);
  } else
    return Promise.all([r(), o(n.delay)]);
}
function Ii(t, e, { delay: n = 0, transitionOverride: s, type: i } = {}) {
  let { transition: r = t.getDefaultTransition(), transitionEnd: o, ...a } = t.makeTargetAnimatable(e);
  const c = t.getValue("willChange");
  s && (r = s);
  const l = [], u = i && t.animationState && t.animationState.getState()[i];
  for (const f in a) {
    const h = t.getValue(f), d = a[f];
    if (!h || d === void 0 || u && mc(u, f))
      continue;
    const p = { delay: n, elapsed: 0, ...r };
    if (window.HandoffAppearAnimations && !h.hasAnimated) {
      const P = t.getProps()[ma];
      P && (p.elapsed = window.HandoffAppearAnimations(P, f, h, R));
    }
    let m = h.start(An(f, h, d, t.shouldReduceMotion && ft.has(f) ? { type: !1 } : p));
    ie(c) && (c.add(f), m = m.then(() => c.remove(f))), l.push(m);
  }
  return Promise.all(l).then(() => {
    o && ua(t, o);
  });
}
function dc(t, e, n = 0, s = 0, i = 1, r) {
  const o = [], a = (t.variantChildren.size - 1) * s, c = i === 1 ? (l = 0) => l * s : (l = 0) => a - l * s;
  return Array.from(t.variantChildren).sort(pc).forEach((l, u) => {
    l.notify("AnimationStart", e), o.push(ze(l, e, {
      ...r,
      delay: n + c(u)
    }).then(() => l.notify("AnimationComplete", e)));
  }), Promise.all(o);
}
function pc(t, e) {
  return t.sortNodePosition(e);
}
function mc({ protectedKeys: t, needsAnimating: e }, n) {
  const s = t.hasOwnProperty(n) && e[n] !== !0;
  return e[n] = !1, s;
}
const gc = [...Qe].reverse(), yc = Qe.length;
function vc(t) {
  return (e) => Promise.all(e.map(({ animation: n, options: s }) => hc(t, n, s)));
}
function xc(t) {
  let e = vc(t);
  const n = Pc();
  let s = !0;
  const i = (c, l) => {
    const u = me(t, l);
    if (u) {
      const { transition: f, transitionEnd: h, ...d } = u;
      c = { ...c, ...d, ...h };
    }
    return c;
  };
  function r(c) {
    e = c(t);
  }
  function o(c, l) {
    const u = t.getProps(), f = t.getVariantContext(!0) || {}, h = [], d = /* @__PURE__ */ new Set();
    let p = {}, m = 1 / 0;
    for (let x = 0; x < yc; x++) {
      const v = gc[x], g = n[v], y = u[v] !== void 0 ? u[v] : f[v], S = Rt(y), V = v === l ? g.isActive : null;
      V === !1 && (m = x);
      let D = y === f[v] && y !== u[v] && S;
      if (D && s && t.manuallyAnimateOnMount && (D = !1), g.protectedKeys = { ...p }, // If it isn't active and hasn't *just* been set as inactive
      !g.isActive && V === null || // If we didn't and don't have any defined prop for this animation type
      !y && !g.prevProp || // Or if the prop doesn't define an animation
      ue(y) || typeof y == "boolean")
        continue;
      const z = bc(g.prevProp, y);
      let F = z || // If we're making this variant active, we want to always make it active
      v === l && g.isActive && !D && S || // If we removed a higher-priority variant (i is in reverse order)
      x > m && S;
      const _ = Array.isArray(y) ? y : [y];
      let J = _.reduce(i, {});
      V === !1 && (J = {});
      const { prevResolvedValues: Wt = {} } = g, ge = {
        ...Wt,
        ...J
      }, Q = (M) => {
        F = !0, d.delete(M), g.needsAnimating[M] = !0;
      };
      for (const M in ge) {
        const k = J[M], dt = Wt[M];
        p.hasOwnProperty(M) || (k !== dt ? ne(k) && ne(dt) ? !fi(k, dt) || z ? Q(M) : g.protectedKeys[M] = !0 : k !== void 0 ? Q(M) : d.add(M) : k !== void 0 && d.has(M) ? Q(M) : g.protectedKeys[M] = !0);
      }
      g.prevProp = y, g.prevResolvedValues = J, g.isActive && (p = { ...p, ...J }), s && t.blockInitialAnimation && (F = !1), F && !D && h.push(..._.map((M) => ({
        animation: M,
        options: { type: v, ...c }
      })));
    }
    if (d.size) {
      const x = {};
      d.forEach((v) => {
        const g = t.getBaseTarget(v);
        g !== void 0 && (x[v] = g);
      }), h.push({ animation: x });
    }
    let P = Boolean(h.length);
    return s && u.initial === !1 && !t.manuallyAnimateOnMount && (P = !1), s = !1, P ? e(h) : Promise.resolve();
  }
  function a(c, l, u) {
    var f;
    if (n[c].isActive === l)
      return Promise.resolve();
    (f = t.variantChildren) === null || f === void 0 || f.forEach((d) => {
      var p;
      return (p = d.animationState) === null || p === void 0 ? void 0 : p.setActive(c, l);
    }), n[c].isActive = l;
    const h = o(u, c);
    for (const d in n)
      n[d].protectedKeys = {};
    return h;
  }
  return {
    animateChanges: o,
    setActive: a,
    setAnimateFunction: r,
    getState: () => n
  };
}
function bc(t, e) {
  return typeof e == "string" ? e !== t : Array.isArray(e) ? !fi(e, t) : !1;
}
function rt(t = !1) {
  return {
    isActive: t,
    protectedKeys: {},
    needsAnimating: {},
    prevResolvedValues: {}
  };
}
function Pc() {
  return {
    animate: rt(!0),
    whileInView: rt(),
    whileHover: rt(),
    whileTap: rt(),
    whileDrag: rt(),
    whileFocus: rt(),
    exit: rt()
  };
}
class Tc extends ot {
  /**
   * We dynamically generate the AnimationState manager as it contains a reference
   * to the underlying animation library. We only want to load that if we load this,
   * so people can optionally code split it out using the `m` component.
   */
  constructor(e) {
    super(e), e.animationState || (e.animationState = xc(e));
  }
  updateAnimationControlsSubscription() {
    const { animate: e } = this.node.getProps();
    this.unmount(), ue(e) && (this.unmount = e.subscribe(this.node));
  }
  /**
   * Subscribe any provided AnimationControls to the component's VisualElement
   */
  mount() {
    this.updateAnimationControlsSubscription();
  }
  update() {
    const { animate: e } = this.node.getProps(), { animate: n } = this.node.prevProps || {};
    e !== n && this.updateAnimationControlsSubscription();
  }
  unmount() {
  }
}
let Sc = 0;
class Cc extends ot {
  constructor() {
    super(...arguments), this.id = Sc++;
  }
  update() {
    if (!this.node.presenceContext)
      return;
    const { isPresent: e, onExitComplete: n, custom: s } = this.node.presenceContext, { isPresent: i } = this.node.prevPresenceContext || {};
    if (!this.node.animationState || e === i)
      return;
    const r = this.node.animationState.setActive("exit", !e, { custom: s ?? this.node.getProps().custom });
    n && !e && r.then(() => n(this.id));
  }
  mount() {
    const { register: e } = this.node.presenceContext || {};
    e && (this.unmount = e(this.id));
  }
  unmount() {
  }
}
const Ac = {
  animation: {
    Feature: Tc
  },
  exit: {
    Feature: Cc
  }
}, Kn = (t, e) => Math.abs(t - e);
function Vc(t, e) {
  const n = Kn(t.x, e.x), s = Kn(t.y, e.y);
  return Math.sqrt(n ** 2 + s ** 2);
}
class Oi {
  constructor(e, n, { transformPagePoint: s } = {}) {
    if (this.startEvent = null, this.lastMoveEvent = null, this.lastMoveEventInfo = null, this.handlers = {}, this.updatePoint = () => {
      if (!(this.lastMoveEvent && this.lastMoveEventInfo))
        return;
      const l = we(this.lastMoveEventInfo, this.history), u = this.startEvent !== null, f = Vc(l.offset, { x: 0, y: 0 }) >= 3;
      if (!u && !f)
        return;
      const { point: h } = l, { timestamp: d } = Y;
      this.history.push({ ...h, timestamp: d });
      const { onStart: p, onMove: m } = this.handlers;
      u || (p && p(this.lastMoveEvent, l), this.startEvent = this.lastMoveEvent), m && m(this.lastMoveEvent, l);
    }, this.handlePointerMove = (l, u) => {
      this.lastMoveEvent = l, this.lastMoveEventInfo = Ve(u, this.transformPagePoint), R.update(this.updatePoint, !0);
    }, this.handlePointerUp = (l, u) => {
      if (this.end(), !(this.lastMoveEvent && this.lastMoveEventInfo))
        return;
      const { onEnd: f, onSessionEnd: h } = this.handlers, d = we(l.type === "pointercancel" ? this.lastMoveEventInfo : Ve(u, this.transformPagePoint), this.history);
      this.startEvent && f && f(l, d), h && h(l, d);
    }, !ri(e))
      return;
    this.handlers = n, this.transformPagePoint = s;
    const i = de(e), r = Ve(i, this.transformPagePoint), { point: o } = r, { timestamp: a } = Y;
    this.history = [{ ...o, timestamp: a }];
    const { onSessionStart: c } = n;
    c && c(e, we(r, this.history)), this.removeListeners = nt(Z(window, "pointermove", this.handlePointerMove), Z(window, "pointerup", this.handlePointerUp), Z(window, "pointercancel", this.handlePointerUp));
  }
  updateHandlers(e) {
    this.handlers = e;
  }
  end() {
    this.removeListeners && this.removeListeners(), st.update(this.updatePoint);
  }
}
function Ve(t, e) {
  return e ? { point: e(t.point) } : t;
}
function Zn(t, e) {
  return { x: t.x - e.x, y: t.y - e.y };
}
function we({ point: t }, e) {
  return {
    point: t,
    delta: Zn(t, Ni(e)),
    offset: Zn(t, wc(e)),
    velocity: Mc(e, 0.1)
  };
}
function wc(t) {
  return t[0];
}
function Ni(t) {
  return t[t.length - 1];
}
function Mc(t, e) {
  if (t.length < 2)
    return { x: 0, y: 0 };
  let n = t.length - 1, s = null;
  const i = Ni(t);
  for (; n >= 0 && (s = t[n], !(i.timestamp - s.timestamp > qt(e))); )
    n--;
  if (!s)
    return { x: 0, y: 0 };
  const r = (i.timestamp - s.timestamp) / 1e3;
  if (r === 0)
    return { x: 0, y: 0 };
  const o = {
    x: (i.x - s.x) / r,
    y: (i.y - s.y) / r
  };
  return o.x === 1 / 0 && (o.x = 0), o.y === 1 / 0 && (o.y = 0), o;
}
function B(t) {
  return t.max - t.min;
}
function We(t, e = 0, n = 0.01) {
  return Math.abs(t - e) <= n;
}
function Yn(t, e, n, s = 0.5) {
  t.origin = s, t.originPoint = C(e.min, e.max, t.origin), t.scale = B(n) / B(e), (We(t.scale, 1, 1e-4) || isNaN(t.scale)) && (t.scale = 1), t.translate = C(n.min, n.max, t.origin) - t.originPoint, (We(t.translate) || isNaN(t.translate)) && (t.translate = 0);
}
function Vt(t, e, n, s) {
  Yn(t.x, e.x, n.x, s ? s.originX : void 0), Yn(t.y, e.y, n.y, s ? s.originY : void 0);
}
function qn(t, e, n) {
  t.min = n.min + e.min, t.max = t.min + B(e);
}
function Dc(t, e, n) {
  qn(t.x, e.x, n.x), qn(t.y, e.y, n.y);
}
function Xn(t, e, n) {
  t.min = e.min - n.min, t.max = t.min + B(e);
}
function wt(t, e, n) {
  Xn(t.x, e.x, n.x), Xn(t.y, e.y, n.y);
}
function Rc(t, { min: e, max: n }, s) {
  return e !== void 0 && t < e ? t = s ? C(e, t, s.min) : Math.max(t, e) : n !== void 0 && t > n && (t = s ? C(n, t, s.max) : Math.min(t, n)), t;
}
function Jn(t, e, n) {
  return {
    min: e !== void 0 ? t.min + e : void 0,
    max: n !== void 0 ? t.max + n - (t.max - t.min) : void 0
  };
}
function Ec(t, { top: e, left: n, bottom: s, right: i }) {
  return {
    x: Jn(t.x, n, i),
    y: Jn(t.y, e, s)
  };
}
function Qn(t, e) {
  let n = e.min - t.min, s = e.max - t.max;
  return e.max - e.min < t.max - t.min && ([n, s] = [s, n]), { min: n, max: s };
}
function Lc(t, e) {
  return {
    x: Qn(t.x, e.x),
    y: Qn(t.y, e.y)
  };
}
function kc(t, e) {
  let n = 0.5;
  const s = B(t), i = B(e);
  return i > s ? n = jt(e.min, e.max - s, t.min) : s > i && (n = jt(t.min, t.max - i, e.min)), vt(0, 1, n);
}
function jc(t, e) {
  const n = {};
  return e.min !== void 0 && (n.min = e.min - t.min), e.max !== void 0 && (n.max = e.max - t.min), n;
}
const He = 0.35;
function Bc(t = He) {
  return t === !1 ? t = 0 : t === !0 && (t = He), {
    x: ts(t, "left", "right"),
    y: ts(t, "top", "bottom")
  };
}
function ts(t, e, n) {
  return {
    min: es(t, e),
    max: es(t, n)
  };
}
function es(t, e) {
  return typeof t == "number" ? t : t[e] || 0;
}
const ns = () => ({
  translate: 0,
  scale: 1,
  origin: 0,
  originPoint: 0
}), Mt = () => ({
  x: ns(),
  y: ns()
}), ss = () => ({ min: 0, max: 0 }), A = () => ({
  x: ss(),
  y: ss()
});
function W(t) {
  return [t("x"), t("y")];
}
function Ui({ top: t, left: e, right: n, bottom: s }) {
  return {
    x: { min: e, max: n },
    y: { min: t, max: s }
  };
}
function Fc({ x: t, y: e }) {
  return { top: e.min, right: t.max, bottom: e.max, left: t.min };
}
function Ic(t, e) {
  if (!e)
    return t;
  const n = e({ x: t.left, y: t.top }), s = e({ x: t.right, y: t.bottom });
  return {
    top: n.y,
    left: n.x,
    bottom: s.y,
    right: s.x
  };
}
function Me(t) {
  return t === void 0 || t === 1;
}
function Ge({ scale: t, scaleX: e, scaleY: n }) {
  return !Me(t) || !Me(e) || !Me(n);
}
function at(t) {
  return Ge(t) || _i(t) || t.z || t.rotate || t.rotateX || t.rotateY;
}
function _i(t) {
  return is(t.x) || is(t.y);
}
function is(t) {
  return t && t !== "0%";
}
function re(t, e, n) {
  const s = t - n, i = e * s;
  return n + i;
}
function os(t, e, n, s, i) {
  return i !== void 0 && (t = re(t, i, s)), re(t, n, s) + e;
}
function Ke(t, e = 0, n = 1, s, i) {
  t.min = os(t.min, e, n, s, i), t.max = os(t.max, e, n, s, i);
}
function $i(t, { x: e, y: n }) {
  Ke(t.x, e.translate, e.scale, e.originPoint), Ke(t.y, n.translate, n.scale, n.originPoint);
}
function Oc(t, e, n, s = !1) {
  const i = n.length;
  if (!i)
    return;
  e.x = e.y = 1;
  let r, o;
  for (let a = 0; a < i; a++) {
    r = n[a], o = r.projectionDelta;
    const c = r.instance;
    c && c.style && c.style.display === "contents" || (s && r.options.layoutScroll && r.scroll && r !== r.root && yt(t, {
      x: -r.scroll.offset.x,
      y: -r.scroll.offset.y
    }), o && (e.x *= o.x.scale, e.y *= o.y.scale, $i(t, o)), s && at(r.latestValues) && yt(t, r.latestValues));
  }
  e.x = rs(e.x), e.y = rs(e.y);
}
function rs(t) {
  return Number.isInteger(t) || t > 1.0000000000001 || t < 0.999999999999 ? t : 1;
}
function et(t, e) {
  t.min = t.min + e, t.max = t.max + e;
}
function as(t, e, [n, s, i]) {
  const r = e[i] !== void 0 ? e[i] : 0.5, o = C(t.min, t.max, r);
  Ke(t, e[n], e[s], o, e.scale);
}
const Nc = ["x", "scaleX", "originX"], Uc = ["y", "scaleY", "originY"];
function yt(t, e) {
  as(t.x, e, Nc), as(t.y, e, Uc);
}
function zi(t, e) {
  return Ui(Ic(t.getBoundingClientRect(), e));
}
function _c(t, e, n) {
  const s = zi(t, n), { scroll: i } = e;
  return i && (et(s.x, i.offset.x), et(s.y, i.offset.y)), s;
}
const $c = /* @__PURE__ */ new WeakMap();
class zc {
  constructor(e) {
    this.openGlobalLock = null, this.isDragging = !1, this.currentDirection = null, this.originPoint = { x: 0, y: 0 }, this.constraints = !1, this.hasMutatedConstraints = !1, this.elastic = A(), this.visualElement = e;
  }
  start(e, { snapToCursor: n = !1 } = {}) {
    const { presenceContext: s } = this.visualElement;
    if (s && s.isPresent === !1)
      return;
    const i = (c) => {
      this.stopAnimation(), n && this.snapToCursor(de(c, "page").point);
    }, r = (c, l) => {
      const { drag: u, dragPropagation: f, onDragStart: h } = this.getProps();
      if (u && !f && (this.openGlobalLock && this.openGlobalLock(), this.openGlobalLock = ci(u), !this.openGlobalLock))
        return;
      this.isDragging = !0, this.currentDirection = null, this.resolveConstraints(), this.visualElement.projection && (this.visualElement.projection.isAnimationBlocked = !0, this.visualElement.projection.target = void 0), W((p) => {
        let m = this.getAxisMotionValue(p).get() || 0;
        if (G.test(m)) {
          const { projection: P } = this.visualElement;
          if (P && P.layout) {
            const x = P.layout.layoutBox[p];
            x && (m = B(x) * (parseFloat(m) / 100));
          }
        }
        this.originPoint[p] = m;
      }), h && h(c, l);
      const { animationState: d } = this.visualElement;
      d && d.setActive("whileDrag", !0);
    }, o = (c, l) => {
      const { dragPropagation: u, dragDirectionLock: f, onDirectionLock: h, onDrag: d } = this.getProps();
      if (!u && !this.openGlobalLock)
        return;
      const { offset: p } = l;
      if (f && this.currentDirection === null) {
        this.currentDirection = Wc(p), this.currentDirection !== null && h && h(this.currentDirection);
        return;
      }
      this.updateAxis("x", l.point, p), this.updateAxis("y", l.point, p), this.visualElement.render(), d && d(c, l);
    }, a = (c, l) => this.stop(c, l);
    this.panSession = new Oi(e, {
      onSessionStart: i,
      onStart: r,
      onMove: o,
      onSessionEnd: a
    }, { transformPagePoint: this.visualElement.getTransformPagePoint() });
  }
  stop(e, n) {
    const s = this.isDragging;
    if (this.cancel(), !s)
      return;
    const { velocity: i } = n;
    this.startAnimation(i);
    const { onDragEnd: r } = this.getProps();
    r && r(e, n);
  }
  cancel() {
    this.isDragging = !1;
    const { projection: e, animationState: n } = this.visualElement;
    e && (e.isAnimationBlocked = !1), this.panSession && this.panSession.end(), this.panSession = void 0;
    const { dragPropagation: s } = this.getProps();
    !s && this.openGlobalLock && (this.openGlobalLock(), this.openGlobalLock = null), n && n.setActive("whileDrag", !1);
  }
  updateAxis(e, n, s) {
    const { drag: i } = this.getProps();
    if (!s || !Kt(e, i, this.currentDirection))
      return;
    const r = this.getAxisMotionValue(e);
    let o = this.originPoint[e] + s[e];
    this.constraints && this.constraints[e] && (o = Rc(o, this.constraints[e], this.elastic[e])), r.set(o);
  }
  resolveConstraints() {
    const { dragConstraints: e, dragElastic: n } = this.getProps(), { layout: s } = this.visualElement.projection || {}, i = this.constraints;
    e && mt(e) ? this.constraints || (this.constraints = this.resolveRefConstraints()) : e && s ? this.constraints = Ec(s.layoutBox, e) : this.constraints = !1, this.elastic = Bc(n), i !== this.constraints && s && this.constraints && !this.hasMutatedConstraints && W((r) => {
      this.getAxisMotionValue(r) && (this.constraints[r] = jc(s.layoutBox[r], this.constraints[r]));
    });
  }
  resolveRefConstraints() {
    const { dragConstraints: e, onMeasureDragConstraints: n } = this.getProps();
    if (!e || !mt(e))
      return !1;
    const s = e.current;
    $(s !== null, "If `dragConstraints` is set as a React ref, that ref must be passed to another component's `ref` prop.");
    const { projection: i } = this.visualElement;
    if (!i || !i.layout)
      return !1;
    const r = _c(s, i.root, this.visualElement.getTransformPagePoint());
    let o = Lc(i.layout.layoutBox, r);
    if (n) {
      const a = n(Fc(o));
      this.hasMutatedConstraints = !!a, a && (o = Ui(a));
    }
    return o;
  }
  startAnimation(e) {
    const { drag: n, dragMomentum: s, dragElastic: i, dragTransition: r, dragSnapToOrigin: o, onDragTransitionEnd: a } = this.getProps(), c = this.constraints || {}, l = W((u) => {
      if (!Kt(u, n, this.currentDirection))
        return;
      let f = c && c[u] || {};
      o && (f = { min: 0, max: 0 });
      const h = i ? 200 : 1e6, d = i ? 40 : 1e7, p = {
        type: "inertia",
        velocity: s ? e[u] : 0,
        bounceStiffness: h,
        bounceDamping: d,
        timeConstant: 750,
        restDelta: 1,
        restSpeed: 10,
        ...r,
        ...f
      };
      return this.startAxisValueAnimation(u, p);
    });
    return Promise.all(l).then(a);
  }
  startAxisValueAnimation(e, n) {
    const s = this.getAxisMotionValue(e);
    return s.start(An(e, s, 0, n));
  }
  stopAnimation() {
    W((e) => this.getAxisMotionValue(e).stop());
  }
  /**
   * Drag works differently depending on which props are provided.
   *
   * - If _dragX and _dragY are provided, we output the gesture delta directly to those motion values.
   * - Otherwise, we apply the delta to the x/y motion values.
   */
  getAxisMotionValue(e) {
    const n = "_drag" + e.toUpperCase(), s = this.visualElement.getProps(), i = s[n];
    return i || this.visualElement.getValue(e, (s.initial ? s.initial[e] : void 0) || 0);
  }
  snapToCursor(e) {
    W((n) => {
      const { drag: s } = this.getProps();
      if (!Kt(n, s, this.currentDirection))
        return;
      const { projection: i } = this.visualElement, r = this.getAxisMotionValue(n);
      if (i && i.layout) {
        const { min: o, max: a } = i.layout.layoutBox[n];
        r.set(e[n] - C(o, a, 0.5));
      }
    });
  }
  /**
   * When the viewport resizes we want to check if the measured constraints
   * have changed and, if so, reposition the element within those new constraints
   * relative to where it was before the resize.
   */
  scalePositionWithinConstraints() {
    if (!this.visualElement.current)
      return;
    const { drag: e, dragConstraints: n } = this.getProps(), { projection: s } = this.visualElement;
    if (!mt(n) || !s || !this.constraints)
      return;
    this.stopAnimation();
    const i = { x: 0, y: 0 };
    W((o) => {
      const a = this.getAxisMotionValue(o);
      if (a) {
        const c = a.get();
        i[o] = kc({ min: c, max: c }, this.constraints[o]);
      }
    });
    const { transformTemplate: r } = this.visualElement.getProps();
    this.visualElement.current.style.transform = r ? r({}, "") : "none", s.root && s.root.updateScroll(), s.updateLayout(), this.resolveConstraints(), W((o) => {
      if (!Kt(o, e, null))
        return;
      const a = this.getAxisMotionValue(o), { min: c, max: l } = this.constraints[o];
      a.set(C(c, l, i[o]));
    });
  }
  addListeners() {
    if (!this.visualElement.current)
      return;
    $c.set(this.visualElement, this);
    const e = this.visualElement.current, n = Z(e, "pointerdown", (c) => {
      const { drag: l, dragListener: u = !0 } = this.getProps();
      l && u && this.start(c);
    }), s = () => {
      const { dragConstraints: c } = this.getProps();
      mt(c) && (this.constraints = this.resolveRefConstraints());
    }, { projection: i } = this.visualElement, r = i.addEventListener("measure", s);
    i && !i.layout && (i.root && i.root.updateScroll(), i.updateLayout()), s();
    const o = K(window, "resize", () => this.scalePositionWithinConstraints()), a = i.addEventListener("didUpdate", ({ delta: c, hasLayoutChanged: l }) => {
      this.isDragging && l && (W((u) => {
        const f = this.getAxisMotionValue(u);
        f && (this.originPoint[u] += c[u].translate, f.set(f.get() + c[u].translate));
      }), this.visualElement.render());
    });
    return () => {
      o(), n(), r(), a && a();
    };
  }
  getProps() {
    const e = this.visualElement.getProps(), { drag: n = !1, dragDirectionLock: s = !1, dragPropagation: i = !1, dragConstraints: r = !1, dragElastic: o = He, dragMomentum: a = !0 } = e;
    return {
      ...e,
      drag: n,
      dragDirectionLock: s,
      dragPropagation: i,
      dragConstraints: r,
      dragElastic: o,
      dragMomentum: a
    };
  }
}
function Kt(t, e, n) {
  return (e === !0 || e === t) && (n === null || n === t);
}
function Wc(t, e = 10) {
  let n = null;
  return Math.abs(t.y) > e ? n = "y" : Math.abs(t.x) > e && (n = "x"), n;
}
class Hc extends ot {
  constructor(e) {
    super(e), this.removeGroupControls = N, this.removeListeners = N, this.controls = new zc(e);
  }
  mount() {
    const { dragControls: e } = this.node.getProps();
    e && (this.removeGroupControls = e.subscribe(this.controls)), this.removeListeners = this.controls.addListeners() || N;
  }
  unmount() {
    this.removeGroupControls(), this.removeListeners();
  }
}
class Gc extends ot {
  constructor() {
    super(...arguments), this.removePointerDownListener = N;
  }
  onPointerDown(e) {
    this.session = new Oi(e, this.createPanHandlers(), { transformPagePoint: this.node.getTransformPagePoint() });
  }
  createPanHandlers() {
    const { onPanSessionStart: e, onPanStart: n, onPan: s, onPanEnd: i } = this.node.getProps();
    return {
      onSessionStart: e,
      onStart: n,
      onMove: s,
      onEnd: (r, o) => {
        delete this.session, i && i(r, o);
      }
    };
  }
  mount() {
    this.removePointerDownListener = Z(this.node.current, "pointerdown", (e) => this.onPointerDown(e));
  }
  update() {
    this.session && this.session.updateHandlers(this.createPanHandlers());
  }
  unmount() {
    this.removePointerDownListener(), this.session && this.session.end();
  }
}
function Kc() {
  const t = w(Ot);
  if (t === null)
    return [!0, null];
  const { isPresent: e, onExitComplete: n, register: s } = t, i = Xe();
  return U(() => s(i), []), !e && n ? [!1, () => n && n(i)] : [!0];
}
function Zc() {
  return Yc(w(Ot));
}
function Yc(t) {
  return t === null ? !0 : t.isPresent;
}
function cs(t, e) {
  return e.max === e.min ? 0 : t / (e.max - e.min) * 100;
}
const Pt = {
  correct: (t, e) => {
    if (!e.target)
      return t;
    if (typeof t == "string")
      if (b.test(t))
        t = parseFloat(t);
      else
        return t;
    const n = cs(t, e.target.x), s = cs(t, e.target.y);
    return `${n}% ${s}%`;
  }
}, Wi = /var\((--[a-zA-Z0-9-_]+),? ?([a-zA-Z0-9 ()%#.,-]+)?\)/;
function qc(t) {
  const e = Wi.exec(t);
  if (!e)
    return [,];
  const [, n, s] = e;
  return [n, s];
}
const Xc = 4;
function Ze(t, e, n = 1) {
  $(n <= Xc, `Max CSS variable fallback depth detected in property "${t}". This may indicate a circular fallback dependency.`);
  const [s, i] = qc(t);
  if (!s)
    return;
  const r = window.getComputedStyle(e).getPropertyValue(s);
  return r ? r.trim() : ke(i) ? Ze(i, e, n + 1) : i;
}
function Jc(t, { ...e }, n) {
  const s = t.current;
  if (!(s instanceof Element))
    return { target: e, transitionEnd: n };
  n && (n = { ...n }), t.values.forEach((i) => {
    const r = i.get();
    if (!ke(r))
      return;
    const o = Ze(r, s);
    o && i.set(o);
  });
  for (const i in e) {
    const r = e[i];
    if (!ke(r))
      continue;
    const o = Ze(r, s);
    o && (e[i] = o, n && n[i] === void 0 && (n[i] = r));
  }
  return { target: e, transitionEnd: n };
}
const ls = "_$css", Qc = {
  correct: (t, { treeScale: e, projectionDelta: n }) => {
    const s = t, i = t.includes("var("), r = [];
    i && (t = t.replace(Wi, (d) => (r.push(d), ls)));
    const o = it.parse(t);
    if (o.length > 5)
      return s;
    const a = it.createTransformer(t), c = typeof o[0] != "number" ? 1 : 0, l = n.x.scale * e.x, u = n.y.scale * e.y;
    o[0 + c] /= l, o[1 + c] /= u;
    const f = C(l, u, 0.5);
    typeof o[2 + c] == "number" && (o[2 + c] /= f), typeof o[3 + c] == "number" && (o[3 + c] /= f);
    let h = a(o);
    if (i) {
      let d = 0;
      h = h.replace(ls, () => {
        const p = r[d];
        return d++, p;
      });
    }
    return h;
  }
};
class tl extends Bs.Component {
  /**
   * This only mounts projection nodes for components that
   * need measuring, we might want to do it for all components
   * in order to incorporate transforms
   */
  componentDidMount() {
    const { visualElement: e, layoutGroup: n, switchLayoutGroup: s, layoutId: i } = this.props, { projection: r } = e;
    tr(el), r && (n.group && n.group.add(r), s && s.register && i && s.register(r), r.root.didUpdate(), r.addEventListener("animationComplete", () => {
      this.safeToRemove();
    }), r.setOptions({
      ...r.options,
      onExitComplete: () => this.safeToRemove()
    })), St.hasEverUpdated = !0;
  }
  getSnapshotBeforeUpdate(e) {
    const { layoutDependency: n, visualElement: s, drag: i, isPresent: r } = this.props, o = s.projection;
    return o && (o.isPresent = r, i || e.layoutDependency !== n || n === void 0 ? o.willUpdate() : this.safeToRemove(), e.isPresent !== r && (r ? o.promote() : o.relegate() || R.postRender(() => {
      const a = o.getStack();
      (!a || !a.members.length) && this.safeToRemove();
    }))), null;
  }
  componentDidUpdate() {
    const { projection: e } = this.props.visualElement;
    e && (e.root.didUpdate(), !e.currentAnimation && e.isLead() && this.safeToRemove());
  }
  componentWillUnmount() {
    const { visualElement: e, layoutGroup: n, switchLayoutGroup: s } = this.props, { projection: i } = e;
    i && (i.scheduleCheckAfterUnmount(), n && n.group && n.group.remove(i), s && s.deregister && s.deregister(i));
  }
  safeToRemove() {
    const { safeToRemove: e } = this.props;
    e && e();
  }
  render() {
    return null;
  }
}
function Hi(t) {
  const [e, n] = Kc(), s = w(nn);
  return Bs.createElement(tl, { ...t, layoutGroup: s, switchLayoutGroup: w(Ks), isPresent: e, safeToRemove: n });
}
const el = {
  borderRadius: {
    ...Pt,
    applyTo: [
      "borderTopLeftRadius",
      "borderTopRightRadius",
      "borderBottomLeftRadius",
      "borderBottomRightRadius"
    ]
  },
  borderTopLeftRadius: Pt,
  borderTopRightRadius: Pt,
  borderBottomLeftRadius: Pt,
  borderBottomRightRadius: Pt,
  boxShadow: Qc
};
function nl(t, e, n = {}) {
  const s = j(t) ? t : xt(t);
  return s.start(An("", s, e, n)), {
    stop: () => s.stop(),
    isAnimating: () => s.isAnimating()
  };
}
const Gi = ["TopLeft", "TopRight", "BottomLeft", "BottomRight"], sl = Gi.length, us = (t) => typeof t == "string" ? parseFloat(t) : t, fs = (t) => typeof t == "number" || b.test(t);
function il(t, e, n, s, i, r) {
  i ? (t.opacity = C(
    0,
    // TODO Reinstate this if only child
    n.opacity !== void 0 ? n.opacity : 1,
    ol(s)
  ), t.opacityExit = C(e.opacity !== void 0 ? e.opacity : 1, 0, rl(s))) : r && (t.opacity = C(e.opacity !== void 0 ? e.opacity : 1, n.opacity !== void 0 ? n.opacity : 1, s));
  for (let o = 0; o < sl; o++) {
    const a = `border${Gi[o]}Radius`;
    let c = hs(e, a), l = hs(n, a);
    if (c === void 0 && l === void 0)
      continue;
    c || (c = 0), l || (l = 0), c === 0 || l === 0 || fs(c) === fs(l) ? (t[a] = Math.max(C(us(c), us(l), s), 0), (G.test(l) || G.test(c)) && (t[a] += "%")) : t[a] = l;
  }
  (e.rotate || n.rotate) && (t.rotate = C(e.rotate || 0, n.rotate || 0, s));
}
function hs(t, e) {
  return t[e] !== void 0 ? t[e] : t.borderRadius;
}
const ol = Ki(0, 0.5, Sn), rl = Ki(0.5, 0.95, N);
function Ki(t, e, n) {
  return (s) => s < t ? 0 : s > e ? 1 : n(jt(t, e, s));
}
function ds(t, e) {
  t.min = e.min, t.max = e.max;
}
function I(t, e) {
  ds(t.x, e.x), ds(t.y, e.y);
}
function ps(t, e, n, s, i) {
  return t -= e, t = re(t, 1 / n, s), i !== void 0 && (t = re(t, 1 / i, s)), t;
}
function al(t, e = 0, n = 1, s = 0.5, i, r = t, o = t) {
  if (G.test(e) && (e = parseFloat(e), e = C(o.min, o.max, e / 100) - o.min), typeof e != "number")
    return;
  let a = C(r.min, r.max, s);
  t === r && (a -= e), t.min = ps(t.min, e, n, a, i), t.max = ps(t.max, e, n, a, i);
}
function ms(t, e, [n, s, i], r, o) {
  al(t, e[n], e[s], e[i], e.scale, r, o);
}
const cl = ["x", "scaleX", "originX"], ll = ["y", "scaleY", "originY"];
function gs(t, e, n, s) {
  ms(t.x, e, cl, n ? n.x : void 0, s ? s.x : void 0), ms(t.y, e, ll, n ? n.y : void 0, s ? s.y : void 0);
}
function ys(t) {
  return t.translate === 0 && t.scale === 1;
}
function Zi(t) {
  return ys(t.x) && ys(t.y);
}
function Ye(t, e) {
  return t.x.min === e.x.min && t.x.max === e.x.max && t.y.min === e.y.min && t.y.max === e.y.max;
}
function vs(t) {
  return B(t.x) / B(t.y);
}
class ul {
  constructor() {
    this.members = [];
  }
  add(e) {
    hn(this.members, e), e.scheduleRender();
  }
  remove(e) {
    if (dn(this.members, e), e === this.prevLead && (this.prevLead = void 0), e === this.lead) {
      const n = this.members[this.members.length - 1];
      n && this.promote(n);
    }
  }
  relegate(e) {
    const n = this.members.findIndex((i) => e === i);
    if (n === 0)
      return !1;
    let s;
    for (let i = n; i >= 0; i--) {
      const r = this.members[i];
      if (r.isPresent !== !1) {
        s = r;
        break;
      }
    }
    return s ? (this.promote(s), !0) : !1;
  }
  promote(e, n) {
    const s = this.lead;
    if (e !== s && (this.prevLead = s, this.lead = e, e.show(), s)) {
      s.instance && s.scheduleRender(), e.scheduleRender(), e.resumeFrom = s, n && (e.resumeFrom.preserveOpacity = !0), s.snapshot && (e.snapshot = s.snapshot, e.snapshot.latestValues = s.animationValues || s.latestValues), e.root && e.root.isUpdating && (e.isLayoutDirty = !0);
      const { crossfade: i } = e.options;
      i === !1 && s.hide();
    }
  }
  exitAnimationComplete() {
    this.members.forEach((e) => {
      const { options: n, resumingFrom: s } = e;
      n.onExitComplete && n.onExitComplete(), s && s.options.onExitComplete && s.options.onExitComplete();
    });
  }
  scheduleRender() {
    this.members.forEach((e) => {
      e.instance && e.scheduleRender(!1);
    });
  }
  /**
   * Clear any leads that have been removed this render to prevent them from being
   * used in future animations and to prevent memory leaks
   */
  removeLeadSnapshot() {
    this.lead && this.lead.snapshot && (this.lead.snapshot = void 0);
  }
}
function xs(t, e, n) {
  let s = "";
  const i = t.x.translate / e.x, r = t.y.translate / e.y;
  if ((i || r) && (s = `translate3d(${i}px, ${r}px, 0) `), (e.x !== 1 || e.y !== 1) && (s += `scale(${1 / e.x}, ${1 / e.y}) `), n) {
    const { rotate: c, rotateX: l, rotateY: u } = n;
    c && (s += `rotate(${c}deg) `), l && (s += `rotateX(${l}deg) `), u && (s += `rotateY(${u}deg) `);
  }
  const o = t.x.scale * e.x, a = t.y.scale * e.y;
  return (o !== 1 || a !== 1) && (s += `scale(${o}, ${a})`), s || "none";
}
const fl = (t, e) => t.depth - e.depth;
class hl {
  constructor() {
    this.children = [], this.isDirty = !1;
  }
  add(e) {
    hn(this.children, e), this.isDirty = !0;
  }
  remove(e) {
    dn(this.children, e), this.isDirty = !0;
  }
  forEach(e) {
    this.isDirty && this.children.sort(fl), this.isDirty = !1, this.children.forEach(e);
  }
}
function dl(t) {
  window.MotionDebug && window.MotionDebug.record(t);
}
const bs = ["", "X", "Y", "Z"], Ps = 1e3;
let pl = 0;
const ct = {
  type: "projectionFrame",
  totalNodes: 0,
  resolvedTargetDeltas: 0,
  recalculatedProjection: 0
};
function Yi({ attachResizeListener: t, defaultParent: e, measureScroll: n, checkIsScrollRoot: s, resetTransform: i }) {
  return class {
    constructor(o, a = {}, c = e == null ? void 0 : e()) {
      this.id = pl++, this.animationId = 0, this.children = /* @__PURE__ */ new Set(), this.options = {}, this.isTreeAnimating = !1, this.isAnimationBlocked = !1, this.isLayoutDirty = !1, this.isProjectionDirty = !1, this.isSharedProjectionDirty = !1, this.isTransformDirty = !1, this.updateManuallyBlocked = !1, this.updateBlockedByResize = !1, this.isUpdating = !1, this.isSVG = !1, this.needsReset = !1, this.shouldResetTransform = !1, this.treeScale = { x: 1, y: 1 }, this.eventHandlers = /* @__PURE__ */ new Map(), this.potentialNodes = /* @__PURE__ */ new Map(), this.checkUpdateFailed = () => {
        this.isUpdating && (this.isUpdating = !1, this.clearAllSnapshots());
      }, this.updateProjection = () => {
        ct.totalNodes = ct.resolvedTargetDeltas = ct.recalculatedProjection = 0, this.nodes.forEach(yl), this.nodes.forEach(Pl), this.nodes.forEach(Tl), this.nodes.forEach(vl), dl(ct);
      }, this.hasProjected = !1, this.isVisible = !0, this.animationProgress = 0, this.sharedNodes = /* @__PURE__ */ new Map(), this.elementId = o, this.latestValues = a, this.root = c ? c.root || c : this, this.path = c ? [...c.path, c] : [], this.parent = c, this.depth = c ? c.depth + 1 : 0, o && this.root.registerPotentialNode(o, this);
      for (let l = 0; l < this.path.length; l++)
        this.path[l].shouldResetTransform = !0;
      this.root === this && (this.nodes = new hl());
    }
    addEventListener(o, a) {
      return this.eventHandlers.has(o) || this.eventHandlers.set(o, new pn()), this.eventHandlers.get(o).add(a);
    }
    notifyListeners(o, ...a) {
      const c = this.eventHandlers.get(o);
      c && c.notify(...a);
    }
    hasListeners(o) {
      return this.eventHandlers.has(o);
    }
    registerPotentialNode(o, a) {
      this.potentialNodes.set(o, a);
    }
    /**
     * Lifecycles
     */
    mount(o, a = !1) {
      if (this.instance)
        return;
      this.isSVG = o instanceof SVGElement && o.tagName !== "svg", this.instance = o;
      const { layoutId: c, layout: l, visualElement: u } = this.options;
      if (u && !u.current && u.mount(o), this.root.nodes.add(this), this.parent && this.parent.children.add(this), this.elementId && this.root.potentialNodes.delete(this.elementId), a && (l || c) && (this.isLayoutDirty = !0), t) {
        let f;
        const h = () => this.root.updateBlockedByResize = !1;
        t(o, () => {
          this.root.updateBlockedByResize = !0, f && f(), f = Bi(h, 250), St.hasAnimatedSinceResize && (St.hasAnimatedSinceResize = !1, this.nodes.forEach(Ss));
        });
      }
      c && this.root.registerSharedNode(c, this), this.options.animate !== !1 && u && (c || l) && this.addEventListener("didUpdate", ({ delta: f, hasLayoutChanged: h, hasRelativeTargetChanged: d, layout: p }) => {
        if (this.isTreeAnimationBlocked()) {
          this.target = void 0, this.relativeTarget = void 0;
          return;
        }
        const m = this.options.transition || u.getDefaultTransition() || wl, { onLayoutAnimationStart: P, onLayoutAnimationComplete: x } = u.getProps(), v = !this.targetLayout || !Ye(this.targetLayout, p) || d, g = !h && d;
        if (this.options.layoutRoot || this.resumeFrom && this.resumeFrom.instance || g || h && (v || !this.currentAnimation)) {
          this.resumeFrom && (this.resumingFrom = this.resumeFrom, this.resumingFrom.resumingFrom = void 0), this.setAnimationOrigin(f, g);
          const y = {
            ...Fi(m, "layout"),
            onPlay: P,
            onComplete: x
          };
          (u.shouldReduceMotion || this.options.layoutRoot) && (y.delay = 0, y.type = !1), this.startAnimation(y);
        } else
          !h && this.animationProgress === 0 && Ss(this), this.isLead() && this.options.onExitComplete && this.options.onExitComplete();
        this.targetLayout = p;
      });
    }
    unmount() {
      this.options.layoutId && this.willUpdate(), this.root.nodes.remove(this);
      const o = this.getStack();
      o && o.remove(this), this.parent && this.parent.children.delete(this), this.instance = void 0, st.preRender(this.updateProjection);
    }
    // only on the root
    blockUpdate() {
      this.updateManuallyBlocked = !0;
    }
    unblockUpdate() {
      this.updateManuallyBlocked = !1;
    }
    isUpdateBlocked() {
      return this.updateManuallyBlocked || this.updateBlockedByResize;
    }
    isTreeAnimationBlocked() {
      return this.isAnimationBlocked || this.parent && this.parent.isTreeAnimationBlocked() || !1;
    }
    // Note: currently only running on root node
    startUpdate() {
      this.isUpdateBlocked() || (this.isUpdating = !0, this.nodes && this.nodes.forEach(Sl), this.animationId++);
    }
    getTransformTemplate() {
      const { visualElement: o } = this.options;
      return o && o.getProps().transformTemplate;
    }
    willUpdate(o = !0) {
      if (this.root.isUpdateBlocked()) {
        this.options.onExitComplete && this.options.onExitComplete();
        return;
      }
      if (!this.root.isUpdating && this.root.startUpdate(), this.isLayoutDirty)
        return;
      this.isLayoutDirty = !0;
      for (let u = 0; u < this.path.length; u++) {
        const f = this.path[u];
        f.shouldResetTransform = !0, f.updateScroll("snapshot"), f.options.layoutRoot && f.willUpdate(!1);
      }
      const { layoutId: a, layout: c } = this.options;
      if (a === void 0 && !c)
        return;
      const l = this.getTransformTemplate();
      this.prevTransformTemplateValue = l ? l(this.latestValues, "") : void 0, this.updateSnapshot(), o && this.notifyListeners("willUpdate");
    }
    // Note: Currently only running on root node
    didUpdate() {
      if (this.isUpdateBlocked()) {
        this.unblockUpdate(), this.clearAllSnapshots(), this.nodes.forEach(Ts);
        return;
      }
      this.isUpdating && (this.isUpdating = !1, this.potentialNodes.size && (this.potentialNodes.forEach(Ml), this.potentialNodes.clear()), this.nodes.forEach(bl), this.nodes.forEach(ml), this.nodes.forEach(gl), this.clearAllSnapshots(), be.update(), be.preRender(), be.render());
    }
    clearAllSnapshots() {
      this.nodes.forEach(xl), this.sharedNodes.forEach(Cl);
    }
    scheduleUpdateProjection() {
      R.preRender(this.updateProjection, !1, !0);
    }
    scheduleCheckAfterUnmount() {
      R.postRender(() => {
        this.isLayoutDirty ? this.root.didUpdate() : this.root.checkUpdateFailed();
      });
    }
    /**
     * Update measurements
     */
    updateSnapshot() {
      this.snapshot || !this.instance || (this.snapshot = this.measure());
    }
    updateLayout() {
      if (!this.instance || (this.updateScroll(), !(this.options.alwaysMeasureLayout && this.isLead()) && !this.isLayoutDirty))
        return;
      if (this.resumeFrom && !this.resumeFrom.instance)
        for (let c = 0; c < this.path.length; c++)
          this.path[c].updateScroll();
      const o = this.layout;
      this.layout = this.measure(!1), this.layoutCorrected = A(), this.isLayoutDirty = !1, this.projectionDelta = void 0, this.notifyListeners("measure", this.layout.layoutBox);
      const { visualElement: a } = this.options;
      a && a.notify("LayoutMeasure", this.layout.layoutBox, o ? o.layoutBox : void 0);
    }
    updateScroll(o = "measure") {
      let a = Boolean(this.options.layoutScroll && this.instance);
      this.scroll && this.scroll.animationId === this.root.animationId && this.scroll.phase === o && (a = !1), a && (this.scroll = {
        animationId: this.root.animationId,
        phase: o,
        isRoot: s(this.instance),
        offset: n(this.instance)
      });
    }
    resetTransform() {
      if (!i)
        return;
      const o = this.isLayoutDirty || this.shouldResetTransform, a = this.projectionDelta && !Zi(this.projectionDelta), c = this.getTransformTemplate(), l = c ? c(this.latestValues, "") : void 0, u = l !== this.prevTransformTemplateValue;
      o && (a || at(this.latestValues) || u) && (i(this.instance, l), this.shouldResetTransform = !1, this.scheduleRender());
    }
    measure(o = !0) {
      const a = this.measurePageBox();
      let c = this.removeElementScroll(a);
      return o && (c = this.removeTransform(c)), Dl(c), {
        animationId: this.root.animationId,
        measuredBox: a,
        layoutBox: c,
        latestValues: {},
        source: this.id
      };
    }
    measurePageBox() {
      const { visualElement: o } = this.options;
      if (!o)
        return A();
      const a = o.measureViewportBox(), { scroll: c } = this.root;
      return c && (et(a.x, c.offset.x), et(a.y, c.offset.y)), a;
    }
    removeElementScroll(o) {
      const a = A();
      I(a, o);
      for (let c = 0; c < this.path.length; c++) {
        const l = this.path[c], { scroll: u, options: f } = l;
        if (l !== this.root && u && f.layoutScroll) {
          if (u.isRoot) {
            I(a, o);
            const { scroll: h } = this.root;
            h && (et(a.x, -h.offset.x), et(a.y, -h.offset.y));
          }
          et(a.x, u.offset.x), et(a.y, u.offset.y);
        }
      }
      return a;
    }
    applyTransform(o, a = !1) {
      const c = A();
      I(c, o);
      for (let l = 0; l < this.path.length; l++) {
        const u = this.path[l];
        !a && u.options.layoutScroll && u.scroll && u !== u.root && yt(c, {
          x: -u.scroll.offset.x,
          y: -u.scroll.offset.y
        }), at(u.latestValues) && yt(c, u.latestValues);
      }
      return at(this.latestValues) && yt(c, this.latestValues), c;
    }
    removeTransform(o) {
      const a = A();
      I(a, o);
      for (let c = 0; c < this.path.length; c++) {
        const l = this.path[c];
        if (!l.instance || !at(l.latestValues))
          continue;
        Ge(l.latestValues) && l.updateSnapshot();
        const u = A(), f = l.measurePageBox();
        I(u, f), gs(a, l.latestValues, l.snapshot ? l.snapshot.layoutBox : void 0, u);
      }
      return at(this.latestValues) && gs(a, this.latestValues), a;
    }
    setTargetDelta(o) {
      this.targetDelta = o, this.root.scheduleUpdateProjection(), this.isProjectionDirty = !0;
    }
    setOptions(o) {
      this.options = {
        ...this.options,
        ...o,
        crossfade: o.crossfade !== void 0 ? o.crossfade : !0
      };
    }
    clearMeasurements() {
      this.scroll = void 0, this.layout = void 0, this.snapshot = void 0, this.prevTransformTemplateValue = void 0, this.targetDelta = void 0, this.target = void 0, this.isLayoutDirty = !1;
    }
    /**
     * Frame calculations
     */
    resolveTargetDelta() {
      var o;
      const a = this.getLead();
      if (this.isProjectionDirty || (this.isProjectionDirty = a.isProjectionDirty), this.isTransformDirty || (this.isTransformDirty = a.isTransformDirty), this.isSharedProjectionDirty || (this.isSharedProjectionDirty = a.isSharedProjectionDirty), !((Boolean(this.resumingFrom) || this !== a) && this.isSharedProjectionDirty || this.isProjectionDirty || !((o = this.parent) === null || o === void 0) && o.isProjectionDirty || this.attemptToResolveRelativeTarget))
        return;
      const { layout: u, layoutId: f } = this.options;
      if (!(!this.layout || !(u || f))) {
        if (!this.targetDelta && !this.relativeTarget) {
          const h = this.getClosestProjectingParent();
          h && h.layout ? (this.relativeParent = h, this.relativeTarget = A(), this.relativeTargetOrigin = A(), wt(this.relativeTargetOrigin, this.layout.layoutBox, h.layout.layoutBox), I(this.relativeTarget, this.relativeTargetOrigin)) : this.relativeParent = this.relativeTarget = void 0;
        }
        if (!(!this.relativeTarget && !this.targetDelta)) {
          if (this.target || (this.target = A(), this.targetWithTransforms = A()), this.relativeTarget && this.relativeTargetOrigin && this.relativeParent && this.relativeParent.target ? Dc(this.target, this.relativeTarget, this.relativeParent.target) : this.targetDelta ? (Boolean(this.resumingFrom) ? this.target = this.applyTransform(this.layout.layoutBox) : I(this.target, this.layout.layoutBox), $i(this.target, this.targetDelta)) : I(this.target, this.layout.layoutBox), this.attemptToResolveRelativeTarget) {
            this.attemptToResolveRelativeTarget = !1;
            const h = this.getClosestProjectingParent();
            h && Boolean(h.resumingFrom) === Boolean(this.resumingFrom) && !h.options.layoutScroll && h.target ? (this.relativeParent = h, this.relativeTarget = A(), this.relativeTargetOrigin = A(), wt(this.relativeTargetOrigin, this.target, h.target), I(this.relativeTarget, this.relativeTargetOrigin)) : this.relativeParent = this.relativeTarget = void 0;
          }
          ct.resolvedTargetDeltas++;
        }
      }
    }
    getClosestProjectingParent() {
      if (!(!this.parent || Ge(this.parent.latestValues) || _i(this.parent.latestValues)))
        return this.parent.isProjecting() ? this.parent : this.parent.getClosestProjectingParent();
    }
    isProjecting() {
      return Boolean((this.relativeTarget || this.targetDelta || this.options.layoutRoot) && this.layout);
    }
    calcProjection() {
      var o;
      const a = this.getLead(), c = Boolean(this.resumingFrom) || this !== a;
      let l = !0;
      if ((this.isProjectionDirty || !((o = this.parent) === null || o === void 0) && o.isProjectionDirty) && (l = !1), c && (this.isSharedProjectionDirty || this.isTransformDirty) && (l = !1), l)
        return;
      const { layout: u, layoutId: f } = this.options;
      if (this.isTreeAnimating = Boolean(this.parent && this.parent.isTreeAnimating || this.currentAnimation || this.pendingAnimation), this.isTreeAnimating || (this.targetDelta = this.relativeTarget = void 0), !this.layout || !(u || f))
        return;
      I(this.layoutCorrected, this.layout.layoutBox), Oc(this.layoutCorrected, this.treeScale, this.path, c);
      const { target: h } = a;
      if (!h)
        return;
      this.projectionDelta || (this.projectionDelta = Mt(), this.projectionDeltaWithTransform = Mt());
      const d = this.treeScale.x, p = this.treeScale.y, m = this.projectionTransform;
      Vt(this.projectionDelta, this.layoutCorrected, h, this.latestValues), this.projectionTransform = xs(this.projectionDelta, this.treeScale), (this.projectionTransform !== m || this.treeScale.x !== d || this.treeScale.y !== p) && (this.hasProjected = !0, this.scheduleRender(), this.notifyListeners("projectionUpdate", h)), ct.recalculatedProjection++;
    }
    hide() {
      this.isVisible = !1;
    }
    show() {
      this.isVisible = !0;
    }
    scheduleRender(o = !0) {
      if (this.options.scheduleRender && this.options.scheduleRender(), o) {
        const a = this.getStack();
        a && a.scheduleRender();
      }
      this.resumingFrom && !this.resumingFrom.instance && (this.resumingFrom = void 0);
    }
    setAnimationOrigin(o, a = !1) {
      const c = this.snapshot, l = c ? c.latestValues : {}, u = { ...this.latestValues }, f = Mt();
      (!this.relativeParent || !this.relativeParent.options.layoutRoot) && (this.relativeTarget = this.relativeTargetOrigin = void 0), this.attemptToResolveRelativeTarget = !a;
      const h = A(), d = c ? c.source : void 0, p = this.layout ? this.layout.source : void 0, m = d !== p, P = this.getStack(), x = !P || P.members.length <= 1, v = Boolean(m && !x && this.options.crossfade === !0 && !this.path.some(Vl));
      this.animationProgress = 0;
      let g;
      this.mixTargetDelta = (y) => {
        const S = y / 1e3;
        Cs(f.x, o.x, S), Cs(f.y, o.y, S), this.setTargetDelta(f), this.relativeTarget && this.relativeTargetOrigin && this.layout && this.relativeParent && this.relativeParent.layout && (wt(h, this.layout.layoutBox, this.relativeParent.layout.layoutBox), Al(this.relativeTarget, this.relativeTargetOrigin, h, S), g && Ye(this.relativeTarget, g) && (this.isProjectionDirty = !1), g || (g = A()), I(g, this.relativeTarget)), m && (this.animationValues = u, il(u, l, this.latestValues, S, v, x)), this.root.scheduleUpdateProjection(), this.scheduleRender(), this.animationProgress = S;
      }, this.mixTargetDelta(this.options.layoutRoot ? 1e3 : 0);
    }
    startAnimation(o) {
      this.notifyListeners("animationStart"), this.currentAnimation && this.currentAnimation.stop(), this.resumingFrom && this.resumingFrom.currentAnimation && this.resumingFrom.currentAnimation.stop(), this.pendingAnimation && (st.update(this.pendingAnimation), this.pendingAnimation = void 0), this.pendingAnimation = R.update(() => {
        St.hasAnimatedSinceResize = !0, this.currentAnimation = nl(0, Ps, {
          ...o,
          onUpdate: (a) => {
            this.mixTargetDelta(a), o.onUpdate && o.onUpdate(a);
          },
          onComplete: () => {
            o.onComplete && o.onComplete(), this.completeAnimation();
          }
        }), this.resumingFrom && (this.resumingFrom.currentAnimation = this.currentAnimation), this.pendingAnimation = void 0;
      });
    }
    completeAnimation() {
      this.resumingFrom && (this.resumingFrom.currentAnimation = void 0, this.resumingFrom.preserveOpacity = void 0);
      const o = this.getStack();
      o && o.exitAnimationComplete(), this.resumingFrom = this.currentAnimation = this.animationValues = void 0, this.notifyListeners("animationComplete");
    }
    finishAnimation() {
      this.currentAnimation && (this.mixTargetDelta && this.mixTargetDelta(Ps), this.currentAnimation.stop()), this.completeAnimation();
    }
    applyTransformsToTarget() {
      const o = this.getLead();
      let { targetWithTransforms: a, target: c, layout: l, latestValues: u } = o;
      if (!(!a || !c || !l)) {
        if (this !== o && this.layout && l && qi(this.options.animationType, this.layout.layoutBox, l.layoutBox)) {
          c = this.target || A();
          const f = B(this.layout.layoutBox.x);
          c.x.min = o.target.x.min, c.x.max = c.x.min + f;
          const h = B(this.layout.layoutBox.y);
          c.y.min = o.target.y.min, c.y.max = c.y.min + h;
        }
        I(a, c), yt(a, u), Vt(this.projectionDeltaWithTransform, this.layoutCorrected, a, u);
      }
    }
    registerSharedNode(o, a) {
      this.sharedNodes.has(o) || this.sharedNodes.set(o, new ul()), this.sharedNodes.get(o).add(a);
      const l = a.options.initialPromotionConfig;
      a.promote({
        transition: l ? l.transition : void 0,
        preserveFollowOpacity: l && l.shouldPreserveFollowOpacity ? l.shouldPreserveFollowOpacity(a) : void 0
      });
    }
    isLead() {
      const o = this.getStack();
      return o ? o.lead === this : !0;
    }
    getLead() {
      var o;
      const { layoutId: a } = this.options;
      return a ? ((o = this.getStack()) === null || o === void 0 ? void 0 : o.lead) || this : this;
    }
    getPrevLead() {
      var o;
      const { layoutId: a } = this.options;
      return a ? (o = this.getStack()) === null || o === void 0 ? void 0 : o.prevLead : void 0;
    }
    getStack() {
      const { layoutId: o } = this.options;
      if (o)
        return this.root.sharedNodes.get(o);
    }
    promote({ needsReset: o, transition: a, preserveFollowOpacity: c } = {}) {
      const l = this.getStack();
      l && l.promote(this, c), o && (this.projectionDelta = void 0, this.needsReset = !0), a && this.setOptions({ transition: a });
    }
    relegate() {
      const o = this.getStack();
      return o ? o.relegate(this) : !1;
    }
    resetRotation() {
      const { visualElement: o } = this.options;
      if (!o)
        return;
      let a = !1;
      const { latestValues: c } = o;
      if ((c.rotate || c.rotateX || c.rotateY || c.rotateZ) && (a = !0), !a)
        return;
      const l = {};
      for (let u = 0; u < bs.length; u++) {
        const f = "rotate" + bs[u];
        c[f] && (l[f] = c[f], o.setStaticValue(f, 0));
      }
      o.render();
      for (const u in l)
        o.setStaticValue(u, l[u]);
      o.scheduleRender();
    }
    getProjectionStyles(o = {}) {
      var a, c;
      const l = {};
      if (!this.instance || this.isSVG)
        return l;
      if (this.isVisible)
        l.visibility = "";
      else
        return { visibility: "hidden" };
      const u = this.getTransformTemplate();
      if (this.needsReset)
        return this.needsReset = !1, l.opacity = "", l.pointerEvents = Yt(o.pointerEvents) || "", l.transform = u ? u(this.latestValues, "") : "none", l;
      const f = this.getLead();
      if (!this.projectionDelta || !this.layout || !f.target) {
        const m = {};
        return this.options.layoutId && (m.opacity = this.latestValues.opacity !== void 0 ? this.latestValues.opacity : 1, m.pointerEvents = Yt(o.pointerEvents) || ""), this.hasProjected && !at(this.latestValues) && (m.transform = u ? u({}, "") : "none", this.hasProjected = !1), m;
      }
      const h = f.animationValues || f.latestValues;
      this.applyTransformsToTarget(), l.transform = xs(this.projectionDeltaWithTransform, this.treeScale, h), u && (l.transform = u(h, l.transform));
      const { x: d, y: p } = this.projectionDelta;
      l.transformOrigin = `${d.origin * 100}% ${p.origin * 100}% 0`, f.animationValues ? l.opacity = f === this ? (c = (a = h.opacity) !== null && a !== void 0 ? a : this.latestValues.opacity) !== null && c !== void 0 ? c : 1 : this.preserveOpacity ? this.latestValues.opacity : h.opacityExit : l.opacity = f === this ? h.opacity !== void 0 ? h.opacity : "" : h.opacityExit !== void 0 ? h.opacityExit : 0;
      for (const m in te) {
        if (h[m] === void 0)
          continue;
        const { correct: P, applyTo: x } = te[m], v = l.transform === "none" ? h[m] : P(h[m], f);
        if (x) {
          const g = x.length;
          for (let y = 0; y < g; y++)
            l[x[y]] = v;
        } else
          l[m] = v;
      }
      return this.options.layoutId && (l.pointerEvents = f === this ? Yt(o.pointerEvents) || "" : "none"), l;
    }
    clearSnapshot() {
      this.resumeFrom = this.snapshot = void 0;
    }
    // Only run on root
    resetTree() {
      this.root.nodes.forEach((o) => {
        var a;
        return (a = o.currentAnimation) === null || a === void 0 ? void 0 : a.stop();
      }), this.root.nodes.forEach(Ts), this.root.sharedNodes.clear();
    }
  };
}
function ml(t) {
  t.updateLayout();
}
function gl(t) {
  var e;
  const n = ((e = t.resumeFrom) === null || e === void 0 ? void 0 : e.snapshot) || t.snapshot;
  if (t.isLead() && t.layout && n && t.hasListeners("didUpdate")) {
    const { layoutBox: s, measuredBox: i } = t.layout, { animationType: r } = t.options, o = n.source !== t.layout.source;
    r === "size" ? W((f) => {
      const h = o ? n.measuredBox[f] : n.layoutBox[f], d = B(h);
      h.min = s[f].min, h.max = h.min + d;
    }) : qi(r, n.layoutBox, s) && W((f) => {
      const h = o ? n.measuredBox[f] : n.layoutBox[f], d = B(s[f]);
      h.max = h.min + d;
    });
    const a = Mt();
    Vt(a, s, n.layoutBox);
    const c = Mt();
    o ? Vt(c, t.applyTransform(i, !0), n.measuredBox) : Vt(c, s, n.layoutBox);
    const l = !Zi(a);
    let u = !1;
    if (!t.resumeFrom) {
      const f = t.getClosestProjectingParent();
      if (f && !f.resumeFrom) {
        const { snapshot: h, layout: d } = f;
        if (h && d) {
          const p = A();
          wt(p, n.layoutBox, h.layoutBox);
          const m = A();
          wt(m, s, d.layoutBox), Ye(p, m) || (u = !0), f.options.layoutRoot && (t.relativeTarget = m, t.relativeTargetOrigin = p, t.relativeParent = f);
        }
      }
    }
    t.notifyListeners("didUpdate", {
      layout: s,
      snapshot: n,
      delta: c,
      layoutDelta: a,
      hasLayoutChanged: l,
      hasRelativeTargetChanged: u
    });
  } else if (t.isLead()) {
    const { onExitComplete: s } = t.options;
    s && s();
  }
  t.options.transition = void 0;
}
function yl(t) {
  ct.totalNodes++, t.parent && (t.isProjecting() || (t.isProjectionDirty = t.parent.isProjectionDirty), t.isSharedProjectionDirty || (t.isSharedProjectionDirty = Boolean(t.isProjectionDirty || t.parent.isProjectionDirty || t.parent.isSharedProjectionDirty)), t.isTransformDirty || (t.isTransformDirty = t.parent.isTransformDirty));
}
function vl(t) {
  t.isProjectionDirty = t.isSharedProjectionDirty = t.isTransformDirty = !1;
}
function xl(t) {
  t.clearSnapshot();
}
function Ts(t) {
  t.clearMeasurements();
}
function bl(t) {
  const { visualElement: e } = t.options;
  e && e.getProps().onBeforeLayoutMeasure && e.notify("BeforeLayoutMeasure"), t.resetTransform();
}
function Ss(t) {
  t.finishAnimation(), t.targetDelta = t.relativeTarget = t.target = void 0;
}
function Pl(t) {
  t.resolveTargetDelta();
}
function Tl(t) {
  t.calcProjection();
}
function Sl(t) {
  t.resetRotation();
}
function Cl(t) {
  t.removeLeadSnapshot();
}
function Cs(t, e, n) {
  t.translate = C(e.translate, 0, n), t.scale = C(e.scale, 1, n), t.origin = e.origin, t.originPoint = e.originPoint;
}
function As(t, e, n, s) {
  t.min = C(e.min, n.min, s), t.max = C(e.max, n.max, s);
}
function Al(t, e, n, s) {
  As(t.x, e.x, n.x, s), As(t.y, e.y, n.y, s);
}
function Vl(t) {
  return t.animationValues && t.animationValues.opacityExit !== void 0;
}
const wl = {
  duration: 0.45,
  ease: [0.4, 0, 0.1, 1]
};
function Ml(t, e) {
  let n = t.root;
  for (let r = t.path.length - 1; r >= 0; r--)
    if (Boolean(t.path[r].instance)) {
      n = t.path[r];
      break;
    }
  const i = (n && n !== t.root ? n.instance : document).querySelector(`[data-projection-id="${e}"]`);
  i && t.mount(i, !0);
}
function Vs(t) {
  t.min = Math.round(t.min), t.max = Math.round(t.max);
}
function Dl(t) {
  Vs(t.x), Vs(t.y);
}
function qi(t, e, n) {
  return t === "position" || t === "preserve-aspect" && !We(vs(e), vs(n), 0.2);
}
const Rl = Yi({
  attachResizeListener: (t, e) => K(t, "resize", e),
  measureScroll: () => ({
    x: document.documentElement.scrollLeft || document.body.scrollLeft,
    y: document.documentElement.scrollTop || document.body.scrollTop
  }),
  checkIsScrollRoot: () => !0
}), De = {
  current: void 0
}, Xi = Yi({
  measureScroll: (t) => ({
    x: t.scrollLeft,
    y: t.scrollTop
  }),
  defaultParent: () => {
    if (!De.current) {
      const t = new Rl(0, {});
      t.mount(window), t.setOptions({ layoutScroll: !0 }), De.current = t;
    }
    return De.current;
  },
  resetTransform: (t, e) => {
    t.style.transform = e !== void 0 ? e : "none";
  },
  checkIsScrollRoot: (t) => Boolean(window.getComputedStyle(t).position === "fixed")
}), El = {
  pan: {
    Feature: Gc
  },
  drag: {
    Feature: Hc,
    ProjectionNode: Xi,
    MeasureLayout: Hi
  }
}, Ll = /* @__PURE__ */ new Set([
  "width",
  "height",
  "top",
  "left",
  "right",
  "bottom",
  "x",
  "y"
]), Ji = (t) => Ll.has(t), kl = (t) => Object.keys(t).some(Ji), Zt = (t) => t === ht || t === b, ws = (t, e) => parseFloat(t.split(", ")[e]), Ms = (t, e) => (n, { transform: s }) => {
  if (s === "none" || !s)
    return 0;
  const i = s.match(/^matrix3d\((.+)\)$/);
  if (i)
    return ws(i[1], e);
  {
    const r = s.match(/^matrix\((.+)\)$/);
    return r ? ws(r[1], t) : 0;
  }
}, jl = /* @__PURE__ */ new Set(["x", "y", "z"]), Bl = he.filter((t) => !jl.has(t));
function Fl(t) {
  const e = [];
  return Bl.forEach((n) => {
    const s = t.getValue(n);
    s !== void 0 && (e.push([n, s.get()]), s.set(n.startsWith("scale") ? 1 : 0));
  }), e.length && t.render(), e;
}
const Ds = {
  // Dimensions
  width: ({ x: t }, { paddingLeft: e = "0", paddingRight: n = "0" }) => t.max - t.min - parseFloat(e) - parseFloat(n),
  height: ({ y: t }, { paddingTop: e = "0", paddingBottom: n = "0" }) => t.max - t.min - parseFloat(e) - parseFloat(n),
  top: (t, { top: e }) => parseFloat(e),
  left: (t, { left: e }) => parseFloat(e),
  bottom: ({ y: t }, { top: e }) => parseFloat(e) + (t.max - t.min),
  right: ({ x: t }, { left: e }) => parseFloat(e) + (t.max - t.min),
  // Transform
  x: Ms(4, 13),
  y: Ms(5, 14)
}, Il = (t, e, n) => {
  const s = e.measureViewportBox(), i = e.current, r = getComputedStyle(i), { display: o } = r, a = {};
  o === "none" && e.setStaticValue("display", t.display || "block"), n.forEach((l) => {
    a[l] = Ds[l](s, r);
  }), e.render();
  const c = e.measureViewportBox();
  return n.forEach((l) => {
    const u = e.getValue(l);
    u && u.jump(a[l]), t[l] = Ds[l](c, r);
  }), t;
}, Ol = (t, e, n = {}, s = {}) => {
  e = { ...e }, s = { ...s };
  const i = Object.keys(e).filter(Ji);
  let r = [], o = !1;
  const a = [];
  if (i.forEach((c) => {
    const l = t.getValue(c);
    if (!t.hasValue(c))
      return;
    let u = n[c], f = bt(u);
    const h = e[c];
    let d;
    if (ne(h)) {
      const p = h.length, m = h[0] === null ? 1 : 0;
      u = h[m], f = bt(u);
      for (let P = m; P < p; P++)
        d ? $(bt(h[P]) === d, "All keyframes must be of the same type") : (d = bt(h[P]), $(d === f || Zt(f) && Zt(d), "Keyframes must be of the same dimension as the current value"));
    } else
      d = bt(h);
    if (f !== d)
      if (Zt(f) && Zt(d)) {
        const p = l.get();
        typeof p == "string" && l.set(parseFloat(p)), typeof h == "string" ? e[c] = parseFloat(h) : Array.isArray(h) && d === b && (e[c] = h.map(parseFloat));
      } else
        f != null && f.transform && (d != null && d.transform) && (u === 0 || h === 0) ? u === 0 ? l.set(d.transform(u)) : e[c] = f.transform(h) : (o || (r = Fl(t), o = !0), a.push(c), s[c] = s[c] !== void 0 ? s[c] : e[c], l.jump(h));
  }), a.length) {
    const c = a.indexOf("height") >= 0 ? window.pageYOffset : null, l = Il(e, t, a);
    return r.length && r.forEach(([u, f]) => {
      t.getValue(u).set(f);
    }), t.render(), le && c !== null && window.scrollTo({ top: c }), { target: l, transitionEnd: s };
  } else
    return { target: e, transitionEnd: s };
};
function Nl(t, e, n, s) {
  return kl(e) ? Ol(t, e, n, s) : { target: e, transitionEnd: s };
}
const Ul = (t, e, n, s) => {
  const i = Jc(t, e, s);
  return e = i.target, s = i.transitionEnd, Nl(t, e, n, s);
}, qe = { current: null }, Qi = { current: !1 };
function _l() {
  if (Qi.current = !0, !!le)
    if (window.matchMedia) {
      const t = window.matchMedia("(prefers-reduced-motion)"), e = () => qe.current = t.matches;
      t.addListener(e), e();
    } else
      qe.current = !1;
}
const Rs = /* @__PURE__ */ new Set();
function Vn(t, e, n) {
  t || Rs.has(e) || (console.warn(e), n && console.warn(n), Rs.add(e));
}
function $l(t, e, n) {
  const { willChange: s } = e;
  for (const i in e) {
    const r = e[i], o = n[i];
    if (j(r))
      t.addValue(i, r), ie(s) && s.add(i), process.env.NODE_ENV === "development" && Vn(r.version === "9.1.7", `Attempting to mix Framer Motion versions ${r.version} with 9.1.7 may not work as expected.`);
    else if (j(o))
      t.addValue(i, xt(r, { owner: t })), ie(s) && s.remove(i);
    else if (o !== r)
      if (t.hasValue(i)) {
        const a = t.getValue(i);
        !a.hasAnimated && a.set(r);
      } else {
        const a = t.getStaticValue(i);
        t.addValue(i, xt(a !== void 0 ? a : r, { owner: t }));
      }
  }
  for (const i in n)
    e[i] === void 0 && t.removeValue(i);
  return e;
}
const to = Object.keys(Et), zl = to.length, Es = [
  "AnimationStart",
  "AnimationComplete",
  "Update",
  "BeforeLayoutMeasure",
  "LayoutMeasure",
  "LayoutAnimationStart",
  "LayoutAnimationComplete"
], Wl = tn.length;
class Hl {
  constructor({ parent: e, props: n, presenceContext: s, reducedMotionConfig: i, visualState: r }, o = {}) {
    this.current = null, this.children = /* @__PURE__ */ new Set(), this.isVariantNode = !1, this.isControllingVariants = !1, this.shouldReduceMotion = null, this.values = /* @__PURE__ */ new Map(), this.features = {}, this.valueSubscriptions = /* @__PURE__ */ new Map(), this.prevMotionValues = {}, this.events = {}, this.propEventSubscriptions = {}, this.notifyUpdate = () => this.notify("Update", this.latestValues), this.render = () => {
      this.current && (this.triggerBuild(), this.renderInstance(this.current, this.renderState, this.props.style, this.projection));
    }, this.scheduleRender = () => R.render(this.render, !1, !0);
    const { latestValues: a, renderState: c } = r;
    this.latestValues = a, this.baseTarget = { ...a }, this.initialValues = n.initial ? { ...a } : {}, this.renderState = c, this.parent = e, this.props = n, this.presenceContext = s, this.depth = e ? e.depth + 1 : 0, this.reducedMotionConfig = i, this.options = o, this.isControllingVariants = fe(n), this.isVariantNode = Gs(n), this.isVariantNode && (this.variantChildren = /* @__PURE__ */ new Set()), this.manuallyAnimateOnMount = Boolean(e && e.current);
    const { willChange: l, ...u } = this.scrapeMotionValuesFromProps(n, {});
    for (const f in u) {
      const h = u[f];
      a[f] !== void 0 && j(h) && (h.set(a[f], !1), ie(l) && l.add(f));
    }
  }
  /**
   * This method takes React props and returns found MotionValues. For example, HTML
   * MotionValues will be found within the style prop, whereas for Three.js within attribute arrays.
   *
   * This isn't an abstract method as it needs calling in the constructor, but it is
   * intended to be one.
   */
  scrapeMotionValuesFromProps(e, n) {
    return {};
  }
  mount(e) {
    this.current = e, this.projection && this.projection.mount(e), this.parent && this.isVariantNode && !this.isControllingVariants && (this.removeFromVariantTree = this.parent.addVariantChild(this)), this.values.forEach((n, s) => this.bindToMotionValue(s, n)), Qi.current || _l(), this.shouldReduceMotion = this.reducedMotionConfig === "never" ? !1 : this.reducedMotionConfig === "always" ? !0 : qe.current, process.env.NODE_ENV !== "production" && Vn(this.shouldReduceMotion !== !0, "You have Reduced Motion enabled on your device. Animations may not appear as expected."), this.parent && this.parent.children.add(this), this.update(this.props, this.presenceContext);
  }
  unmount() {
    this.projection && this.projection.unmount(), st.update(this.notifyUpdate), st.render(this.render), this.valueSubscriptions.forEach((e) => e()), this.removeFromVariantTree && this.removeFromVariantTree(), this.parent && this.parent.children.delete(this);
    for (const e in this.events)
      this.events[e].clear();
    for (const e in this.features)
      this.features[e].unmount();
    this.current = null;
  }
  bindToMotionValue(e, n) {
    const s = ft.has(e), i = n.on("change", (o) => {
      this.latestValues[e] = o, this.props.onUpdate && R.update(this.notifyUpdate, !1, !0), s && this.projection && (this.projection.isTransformDirty = !0);
    }), r = n.on("renderRequest", this.scheduleRender);
    this.valueSubscriptions.set(e, () => {
      i(), r();
    });
  }
  sortNodePosition(e) {
    return !this.current || !this.sortInstanceNodePosition || this.type !== e.type ? 0 : this.sortInstanceNodePosition(this.current, e.current);
  }
  loadFeatures({ children: e, ...n }, s, i, r, o) {
    let a, c;
    if (process.env.NODE_ENV !== "production" && i && s) {
      const l = "You have rendered a `motion` component within a `LazyMotion` component. This will break tree shaking. Import and render a `m` component instead.";
      n.ignoreStrict ? $t(!1, l) : $(!1, l);
    }
    for (let l = 0; l < zl; l++) {
      const u = to[l], { isEnabled: f, Feature: h, ProjectionNode: d, MeasureLayout: p } = Et[u];
      d && (a = d), f(n) && (!this.features[u] && h && (this.features[u] = new h(this)), p && (c = p));
    }
    if (!this.projection && a) {
      this.projection = new a(r, this.latestValues, this.parent && this.parent.projection);
      const { layoutId: l, layout: u, drag: f, dragConstraints: h, layoutScroll: d, layoutRoot: p } = n;
      this.projection.setOptions({
        layoutId: l,
        layout: u,
        alwaysMeasureLayout: Boolean(f) || h && mt(h),
        visualElement: this,
        scheduleRender: () => this.scheduleRender(),
        /**
         * TODO: Update options in an effect. This could be tricky as it'll be too late
         * to update by the time layout animations run.
         * We also need to fix this safeToRemove by linking it up to the one returned by usePresence,
         * ensuring it gets called if there's no potential layout animations.
         *
         */
        animationType: typeof u == "string" ? u : "both",
        initialPromotionConfig: o,
        layoutScroll: d,
        layoutRoot: p
      });
    }
    return c;
  }
  updateFeatures() {
    for (const e in this.features) {
      const n = this.features[e];
      n.isMounted ? n.update(this.props, this.prevProps) : (n.mount(), n.isMounted = !0);
    }
  }
  triggerBuild() {
    this.build(this.renderState, this.latestValues, this.options, this.props);
  }
  /**
   * Measure the current viewport box with or without transforms.
   * Only measures axis-aligned boxes, rotate and skew must be manually
   * removed with a re-render to work.
   */
  measureViewportBox() {
    return this.current ? this.measureInstanceViewportBox(this.current, this.props) : A();
  }
  getStaticValue(e) {
    return this.latestValues[e];
  }
  setStaticValue(e, n) {
    this.latestValues[e] = n;
  }
  /**
   * Make a target animatable by Popmotion. For instance, if we're
   * trying to animate width from 100px to 100vw we need to measure 100vw
   * in pixels to determine what we really need to animate to. This is also
   * pluggable to support Framer's custom value types like Color,
   * and CSS variables.
   */
  makeTargetAnimatable(e, n = !0) {
    return this.makeTargetAnimatableFromInstance(e, this.props, n);
  }
  /**
   * Update the provided props. Ensure any newly-added motion values are
   * added to our map, old ones removed, and listeners updated.
   */
  update(e, n) {
    (e.transformTemplate || this.props.transformTemplate) && this.scheduleRender(), this.prevProps = this.props, this.props = e, this.prevPresenceContext = this.presenceContext, this.presenceContext = n;
    for (let s = 0; s < Es.length; s++) {
      const i = Es[s];
      this.propEventSubscriptions[i] && (this.propEventSubscriptions[i](), delete this.propEventSubscriptions[i]);
      const r = e["on" + i];
      r && (this.propEventSubscriptions[i] = this.on(i, r));
    }
    this.prevMotionValues = $l(this, this.scrapeMotionValuesFromProps(e, this.prevProps), this.prevMotionValues), this.handleChildMotionValue && this.handleChildMotionValue();
  }
  getProps() {
    return this.props;
  }
  /**
   * Returns the variant definition with a given name.
   */
  getVariant(e) {
    return this.props.variants ? this.props.variants[e] : void 0;
  }
  /**
   * Returns the defined default transition on this component.
   */
  getDefaultTransition() {
    return this.props.transition;
  }
  getTransformPagePoint() {
    return this.props.transformPagePoint;
  }
  getClosestVariantNode() {
    return this.isVariantNode ? this : this.parent ? this.parent.getClosestVariantNode() : void 0;
  }
  getVariantContext(e = !1) {
    if (e)
      return this.parent ? this.parent.getVariantContext() : void 0;
    if (!this.isControllingVariants) {
      const s = this.parent ? this.parent.getVariantContext() || {} : {};
      return this.props.initial !== void 0 && (s.initial = this.props.initial), s;
    }
    const n = {};
    for (let s = 0; s < Wl; s++) {
      const i = tn[s], r = this.props[i];
      (Rt(r) || r === !1) && (n[i] = r);
    }
    return n;
  }
  /**
   * Add a child visual element to our set of children.
   */
  addVariantChild(e) {
    const n = this.getClosestVariantNode();
    if (n)
      return n.variantChildren && n.variantChildren.add(e), () => n.variantChildren.delete(e);
  }
  /**
   * Add a motion value and bind it to this visual element.
   */
  addValue(e, n) {
    n !== this.values.get(e) && (this.removeValue(e), this.bindToMotionValue(e, n)), this.values.set(e, n), this.latestValues[e] = n.get();
  }
  /**
   * Remove a motion value and unbind any active subscriptions.
   */
  removeValue(e) {
    this.values.delete(e);
    const n = this.valueSubscriptions.get(e);
    n && (n(), this.valueSubscriptions.delete(e)), delete this.latestValues[e], this.removeValueFromRenderState(e, this.renderState);
  }
  /**
   * Check whether we have a motion value for this key
   */
  hasValue(e) {
    return this.values.has(e);
  }
  getValue(e, n) {
    if (this.props.values && this.props.values[e])
      return this.props.values[e];
    let s = this.values.get(e);
    return s === void 0 && n !== void 0 && (s = xt(n, { owner: this }), this.addValue(e, s)), s;
  }
  /**
   * If we're trying to animate to a previously unencountered value,
   * we need to check for it in our state and as a last resort read it
   * directly from the instance (which might have performance implications).
   */
  readValue(e) {
    return this.latestValues[e] !== void 0 || !this.current ? this.latestValues[e] : this.readValueFromInstance(this.current, e, this.options);
  }
  /**
   * Set the base target to later animate back to. This is currently
   * only hydrated on creation and when we first read a value.
   */
  setBaseTarget(e, n) {
    this.baseTarget[e] = n;
  }
  /**
   * Find the base target for a value thats been removed from all animation
   * props.
   */
  getBaseTarget(e) {
    var n;
    const { initial: s } = this.props, i = typeof s == "string" || typeof s == "object" ? (n = fn(this.props, s)) === null || n === void 0 ? void 0 : n[e] : void 0;
    if (s && i !== void 0)
      return i;
    const r = this.getBaseTargetFromProps(this.props, e);
    return r !== void 0 && !j(r) ? r : this.initialValues[e] !== void 0 && i === void 0 ? void 0 : this.baseTarget[e];
  }
  on(e, n) {
    return this.events[e] || (this.events[e] = new pn()), this.events[e].add(n);
  }
  notify(e, ...n) {
    this.events[e] && this.events[e].notify(...n);
  }
}
class eo extends Hl {
  sortInstanceNodePosition(e, n) {
    return e.compareDocumentPosition(n) & 2 ? 1 : -1;
  }
  getBaseTargetFromProps(e, n) {
    return e.style ? e.style[n] : void 0;
  }
  removeValueFromRenderState(e, { vars: n, style: s }) {
    delete n[e], delete s[e];
  }
  makeTargetAnimatableFromInstance({ transition: e, transitionEnd: n, ...s }, { transformValues: i }, r) {
    let o = da(s, e || {}, this);
    if (i && (n && (n = i(n)), s && (s = i(s)), o && (o = i(o))), r) {
      fa(this, s, o);
      const a = Ul(this, s, o, n);
      n = a.transitionEnd, s = a.target;
    }
    return {
      transition: e,
      transitionEnd: n,
      ...s
    };
  }
}
function Gl(t) {
  return window.getComputedStyle(t);
}
class Kl extends eo {
  readValueFromInstance(e, n) {
    if (ft.has(n)) {
      const s = yn(n);
      return s && s.default || 0;
    } else {
      const s = Gl(e), i = (qs(n) ? s.getPropertyValue(n) : s[n]) || 0;
      return typeof i == "string" ? i.trim() : i;
    }
  }
  measureInstanceViewportBox(e, { transformPagePoint: n }) {
    return zi(e, n);
  }
  build(e, n, s, i) {
    on(e, n, s, i.transformTemplate);
  }
  scrapeMotionValuesFromProps(e, n) {
    return un(e, n);
  }
  handleChildMotionValue() {
    this.childSubscription && (this.childSubscription(), delete this.childSubscription);
    const { children: e } = this.props;
    j(e) && (this.childSubscription = e.on("change", (n) => {
      this.current && (this.current.textContent = `${n}`);
    }));
  }
  renderInstance(e, n, s, i) {
    ei(e, n, s, i);
  }
}
class Zl extends eo {
  constructor() {
    super(...arguments), this.isSVGTag = !1;
  }
  getBaseTargetFromProps(e, n) {
    return e[n];
  }
  readValueFromInstance(e, n) {
    if (ft.has(n)) {
      const s = yn(n);
      return s && s.default || 0;
    }
    return n = ni.has(n) ? n : ln(n), e.getAttribute(n);
  }
  measureInstanceViewportBox() {
    return A();
  }
  scrapeMotionValuesFromProps(e, n) {
    return ii(e, n);
  }
  build(e, n, s, i) {
    an(e, n, s, this.isSVGTag, i.transformTemplate);
  }
  renderInstance(e, n, s, i) {
    si(e, n, s, i);
  }
  mount(e) {
    this.isSVGTag = cn(e.tagName), super.mount(e);
  }
}
const Yl = (t, e) => sn(t) ? new Zl(e, { enableHardwareAcceleration: !1 }) : new Kl(e, { enableHardwareAcceleration: !0 }), ql = {
  layout: {
    ProjectionNode: Xi,
    MeasureLayout: Hi
  }
}, Xl = {
  ...Ac,
  ...Nr,
  ...El,
  ...ql
}, Jl = /* @__PURE__ */ Jo((t, e) => Vr(t, e, Xl, Yl));
function no() {
  const t = O(!1);
  return Qt(() => (t.current = !0, () => {
    t.current = !1;
  }), []), t;
}
function Ql() {
  const t = no(), [e, n] = Dt(0), s = Xt(() => {
    t.current && n(e + 1);
  }, [e]);
  return [Xt(() => R.postRender(s), [s]), e];
}
class tu extends L.Component {
  getSnapshotBeforeUpdate(e) {
    const n = this.props.childRef.current;
    if (n && e.isPresent && !this.props.isPresent) {
      const s = this.props.sizeRef.current;
      s.height = n.offsetHeight || 0, s.width = n.offsetWidth || 0, s.top = n.offsetTop, s.left = n.offsetLeft;
    }
    return null;
  }
  /**
   * Required with getSnapshotBeforeUpdate to stop React complaining.
   */
  componentDidUpdate() {
  }
  render() {
    return this.props.children;
  }
}
function eu({ children: t, isPresent: e }) {
  const n = Xe(), s = O(null), i = O({
    width: 0,
    height: 0,
    top: 0,
    left: 0
  });
  return Is(() => {
    const { width: r, height: o, top: a, left: c } = i.current;
    if (e || !s.current || !r || !o)
      return;
    s.current.dataset.motionPopId = n;
    const l = document.createElement("style");
    return document.head.appendChild(l), l.sheet && l.sheet.insertRule(`
          [data-motion-pop-id="${n}"] {
            position: absolute !important;
            width: ${r}px !important;
            height: ${o}px !important;
            top: ${a}px !important;
            left: ${c}px !important;
          }
        `), () => {
      document.head.removeChild(l);
    };
  }, [e]), L.createElement(tu, { isPresent: e, childRef: s, sizeRef: i }, L.cloneElement(t, { ref: s }));
}
const Re = ({ children: t, initial: e, isPresent: n, onExitComplete: s, custom: i, presenceAffectsLayout: r, mode: o }) => {
  const a = en(nu), c = Xe(), l = q(
    () => ({
      id: c,
      initial: e,
      isPresent: n,
      custom: i,
      onExitComplete: (u) => {
        a.set(u, !0);
        for (const f of a.values())
          if (!f)
            return;
        s && s();
      },
      register: (u) => (a.set(u, !1), () => a.delete(u))
    }),
    /**
     * If the presence of a child affects the layout of the components around it,
     * we want to make a new context value to ensure they get re-rendered
     * so they can detect that layout change.
     */
    r ? void 0 : [n]
  );
  return q(() => {
    a.forEach((u, f) => a.set(f, !1));
  }, [n]), L.useEffect(() => {
    !n && !a.size && s && s();
  }, [n]), o === "popLayout" && (t = L.createElement(eu, { isPresent: n }, t)), L.createElement(Ot.Provider, { value: l }, t);
};
function nu() {
  return /* @__PURE__ */ new Map();
}
function su(t) {
  return U(() => () => t(), []);
}
const pt = (t) => t.key || "";
function iu(t, e) {
  t.forEach((n) => {
    const s = pt(n);
    e.set(s, n);
  });
}
function ou(t) {
  const e = [];
  return go.forEach(t, (n) => {
    yo(n) && e.push(n);
  }), e;
}
const ru = ({ children: t, custom: e, initial: n = !0, onExitComplete: s, exitBeforeEnter: i, presenceAffectsLayout: r = !0, mode: o = "sync" }) => {
  i && (o = "wait", process.env.NODE_ENV !== "production" && Vn(!1, "Replace exitBeforeEnter with mode='wait'"));
  let [a] = Ql();
  const c = w(nn).forceRender;
  c && (a = c);
  const l = no(), u = ou(t);
  let f = u;
  const h = /* @__PURE__ */ new Set(), d = O(f), p = O(/* @__PURE__ */ new Map()).current, m = O(!0);
  if (Qt(() => {
    m.current = !1, iu(u, p), d.current = f;
  }), su(() => {
    m.current = !0, p.clear(), h.clear();
  }), m.current)
    return L.createElement(L.Fragment, null, f.map((g) => L.createElement(Re, { key: pt(g), isPresent: !0, initial: n ? void 0 : !1, presenceAffectsLayout: r, mode: o }, g)));
  f = [...f];
  const P = d.current.map(pt), x = u.map(pt), v = P.length;
  for (let g = 0; g < v; g++) {
    const y = P[g];
    x.indexOf(y) === -1 && h.add(y);
  }
  return o === "wait" && h.size && (f = []), h.forEach((g) => {
    if (x.indexOf(g) !== -1)
      return;
    const y = p.get(g);
    if (!y)
      return;
    const S = P.indexOf(g), V = () => {
      p.delete(g), h.delete(g);
      const D = d.current.findIndex((z) => z.key === g);
      if (d.current.splice(D, 1), !h.size) {
        if (d.current = u, l.current === !1)
          return;
        a(), s && s();
      }
    };
    f.splice(S, 0, L.createElement(Re, { key: pt(y), isPresent: !1, onExitComplete: V, custom: e, presenceAffectsLayout: r, mode: o }, y));
  }), f = f.map((g) => {
    const y = g.key;
    return h.has(y) ? g : L.createElement(Re, { key: pt(g), isPresent: !0, presenceAffectsLayout: r, mode: o }, g);
  }), process.env.NODE_ENV !== "production" && o === "wait" && f.length > 1 && console.warn(`You're attempting to animate multiple children within AnimatePresence, but its mode is set to "wait". This will lead to odd visual behaviour.`), L.createElement(L.Fragment, null, h.size ? f : f.map((g) => mo(g)));
};
var au = {
  initial: (t) => {
    const { position: e } = t, n = ["top", "bottom"].includes(e) ? "y" : "x";
    let s = ["top-right", "bottom-right"].includes(e) ? 1 : -1;
    return e === "bottom" && (s = 1), {
      opacity: 0,
      [n]: s * 24
    };
  },
  animate: {
    opacity: 1,
    y: 0,
    x: 0,
    scale: 1,
    transition: {
      duration: 0.4,
      ease: [0.4, 0, 0.2, 1]
    }
  },
  exit: {
    opacity: 0,
    scale: 0.85,
    transition: {
      duration: 0.2,
      ease: [0.4, 0, 1, 1]
    }
  }
}, so = vo((t) => {
  const {
    id: e,
    message: n,
    onCloseComplete: s,
    onRequestRemove: i,
    requestClose: r = !1,
    position: o = "bottom",
    duration: a = 5e3,
    containerStyle: c,
    motionVariants: l = au,
    toastSpacing: u = "0.5rem"
  } = t, [f, h] = Dt(a), d = Zc();
  Dn(() => {
    d || s == null || s();
  }, [d]), Dn(() => {
    h(a);
  }, [a]);
  const p = () => h(null), m = () => h(a), P = () => {
    d && i();
  };
  U(() => {
    d && r && i();
  }, [d, r, i]), Uo(P, f);
  const x = q(
    () => ({
      pointerEvents: "auto",
      maxWidth: 560,
      minWidth: 300,
      margin: u,
      ...c
    }),
    [c, u]
  ), v = q(() => Io(o), [o]);
  return /* @__PURE__ */ T.jsx(
    Jl.li,
    {
      layout: !0,
      className: "chakra-toast",
      variants: l,
      initial: "initial",
      animate: "animate",
      exit: "exit",
      onHoverStart: p,
      onHoverEnd: m,
      custom: { position: o },
      style: v,
      children: /* @__PURE__ */ T.jsx(
        X.div,
        {
          role: "status",
          "aria-atomic": "true",
          className: "chakra-toast__inner",
          __css: x,
          children: Ee(n, { id: e, onClose: P })
        }
      )
    }
  );
});
so.displayName = "ToastComponent";
function cu(t, e) {
  var n;
  const s = t ?? "bottom", r = {
    "top-start": { ltr: "top-left", rtl: "top-right" },
    "top-end": { ltr: "top-right", rtl: "top-left" },
    "bottom-start": { ltr: "bottom-left", rtl: "bottom-right" },
    "bottom-end": { ltr: "bottom-right", rtl: "bottom-left" }
  }[s];
  return (n = r == null ? void 0 : r[e]) != null ? n : s;
}
var Ls = {
  path: /* @__PURE__ */ T.jsxs("g", { stroke: "currentColor", strokeWidth: "1.5", children: [
    /* @__PURE__ */ T.jsx(
      "path",
      {
        strokeLinecap: "round",
        fill: "none",
        d: "M9,9a3,3,0,1,1,4,2.829,1.5,1.5,0,0,0-1,1.415V14.25"
      }
    ),
    /* @__PURE__ */ T.jsx(
      "path",
      {
        fill: "currentColor",
        strokeLinecap: "round",
        d: "M12,17.25a.375.375,0,1,0,.375.375A.375.375,0,0,0,12,17.25h0"
      }
    ),
    /* @__PURE__ */ T.jsx("circle", { fill: "none", strokeMiterlimit: "10", cx: "12", cy: "12", r: "11.25" })
  ] }),
  viewBox: "0 0 24 24"
}, zt = Bt((t, e) => {
  const {
    as: n,
    viewBox: s,
    color: i = "currentColor",
    focusable: r = !1,
    children: o,
    className: a,
    __css: c,
    ...l
  } = t, u = Ft("chakra-icon", a), f = Os("Icon", t), h = {
    w: "1em",
    h: "1em",
    display: "inline-block",
    lineHeight: "1em",
    flexShrink: 0,
    color: i,
    ...c,
    ...f
  }, d = {
    ref: e,
    focusable: r,
    className: u,
    __css: h
  }, p = s ?? Ls.viewBox;
  if (n && typeof n != "string")
    return /* @__PURE__ */ T.jsx(X.svg, { as: n, ...d, ...l });
  const m = o ?? Ls.path;
  return /* @__PURE__ */ T.jsx(X.svg, { verticalAlign: "middle", viewBox: p, ...d, ...l, children: m });
});
zt.displayName = "Icon";
function lu(t) {
  return /* @__PURE__ */ T.jsx(zt, { viewBox: "0 0 24 24", ...t, children: /* @__PURE__ */ T.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M12,0A12,12,0,1,0,24,12,12.014,12.014,0,0,0,12,0Zm6.927,8.2-6.845,9.289a1.011,1.011,0,0,1-1.43.188L5.764,13.769a1,1,0,1,1,1.25-1.562l4.076,3.261,6.227-8.451A1,1,0,1,1,18.927,8.2Z"
    }
  ) });
}
function uu(t) {
  return /* @__PURE__ */ T.jsx(zt, { viewBox: "0 0 24 24", ...t, children: /* @__PURE__ */ T.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M12,0A12,12,0,1,0,24,12,12.013,12.013,0,0,0,12,0Zm.25,5a1.5,1.5,0,1,1-1.5,1.5A1.5,1.5,0,0,1,12.25,5ZM14.5,18.5h-4a1,1,0,0,1,0-2h.75a.25.25,0,0,0,.25-.25v-4.5a.25.25,0,0,0-.25-.25H10.5a1,1,0,0,1,0-2h1a2,2,0,0,1,2,2v4.75a.25.25,0,0,0,.25.25h.75a1,1,0,1,1,0,2Z"
    }
  ) });
}
function ks(t) {
  return /* @__PURE__ */ T.jsx(zt, { viewBox: "0 0 24 24", ...t, children: /* @__PURE__ */ T.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M11.983,0a12.206,12.206,0,0,0-8.51,3.653A11.8,11.8,0,0,0,0,12.207,11.779,11.779,0,0,0,11.8,24h.214A12.111,12.111,0,0,0,24,11.791h0A11.766,11.766,0,0,0,11.983,0ZM10.5,16.542a1.476,1.476,0,0,1,1.449-1.53h.027a1.527,1.527,0,0,1,1.523,1.47,1.475,1.475,0,0,1-1.449,1.53h-.027A1.529,1.529,0,0,1,10.5,16.542ZM11,12.5v-6a1,1,0,0,1,2,0v6a1,1,0,1,1-2,0Z"
    }
  ) });
}
var [fu, hu] = It({
  name: "AlertContext",
  hookName: "useAlertContext",
  providerName: "<Alert />"
}), [du, wn] = It({
  name: "AlertStylesContext",
  hookName: "useAlertStyles",
  providerName: "<Alert />"
}), io = {
  info: { icon: uu, colorScheme: "blue" },
  warning: { icon: ks, colorScheme: "orange" },
  success: { icon: lu, colorScheme: "green" },
  error: { icon: ks, colorScheme: "red" },
  loading: { icon: Po, colorScheme: "blue" }
};
function pu(t) {
  return io[t].colorScheme;
}
function mu(t) {
  return io[t].icon;
}
var oo = Bt(
  function(e, n) {
    const i = {
      display: "inline",
      ...wn().description
    };
    return /* @__PURE__ */ T.jsx(
      X.div,
      {
        ref: n,
        ...e,
        className: Ft("chakra-alert__desc", e.className),
        __css: i
      }
    );
  }
);
oo.displayName = "AlertDescription";
function ro(t) {
  const { status: e } = hu(), n = mu(e), s = wn(), i = e === "loading" ? s.spinner : s.icon;
  return /* @__PURE__ */ T.jsx(
    X.span,
    {
      display: "inherit",
      ...t,
      className: Ft("chakra-alert__icon", t.className),
      __css: i,
      children: t.children || /* @__PURE__ */ T.jsx(n, { h: "100%", w: "100%" })
    }
  );
}
ro.displayName = "AlertIcon";
var ao = Bt(
  function(e, n) {
    const s = wn();
    return /* @__PURE__ */ T.jsx(
      X.div,
      {
        ref: n,
        ...e,
        className: Ft("chakra-alert__title", e.className),
        __css: s.title
      }
    );
  }
);
ao.displayName = "AlertTitle";
var co = Bt(function(e, n) {
  var s;
  const { status: i = "info", addRole: r = !0, ...o } = Ns(e), a = (s = e.colorScheme) != null ? s : pu(i), c = To("Alert", { ...e, colorScheme: a }), l = {
    width: "100%",
    display: "flex",
    alignItems: "center",
    position: "relative",
    overflow: "hidden",
    ...c.container
  };
  return /* @__PURE__ */ T.jsx(fu, { value: { status: i }, children: /* @__PURE__ */ T.jsx(du, { value: c, children: /* @__PURE__ */ T.jsx(
    X.div,
    {
      role: r ? "alert" : void 0,
      ref: n,
      ...o,
      className: Ft("chakra-alert", e.className),
      __css: l
    }
  ) }) });
});
co.displayName = "Alert";
function gu(t) {
  return /* @__PURE__ */ T.jsx(zt, { focusable: "false", "aria-hidden": !0, ...t, children: /* @__PURE__ */ T.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M.439,21.44a1.5,1.5,0,0,0,2.122,2.121L11.823,14.3a.25.25,0,0,1,.354,0l9.262,9.263a1.5,1.5,0,1,0,2.122-2.121L14.3,12.177a.25.25,0,0,1,0-.354l9.263-9.262A1.5,1.5,0,0,0,21.439.44L12.177,9.7a.25.25,0,0,1-.354,0L2.561.44A1.5,1.5,0,0,0,.439,2.561L9.7,11.823a.25.25,0,0,1,0,.354Z"
    }
  ) });
}
var lo = Bt(
  function(e, n) {
    const s = Os("CloseButton", e), { children: i, isDisabled: r, __css: o, ...a } = Ns(e), c = {
      outline: 0,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0
    };
    return /* @__PURE__ */ T.jsx(
      X.button,
      {
        type: "button",
        "aria-label": "Close",
        ref: n,
        disabled: r,
        __css: {
          ...c,
          ...s,
          ...o
        },
        ...a,
        children: i || /* @__PURE__ */ T.jsx(gu, { width: "1em", height: "1em" })
      }
    );
  }
);
lo.displayName = "CloseButton";
var yu = {
  top: [],
  "top-left": [],
  "top-right": [],
  "bottom-left": [],
  bottom: [],
  "bottom-right": []
}, H = vu(yu);
function vu(t) {
  let e = t;
  const n = /* @__PURE__ */ new Set(), s = (i) => {
    e = i(e), n.forEach((r) => r());
  };
  return {
    getState: () => e,
    subscribe: (i) => (n.add(i), () => {
      s(() => t), n.delete(i);
    }),
    removeToast: (i, r) => {
      s((o) => ({
        ...o,
        [r]: o[r].filter((a) => a.id != i)
      }));
    },
    notify: (i, r) => {
      const o = xu(i, r), { position: a, id: c } = o;
      return s((l) => {
        var u, f;
        const d = a.includes("top") ? [o, ...(u = l[a]) != null ? u : []] : [...(f = l[a]) != null ? f : [], o];
        return {
          ...l,
          [a]: d
        };
      }), c;
    },
    update: (i, r) => {
      i && s((o) => {
        const a = { ...o }, { position: c, index: l } = Mn(a, i);
        return c && l !== -1 && (a[c][l] = {
          ...a[c][l],
          ...r,
          message: uo(r)
        }), a;
      });
    },
    closeAll: ({ positions: i } = {}) => {
      s((r) => (i ?? [
        "bottom",
        "bottom-right",
        "bottom-left",
        "top",
        "top-left",
        "top-right"
      ]).reduce(
        (c, l) => (c[l] = r[l].map((u) => ({
          ...u,
          requestClose: !0
        })), c),
        { ...r }
      ));
    },
    close: (i) => {
      s((r) => {
        const o = zs(r, i);
        return o ? {
          ...r,
          [o]: r[o].map((a) => a.id == i ? {
            ...a,
            requestClose: !0
          } : a)
        } : r;
      });
    },
    isActive: (i) => Boolean(Mn(H.getState(), i).position)
  };
}
var js = 0;
function xu(t, e = {}) {
  var n, s;
  js += 1;
  const i = (n = e.id) != null ? n : js, r = (s = e.position) != null ? s : "bottom";
  return {
    id: i,
    message: t,
    position: r,
    duration: e.duration,
    onCloseComplete: e.onCloseComplete,
    onRequestRemove: () => H.removeToast(String(i), r),
    status: e.status,
    requestClose: !1,
    containerStyle: e.containerStyle
  };
}
var bu = (t) => {
  const {
    status: e,
    variant: n = "solid",
    id: s,
    title: i,
    isClosable: r,
    onClose: o,
    description: a,
    icon: c
  } = t, l = s ? {
    root: `toast-${s}`,
    title: `toast-${s}-title`,
    description: `toast-${s}-description`
  } : void 0;
  return /* @__PURE__ */ T.jsxs(
    co,
    {
      addRole: !1,
      status: e,
      variant: n,
      id: l == null ? void 0 : l.root,
      alignItems: "start",
      borderRadius: "md",
      boxShadow: "lg",
      paddingEnd: 8,
      textAlign: "start",
      width: "auto",
      children: [
        /* @__PURE__ */ T.jsx(ro, { children: c }),
        /* @__PURE__ */ T.jsxs(X.div, { flex: "1", maxWidth: "100%", children: [
          i && /* @__PURE__ */ T.jsx(ao, { id: l == null ? void 0 : l.title, children: i }),
          a && /* @__PURE__ */ T.jsx(oo, { id: l == null ? void 0 : l.description, display: "block", children: a })
        ] }),
        r && /* @__PURE__ */ T.jsx(
          lo,
          {
            size: "sm",
            onClick: o,
            position: "absolute",
            insetEnd: 1,
            top: 1
          }
        )
      ]
    }
  );
};
function uo(t = {}) {
  const { render: e, toastComponent: n = bu } = t;
  return (i) => typeof e == "function" ? e({ ...i, ...t }) : /* @__PURE__ */ T.jsx(n, { ...i, ...t });
}
function Au(t, e) {
  const n = (i) => {
    var r;
    return {
      ...e,
      ...i,
      position: cu(
        (r = i == null ? void 0 : i.position) != null ? r : e == null ? void 0 : e.position,
        t
      )
    };
  }, s = (i) => {
    const r = n(i), o = uo(r);
    return H.notify(o, r);
  };
  return s.update = (i, r) => {
    H.update(i, n(r));
  }, s.promise = (i, r) => {
    const o = s({
      ...r.loading,
      status: "loading",
      duration: null
    });
    i.then(
      (a) => s.update(o, {
        status: "success",
        duration: 5e3,
        ...Ee(r.success, a)
      })
    ).catch(
      (a) => s.update(o, {
        status: "error",
        duration: 5e3,
        ...Ee(r.error, a)
      })
    );
  }, s.closeAll = H.closeAll, s.close = H.close, s.isActive = H.isActive, s;
}
var [Vu, wu] = It({
  name: "ToastOptionsContext",
  strict: !1
}), Mu = (t) => {
  const e = xo(
    H.subscribe,
    H.getState,
    H.getState
  ), {
    motionVariants: n,
    component: s = so,
    portalProps: i
  } = t, o = Object.keys(e).map((a) => {
    const c = e[a];
    return /* @__PURE__ */ T.jsx(
      "ul",
      {
        role: "region",
        "aria-live": "polite",
        id: `chakra-toast-manager-${a}`,
        style: Oo(a),
        children: /* @__PURE__ */ T.jsx(ru, { initial: !1, children: c.map((l) => /* @__PURE__ */ T.jsx(
          s,
          {
            motionVariants: n,
            ...l
          },
          l.id
        )) })
      },
      a
    );
  });
  return /* @__PURE__ */ T.jsx(ae, { ...i, children: o });
};
const Du = Vo, Ru = So, Eu = {
  "::-webkit-scrollbar": {
    display: "none"
  },
  scrollbarWidth: "none"
}, Lu = {
  scrollbarColor: "accent.600 transparent",
  scrollbarWidth: "thick",
  "::-webkit-scrollbar": {
    width: "6px",
    // Vertical Scrollbar Width
    height: "6px"
    // Horizontal Scrollbar Height
  },
  "::-webkit-scrollbar-track": {
    background: "transparent"
  },
  "::-webkit-scrollbar-thumb": {
    background: "accent.600",
    borderRadius: "8px",
    borderWidth: "4px",
    borderColor: "accent.600"
  },
  "::-webkit-scrollbar-thumb:hover": {
    background: "accent.500",
    borderColor: "accent.500"
  },
  "::-webkit-scrollbar-button": {
    background: "transaprent"
  }
};
export {
  ru as A,
  lo as C,
  zt as I,
  ae as P,
  Vu as T,
  It as a,
  No as b,
  Au as c,
  Le as d,
  Dn as e,
  Kc as f,
  Ru as g,
  Du as h,
  Ro as i,
  Mu as j,
  Jl as m,
  Eu as n,
  Lu as s,
  wu as u
};
