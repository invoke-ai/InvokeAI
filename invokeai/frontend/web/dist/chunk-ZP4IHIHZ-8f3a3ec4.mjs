import { useEffect as M, memo as F, useState as K, useMemo as _, useSyncExternalStore as U } from "react";
import { d as J, eg as G, p as k, j as a, g as Q, a as h, J as S, f as p, b as g, k as P, c as A, ar as X, o as E, h as Y, A as tt, P as et } from "./exports-21bec2df.mjs";
var ot = (e, o) => e.find((s) => s.id === o);
function I(e, o) {
  const s = L(e, o), t = s ? e[s].findIndex((n) => n.id === o) : -1;
  return {
    position: s,
    index: t
  };
}
function L(e, o) {
  for (const [s, t] of Object.entries(e))
    if (ot(t, o))
      return s;
}
function nt(e) {
  const o = e.includes("right"), s = e.includes("left");
  let t = "center";
  return o && (t = "flex-end"), s && (t = "flex-start"), {
    display: "flex",
    flexDirection: "column",
    alignItems: t
  };
}
function st(e) {
  const s = e === "top" || e === "bottom" ? "0 auto" : void 0, t = e.includes("top") ? "env(safe-area-inset-top, 0px)" : void 0, n = e.includes("bottom") ? "env(safe-area-inset-bottom, 0px)" : void 0, r = e.includes("left") ? void 0 : "env(safe-area-inset-right, 0px)", l = e.includes("right") ? void 0 : "env(safe-area-inset-left, 0px)";
  return {
    position: "fixed",
    zIndex: 5500,
    pointerEvents: "none",
    display: "flex",
    flexDirection: "column",
    margin: s,
    top: t,
    bottom: n,
    right: r,
    left: l
  };
}
function rt(e, o) {
  const s = J(e);
  M(() => {
    if (o == null)
      return;
    let t = null;
    return t = window.setTimeout(() => {
      s();
    }, o), () => {
      t && window.clearTimeout(t);
    };
  }, [o, s]);
}
var it = {
  initial: (e) => {
    const { position: o } = e, s = ["top", "bottom"].includes(o) ? "y" : "x";
    let t = ["top-right", "bottom-right"].includes(o) ? 1 : -1;
    return o === "bottom" && (t = 1), {
      opacity: 0,
      [s]: t * 24
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
}, R = F((e) => {
  const {
    id: o,
    message: s,
    onCloseComplete: t,
    onRequestRemove: n,
    requestClose: r = !1,
    position: l = "bottom",
    duration: i = 5e3,
    containerStyle: u,
    motionVariants: c = it,
    toastSpacing: f = "0.5rem"
  } = e, [x, v] = K(i), d = G();
  k(() => {
    d || t == null || t();
  }, [d]), k(() => {
    v(i);
  }, [i]);
  const b = () => v(null), C = () => v(i), T = () => {
    d && n();
  };
  M(() => {
    d && r && n();
  }, [d, r, n]), rt(T, x);
  const V = _(
    () => ({
      pointerEvents: "auto",
      maxWidth: 560,
      minWidth: 300,
      margin: f,
      ...u
    }),
    [u, f]
  ), $ = _(() => nt(l), [l]);
  return /* @__PURE__ */ a.jsx(
    Q.li,
    {
      layout: !0,
      className: "chakra-toast",
      variants: c,
      initial: "initial",
      animate: "animate",
      exit: "exit",
      onHoverStart: b,
      onHoverEnd: C,
      custom: { position: l },
      style: $,
      children: /* @__PURE__ */ a.jsx(
        h.div,
        {
          role: "status",
          "aria-atomic": "true",
          className: "chakra-toast__inner",
          __css: V,
          children: S(s, { id: o, onClose: T })
        }
      )
    }
  );
});
R.displayName = "ToastComponent";
function at(e, o) {
  var s;
  const t = e ?? "bottom", r = {
    "top-start": { ltr: "top-left", rtl: "top-right" },
    "top-end": { ltr: "top-right", rtl: "top-left" },
    "bottom-start": { ltr: "bottom-left", rtl: "bottom-right" },
    "bottom-end": { ltr: "bottom-right", rtl: "bottom-left" }
  }[t];
  return (s = r == null ? void 0 : r[o]) != null ? s : t;
}
var N = {
  path: /* @__PURE__ */ a.jsxs("g", { stroke: "currentColor", strokeWidth: "1.5", children: [
    /* @__PURE__ */ a.jsx(
      "path",
      {
        strokeLinecap: "round",
        fill: "none",
        d: "M9,9a3,3,0,1,1,4,2.829,1.5,1.5,0,0,0-1,1.415V14.25"
      }
    ),
    /* @__PURE__ */ a.jsx(
      "path",
      {
        fill: "currentColor",
        strokeLinecap: "round",
        d: "M12,17.25a.375.375,0,1,0,.375.375A.375.375,0,0,0,12,17.25h0"
      }
    ),
    /* @__PURE__ */ a.jsx("circle", { fill: "none", strokeMiterlimit: "10", cx: "12", cy: "12", r: "11.25" })
  ] }),
  viewBox: "0 0 24 24"
}, y = p((e, o) => {
  const {
    as: s,
    viewBox: t,
    color: n = "currentColor",
    focusable: r = !1,
    children: l,
    className: i,
    __css: u,
    ...c
  } = e, f = g("chakra-icon", i), x = P("Icon", e), v = {
    w: "1em",
    h: "1em",
    display: "inline-block",
    lineHeight: "1em",
    flexShrink: 0,
    color: n,
    ...u,
    ...x
  }, d = {
    ref: o,
    focusable: r,
    className: f,
    __css: v
  }, b = t ?? N.viewBox;
  if (s && typeof s != "string")
    return /* @__PURE__ */ a.jsx(h.svg, { as: s, ...d, ...c });
  const C = l ?? N.path;
  return /* @__PURE__ */ a.jsx(h.svg, { verticalAlign: "middle", viewBox: b, ...d, ...c, children: C });
});
y.displayName = "Icon";
function lt(e) {
  return /* @__PURE__ */ a.jsx(y, { viewBox: "0 0 24 24", ...e, children: /* @__PURE__ */ a.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M12,0A12,12,0,1,0,24,12,12.014,12.014,0,0,0,12,0Zm6.927,8.2-6.845,9.289a1.011,1.011,0,0,1-1.43.188L5.764,13.769a1,1,0,1,1,1.25-1.562l4.076,3.261,6.227-8.451A1,1,0,1,1,18.927,8.2Z"
    }
  ) });
}
function ct(e) {
  return /* @__PURE__ */ a.jsx(y, { viewBox: "0 0 24 24", ...e, children: /* @__PURE__ */ a.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M12,0A12,12,0,1,0,24,12,12.013,12.013,0,0,0,12,0Zm.25,5a1.5,1.5,0,1,1-1.5,1.5A1.5,1.5,0,0,1,12.25,5ZM14.5,18.5h-4a1,1,0,0,1,0-2h.75a.25.25,0,0,0,.25-.25v-4.5a.25.25,0,0,0-.25-.25H10.5a1,1,0,0,1,0-2h1a2,2,0,0,1,2,2v4.75a.25.25,0,0,0,.25.25h.75a1,1,0,1,1,0,2Z"
    }
  ) });
}
function w(e) {
  return /* @__PURE__ */ a.jsx(y, { viewBox: "0 0 24 24", ...e, children: /* @__PURE__ */ a.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M11.983,0a12.206,12.206,0,0,0-8.51,3.653A11.8,11.8,0,0,0,0,12.207,11.779,11.779,0,0,0,11.8,24h.214A12.111,12.111,0,0,0,24,11.791h0A11.766,11.766,0,0,0,11.983,0ZM10.5,16.542a1.476,1.476,0,0,1,1.449-1.53h.027a1.527,1.527,0,0,1,1.523,1.47,1.475,1.475,0,0,1-1.449,1.53h-.027A1.529,1.529,0,0,1,10.5,16.542ZM11,12.5v-6a1,1,0,0,1,2,0v6a1,1,0,1,1-2,0Z"
    }
  ) });
}
var [ut, dt] = A({
  name: "AlertContext",
  hookName: "useAlertContext",
  providerName: "<Alert />"
}), [mt, j] = A({
  name: "AlertStylesContext",
  hookName: "useAlertStyles",
  providerName: "<Alert />"
}), Z = {
  info: { icon: ct, colorScheme: "blue" },
  warning: { icon: w, colorScheme: "orange" },
  success: { icon: lt, colorScheme: "green" },
  error: { icon: w, colorScheme: "red" },
  loading: { icon: X, colorScheme: "blue" }
};
function ft(e) {
  return Z[e].colorScheme;
}
function ht(e) {
  return Z[e].icon;
}
var D = p(
  function(o, s) {
    const n = {
      display: "inline",
      ...j().description
    };
    return /* @__PURE__ */ a.jsx(
      h.div,
      {
        ref: s,
        ...o,
        className: g("chakra-alert__desc", o.className),
        __css: n
      }
    );
  }
);
D.displayName = "AlertDescription";
function O(e) {
  const { status: o } = dt(), s = ht(o), t = j(), n = o === "loading" ? t.spinner : t.icon;
  return /* @__PURE__ */ a.jsx(
    h.span,
    {
      display: "inherit",
      ...e,
      className: g("chakra-alert__icon", e.className),
      __css: n,
      children: e.children || /* @__PURE__ */ a.jsx(s, { h: "100%", w: "100%" })
    }
  );
}
O.displayName = "AlertIcon";
var q = p(
  function(o, s) {
    const t = j();
    return /* @__PURE__ */ a.jsx(
      h.div,
      {
        ref: s,
        ...o,
        className: g("chakra-alert__title", o.className),
        __css: t.title
      }
    );
  }
);
q.displayName = "AlertTitle";
var W = p(function(o, s) {
  var t;
  const { status: n = "info", addRole: r = !0, ...l } = E(o), i = (t = o.colorScheme) != null ? t : ft(n), u = Y("Alert", { ...o, colorScheme: i }), c = {
    width: "100%",
    display: "flex",
    alignItems: "center",
    position: "relative",
    overflow: "hidden",
    ...u.container
  };
  return /* @__PURE__ */ a.jsx(ut, { value: { status: n }, children: /* @__PURE__ */ a.jsx(mt, { value: u, children: /* @__PURE__ */ a.jsx(
    h.div,
    {
      role: r ? "alert" : void 0,
      ref: s,
      ...l,
      className: g("chakra-alert", o.className),
      __css: c
    }
  ) }) });
});
W.displayName = "Alert";
function xt(e) {
  return /* @__PURE__ */ a.jsx(y, { focusable: "false", "aria-hidden": !0, ...e, children: /* @__PURE__ */ a.jsx(
    "path",
    {
      fill: "currentColor",
      d: "M.439,21.44a1.5,1.5,0,0,0,2.122,2.121L11.823,14.3a.25.25,0,0,1,.354,0l9.262,9.263a1.5,1.5,0,1,0,2.122-2.121L14.3,12.177a.25.25,0,0,1,0-.354l9.263-9.262A1.5,1.5,0,0,0,21.439.44L12.177,9.7a.25.25,0,0,1-.354,0L2.561.44A1.5,1.5,0,0,0,.439,2.561L9.7,11.823a.25.25,0,0,1,0,.354Z"
    }
  ) });
}
var z = p(
  function(o, s) {
    const t = P("CloseButton", o), { children: n, isDisabled: r, __css: l, ...i } = E(o), u = {
      outline: 0,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0
    };
    return /* @__PURE__ */ a.jsx(
      h.button,
      {
        type: "button",
        "aria-label": "Close",
        ref: s,
        disabled: r,
        __css: {
          ...u,
          ...t,
          ...l
        },
        ...i,
        children: n || /* @__PURE__ */ a.jsx(xt, { width: "1em", height: "1em" })
      }
    );
  }
);
z.displayName = "CloseButton";
var vt = {
  top: [],
  "top-left": [],
  "top-right": [],
  "bottom-left": [],
  bottom: [],
  "bottom-right": []
}, m = pt(vt);
function pt(e) {
  let o = e;
  const s = /* @__PURE__ */ new Set(), t = (n) => {
    o = n(o), s.forEach((r) => r());
  };
  return {
    getState: () => o,
    subscribe: (n) => (s.add(n), () => {
      t(() => e), s.delete(n);
    }),
    removeToast: (n, r) => {
      t((l) => ({
        ...l,
        [r]: l[r].filter((i) => i.id != n)
      }));
    },
    notify: (n, r) => {
      const l = gt(n, r), { position: i, id: u } = l;
      return t((c) => {
        var f, x;
        const d = i.includes("top") ? [l, ...(f = c[i]) != null ? f : []] : [...(x = c[i]) != null ? x : [], l];
        return {
          ...c,
          [i]: d
        };
      }), u;
    },
    update: (n, r) => {
      n && t((l) => {
        const i = { ...l }, { position: u, index: c } = I(i, n);
        return u && c !== -1 && (i[u][c] = {
          ...i[u][c],
          ...r,
          message: H(r)
        }), i;
      });
    },
    closeAll: ({ positions: n } = {}) => {
      t((r) => (n ?? [
        "bottom",
        "bottom-right",
        "bottom-left",
        "top",
        "top-left",
        "top-right"
      ]).reduce(
        (u, c) => (u[c] = r[c].map((f) => ({
          ...f,
          requestClose: !0
        })), u),
        { ...r }
      ));
    },
    close: (n) => {
      t((r) => {
        const l = L(r, n);
        return l ? {
          ...r,
          [l]: r[l].map((i) => i.id == n ? {
            ...i,
            requestClose: !0
          } : i)
        } : r;
      });
    },
    isActive: (n) => Boolean(I(m.getState(), n).position)
  };
}
var B = 0;
function gt(e, o = {}) {
  var s, t;
  B += 1;
  const n = (s = o.id) != null ? s : B, r = (t = o.position) != null ? t : "bottom";
  return {
    id: n,
    message: e,
    position: r,
    duration: o.duration,
    onCloseComplete: o.onCloseComplete,
    onRequestRemove: () => m.removeToast(String(n), r),
    status: o.status,
    requestClose: !1,
    containerStyle: o.containerStyle
  };
}
var yt = (e) => {
  const {
    status: o,
    variant: s = "solid",
    id: t,
    title: n,
    isClosable: r,
    onClose: l,
    description: i,
    icon: u
  } = e, c = t ? {
    root: `toast-${t}`,
    title: `toast-${t}-title`,
    description: `toast-${t}-description`
  } : void 0;
  return /* @__PURE__ */ a.jsxs(
    W,
    {
      addRole: !1,
      status: o,
      variant: s,
      id: c == null ? void 0 : c.root,
      alignItems: "start",
      borderRadius: "md",
      boxShadow: "lg",
      paddingEnd: 8,
      textAlign: "start",
      width: "auto",
      children: [
        /* @__PURE__ */ a.jsx(O, { children: u }),
        /* @__PURE__ */ a.jsxs(h.div, { flex: "1", maxWidth: "100%", children: [
          n && /* @__PURE__ */ a.jsx(q, { id: c == null ? void 0 : c.title, children: n }),
          i && /* @__PURE__ */ a.jsx(D, { id: c == null ? void 0 : c.description, display: "block", children: i })
        ] }),
        r && /* @__PURE__ */ a.jsx(
          z,
          {
            size: "sm",
            onClick: l,
            position: "absolute",
            insetEnd: 1,
            top: 1
          }
        )
      ]
    }
  );
};
function H(e = {}) {
  const { render: o, toastComponent: s = yt } = e;
  return (n) => typeof o == "function" ? o({ ...n, ...e }) : /* @__PURE__ */ a.jsx(s, { ...n, ...e });
}
function St(e, o) {
  const s = (n) => {
    var r;
    return {
      ...o,
      ...n,
      position: at(
        (r = n == null ? void 0 : n.position) != null ? r : o == null ? void 0 : o.position,
        e
      )
    };
  }, t = (n) => {
    const r = s(n), l = H(r);
    return m.notify(l, r);
  };
  return t.update = (n, r) => {
    m.update(n, s(r));
  }, t.promise = (n, r) => {
    const l = t({
      ...r.loading,
      status: "loading",
      duration: null
    });
    n.then(
      (i) => t.update(l, {
        status: "success",
        duration: 5e3,
        ...S(r.success, i)
      })
    ).catch(
      (i) => t.update(l, {
        status: "error",
        duration: 5e3,
        ...S(r.error, i)
      })
    );
  }, t.closeAll = m.closeAll, t.close = m.close, t.isActive = m.isActive, t;
}
var [At, jt] = A({
  name: "ToastOptionsContext",
  strict: !1
}), Tt = (e) => {
  const o = U(
    m.subscribe,
    m.getState,
    m.getState
  ), {
    motionVariants: s,
    component: t = R,
    portalProps: n
  } = e, l = Object.keys(o).map((i) => {
    const u = o[i];
    return /* @__PURE__ */ a.jsx(
      "ul",
      {
        role: "region",
        "aria-live": "polite",
        id: `chakra-toast-manager-${i}`,
        style: st(i),
        children: /* @__PURE__ */ a.jsx(tt, { initial: !1, children: u.map((c) => /* @__PURE__ */ a.jsx(
          t,
          {
            motionVariants: s,
            ...c
          },
          c.id
        )) })
      },
      i
    );
  });
  return /* @__PURE__ */ a.jsx(et, { ...n, children: l });
};
export {
  z as C,
  y as I,
  At as T,
  Tt as a,
  St as c,
  jt as u
};
