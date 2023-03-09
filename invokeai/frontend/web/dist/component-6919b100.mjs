import * as ep from "react";
import ln, { createContext as vo, useContext as cn, useDebugValue as VC, useLayoutEffect as Mv, useEffect as fl, useMemo as tp, PureComponent as qC, useState as YC, useRef as so, forwardRef as Nv, createElement as Qn, Fragment as vh, lazy as Bv } from "react";
import { unstable_batchedUpdates as KC } from "react-dom";
function XC(t, e) {
  for (var r = 0; r < e.length; r++) {
    const n = e[r];
    if (typeof n != "string" && !Array.isArray(n)) {
      for (const i in n)
        if (i !== "default" && !(i in t)) {
          const s = Object.getOwnPropertyDescriptor(n, i);
          s && Object.defineProperty(t, i, s.get ? s : {
            enumerable: !0,
            get: () => n[i]
          });
        }
    }
  }
  return Object.freeze(Object.defineProperty(t, Symbol.toStringTag, { value: "Module" }));
}
var dr = typeof globalThis < "u" ? globalThis : typeof window < "u" ? window : typeof global < "u" ? global : typeof self < "u" ? self : {};
function DD(t) {
  return t && t.__esModule && Object.prototype.hasOwnProperty.call(t, "default") ? t.default : t;
}
var Qt = {}, JC = {
  get exports() {
    return Qt;
  },
  set exports(t) {
    Qt = t;
  }
}, Is = {};
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var rp;
function QC() {
  if (rp)
    return Is;
  rp = 1;
  var t = ln, e = Symbol.for("react.element"), r = Symbol.for("react.fragment"), n = Object.prototype.hasOwnProperty, i = t.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner, s = { key: !0, ref: !0, __self: !0, __source: !0 };
  function o(l, c, h) {
    var d, g = {}, p = null, y = null;
    h !== void 0 && (p = "" + h), c.key !== void 0 && (p = "" + c.key), c.ref !== void 0 && (y = c.ref);
    for (d in c)
      n.call(c, d) && !s.hasOwnProperty(d) && (g[d] = c[d]);
    if (l && l.defaultProps)
      for (d in c = l.defaultProps, c)
        g[d] === void 0 && (g[d] = c[d]);
    return { $$typeof: e, type: l, key: p, ref: y, props: g, _owner: i.current };
  }
  return Is.Fragment = r, Is.jsx = o, Is.jsxs = o, Is;
}
var Ls = {};
/**
 * @license React
 * react-jsx-runtime.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var np;
function ZC() {
  return np || (np = 1, process.env.NODE_ENV !== "production" && function() {
    var t = ln, e = Symbol.for("react.element"), r = Symbol.for("react.portal"), n = Symbol.for("react.fragment"), i = Symbol.for("react.strict_mode"), s = Symbol.for("react.profiler"), o = Symbol.for("react.provider"), l = Symbol.for("react.context"), c = Symbol.for("react.forward_ref"), h = Symbol.for("react.suspense"), d = Symbol.for("react.suspense_list"), g = Symbol.for("react.memo"), p = Symbol.for("react.lazy"), y = Symbol.for("react.offscreen"), v = Symbol.iterator, _ = "@@iterator";
    function b(L) {
      if (L === null || typeof L != "object")
        return null;
      var ie = v && L[v] || L[_];
      return typeof ie == "function" ? ie : null;
    }
    var C = t.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;
    function w(L) {
      {
        for (var ie = arguments.length, he = new Array(ie > 1 ? ie - 1 : 0), Se = 1; Se < ie; Se++)
          he[Se - 1] = arguments[Se];
        O("error", L, he);
      }
    }
    function O(L, ie, he) {
      {
        var Se = C.ReactDebugCurrentFrame, je = Se.getStackAddendum();
        je !== "" && (ie += "%s", he = he.concat([je]));
        var Ne = he.map(function(Me) {
          return String(Me);
        });
        Ne.unshift("Warning: " + ie), Function.prototype.apply.call(console[L], console, Ne);
      }
    }
    var k = !1, P = !1, M = !1, U = !1, V = !1, Y;
    Y = Symbol.for("react.module.reference");
    function X(L) {
      return !!(typeof L == "string" || typeof L == "function" || L === n || L === s || V || L === i || L === h || L === d || U || L === y || k || P || M || typeof L == "object" && L !== null && (L.$$typeof === p || L.$$typeof === g || L.$$typeof === o || L.$$typeof === l || L.$$typeof === c || // This needs to include all possible module reference object
      // types supported by any Flight configuration anywhere since
      // we don't know which Flight build this will end up being used
      // with.
      L.$$typeof === Y || L.getModuleId !== void 0));
    }
    function Z(L, ie, he) {
      var Se = L.displayName;
      if (Se)
        return Se;
      var je = ie.displayName || ie.name || "";
      return je !== "" ? he + "(" + je + ")" : he;
    }
    function ce(L) {
      return L.displayName || "Context";
    }
    function $(L) {
      if (L == null)
        return null;
      if (typeof L.tag == "number" && w("Received an unexpected object in getComponentNameFromType(). This is likely a bug in React. Please file an issue."), typeof L == "function")
        return L.displayName || L.name || null;
      if (typeof L == "string")
        return L;
      switch (L) {
        case n:
          return "Fragment";
        case r:
          return "Portal";
        case s:
          return "Profiler";
        case i:
          return "StrictMode";
        case h:
          return "Suspense";
        case d:
          return "SuspenseList";
      }
      if (typeof L == "object")
        switch (L.$$typeof) {
          case l:
            var ie = L;
            return ce(ie) + ".Consumer";
          case o:
            var he = L;
            return ce(he._context) + ".Provider";
          case c:
            return Z(L, L.render, "ForwardRef");
          case g:
            var Se = L.displayName || null;
            return Se !== null ? Se : $(L.type) || "Memo";
          case p: {
            var je = L, Ne = je._payload, Me = je._init;
            try {
              return $(Me(Ne));
            } catch {
              return null;
            }
          }
        }
      return null;
    }
    var R = Object.assign, F = 0, H, re, ae, ne, oe, de, Re;
    function Ce() {
    }
    Ce.__reactDisabledLog = !0;
    function rt() {
      {
        if (F === 0) {
          H = console.log, re = console.info, ae = console.warn, ne = console.error, oe = console.group, de = console.groupCollapsed, Re = console.groupEnd;
          var L = {
            configurable: !0,
            enumerable: !0,
            value: Ce,
            writable: !0
          };
          Object.defineProperties(console, {
            info: L,
            log: L,
            warn: L,
            error: L,
            group: L,
            groupCollapsed: L,
            groupEnd: L
          });
        }
        F++;
      }
    }
    function xt() {
      {
        if (F--, F === 0) {
          var L = {
            configurable: !0,
            enumerable: !0,
            writable: !0
          };
          Object.defineProperties(console, {
            log: R({}, L, {
              value: H
            }),
            info: R({}, L, {
              value: re
            }),
            warn: R({}, L, {
              value: ae
            }),
            error: R({}, L, {
              value: ne
            }),
            group: R({}, L, {
              value: oe
            }),
            groupCollapsed: R({}, L, {
              value: de
            }),
            groupEnd: R({}, L, {
              value: Re
            })
          });
        }
        F < 0 && w("disabledDepth fell below zero. This is a bug in React. Please file an issue.");
      }
    }
    var be = C.ReactCurrentDispatcher, At;
    function ot(L, ie, he) {
      {
        if (At === void 0)
          try {
            throw Error();
          } catch (je) {
            var Se = je.stack.trim().match(/\n( *(at )?)/);
            At = Se && Se[1] || "";
          }
        return `
` + At + L;
      }
    }
    var wt = !1, kt;
    {
      var dt = typeof WeakMap == "function" ? WeakMap : Map;
      kt = new dt();
    }
    function te(L, ie) {
      if (!L || wt)
        return "";
      {
        var he = kt.get(L);
        if (he !== void 0)
          return he;
      }
      var Se;
      wt = !0;
      var je = Error.prepareStackTrace;
      Error.prepareStackTrace = void 0;
      var Ne;
      Ne = be.current, be.current = null, rt();
      try {
        if (ie) {
          var Me = function() {
            throw Error();
          };
          if (Object.defineProperty(Me.prototype, "props", {
            set: function() {
              throw Error();
            }
          }), typeof Reflect == "object" && Reflect.construct) {
            try {
              Reflect.construct(Me, []);
            } catch (Bt) {
              Se = Bt;
            }
            Reflect.construct(L, [], Me);
          } else {
            try {
              Me.call();
            } catch (Bt) {
              Se = Bt;
            }
            L.call(Me.prototype);
          }
        } else {
          try {
            throw Error();
          } catch (Bt) {
            Se = Bt;
          }
          L();
        }
      } catch (Bt) {
        if (Bt && Se && typeof Bt.stack == "string") {
          for (var Ae = Bt.stack.split(`
`), Nt = Se.stack.split(`
`), ft = Ae.length - 1, gt = Nt.length - 1; ft >= 1 && gt >= 0 && Ae[ft] !== Nt[gt]; )
            gt--;
          for (; ft >= 1 && gt >= 0; ft--, gt--)
            if (Ae[ft] !== Nt[gt]) {
              if (ft !== 1 || gt !== 1)
                do
                  if (ft--, gt--, gt < 0 || Ae[ft] !== Nt[gt]) {
                    var tr = `
` + Ae[ft].replace(" at new ", " at ");
                    return L.displayName && tr.includes("<anonymous>") && (tr = tr.replace("<anonymous>", L.displayName)), typeof L == "function" && kt.set(L, tr), tr;
                  }
                while (ft >= 1 && gt >= 0);
              break;
            }
        }
      } finally {
        wt = !1, be.current = Ne, xt(), Error.prepareStackTrace = je;
      }
      var or = L ? L.displayName || L.name : "", ca = or ? ot(or) : "";
      return typeof L == "function" && kt.set(L, ca), ca;
    }
    function $t(L, ie, he) {
      return te(L, !1);
    }
    function fe(L) {
      var ie = L.prototype;
      return !!(ie && ie.isReactComponent);
    }
    function Ot(L, ie, he) {
      if (L == null)
        return "";
      if (typeof L == "function")
        return te(L, fe(L));
      if (typeof L == "string")
        return ot(L);
      switch (L) {
        case h:
          return ot("Suspense");
        case d:
          return ot("SuspenseList");
      }
      if (typeof L == "object")
        switch (L.$$typeof) {
          case c:
            return $t(L.render);
          case g:
            return Ot(L.type, ie, he);
          case p: {
            var Se = L, je = Se._payload, Ne = Se._init;
            try {
              return Ot(Ne(je), ie, he);
            } catch {
            }
          }
        }
      return "";
    }
    var sr = Object.prototype.hasOwnProperty, Dt = {}, dn = C.ReactDebugCurrentFrame;
    function gn(L) {
      if (L) {
        var ie = L._owner, he = Ot(L.type, L._source, ie ? ie.type : null);
        dn.setExtraStackFrame(he);
      } else
        dn.setExtraStackFrame(null);
    }
    function gr(L, ie, he, Se, je) {
      {
        var Ne = Function.call.bind(sr);
        for (var Me in L)
          if (Ne(L, Me)) {
            var Ae = void 0;
            try {
              if (typeof L[Me] != "function") {
                var Nt = Error((Se || "React class") + ": " + he + " type `" + Me + "` is invalid; it must be a function, usually from the `prop-types` package, but received `" + typeof L[Me] + "`.This often happens because of typos such as `PropTypes.function` instead of `PropTypes.func`.");
                throw Nt.name = "Invariant Violation", Nt;
              }
              Ae = L[Me](ie, Me, Se, he, null, "SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED");
            } catch (ft) {
              Ae = ft;
            }
            Ae && !(Ae instanceof Error) && (gn(je), w("%s: type specification of %s `%s` is invalid; the type checker function must return `null` or an `Error` but returned a %s. You may have forgotten to pass an argument to the type checker creator (arrayOf, instanceOf, objectOf, oneOf, oneOfType, and shape all require an argument).", Se || "React class", he, Me, typeof Ae), gn(null)), Ae instanceof Error && !(Ae.message in Dt) && (Dt[Ae.message] = !0, gn(je), w("Failed %s type: %s", he, Ae.message), gn(null));
          }
      }
    }
    var Mt = Array.isArray;
    function qt(L) {
      return Mt(L);
    }
    function yt(L) {
      {
        var ie = typeof Symbol == "function" && Symbol.toStringTag, he = ie && L[Symbol.toStringTag] || L.constructor.name || "Object";
        return he;
      }
    }
    function Dn(L) {
      try {
        return Sr(L), !1;
      } catch {
        return !0;
      }
    }
    function Sr(L) {
      return "" + L;
    }
    function Mn(L) {
      if (Dn(L))
        return w("The provided key is an unsupported type %s. This value must be coerced to a string before before using it here.", yt(L)), Sr(L);
    }
    var er = C.ReactCurrentOwner, wr = {
      key: !0,
      ref: !0,
      __self: !0,
      __source: !0
    }, Nn, Bn, pn;
    pn = {};
    function ii(L) {
      if (sr.call(L, "ref")) {
        var ie = Object.getOwnPropertyDescriptor(L, "ref").get;
        if (ie && ie.isReactWarning)
          return !1;
      }
      return L.ref !== void 0;
    }
    function ai(L) {
      if (sr.call(L, "key")) {
        var ie = Object.getOwnPropertyDescriptor(L, "key").get;
        if (ie && ie.isReactWarning)
          return !1;
      }
      return L.key !== void 0;
    }
    function Ai(L, ie) {
      if (typeof L.ref == "string" && er.current && ie && er.current.stateNode !== ie) {
        var he = $(er.current.type);
        pn[he] || (w('Component "%s" contains the string ref "%s". Support for string refs will be removed in a future major release. This case cannot be automatically converted to an arrow function. We ask you to manually fix this case by using useRef() or createRef() instead. Learn more about using refs safely here: https://reactjs.org/link/strict-mode-string-ref', $(er.current.type), L.ref), pn[he] = !0);
      }
    }
    function ki(L, ie) {
      {
        var he = function() {
          Nn || (Nn = !0, w("%s: `key` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://reactjs.org/link/special-props)", ie));
        };
        he.isReactWarning = !0, Object.defineProperty(L, "key", {
          get: he,
          configurable: !0
        });
      }
    }
    function Er(L, ie) {
      {
        var he = function() {
          Bn || (Bn = !0, w("%s: `ref` is not a prop. Trying to access it will result in `undefined` being returned. If you need to access the same value within the child component, you should pass it as a different prop. (https://reactjs.org/link/special-props)", ie));
        };
        he.isReactWarning = !0, Object.defineProperty(L, "ref", {
          get: he,
          configurable: !0
        });
      }
    }
    var mn = function(L, ie, he, Se, je, Ne, Me) {
      var Ae = {
        // This tag allows us to uniquely identify this as a React Element
        $$typeof: e,
        // Built-in properties that belong on the element
        type: L,
        key: ie,
        ref: he,
        props: Me,
        // Record the component responsible for creating this element.
        _owner: Ne
      };
      return Ae._store = {}, Object.defineProperty(Ae._store, "validated", {
        configurable: !1,
        enumerable: !1,
        writable: !0,
        value: !1
      }), Object.defineProperty(Ae, "_self", {
        configurable: !1,
        enumerable: !1,
        writable: !1,
        value: Se
      }), Object.defineProperty(Ae, "_source", {
        configurable: !1,
        enumerable: !1,
        writable: !1,
        value: je
      }), Object.freeze && (Object.freeze(Ae.props), Object.freeze(Ae)), Ae;
    };
    function ns(L, ie, he, Se, je) {
      {
        var Ne, Me = {}, Ae = null, Nt = null;
        he !== void 0 && (Mn(he), Ae = "" + he), ai(ie) && (Mn(ie.key), Ae = "" + ie.key), ii(ie) && (Nt = ie.ref, Ai(ie, je));
        for (Ne in ie)
          sr.call(ie, Ne) && !wr.hasOwnProperty(Ne) && (Me[Ne] = ie[Ne]);
        if (L && L.defaultProps) {
          var ft = L.defaultProps;
          for (Ne in ft)
            Me[Ne] === void 0 && (Me[Ne] = ft[Ne]);
        }
        if (Ae || Nt) {
          var gt = typeof L == "function" ? L.displayName || L.name || "Unknown" : L;
          Ae && ki(Me, gt), Nt && Er(Me, gt);
        }
        return mn(L, Ae, Nt, je, Se, er.current, Me);
      }
    }
    var si = C.ReactCurrentOwner, ia = C.ReactDebugCurrentFrame;
    function Nr(L) {
      if (L) {
        var ie = L._owner, he = Ot(L.type, L._source, ie ? ie.type : null);
        ia.setExtraStackFrame(he);
      } else
        ia.setExtraStackFrame(null);
    }
    var Br;
    Br = !1;
    function Ii(L) {
      return typeof L == "object" && L !== null && L.$$typeof === e;
    }
    function Fr() {
      {
        if (si.current) {
          var L = $(si.current.type);
          if (L)
            return `

Check the render method of \`` + L + "`.";
        }
        return "";
      }
    }
    function is(L) {
      {
        if (L !== void 0) {
          var ie = L.fileName.replace(/^.*[\\\/]/, ""), he = L.lineNumber;
          return `

Check your code at ` + ie + ":" + he + ".";
        }
        return "";
      }
    }
    var aa = {};
    function sa(L) {
      {
        var ie = Fr();
        if (!ie) {
          var he = typeof L == "string" ? L : L.displayName || L.name;
          he && (ie = `

Check the top-level render call using <` + he + ">.");
        }
        return ie;
      }
    }
    function oa(L, ie) {
      {
        if (!L._store || L._store.validated || L.key != null)
          return;
        L._store.validated = !0;
        var he = sa(ie);
        if (aa[he])
          return;
        aa[he] = !0;
        var Se = "";
        L && L._owner && L._owner !== si.current && (Se = " It was passed a child from " + $(L._owner.type) + "."), Nr(L), w('Each child in a list should have a unique "key" prop.%s%s See https://reactjs.org/link/warning-keys for more information.', he, Se), Nr(null);
      }
    }
    function ua(L, ie) {
      {
        if (typeof L != "object")
          return;
        if (qt(L))
          for (var he = 0; he < L.length; he++) {
            var Se = L[he];
            Ii(Se) && oa(Se, ie);
          }
        else if (Ii(L))
          L._store && (L._store.validated = !0);
        else if (L) {
          var je = b(L);
          if (typeof je == "function" && je !== L.entries)
            for (var Ne = je.call(L), Me; !(Me = Ne.next()).done; )
              Ii(Me.value) && oa(Me.value, ie);
        }
      }
    }
    function Cr(L) {
      {
        var ie = L.type;
        if (ie == null || typeof ie == "string")
          return;
        var he;
        if (typeof ie == "function")
          he = ie.propTypes;
        else if (typeof ie == "object" && (ie.$$typeof === c || // Note: Memo only checks outer props here.
        // Inner props are checked in the reconciler.
        ie.$$typeof === g))
          he = ie.propTypes;
        else
          return;
        if (he) {
          var Se = $(ie);
          gr(he, L.props, "prop", Se, L);
        } else if (ie.PropTypes !== void 0 && !Br) {
          Br = !0;
          var je = $(ie);
          w("Component %s declared `PropTypes` instead of `propTypes`. Did you misspell the property assignment?", je || "Unknown");
        }
        typeof ie.getDefaultProps == "function" && !ie.getDefaultProps.isReactClassApproved && w("getDefaultProps is only used on classic React.createClass definitions. Use a static property named `defaultProps` instead.");
      }
    }
    function Li(L) {
      {
        for (var ie = Object.keys(L.props), he = 0; he < ie.length; he++) {
          var Se = ie[he];
          if (Se !== "children" && Se !== "key") {
            Nr(L), w("Invalid prop `%s` supplied to `React.Fragment`. React.Fragment can only have `key` and `children` props.", Se), Nr(null);
            break;
          }
        }
        L.ref !== null && (Nr(L), w("Invalid attribute `ref` supplied to `React.Fragment`."), Nr(null));
      }
    }
    function la(L, ie, he, Se, je, Ne) {
      {
        var Me = X(L);
        if (!Me) {
          var Ae = "";
          (L === void 0 || typeof L == "object" && L !== null && Object.keys(L).length === 0) && (Ae += " You likely forgot to export your component from the file it's defined in, or you might have mixed up default and named imports.");
          var Nt = is(je);
          Nt ? Ae += Nt : Ae += Fr();
          var ft;
          L === null ? ft = "null" : qt(L) ? ft = "array" : L !== void 0 && L.$$typeof === e ? (ft = "<" + ($(L.type) || "Unknown") + " />", Ae = " Did you accidentally export a JSX literal instead of a component?") : ft = typeof L, w("React.jsx: type is invalid -- expected a string (for built-in components) or a class/function (for composite components) but got: %s.%s", ft, Ae);
        }
        var gt = ns(L, ie, he, je, Ne);
        if (gt == null)
          return gt;
        if (Me) {
          var tr = ie.children;
          if (tr !== void 0)
            if (Se)
              if (qt(tr)) {
                for (var or = 0; or < tr.length; or++)
                  ua(tr[or], L);
                Object.freeze && Object.freeze(tr);
              } else
                w("React.jsx: Static children should always be an array. You are likely explicitly calling React.jsxs or React.jsxDEV. Use the Babel transform instead.");
            else
              ua(tr, L);
        }
        return L === n ? Li(gt) : Cr(gt), gt;
      }
    }
    function Di(L, ie, he) {
      return la(L, ie, he, !0);
    }
    function as(L, ie, he) {
      return la(L, ie, he, !1);
    }
    var ss = as, vn = Di;
    Ls.Fragment = n, Ls.jsx = ss, Ls.jsxs = vn;
  }()), Ls;
}
(function(t) {
  process.env.NODE_ENV === "production" ? t.exports = QC() : t.exports = ZC();
})(JC);
var hl = {}, ex = {
  get exports() {
    return hl;
  },
  set exports(t) {
    hl = t;
  }
}, nf = {};
/**
 * @license React
 * use-sync-external-store-shim.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var ip;
function tx() {
  if (ip)
    return nf;
  ip = 1;
  var t = ln;
  function e(g, p) {
    return g === p && (g !== 0 || 1 / g === 1 / p) || g !== g && p !== p;
  }
  var r = typeof Object.is == "function" ? Object.is : e, n = t.useState, i = t.useEffect, s = t.useLayoutEffect, o = t.useDebugValue;
  function l(g, p) {
    var y = p(), v = n({ inst: { value: y, getSnapshot: p } }), _ = v[0].inst, b = v[1];
    return s(function() {
      _.value = y, _.getSnapshot = p, c(_) && b({ inst: _ });
    }, [g, y, p]), i(function() {
      return c(_) && b({ inst: _ }), g(function() {
        c(_) && b({ inst: _ });
      });
    }, [g]), o(y), y;
  }
  function c(g) {
    var p = g.getSnapshot;
    g = g.value;
    try {
      var y = p();
      return !r(g, y);
    } catch {
      return !0;
    }
  }
  function h(g, p) {
    return p();
  }
  var d = typeof window > "u" || typeof window.document > "u" || typeof window.document.createElement > "u" ? h : l;
  return nf.useSyncExternalStore = t.useSyncExternalStore !== void 0 ? t.useSyncExternalStore : d, nf;
}
var af = {};
/**
 * @license React
 * use-sync-external-store-shim.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var ap;
function rx() {
  return ap || (ap = 1, process.env.NODE_ENV !== "production" && function() {
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(new Error());
    var t = ln, e = t.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;
    function r(O) {
      {
        for (var k = arguments.length, P = new Array(k > 1 ? k - 1 : 0), M = 1; M < k; M++)
          P[M - 1] = arguments[M];
        n("error", O, P);
      }
    }
    function n(O, k, P) {
      {
        var M = e.ReactDebugCurrentFrame, U = M.getStackAddendum();
        U !== "" && (k += "%s", P = P.concat([U]));
        var V = P.map(function(Y) {
          return String(Y);
        });
        V.unshift("Warning: " + k), Function.prototype.apply.call(console[O], console, V);
      }
    }
    function i(O, k) {
      return O === k && (O !== 0 || 1 / O === 1 / k) || O !== O && k !== k;
    }
    var s = typeof Object.is == "function" ? Object.is : i, o = t.useState, l = t.useEffect, c = t.useLayoutEffect, h = t.useDebugValue, d = !1, g = !1;
    function p(O, k, P) {
      d || t.startTransition !== void 0 && (d = !0, r("You are using an outdated, pre-release alpha of React 18 that does not support useSyncExternalStore. The use-sync-external-store shim will not work correctly. Upgrade to a newer pre-release."));
      var M = k();
      if (!g) {
        var U = k();
        s(M, U) || (r("The result of getSnapshot should be cached to avoid an infinite loop"), g = !0);
      }
      var V = o({
        inst: {
          value: M,
          getSnapshot: k
        }
      }), Y = V[0].inst, X = V[1];
      return c(function() {
        Y.value = M, Y.getSnapshot = k, y(Y) && X({
          inst: Y
        });
      }, [O, M, k]), l(function() {
        y(Y) && X({
          inst: Y
        });
        var Z = function() {
          y(Y) && X({
            inst: Y
          });
        };
        return O(Z);
      }, [O]), h(M), M;
    }
    function y(O) {
      var k = O.getSnapshot, P = O.value;
      try {
        var M = k();
        return !s(P, M);
      } catch {
        return !0;
      }
    }
    function v(O, k, P) {
      return k();
    }
    var _ = typeof window < "u" && typeof window.document < "u" && typeof window.document.createElement < "u", b = !_, C = b ? v : p, w = t.useSyncExternalStore !== void 0 ? t.useSyncExternalStore : C;
    af.useSyncExternalStore = w, typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(new Error());
  }()), af;
}
(function(t) {
  process.env.NODE_ENV === "production" ? t.exports = tx() : t.exports = rx();
})(ex);
var Lf = {}, nx = {
  get exports() {
    return Lf;
  },
  set exports(t) {
    Lf = t;
  }
}, sf = {};
/**
 * @license React
 * use-sync-external-store-shim/with-selector.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var sp;
function ix() {
  if (sp)
    return sf;
  sp = 1;
  var t = ln, e = hl;
  function r(h, d) {
    return h === d && (h !== 0 || 1 / h === 1 / d) || h !== h && d !== d;
  }
  var n = typeof Object.is == "function" ? Object.is : r, i = e.useSyncExternalStore, s = t.useRef, o = t.useEffect, l = t.useMemo, c = t.useDebugValue;
  return sf.useSyncExternalStoreWithSelector = function(h, d, g, p, y) {
    var v = s(null);
    if (v.current === null) {
      var _ = { hasValue: !1, value: null };
      v.current = _;
    } else
      _ = v.current;
    v = l(function() {
      function C(M) {
        if (!w) {
          if (w = !0, O = M, M = p(M), y !== void 0 && _.hasValue) {
            var U = _.value;
            if (y(U, M))
              return k = U;
          }
          return k = M;
        }
        if (U = k, n(O, M))
          return U;
        var V = p(M);
        return y !== void 0 && y(U, V) ? U : (O = M, k = V);
      }
      var w = !1, O, k, P = g === void 0 ? null : g;
      return [function() {
        return C(d());
      }, P === null ? void 0 : function() {
        return C(P());
      }];
    }, [d, g, p, y]);
    var b = i(h, v[0], v[1]);
    return o(function() {
      _.hasValue = !0, _.value = b;
    }, [b]), c(b), b;
  }, sf;
}
var of = {};
/**
 * @license React
 * use-sync-external-store-shim/with-selector.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var op;
function ax() {
  return op || (op = 1, process.env.NODE_ENV !== "production" && function() {
    typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStart(new Error());
    var t = ln, e = hl;
    function r(d, g) {
      return d === g && (d !== 0 || 1 / d === 1 / g) || d !== d && g !== g;
    }
    var n = typeof Object.is == "function" ? Object.is : r, i = e.useSyncExternalStore, s = t.useRef, o = t.useEffect, l = t.useMemo, c = t.useDebugValue;
    function h(d, g, p, y, v) {
      var _ = s(null), b;
      _.current === null ? (b = {
        hasValue: !1,
        value: null
      }, _.current = b) : b = _.current;
      var C = l(function() {
        var P = !1, M, U, V = function(ce) {
          if (!P) {
            P = !0, M = ce;
            var $ = y(ce);
            if (v !== void 0 && b.hasValue) {
              var R = b.value;
              if (v(R, $))
                return U = R, R;
            }
            return U = $, $;
          }
          var F = M, H = U;
          if (n(F, ce))
            return H;
          var re = y(ce);
          return v !== void 0 && v(H, re) ? H : (M = ce, U = re, re);
        }, Y = p === void 0 ? null : p, X = function() {
          return V(g());
        }, Z = Y === null ? void 0 : function() {
          return V(Y());
        };
        return [X, Z];
      }, [g, p, y, v]), w = C[0], O = C[1], k = i(d, w, O);
      return o(function() {
        b.hasValue = !0, b.value = k;
      }, [k]), c(k), k;
    }
    of.useSyncExternalStoreWithSelector = h, typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < "u" && typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop == "function" && __REACT_DEVTOOLS_GLOBAL_HOOK__.registerInternalModuleStop(new Error());
  }()), of;
}
(function(t) {
  process.env.NODE_ENV === "production" ? t.exports = ix() : t.exports = ax();
})(nx);
function sx(t) {
  t();
}
let Fv = sx;
const ox = (t) => Fv = t, ux = () => Fv, oo = /* @__PURE__ */ vo(null);
process.env.NODE_ENV !== "production" && (oo.displayName = "ReactRedux");
function lx() {
  const t = cn(oo);
  if (process.env.NODE_ENV !== "production" && !t)
    throw new Error("could not find react-redux context value; please ensure the component is wrapped in a <Provider>");
  return t;
}
const cx = () => {
  throw new Error("uSES not initialized!");
};
let Gv = cx;
const fx = (t) => {
  Gv = t;
}, hx = (t, e) => t === e;
function dx(t = oo) {
  const e = t === oo ? lx : () => cn(t);
  return function(n, i = hx) {
    if (process.env.NODE_ENV !== "production") {
      if (!n)
        throw new Error("You must pass a selector to useSelector");
      if (typeof n != "function")
        throw new Error("You must pass a function as a selector to useSelector");
      if (typeof i != "function")
        throw new Error("You must pass a function as an equality function to useSelector");
    }
    const {
      store: s,
      subscription: o,
      getServerState: l
    } = e(), c = Gv(o.addNestedSub, s.getState, l || s.getState, n, i);
    return VC(c), c;
  };
}
const MD = /* @__PURE__ */ dx();
function dl() {
  return dl = Object.assign ? Object.assign.bind() : function(t) {
    for (var e = 1; e < arguments.length; e++) {
      var r = arguments[e];
      for (var n in r)
        Object.prototype.hasOwnProperty.call(r, n) && (t[n] = r[n]);
    }
    return t;
  }, dl.apply(this, arguments);
}
var Df = {}, gx = {
  get exports() {
    return Df;
  },
  set exports(t) {
    Df = t;
  }
}, Xe = {};
/** @license React v16.13.1
 * react-is.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var up;
function px() {
  if (up)
    return Xe;
  up = 1;
  var t = typeof Symbol == "function" && Symbol.for, e = t ? Symbol.for("react.element") : 60103, r = t ? Symbol.for("react.portal") : 60106, n = t ? Symbol.for("react.fragment") : 60107, i = t ? Symbol.for("react.strict_mode") : 60108, s = t ? Symbol.for("react.profiler") : 60114, o = t ? Symbol.for("react.provider") : 60109, l = t ? Symbol.for("react.context") : 60110, c = t ? Symbol.for("react.async_mode") : 60111, h = t ? Symbol.for("react.concurrent_mode") : 60111, d = t ? Symbol.for("react.forward_ref") : 60112, g = t ? Symbol.for("react.suspense") : 60113, p = t ? Symbol.for("react.suspense_list") : 60120, y = t ? Symbol.for("react.memo") : 60115, v = t ? Symbol.for("react.lazy") : 60116, _ = t ? Symbol.for("react.block") : 60121, b = t ? Symbol.for("react.fundamental") : 60117, C = t ? Symbol.for("react.responder") : 60118, w = t ? Symbol.for("react.scope") : 60119;
  function O(P) {
    if (typeof P == "object" && P !== null) {
      var M = P.$$typeof;
      switch (M) {
        case e:
          switch (P = P.type, P) {
            case c:
            case h:
            case n:
            case s:
            case i:
            case g:
              return P;
            default:
              switch (P = P && P.$$typeof, P) {
                case l:
                case d:
                case v:
                case y:
                case o:
                  return P;
                default:
                  return M;
              }
          }
        case r:
          return M;
      }
    }
  }
  function k(P) {
    return O(P) === h;
  }
  return Xe.AsyncMode = c, Xe.ConcurrentMode = h, Xe.ContextConsumer = l, Xe.ContextProvider = o, Xe.Element = e, Xe.ForwardRef = d, Xe.Fragment = n, Xe.Lazy = v, Xe.Memo = y, Xe.Portal = r, Xe.Profiler = s, Xe.StrictMode = i, Xe.Suspense = g, Xe.isAsyncMode = function(P) {
    return k(P) || O(P) === c;
  }, Xe.isConcurrentMode = k, Xe.isContextConsumer = function(P) {
    return O(P) === l;
  }, Xe.isContextProvider = function(P) {
    return O(P) === o;
  }, Xe.isElement = function(P) {
    return typeof P == "object" && P !== null && P.$$typeof === e;
  }, Xe.isForwardRef = function(P) {
    return O(P) === d;
  }, Xe.isFragment = function(P) {
    return O(P) === n;
  }, Xe.isLazy = function(P) {
    return O(P) === v;
  }, Xe.isMemo = function(P) {
    return O(P) === y;
  }, Xe.isPortal = function(P) {
    return O(P) === r;
  }, Xe.isProfiler = function(P) {
    return O(P) === s;
  }, Xe.isStrictMode = function(P) {
    return O(P) === i;
  }, Xe.isSuspense = function(P) {
    return O(P) === g;
  }, Xe.isValidElementType = function(P) {
    return typeof P == "string" || typeof P == "function" || P === n || P === h || P === s || P === i || P === g || P === p || typeof P == "object" && P !== null && (P.$$typeof === v || P.$$typeof === y || P.$$typeof === o || P.$$typeof === l || P.$$typeof === d || P.$$typeof === b || P.$$typeof === C || P.$$typeof === w || P.$$typeof === _);
  }, Xe.typeOf = O, Xe;
}
var Je = {};
/** @license React v16.13.1
 * react-is.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var lp;
function mx() {
  return lp || (lp = 1, process.env.NODE_ENV !== "production" && function() {
    var t = typeof Symbol == "function" && Symbol.for, e = t ? Symbol.for("react.element") : 60103, r = t ? Symbol.for("react.portal") : 60106, n = t ? Symbol.for("react.fragment") : 60107, i = t ? Symbol.for("react.strict_mode") : 60108, s = t ? Symbol.for("react.profiler") : 60114, o = t ? Symbol.for("react.provider") : 60109, l = t ? Symbol.for("react.context") : 60110, c = t ? Symbol.for("react.async_mode") : 60111, h = t ? Symbol.for("react.concurrent_mode") : 60111, d = t ? Symbol.for("react.forward_ref") : 60112, g = t ? Symbol.for("react.suspense") : 60113, p = t ? Symbol.for("react.suspense_list") : 60120, y = t ? Symbol.for("react.memo") : 60115, v = t ? Symbol.for("react.lazy") : 60116, _ = t ? Symbol.for("react.block") : 60121, b = t ? Symbol.for("react.fundamental") : 60117, C = t ? Symbol.for("react.responder") : 60118, w = t ? Symbol.for("react.scope") : 60119;
    function O(te) {
      return typeof te == "string" || typeof te == "function" || // Note: its typeof might be other than 'symbol' or 'number' if it's a polyfill.
      te === n || te === h || te === s || te === i || te === g || te === p || typeof te == "object" && te !== null && (te.$$typeof === v || te.$$typeof === y || te.$$typeof === o || te.$$typeof === l || te.$$typeof === d || te.$$typeof === b || te.$$typeof === C || te.$$typeof === w || te.$$typeof === _);
    }
    function k(te) {
      if (typeof te == "object" && te !== null) {
        var $t = te.$$typeof;
        switch ($t) {
          case e:
            var fe = te.type;
            switch (fe) {
              case c:
              case h:
              case n:
              case s:
              case i:
              case g:
                return fe;
              default:
                var Ot = fe && fe.$$typeof;
                switch (Ot) {
                  case l:
                  case d:
                  case v:
                  case y:
                  case o:
                    return Ot;
                  default:
                    return $t;
                }
            }
          case r:
            return $t;
        }
      }
    }
    var P = c, M = h, U = l, V = o, Y = e, X = d, Z = n, ce = v, $ = y, R = r, F = s, H = i, re = g, ae = !1;
    function ne(te) {
      return ae || (ae = !0, console.warn("The ReactIs.isAsyncMode() alias has been deprecated, and will be removed in React 17+. Update your code to use ReactIs.isConcurrentMode() instead. It has the exact same API.")), oe(te) || k(te) === c;
    }
    function oe(te) {
      return k(te) === h;
    }
    function de(te) {
      return k(te) === l;
    }
    function Re(te) {
      return k(te) === o;
    }
    function Ce(te) {
      return typeof te == "object" && te !== null && te.$$typeof === e;
    }
    function rt(te) {
      return k(te) === d;
    }
    function xt(te) {
      return k(te) === n;
    }
    function be(te) {
      return k(te) === v;
    }
    function At(te) {
      return k(te) === y;
    }
    function ot(te) {
      return k(te) === r;
    }
    function wt(te) {
      return k(te) === s;
    }
    function kt(te) {
      return k(te) === i;
    }
    function dt(te) {
      return k(te) === g;
    }
    Je.AsyncMode = P, Je.ConcurrentMode = M, Je.ContextConsumer = U, Je.ContextProvider = V, Je.Element = Y, Je.ForwardRef = X, Je.Fragment = Z, Je.Lazy = ce, Je.Memo = $, Je.Portal = R, Je.Profiler = F, Je.StrictMode = H, Je.Suspense = re, Je.isAsyncMode = ne, Je.isConcurrentMode = oe, Je.isContextConsumer = de, Je.isContextProvider = Re, Je.isElement = Ce, Je.isForwardRef = rt, Je.isFragment = xt, Je.isLazy = be, Je.isMemo = At, Je.isPortal = ot, Je.isProfiler = wt, Je.isStrictMode = kt, Je.isSuspense = dt, Je.isValidElementType = O, Je.typeOf = k;
  }()), Je;
}
(function(t) {
  process.env.NODE_ENV === "production" ? t.exports = px() : t.exports = mx();
})(gx);
var jv = Df, vx = {
  $$typeof: !0,
  render: !0,
  defaultProps: !0,
  displayName: !0,
  propTypes: !0
}, yx = {
  $$typeof: !0,
  compare: !0,
  defaultProps: !0,
  displayName: !0,
  propTypes: !0,
  type: !0
}, $v = {};
$v[jv.ForwardRef] = vx;
$v[jv.Memo] = yx;
var cp = {}, bx = {
  get exports() {
    return cp;
  },
  set exports(t) {
    cp = t;
  }
}, Qe = {};
/**
 * @license React
 * react-is.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var fp;
function _x() {
  if (fp)
    return Qe;
  fp = 1;
  var t = Symbol.for("react.element"), e = Symbol.for("react.portal"), r = Symbol.for("react.fragment"), n = Symbol.for("react.strict_mode"), i = Symbol.for("react.profiler"), s = Symbol.for("react.provider"), o = Symbol.for("react.context"), l = Symbol.for("react.server_context"), c = Symbol.for("react.forward_ref"), h = Symbol.for("react.suspense"), d = Symbol.for("react.suspense_list"), g = Symbol.for("react.memo"), p = Symbol.for("react.lazy"), y = Symbol.for("react.offscreen"), v;
  v = Symbol.for("react.module.reference");
  function _(b) {
    if (typeof b == "object" && b !== null) {
      var C = b.$$typeof;
      switch (C) {
        case t:
          switch (b = b.type, b) {
            case r:
            case i:
            case n:
            case h:
            case d:
              return b;
            default:
              switch (b = b && b.$$typeof, b) {
                case l:
                case o:
                case c:
                case p:
                case g:
                case s:
                  return b;
                default:
                  return C;
              }
          }
        case e:
          return C;
      }
    }
  }
  return Qe.ContextConsumer = o, Qe.ContextProvider = s, Qe.Element = t, Qe.ForwardRef = c, Qe.Fragment = r, Qe.Lazy = p, Qe.Memo = g, Qe.Portal = e, Qe.Profiler = i, Qe.StrictMode = n, Qe.Suspense = h, Qe.SuspenseList = d, Qe.isAsyncMode = function() {
    return !1;
  }, Qe.isConcurrentMode = function() {
    return !1;
  }, Qe.isContextConsumer = function(b) {
    return _(b) === o;
  }, Qe.isContextProvider = function(b) {
    return _(b) === s;
  }, Qe.isElement = function(b) {
    return typeof b == "object" && b !== null && b.$$typeof === t;
  }, Qe.isForwardRef = function(b) {
    return _(b) === c;
  }, Qe.isFragment = function(b) {
    return _(b) === r;
  }, Qe.isLazy = function(b) {
    return _(b) === p;
  }, Qe.isMemo = function(b) {
    return _(b) === g;
  }, Qe.isPortal = function(b) {
    return _(b) === e;
  }, Qe.isProfiler = function(b) {
    return _(b) === i;
  }, Qe.isStrictMode = function(b) {
    return _(b) === n;
  }, Qe.isSuspense = function(b) {
    return _(b) === h;
  }, Qe.isSuspenseList = function(b) {
    return _(b) === d;
  }, Qe.isValidElementType = function(b) {
    return typeof b == "string" || typeof b == "function" || b === r || b === i || b === n || b === h || b === d || b === y || typeof b == "object" && b !== null && (b.$$typeof === p || b.$$typeof === g || b.$$typeof === s || b.$$typeof === o || b.$$typeof === c || b.$$typeof === v || b.getModuleId !== void 0);
  }, Qe.typeOf = _, Qe;
}
var Ze = {};
/**
 * @license React
 * react-is.development.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var hp;
function Sx() {
  return hp || (hp = 1, process.env.NODE_ENV !== "production" && function() {
    var t = Symbol.for("react.element"), e = Symbol.for("react.portal"), r = Symbol.for("react.fragment"), n = Symbol.for("react.strict_mode"), i = Symbol.for("react.profiler"), s = Symbol.for("react.provider"), o = Symbol.for("react.context"), l = Symbol.for("react.server_context"), c = Symbol.for("react.forward_ref"), h = Symbol.for("react.suspense"), d = Symbol.for("react.suspense_list"), g = Symbol.for("react.memo"), p = Symbol.for("react.lazy"), y = Symbol.for("react.offscreen"), v = !1, _ = !1, b = !1, C = !1, w = !1, O;
    O = Symbol.for("react.module.reference");
    function k(fe) {
      return !!(typeof fe == "string" || typeof fe == "function" || fe === r || fe === i || w || fe === n || fe === h || fe === d || C || fe === y || v || _ || b || typeof fe == "object" && fe !== null && (fe.$$typeof === p || fe.$$typeof === g || fe.$$typeof === s || fe.$$typeof === o || fe.$$typeof === c || // This needs to include all possible module reference object
      // types supported by any Flight configuration anywhere since
      // we don't know which Flight build this will end up being used
      // with.
      fe.$$typeof === O || fe.getModuleId !== void 0));
    }
    function P(fe) {
      if (typeof fe == "object" && fe !== null) {
        var Ot = fe.$$typeof;
        switch (Ot) {
          case t:
            var sr = fe.type;
            switch (sr) {
              case r:
              case i:
              case n:
              case h:
              case d:
                return sr;
              default:
                var Dt = sr && sr.$$typeof;
                switch (Dt) {
                  case l:
                  case o:
                  case c:
                  case p:
                  case g:
                  case s:
                    return Dt;
                  default:
                    return Ot;
                }
            }
          case e:
            return Ot;
        }
      }
    }
    var M = o, U = s, V = t, Y = c, X = r, Z = p, ce = g, $ = e, R = i, F = n, H = h, re = d, ae = !1, ne = !1;
    function oe(fe) {
      return ae || (ae = !0, console.warn("The ReactIs.isAsyncMode() alias has been deprecated, and will be removed in React 18+.")), !1;
    }
    function de(fe) {
      return ne || (ne = !0, console.warn("The ReactIs.isConcurrentMode() alias has been deprecated, and will be removed in React 18+.")), !1;
    }
    function Re(fe) {
      return P(fe) === o;
    }
    function Ce(fe) {
      return P(fe) === s;
    }
    function rt(fe) {
      return typeof fe == "object" && fe !== null && fe.$$typeof === t;
    }
    function xt(fe) {
      return P(fe) === c;
    }
    function be(fe) {
      return P(fe) === r;
    }
    function At(fe) {
      return P(fe) === p;
    }
    function ot(fe) {
      return P(fe) === g;
    }
    function wt(fe) {
      return P(fe) === e;
    }
    function kt(fe) {
      return P(fe) === i;
    }
    function dt(fe) {
      return P(fe) === n;
    }
    function te(fe) {
      return P(fe) === h;
    }
    function $t(fe) {
      return P(fe) === d;
    }
    Ze.ContextConsumer = M, Ze.ContextProvider = U, Ze.Element = V, Ze.ForwardRef = Y, Ze.Fragment = X, Ze.Lazy = Z, Ze.Memo = ce, Ze.Portal = $, Ze.Profiler = R, Ze.StrictMode = F, Ze.Suspense = H, Ze.SuspenseList = re, Ze.isAsyncMode = oe, Ze.isConcurrentMode = de, Ze.isContextConsumer = Re, Ze.isContextProvider = Ce, Ze.isElement = rt, Ze.isForwardRef = xt, Ze.isFragment = be, Ze.isLazy = At, Ze.isMemo = ot, Ze.isPortal = wt, Ze.isProfiler = kt, Ze.isStrictMode = dt, Ze.isSuspense = te, Ze.isSuspenseList = $t, Ze.isValidElementType = k, Ze.typeOf = P;
  }()), Ze;
}
(function(t) {
  process.env.NODE_ENV === "production" ? t.exports = _x() : t.exports = Sx();
})(bx);
function wx() {
  const t = ux();
  let e = null, r = null;
  return {
    clear() {
      e = null, r = null;
    },
    notify() {
      t(() => {
        let n = e;
        for (; n; )
          n.callback(), n = n.next;
      });
    },
    get() {
      let n = [], i = e;
      for (; i; )
        n.push(i), i = i.next;
      return n;
    },
    subscribe(n) {
      let i = !0, s = r = {
        callback: n,
        next: null,
        prev: r
      };
      return s.prev ? s.prev.next = s : e = s, function() {
        !i || e === null || (i = !1, s.next ? s.next.prev = s.prev : r = s.prev, s.prev ? s.prev.next = s.next : e = s.next);
      };
    }
  };
}
const dp = {
  notify() {
  },
  get: () => []
};
function Ex(t, e) {
  let r, n = dp;
  function i(g) {
    return c(), n.subscribe(g);
  }
  function s() {
    n.notify();
  }
  function o() {
    d.onStateChange && d.onStateChange();
  }
  function l() {
    return Boolean(r);
  }
  function c() {
    r || (r = e ? e.addNestedSub(o) : t.subscribe(o), n = wx());
  }
  function h() {
    r && (r(), r = void 0, n.clear(), n = dp);
  }
  const d = {
    addNestedSub: i,
    notifyNestedSubs: s,
    handleChangeWrapper: o,
    isSubscribed: l,
    trySubscribe: c,
    tryUnsubscribe: h,
    getListeners: () => n
  };
  return d;
}
const Cx = typeof window < "u" && typeof window.document < "u" && typeof window.document.createElement < "u", xx = Cx ? Mv : fl;
function Ox({
  store: t,
  context: e,
  children: r,
  serverState: n
}) {
  const i = tp(() => {
    const l = Ex(t);
    return {
      store: t,
      subscription: l,
      getServerState: n ? () => n : void 0
    };
  }, [t, n]), s = tp(() => t.getState(), [t]);
  xx(() => {
    const {
      subscription: l
    } = i;
    return l.onStateChange = l.notifyNestedSubs, l.trySubscribe(), s !== t.getState() && l.notifyNestedSubs(), () => {
      l.tryUnsubscribe(), l.onStateChange = void 0;
    };
  }, [i, s]);
  const o = e || oo;
  return /* @__PURE__ */ ln.createElement(o.Provider, {
    value: i
  }, r);
}
fx(Lf.useSyncExternalStoreWithSelector);
ox(KC);
function Ku(t) {
  return typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? Ku = function(r) {
    return typeof r;
  } : Ku = function(r) {
    return r && typeof Symbol == "function" && r.constructor === Symbol && r !== Symbol.prototype ? "symbol" : typeof r;
  }, Ku(t);
}
function Tx(t, e) {
  if (!(t instanceof e))
    throw new TypeError("Cannot call a class as a function");
}
function gp(t, e) {
  for (var r = 0; r < e.length; r++) {
    var n = e[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(t, n.key, n);
  }
}
function Px(t, e, r) {
  return e && gp(t.prototype, e), r && gp(t, r), t;
}
function Rx(t, e) {
  return e && (Ku(e) === "object" || typeof e == "function") ? e : Xu(t);
}
function Mf(t) {
  return Mf = Object.setPrototypeOf ? Object.getPrototypeOf : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, Mf(t);
}
function Xu(t) {
  if (t === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return t;
}
function Ax(t, e) {
  if (typeof e != "function" && e !== null)
    throw new TypeError("Super expression must either be null or a function");
  t.prototype = Object.create(e && e.prototype, { constructor: { value: t, writable: !0, configurable: !0 } }), e && Nf(t, e);
}
function Nf(t, e) {
  return Nf = Object.setPrototypeOf || function(n, i) {
    return n.__proto__ = i, n;
  }, Nf(t, e);
}
function Ju(t, e, r) {
  return e in t ? Object.defineProperty(t, e, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = r, t;
}
var Wv = /* @__PURE__ */ function(t) {
  Ax(e, t);
  function e() {
    var r, n;
    Tx(this, e);
    for (var i = arguments.length, s = new Array(i), o = 0; o < i; o++)
      s[o] = arguments[o];
    return n = Rx(this, (r = Mf(e)).call.apply(r, [this].concat(s))), Ju(Xu(n), "state", {
      bootstrapped: !1
    }), Ju(Xu(n), "_unsubscribe", void 0), Ju(Xu(n), "handlePersistorState", function() {
      var l = n.props.persistor, c = l.getState(), h = c.bootstrapped;
      h && (n.props.onBeforeLift ? Promise.resolve(n.props.onBeforeLift()).finally(function() {
        return n.setState({
          bootstrapped: !0
        });
      }) : n.setState({
        bootstrapped: !0
      }), n._unsubscribe && n._unsubscribe());
    }), n;
  }
  return Px(e, [{
    key: "componentDidMount",
    value: function() {
      this._unsubscribe = this.props.persistor.subscribe(this.handlePersistorState), this.handlePersistorState();
    }
  }, {
    key: "componentWillUnmount",
    value: function() {
      this._unsubscribe && this._unsubscribe();
    }
  }, {
    key: "render",
    value: function() {
      return process.env.NODE_ENV !== "production" && typeof this.props.children == "function" && this.props.loading && console.error("redux-persist: PersistGate expects either a function child or loading prop, but not both. The loading prop will be ignored."), typeof this.props.children == "function" ? this.props.children(this.state.bootstrapped) : this.state.bootstrapped ? this.props.children : this.props.loading;
    }
  }]), e;
}(qC);
Ju(Wv, "defaultProps", {
  children: null,
  loading: null
});
function Vt(t) {
  for (var e = arguments.length, r = Array(e > 1 ? e - 1 : 0), n = 1; n < e; n++)
    r[n - 1] = arguments[n];
  if (process.env.NODE_ENV !== "production") {
    var i = Gx[t], s = i ? typeof i == "function" ? i.apply(null, r) : i : "unknown error nr: " + t;
    throw Error("[Immer] " + s);
  }
  throw Error("[Immer] minified error nr: " + t + (r.length ? " " + r.map(function(o) {
    return "'" + o + "'";
  }).join(",") : "") + ". Find the full error at: https://bit.ly/3cXEKWf");
}
function Ci(t) {
  return !!t && !!t[mt];
}
function ei(t) {
  var e;
  return !!t && (function(r) {
    if (!r || typeof r != "object")
      return !1;
    var n = Object.getPrototypeOf(r);
    if (n === null)
      return !0;
    var i = Object.hasOwnProperty.call(n, "constructor") && n.constructor;
    return i === Object || typeof i == "function" && Function.toString.call(i) === jx;
  }(t) || Array.isArray(t) || !!t[Sp] || !!(!((e = t.constructor) === null || e === void 0) && e[Sp]) || yh(t) || bh(t));
}
function Xi(t, e, r) {
  r === void 0 && (r = !1), Qa(t) === 0 ? (r ? Object.keys : Ha)(t).forEach(function(n) {
    r && typeof n == "symbol" || e(n, t[n], t);
  }) : t.forEach(function(n, i) {
    return e(i, n, t);
  });
}
function Qa(t) {
  var e = t[mt];
  return e ? e.i > 3 ? e.i - 4 : e.i : Array.isArray(t) ? 1 : yh(t) ? 2 : bh(t) ? 3 : 0;
}
function Ua(t, e) {
  return Qa(t) === 2 ? t.has(e) : Object.prototype.hasOwnProperty.call(t, e);
}
function kx(t, e) {
  return Qa(t) === 2 ? t.get(e) : t[e];
}
function Uv(t, e, r) {
  var n = Qa(t);
  n === 2 ? t.set(e, r) : n === 3 ? t.add(r) : t[e] = r;
}
function Hv(t, e) {
  return t === e ? t !== 0 || 1 / t == 1 / e : t != t && e != e;
}
function yh(t) {
  return Bx && t instanceof Map;
}
function bh(t) {
  return Fx && t instanceof Set;
}
function bi(t) {
  return t.o || t.t;
}
function _h(t) {
  if (Array.isArray(t))
    return Array.prototype.slice.call(t);
  var e = Vv(t);
  delete e[mt];
  for (var r = Ha(e), n = 0; n < r.length; n++) {
    var i = r[n], s = e[i];
    s.writable === !1 && (s.writable = !0, s.configurable = !0), (s.get || s.set) && (e[i] = { configurable: !0, writable: !0, enumerable: s.enumerable, value: t[i] });
  }
  return Object.create(Object.getPrototypeOf(t), e);
}
function Sh(t, e) {
  return e === void 0 && (e = !1), wh(t) || Ci(t) || !ei(t) || (Qa(t) > 1 && (t.set = t.add = t.clear = t.delete = Ix), Object.freeze(t), e && Xi(t, function(r, n) {
    return Sh(n, !0);
  }, !0)), t;
}
function Ix() {
  Vt(2);
}
function wh(t) {
  return t == null || typeof t != "object" || Object.isFrozen(t);
}
function Tn(t) {
  var e = jf[t];
  return e || Vt(18, t), e;
}
function Lx(t, e) {
  jf[t] || (jf[t] = e);
}
function Bf() {
  return process.env.NODE_ENV === "production" || Ya || Vt(0), Ya;
}
function uf(t, e) {
  e && (Tn("Patches"), t.u = [], t.s = [], t.v = e);
}
function gl(t) {
  Ff(t), t.p.forEach(Dx), t.p = null;
}
function Ff(t) {
  t === Ya && (Ya = t.l);
}
function pp(t) {
  return Ya = { p: [], l: Ya, h: t, m: !0, _: 0 };
}
function Dx(t) {
  var e = t[mt];
  e.i === 0 || e.i === 1 ? e.j() : e.O = !0;
}
function lf(t, e) {
  e._ = e.p.length;
  var r = e.p[0], n = t !== void 0 && t !== r;
  return e.h.g || Tn("ES5").S(e, t, n), n ? (r[mt].P && (gl(e), Vt(4)), ei(t) && (t = pl(e, t), e.l || ml(e, t)), e.u && Tn("Patches").M(r[mt].t, t, e.u, e.s)) : t = pl(e, r, []), gl(e), e.u && e.v(e.u, e.s), t !== zv ? t : void 0;
}
function pl(t, e, r) {
  if (wh(e))
    return e;
  var n = e[mt];
  if (!n)
    return Xi(e, function(l, c) {
      return mp(t, n, e, l, c, r);
    }, !0), e;
  if (n.A !== t)
    return e;
  if (!n.P)
    return ml(t, n.t, !0), n.t;
  if (!n.I) {
    n.I = !0, n.A._--;
    var i = n.i === 4 || n.i === 5 ? n.o = _h(n.k) : n.o, s = i, o = !1;
    n.i === 3 && (s = new Set(i), i.clear(), o = !0), Xi(s, function(l, c) {
      return mp(t, n, i, l, c, r, o);
    }), ml(t, i, !1), r && t.u && Tn("Patches").N(n, r, t.u, t.s);
  }
  return n.o;
}
function mp(t, e, r, n, i, s, o) {
  if (process.env.NODE_ENV !== "production" && i === r && Vt(5), Ci(i)) {
    var l = pl(t, i, s && e && e.i !== 3 && !Ua(e.R, n) ? s.concat(n) : void 0);
    if (Uv(r, n, l), !Ci(l))
      return;
    t.m = !1;
  } else
    o && r.add(i);
  if (ei(i) && !wh(i)) {
    if (!t.h.D && t._ < 1)
      return;
    pl(t, i), e && e.A.l || ml(t, i);
  }
}
function ml(t, e, r) {
  r === void 0 && (r = !1), !t.l && t.h.D && t.m && Sh(e, r);
}
function cf(t, e) {
  var r = t[mt];
  return (r ? bi(r) : t)[e];
}
function vp(t, e) {
  if (e in t)
    for (var r = Object.getPrototypeOf(t); r; ) {
      var n = Object.getOwnPropertyDescriptor(r, e);
      if (n)
        return n;
      r = Object.getPrototypeOf(r);
    }
}
function _i(t) {
  t.P || (t.P = !0, t.l && _i(t.l));
}
function ff(t) {
  t.o || (t.o = _h(t.t));
}
function Gf(t, e, r) {
  var n = yh(e) ? Tn("MapSet").F(e, r) : bh(e) ? Tn("MapSet").T(e, r) : t.g ? function(i, s) {
    var o = Array.isArray(i), l = { i: o ? 1 : 0, A: s ? s.A : Bf(), P: !1, I: !1, R: {}, l: s, t: i, k: null, o: null, j: null, C: !1 }, c = l, h = uo;
    o && (c = [l], h = Vs);
    var d = Proxy.revocable(c, h), g = d.revoke, p = d.proxy;
    return l.k = p, l.j = g, p;
  }(e, r) : Tn("ES5").J(e, r);
  return (r ? r.A : Bf()).p.push(n), n;
}
function Mx(t) {
  return Ci(t) || Vt(22, t), function e(r) {
    if (!ei(r))
      return r;
    var n, i = r[mt], s = Qa(r);
    if (i) {
      if (!i.P && (i.i < 4 || !Tn("ES5").K(i)))
        return i.t;
      i.I = !0, n = yp(r, s), i.I = !1;
    } else
      n = yp(r, s);
    return Xi(n, function(o, l) {
      i && kx(i.t, o) === l || Uv(n, o, e(l));
    }), s === 3 ? new Set(n) : n;
  }(t);
}
function yp(t, e) {
  switch (e) {
    case 2:
      return new Map(t);
    case 3:
      return Array.from(t);
  }
  return _h(t);
}
function Nx() {
  function t(o, l) {
    var c = s[o];
    return c ? c.enumerable = l : s[o] = c = { configurable: !0, enumerable: l, get: function() {
      var h = this[mt];
      return process.env.NODE_ENV !== "production" && i(h), uo.get(h, o);
    }, set: function(h) {
      var d = this[mt];
      process.env.NODE_ENV !== "production" && i(d), uo.set(d, o, h);
    } }, c;
  }
  function e(o) {
    for (var l = o.length - 1; l >= 0; l--) {
      var c = o[l][mt];
      if (!c.P)
        switch (c.i) {
          case 5:
            n(c) && _i(c);
            break;
          case 4:
            r(c) && _i(c);
        }
    }
  }
  function r(o) {
    for (var l = o.t, c = o.k, h = Ha(c), d = h.length - 1; d >= 0; d--) {
      var g = h[d];
      if (g !== mt) {
        var p = l[g];
        if (p === void 0 && !Ua(l, g))
          return !0;
        var y = c[g], v = y && y[mt];
        if (v ? v.t !== p : !Hv(y, p))
          return !0;
      }
    }
    var _ = !!l[mt];
    return h.length !== Ha(l).length + (_ ? 0 : 1);
  }
  function n(o) {
    var l = o.k;
    if (l.length !== o.t.length)
      return !0;
    var c = Object.getOwnPropertyDescriptor(l, l.length - 1);
    if (c && !c.get)
      return !0;
    for (var h = 0; h < l.length; h++)
      if (!l.hasOwnProperty(h))
        return !0;
    return !1;
  }
  function i(o) {
    o.O && Vt(3, JSON.stringify(bi(o)));
  }
  var s = {};
  Lx("ES5", { J: function(o, l) {
    var c = Array.isArray(o), h = function(g, p) {
      if (g) {
        for (var y = Array(p.length), v = 0; v < p.length; v++)
          Object.defineProperty(y, "" + v, t(v, !0));
        return y;
      }
      var _ = Vv(p);
      delete _[mt];
      for (var b = Ha(_), C = 0; C < b.length; C++) {
        var w = b[C];
        _[w] = t(w, g || !!_[w].enumerable);
      }
      return Object.create(Object.getPrototypeOf(p), _);
    }(c, o), d = { i: c ? 5 : 4, A: l ? l.A : Bf(), P: !1, I: !1, R: {}, l, t: o, k: h, o: null, O: !1, C: !1 };
    return Object.defineProperty(h, mt, { value: d, writable: !0 }), h;
  }, S: function(o, l, c) {
    c ? Ci(l) && l[mt].A === o && e(o.p) : (o.u && function h(d) {
      if (d && typeof d == "object") {
        var g = d[mt];
        if (g) {
          var p = g.t, y = g.k, v = g.R, _ = g.i;
          if (_ === 4)
            Xi(y, function(k) {
              k !== mt && (p[k] !== void 0 || Ua(p, k) ? v[k] || h(y[k]) : (v[k] = !0, _i(g)));
            }), Xi(p, function(k) {
              y[k] !== void 0 || Ua(y, k) || (v[k] = !1, _i(g));
            });
          else if (_ === 5) {
            if (n(g) && (_i(g), v.length = !0), y.length < p.length)
              for (var b = y.length; b < p.length; b++)
                v[b] = !1;
            else
              for (var C = p.length; C < y.length; C++)
                v[C] = !0;
            for (var w = Math.min(y.length, p.length), O = 0; O < w; O++)
              y.hasOwnProperty(O) || (v[O] = !0), v[O] === void 0 && h(y[O]);
          }
        }
      }
    }(o.p[0]), e(o.p));
  }, K: function(o) {
    return o.i === 4 ? r(o) : n(o);
  } });
}
var bp, Ya, Eh = typeof Symbol < "u" && typeof Symbol("x") == "symbol", Bx = typeof Map < "u", Fx = typeof Set < "u", _p = typeof Proxy < "u" && Proxy.revocable !== void 0 && typeof Reflect < "u", zv = Eh ? Symbol.for("immer-nothing") : ((bp = {})["immer-nothing"] = !0, bp), Sp = Eh ? Symbol.for("immer-draftable") : "__$immer_draftable", mt = Eh ? Symbol.for("immer-state") : "__$immer_state", Gx = { 0: "Illegal state", 1: "Immer drafts cannot have computed properties", 2: "This object has been frozen and should not be mutated", 3: function(t) {
  return "Cannot use a proxy that has been revoked. Did you pass an object from inside an immer function to an async process? " + t;
}, 4: "An immer producer returned a new value *and* modified its draft. Either return a new value *or* modify the draft.", 5: "Immer forbids circular references", 6: "The first or second argument to `produce` must be a function", 7: "The third argument to `produce` must be a function or undefined", 8: "First argument to `createDraft` must be a plain object, an array, or an immerable object", 9: "First argument to `finishDraft` must be a draft returned by `createDraft`", 10: "The given draft is already finalized", 11: "Object.defineProperty() cannot be used on an Immer draft", 12: "Object.setPrototypeOf() cannot be used on an Immer draft", 13: "Immer only supports deleting array indices", 14: "Immer only supports setting array indices and the 'length' property", 15: function(t) {
  return "Cannot apply patch, path doesn't resolve: " + t;
}, 16: 'Sets cannot have "replace" patches.', 17: function(t) {
  return "Unsupported patch operation: " + t;
}, 18: function(t) {
  return "The plugin for '" + t + "' has not been loaded into Immer. To enable the plugin, import and call `enable" + t + "()` when initializing your application.";
}, 20: "Cannot use proxies if Proxy, Proxy.revocable or Reflect are not available", 21: function(t) {
  return "produce can only be called on things that are draftable: plain objects, arrays, Map, Set or classes that are marked with '[immerable]: true'. Got '" + t + "'";
}, 22: function(t) {
  return "'current' expects a draft, got: " + t;
}, 23: function(t) {
  return "'original' expects a draft, got: " + t;
}, 24: "Patching reserved attributes like __proto__, prototype and constructor is not allowed" }, jx = "" + Object.prototype.constructor, Ha = typeof Reflect < "u" && Reflect.ownKeys ? Reflect.ownKeys : Object.getOwnPropertySymbols !== void 0 ? function(t) {
  return Object.getOwnPropertyNames(t).concat(Object.getOwnPropertySymbols(t));
} : Object.getOwnPropertyNames, Vv = Object.getOwnPropertyDescriptors || function(t) {
  var e = {};
  return Ha(t).forEach(function(r) {
    e[r] = Object.getOwnPropertyDescriptor(t, r);
  }), e;
}, jf = {}, uo = { get: function(t, e) {
  if (e === mt)
    return t;
  var r = bi(t);
  if (!Ua(r, e))
    return function(i, s, o) {
      var l, c = vp(s, o);
      return c ? "value" in c ? c.value : (l = c.get) === null || l === void 0 ? void 0 : l.call(i.k) : void 0;
    }(t, r, e);
  var n = r[e];
  return t.I || !ei(n) ? n : n === cf(t.t, e) ? (ff(t), t.o[e] = Gf(t.A.h, n, t)) : n;
}, has: function(t, e) {
  return e in bi(t);
}, ownKeys: function(t) {
  return Reflect.ownKeys(bi(t));
}, set: function(t, e, r) {
  var n = vp(bi(t), e);
  if (n != null && n.set)
    return n.set.call(t.k, r), !0;
  if (!t.P) {
    var i = cf(bi(t), e), s = i == null ? void 0 : i[mt];
    if (s && s.t === r)
      return t.o[e] = r, t.R[e] = !1, !0;
    if (Hv(r, i) && (r !== void 0 || Ua(t.t, e)))
      return !0;
    ff(t), _i(t);
  }
  return t.o[e] === r && (r !== void 0 || e in t.o) || Number.isNaN(r) && Number.isNaN(t.o[e]) || (t.o[e] = r, t.R[e] = !0), !0;
}, deleteProperty: function(t, e) {
  return cf(t.t, e) !== void 0 || e in t.t ? (t.R[e] = !1, ff(t), _i(t)) : delete t.R[e], t.o && delete t.o[e], !0;
}, getOwnPropertyDescriptor: function(t, e) {
  var r = bi(t), n = Reflect.getOwnPropertyDescriptor(r, e);
  return n && { writable: !0, configurable: t.i !== 1 || e !== "length", enumerable: n.enumerable, value: r[e] };
}, defineProperty: function() {
  Vt(11);
}, getPrototypeOf: function(t) {
  return Object.getPrototypeOf(t.t);
}, setPrototypeOf: function() {
  Vt(12);
} }, Vs = {};
Xi(uo, function(t, e) {
  Vs[t] = function() {
    return arguments[0] = arguments[0][0], e.apply(this, arguments);
  };
}), Vs.deleteProperty = function(t, e) {
  return process.env.NODE_ENV !== "production" && isNaN(parseInt(e)) && Vt(13), Vs.set.call(this, t, e, void 0);
}, Vs.set = function(t, e, r) {
  return process.env.NODE_ENV !== "production" && e !== "length" && isNaN(parseInt(e)) && Vt(14), uo.set.call(this, t[0], e, r, t[0]);
};
var $x = function() {
  function t(r) {
    var n = this;
    this.g = _p, this.D = !0, this.produce = function(i, s, o) {
      if (typeof i == "function" && typeof s != "function") {
        var l = s;
        s = i;
        var c = n;
        return function(_) {
          var b = this;
          _ === void 0 && (_ = l);
          for (var C = arguments.length, w = Array(C > 1 ? C - 1 : 0), O = 1; O < C; O++)
            w[O - 1] = arguments[O];
          return c.produce(_, function(k) {
            var P;
            return (P = s).call.apply(P, [b, k].concat(w));
          });
        };
      }
      var h;
      if (typeof s != "function" && Vt(6), o !== void 0 && typeof o != "function" && Vt(7), ei(i)) {
        var d = pp(n), g = Gf(n, i, void 0), p = !0;
        try {
          h = s(g), p = !1;
        } finally {
          p ? gl(d) : Ff(d);
        }
        return typeof Promise < "u" && h instanceof Promise ? h.then(function(_) {
          return uf(d, o), lf(_, d);
        }, function(_) {
          throw gl(d), _;
        }) : (uf(d, o), lf(h, d));
      }
      if (!i || typeof i != "object") {
        if ((h = s(i)) === void 0 && (h = i), h === zv && (h = void 0), n.D && Sh(h, !0), o) {
          var y = [], v = [];
          Tn("Patches").M(i, h, y, v), o(y, v);
        }
        return h;
      }
      Vt(21, i);
    }, this.produceWithPatches = function(i, s) {
      if (typeof i == "function")
        return function(h) {
          for (var d = arguments.length, g = Array(d > 1 ? d - 1 : 0), p = 1; p < d; p++)
            g[p - 1] = arguments[p];
          return n.produceWithPatches(h, function(y) {
            return i.apply(void 0, [y].concat(g));
          });
        };
      var o, l, c = n.produce(i, s, function(h, d) {
        o = h, l = d;
      });
      return typeof Promise < "u" && c instanceof Promise ? c.then(function(h) {
        return [h, o, l];
      }) : [c, o, l];
    }, typeof (r == null ? void 0 : r.useProxies) == "boolean" && this.setUseProxies(r.useProxies), typeof (r == null ? void 0 : r.autoFreeze) == "boolean" && this.setAutoFreeze(r.autoFreeze);
  }
  var e = t.prototype;
  return e.createDraft = function(r) {
    ei(r) || Vt(8), Ci(r) && (r = Mx(r));
    var n = pp(this), i = Gf(this, r, void 0);
    return i[mt].C = !0, Ff(n), i;
  }, e.finishDraft = function(r, n) {
    var i = r && r[mt];
    process.env.NODE_ENV !== "production" && (i && i.C || Vt(9), i.I && Vt(10));
    var s = i.A;
    return uf(s, n), lf(void 0, s);
  }, e.setAutoFreeze = function(r) {
    this.D = r;
  }, e.setUseProxies = function(r) {
    r && !_p && Vt(20), this.g = r;
  }, e.applyPatches = function(r, n) {
    var i;
    for (i = n.length - 1; i >= 0; i--) {
      var s = n[i];
      if (s.path.length === 0 && s.op === "replace") {
        r = s.value;
        break;
      }
    }
    i > -1 && (n = n.slice(i + 1));
    var o = Tn("Patches").$;
    return Ci(r) ? o(r, n) : this.produce(r, function(l) {
      return o(l, n);
    });
  }, t;
}(), Mr = new $x(), qv = Mr.produce;
Mr.produceWithPatches.bind(Mr);
Mr.setAutoFreeze.bind(Mr);
Mr.setUseProxies.bind(Mr);
Mr.applyPatches.bind(Mr);
Mr.createDraft.bind(Mr);
Mr.finishDraft.bind(Mr);
function Kr(t) {
  return Kr = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(e) {
    return typeof e;
  } : function(e) {
    return e && typeof Symbol == "function" && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e;
  }, Kr(t);
}
function Wx(t, e) {
  if (Kr(t) !== "object" || t === null)
    return t;
  var r = t[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(t, e || "default");
    if (Kr(n) !== "object")
      return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (e === "string" ? String : Number)(t);
}
function Yv(t) {
  var e = Wx(t, "string");
  return Kr(e) === "symbol" ? e : String(e);
}
function fn(t, e, r) {
  return e = Yv(e), e in t ? Object.defineProperty(t, e, {
    value: r,
    enumerable: !0,
    configurable: !0,
    writable: !0
  }) : t[e] = r, t;
}
function wp(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Ep(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? wp(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : wp(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function ar(t) {
  return "Minified Redux error #" + t + "; visit https://redux.js.org/Errors?code=" + t + " for the full message or use the non-minified dev environment for full errors. ";
}
var Cp = function() {
  return typeof Symbol == "function" && Symbol.observable || "@@observable";
}(), hf = function() {
  return Math.random().toString(36).substring(7).split("").join(".");
}, Ki = {
  INIT: "@@redux/INIT" + hf(),
  REPLACE: "@@redux/REPLACE" + hf(),
  PROBE_UNKNOWN_ACTION: function() {
    return "@@redux/PROBE_UNKNOWN_ACTION" + hf();
  }
};
function Kv(t) {
  if (typeof t != "object" || t === null)
    return !1;
  for (var e = t; Object.getPrototypeOf(e) !== null; )
    e = Object.getPrototypeOf(e);
  return Object.getPrototypeOf(t) === e;
}
function Ux(t) {
  if (t === void 0)
    return "undefined";
  if (t === null)
    return "null";
  var e = typeof t;
  switch (e) {
    case "boolean":
    case "string":
    case "number":
    case "symbol":
    case "function":
      return e;
  }
  if (Array.isArray(t))
    return "array";
  if (Vx(t))
    return "date";
  if (zx(t))
    return "error";
  var r = Hx(t);
  switch (r) {
    case "Symbol":
    case "Promise":
    case "WeakMap":
    case "WeakSet":
    case "Map":
    case "Set":
      return r;
  }
  return e.slice(8, -1).toLowerCase().replace(/\s/g, "");
}
function Hx(t) {
  return typeof t.constructor == "function" ? t.constructor.name : null;
}
function zx(t) {
  return t instanceof Error || typeof t.message == "string" && t.constructor && typeof t.constructor.stackTraceLimit == "number";
}
function Vx(t) {
  return t instanceof Date ? !0 : typeof t.toDateString == "function" && typeof t.getDate == "function" && typeof t.setDate == "function";
}
function Vi(t) {
  var e = typeof t;
  return process.env.NODE_ENV !== "production" && (e = Ux(t)), e;
}
function Ch(t, e, r) {
  var n;
  if (typeof e == "function" && typeof r == "function" || typeof r == "function" && typeof arguments[3] == "function")
    throw new Error(process.env.NODE_ENV === "production" ? ar(0) : "It looks like you are passing several store enhancers to createStore(). This is not supported. Instead, compose them together to a single function. See https://redux.js.org/tutorials/fundamentals/part-4-store#creating-a-store-with-enhancers for an example.");
  if (typeof e == "function" && typeof r > "u" && (r = e, e = void 0), typeof r < "u") {
    if (typeof r != "function")
      throw new Error(process.env.NODE_ENV === "production" ? ar(1) : "Expected the enhancer to be a function. Instead, received: '" + Vi(r) + "'");
    return r(Ch)(t, e);
  }
  if (typeof t != "function")
    throw new Error(process.env.NODE_ENV === "production" ? ar(2) : "Expected the root reducer to be a function. Instead, received: '" + Vi(t) + "'");
  var i = t, s = e, o = [], l = o, c = !1;
  function h() {
    l === o && (l = o.slice());
  }
  function d() {
    if (c)
      throw new Error(process.env.NODE_ENV === "production" ? ar(3) : "You may not call store.getState() while the reducer is executing. The reducer has already received the state as an argument. Pass it down from the top reducer instead of reading it from the store.");
    return s;
  }
  function g(_) {
    if (typeof _ != "function")
      throw new Error(process.env.NODE_ENV === "production" ? ar(4) : "Expected the listener to be a function. Instead, received: '" + Vi(_) + "'");
    if (c)
      throw new Error(process.env.NODE_ENV === "production" ? ar(5) : "You may not call store.subscribe() while the reducer is executing. If you would like to be notified after the store has been updated, subscribe from a component and invoke store.getState() in the callback to access the latest state. See https://redux.js.org/api/store#subscribelistener for more details.");
    var b = !0;
    return h(), l.push(_), function() {
      if (b) {
        if (c)
          throw new Error(process.env.NODE_ENV === "production" ? ar(6) : "You may not unsubscribe from a store listener while the reducer is executing. See https://redux.js.org/api/store#subscribelistener for more details.");
        b = !1, h();
        var w = l.indexOf(_);
        l.splice(w, 1), o = null;
      }
    };
  }
  function p(_) {
    if (!Kv(_))
      throw new Error(process.env.NODE_ENV === "production" ? ar(7) : "Actions must be plain objects. Instead, the actual type was: '" + Vi(_) + "'. You may need to add middleware to your store setup to handle dispatching other values, such as 'redux-thunk' to handle dispatching functions. See https://redux.js.org/tutorials/fundamentals/part-4-store#middleware and https://redux.js.org/tutorials/fundamentals/part-6-async-logic#using-the-redux-thunk-middleware for examples.");
    if (typeof _.type > "u")
      throw new Error(process.env.NODE_ENV === "production" ? ar(8) : 'Actions may not have an undefined "type" property. You may have misspelled an action type string constant.');
    if (c)
      throw new Error(process.env.NODE_ENV === "production" ? ar(9) : "Reducers may not dispatch actions.");
    try {
      c = !0, s = i(s, _);
    } finally {
      c = !1;
    }
    for (var b = o = l, C = 0; C < b.length; C++) {
      var w = b[C];
      w();
    }
    return _;
  }
  function y(_) {
    if (typeof _ != "function")
      throw new Error(process.env.NODE_ENV === "production" ? ar(10) : "Expected the nextReducer to be a function. Instead, received: '" + Vi(_));
    i = _, p({
      type: Ki.REPLACE
    });
  }
  function v() {
    var _, b = g;
    return _ = {
      /**
       * The minimal observable subscription method.
       * @param {Object} observer Any object that can be used as an observer.
       * The observer object should have a `next` method.
       * @returns {subscription} An object with an `unsubscribe` method that can
       * be used to unsubscribe the observable from the store, and prevent further
       * emission of values from the observable.
       */
      subscribe: function(w) {
        if (typeof w != "object" || w === null)
          throw new Error(process.env.NODE_ENV === "production" ? ar(11) : "Expected the observer to be an object. Instead, received: '" + Vi(w) + "'");
        function O() {
          w.next && w.next(d());
        }
        O();
        var k = b(O);
        return {
          unsubscribe: k
        };
      }
    }, _[Cp] = function() {
      return this;
    }, _;
  }
  return p({
    type: Ki.INIT
  }), n = {
    dispatch: p,
    subscribe: g,
    getState: d,
    replaceReducer: y
  }, n[Cp] = v, n;
}
function xp(t) {
  typeof console < "u" && typeof console.error == "function" && console.error(t);
  try {
    throw new Error(t);
  } catch {
  }
}
function qx(t, e, r, n) {
  var i = Object.keys(e), s = r && r.type === Ki.INIT ? "preloadedState argument passed to createStore" : "previous state received by the reducer";
  if (i.length === 0)
    return "Store does not have a valid reducer. Make sure the argument passed to combineReducers is an object whose values are reducers.";
  if (!Kv(t))
    return "The " + s + ' has unexpected type of "' + Vi(t) + '". Expected argument to be an object with the following ' + ('keys: "' + i.join('", "') + '"');
  var o = Object.keys(t).filter(function(l) {
    return !e.hasOwnProperty(l) && !n[l];
  });
  if (o.forEach(function(l) {
    n[l] = !0;
  }), !(r && r.type === Ki.REPLACE) && o.length > 0)
    return "Unexpected " + (o.length > 1 ? "keys" : "key") + " " + ('"' + o.join('", "') + '" found in ' + s + ". ") + "Expected to find one of the known reducer keys instead: " + ('"' + i.join('", "') + '". Unexpected keys will be ignored.');
}
function Yx(t) {
  Object.keys(t).forEach(function(e) {
    var r = t[e], n = r(void 0, {
      type: Ki.INIT
    });
    if (typeof n > "u")
      throw new Error(process.env.NODE_ENV === "production" ? ar(12) : 'The slice reducer for key "' + e + `" returned undefined during initialization. If the state passed to the reducer is undefined, you must explicitly return the initial state. The initial state may not be undefined. If you don't want to set a value for this reducer, you can use null instead of undefined.`);
    if (typeof r(void 0, {
      type: Ki.PROBE_UNKNOWN_ACTION()
    }) > "u")
      throw new Error(process.env.NODE_ENV === "production" ? ar(13) : 'The slice reducer for key "' + e + '" returned undefined when probed with a random type. ' + ("Don't try to handle '" + Ki.INIT + `' or other actions in "redux/*" `) + "namespace. They are considered private. Instead, you must return the current state for any unknown actions, unless it is undefined, in which case you must return the initial state, regardless of the action type. The initial state may not be undefined, but can be null.");
  });
}
function Xv(t) {
  for (var e = Object.keys(t), r = {}, n = 0; n < e.length; n++) {
    var i = e[n];
    process.env.NODE_ENV !== "production" && typeof t[i] > "u" && xp('No reducer provided for key "' + i + '"'), typeof t[i] == "function" && (r[i] = t[i]);
  }
  var s = Object.keys(r), o;
  process.env.NODE_ENV !== "production" && (o = {});
  var l;
  try {
    Yx(r);
  } catch (c) {
    l = c;
  }
  return function(h, d) {
    if (h === void 0 && (h = {}), l)
      throw l;
    if (process.env.NODE_ENV !== "production") {
      var g = qx(h, r, d, o);
      g && xp(g);
    }
    for (var p = !1, y = {}, v = 0; v < s.length; v++) {
      var _ = s[v], b = r[_], C = h[_], w = b(C, d);
      if (typeof w > "u") {
        var O = d && d.type;
        throw new Error(process.env.NODE_ENV === "production" ? ar(14) : "When called with an action of type " + (O ? '"' + String(O) + '"' : "(unknown type)") + ', the slice reducer for key "' + _ + '" returned undefined. To ignore an action, you must explicitly return the previous state. If you want this reducer to hold no value, you can return null instead of undefined.');
      }
      y[_] = w, p = p || w !== C;
    }
    return p = p || s.length !== Object.keys(h).length, p ? y : h;
  };
}
function vl() {
  for (var t = arguments.length, e = new Array(t), r = 0; r < t; r++)
    e[r] = arguments[r];
  return e.length === 0 ? function(n) {
    return n;
  } : e.length === 1 ? e[0] : e.reduce(function(n, i) {
    return function() {
      return n(i.apply(void 0, arguments));
    };
  });
}
function Kx() {
  for (var t = arguments.length, e = new Array(t), r = 0; r < t; r++)
    e[r] = arguments[r];
  return function(n) {
    return function() {
      var i = n.apply(void 0, arguments), s = function() {
        throw new Error(process.env.NODE_ENV === "production" ? ar(15) : "Dispatching while constructing your middleware is not allowed. Other middleware would not be applied to this dispatch.");
      }, o = {
        getState: i.getState,
        dispatch: function() {
          return s.apply(void 0, arguments);
        }
      }, l = e.map(function(c) {
        return c(o);
      });
      return s = vl.apply(void 0, l)(i.dispatch), Ep(Ep({}, i), {}, {
        dispatch: s
      });
    };
  };
}
function Jv(t) {
  var e = function(n) {
    var i = n.dispatch, s = n.getState;
    return function(o) {
      return function(l) {
        return typeof l == "function" ? l(i, s, t) : o(l);
      };
    };
  };
  return e;
}
var Qv = Jv();
Qv.withExtraArgument = Jv;
const Op = Qv;
var Xx = globalThis && globalThis.__extends || function() {
  var t = function(e, r) {
    return t = Object.setPrototypeOf || { __proto__: [] } instanceof Array && function(n, i) {
      n.__proto__ = i;
    } || function(n, i) {
      for (var s in i)
        Object.prototype.hasOwnProperty.call(i, s) && (n[s] = i[s]);
    }, t(e, r);
  };
  return function(e, r) {
    if (typeof r != "function" && r !== null)
      throw new TypeError("Class extends value " + String(r) + " is not a constructor or null");
    t(e, r);
    function n() {
      this.constructor = e;
    }
    e.prototype = r === null ? Object.create(r) : (n.prototype = r.prototype, new n());
  };
}(), Jx = globalThis && globalThis.__generator || function(t, e) {
  var r = { label: 0, sent: function() {
    if (s[0] & 1)
      throw s[1];
    return s[1];
  }, trys: [], ops: [] }, n, i, s, o;
  return o = { next: l(0), throw: l(1), return: l(2) }, typeof Symbol == "function" && (o[Symbol.iterator] = function() {
    return this;
  }), o;
  function l(h) {
    return function(d) {
      return c([h, d]);
    };
  }
  function c(h) {
    if (n)
      throw new TypeError("Generator is already executing.");
    for (; r; )
      try {
        if (n = 1, i && (s = h[0] & 2 ? i.return : h[0] ? i.throw || ((s = i.return) && s.call(i), 0) : i.next) && !(s = s.call(i, h[1])).done)
          return s;
        switch (i = 0, s && (h = [h[0] & 2, s.value]), h[0]) {
          case 0:
          case 1:
            s = h;
            break;
          case 4:
            return r.label++, { value: h[1], done: !1 };
          case 5:
            r.label++, i = h[1], h = [0];
            continue;
          case 7:
            h = r.ops.pop(), r.trys.pop();
            continue;
          default:
            if (s = r.trys, !(s = s.length > 0 && s[s.length - 1]) && (h[0] === 6 || h[0] === 2)) {
              r = 0;
              continue;
            }
            if (h[0] === 3 && (!s || h[1] > s[0] && h[1] < s[3])) {
              r.label = h[1];
              break;
            }
            if (h[0] === 6 && r.label < s[1]) {
              r.label = s[1], s = h;
              break;
            }
            if (s && r.label < s[2]) {
              r.label = s[2], r.ops.push(h);
              break;
            }
            s[2] && r.ops.pop(), r.trys.pop();
            continue;
        }
        h = e.call(t, r);
      } catch (d) {
        h = [6, d], i = 0;
      } finally {
        n = s = 0;
      }
    if (h[0] & 5)
      throw h[1];
    return { value: h[0] ? h[1] : void 0, done: !0 };
  }
}, yl = globalThis && globalThis.__spreadArray || function(t, e) {
  for (var r = 0, n = e.length, i = t.length; r < n; r++, i++)
    t[i] = e[r];
  return t;
}, Qx = Object.defineProperty, Zx = Object.defineProperties, e2 = Object.getOwnPropertyDescriptors, Tp = Object.getOwnPropertySymbols, t2 = Object.prototype.hasOwnProperty, r2 = Object.prototype.propertyIsEnumerable, Pp = function(t, e, r) {
  return e in t ? Qx(t, e, { enumerable: !0, configurable: !0, writable: !0, value: r }) : t[e] = r;
}, wi = function(t, e) {
  for (var r in e || (e = {}))
    t2.call(e, r) && Pp(t, r, e[r]);
  if (Tp)
    for (var n = 0, i = Tp(e); n < i.length; n++) {
      var r = i[n];
      r2.call(e, r) && Pp(t, r, e[r]);
    }
  return t;
}, df = function(t, e) {
  return Zx(t, e2(e));
}, n2 = function(t, e, r) {
  return new Promise(function(n, i) {
    var s = function(c) {
      try {
        l(r.next(c));
      } catch (h) {
        i(h);
      }
    }, o = function(c) {
      try {
        l(r.throw(c));
      } catch (h) {
        i(h);
      }
    }, l = function(c) {
      return c.done ? n(c.value) : Promise.resolve(c.value).then(s, o);
    };
    l((r = r.apply(t, e)).next());
  });
}, i2 = typeof window < "u" && window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ ? window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ : function() {
  if (arguments.length !== 0)
    return typeof arguments[0] == "object" ? vl : vl.apply(null, arguments);
};
function Zv(t) {
  if (typeof t != "object" || t === null)
    return !1;
  var e = Object.getPrototypeOf(t);
  if (e === null)
    return !0;
  for (var r = e; Object.getPrototypeOf(r) !== null; )
    r = Object.getPrototypeOf(r);
  return e === r;
}
function ey(t, e) {
  var r = 0;
  return {
    measureTime: function(n) {
      var i = Date.now();
      try {
        return n();
      } finally {
        var s = Date.now();
        r += s - i;
      }
    },
    warnIfExceeded: function() {
      r > t && console.warn(e + " took " + r + "ms, which is more than the warning threshold of " + t + `ms. 
If your state or actions are very large, you may want to disable the middleware as it might cause too much of a slowdown in development mode. See https://redux-toolkit.js.org/api/getDefaultMiddleware for instructions.
It is disabled in production builds, so you don't need to worry about that.`);
    }
  };
}
var a2 = (
  /** @class */
  function(t) {
    Xx(e, t);
    function e() {
      for (var r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      var i = t.apply(this, r) || this;
      return Object.setPrototypeOf(i, e.prototype), i;
    }
    return Object.defineProperty(e, Symbol.species, {
      get: function() {
        return e;
      },
      enumerable: !1,
      configurable: !0
    }), e.prototype.concat = function() {
      for (var r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return t.prototype.concat.apply(this, r);
    }, e.prototype.prepend = function() {
      for (var r = [], n = 0; n < arguments.length; n++)
        r[n] = arguments[n];
      return r.length === 1 && Array.isArray(r[0]) ? new (e.bind.apply(e, yl([void 0], r[0].concat(this))))() : new (e.bind.apply(e, yl([void 0], r.concat(this))))();
    }, e;
  }(Array)
);
function $f(t) {
  return ei(t) ? qv(t, function() {
  }) : t;
}
var s2 = process.env.NODE_ENV === "production", Rp = "Invariant failed";
function Ap(t, e) {
  if (!t)
    throw s2 ? new Error(Rp) : new Error(Rp + ": " + (e || ""));
}
function o2(t, e, r, n) {
  return JSON.stringify(t, u2(e, n), r);
}
function u2(t, e) {
  var r = [], n = [];
  return e || (e = function(i, s) {
    return r[0] === s ? "[Circular ~]" : "[Circular ~." + n.slice(0, r.indexOf(s)).join(".") + "]";
  }), function(i, s) {
    if (r.length > 0) {
      var o = r.indexOf(this);
      ~o ? r.splice(o + 1) : r.push(this), ~o ? n.splice(o, 1 / 0, i) : n.push(i), ~r.indexOf(s) && (s = e.call(this, i, s));
    } else
      r.push(s);
    return t == null ? s : t.call(this, i, s);
  };
}
function l2(t) {
  return typeof t != "object" || t == null || Object.isFrozen(t);
}
function c2(t, e, r) {
  var n = ty(t, e, r);
  return {
    detectMutations: function() {
      return ry(t, e, n, r);
    }
  };
}
function ty(t, e, r, n) {
  e === void 0 && (e = []), n === void 0 && (n = "");
  var i = { value: r };
  if (!t(r)) {
    i.children = {};
    for (var s in r) {
      var o = n ? n + "." + s : s;
      e.length && e.indexOf(o) !== -1 || (i.children[s] = ty(t, e, r[s], o));
    }
  }
  return i;
}
function ry(t, e, r, n, i, s) {
  e === void 0 && (e = []), i === void 0 && (i = !1), s === void 0 && (s = "");
  var o = r ? r.value : void 0, l = o === n;
  if (i && !l && !Number.isNaN(n))
    return { wasMutated: !0, path: s };
  if (t(o) || t(n))
    return { wasMutated: !1 };
  var c = {};
  for (var h in r.children)
    c[h] = !0;
  for (var h in n)
    c[h] = !0;
  var d = e.length > 0, g = function(y) {
    var v = s ? s + "." + y : y;
    if (d) {
      var _ = e.some(function(C) {
        return C instanceof RegExp ? C.test(v) : v === C;
      });
      if (_)
        return "continue";
    }
    var b = ry(t, e, r.children[y], n[y], l, v);
    if (b.wasMutated)
      return { value: b };
  };
  for (var h in c) {
    var p = g(h);
    if (typeof p == "object")
      return p.value;
  }
  return { wasMutated: !1 };
}
function f2(t) {
  if (t === void 0 && (t = {}), process.env.NODE_ENV === "production")
    return function() {
      return function(c) {
        return function(h) {
          return c(h);
        };
      };
    };
  var e = t.isImmutable, r = e === void 0 ? l2 : e, n = t.ignoredPaths, i = t.warnAfter, s = i === void 0 ? 32 : i, o = t.ignore;
  n = n || o;
  var l = c2.bind(null, r, n);
  return function(c) {
    var h = c.getState, d = h(), g = l(d), p;
    return function(y) {
      return function(v) {
        var _ = ey(s, "ImmutableStateInvariantMiddleware");
        _.measureTime(function() {
          d = h(), p = g.detectMutations(), g = l(d), Ap(!p.wasMutated, "A state mutation was detected between dispatches, in the path '" + (p.path || "") + "'.  This may cause incorrect behavior. (https://redux.js.org/style-guide/style-guide#do-not-mutate-state)");
        });
        var b = y(v);
        return _.measureTime(function() {
          d = h(), p = g.detectMutations(), g = l(d), p.wasMutated && Ap(!p.wasMutated, "A state mutation was detected inside a dispatch, in the path: " + (p.path || "") + ". Take a look at the reducer(s) handling the action " + o2(v) + ". (https://redux.js.org/style-guide/style-guide#do-not-mutate-state)");
        }), _.warnIfExceeded(), b;
      };
    };
  };
}
function ny(t) {
  var e = typeof t;
  return t == null || e === "string" || e === "boolean" || e === "number" || Array.isArray(t) || Zv(t);
}
function Wf(t, e, r, n, i, s) {
  e === void 0 && (e = ""), r === void 0 && (r = ny), i === void 0 && (i = []);
  var o;
  if (!r(t))
    return {
      keyPath: e || "<root>",
      value: t
    };
  if (typeof t != "object" || t === null || s != null && s.has(t))
    return !1;
  for (var l = n != null ? n(t) : Object.entries(t), c = i.length > 0, h = function(b, C) {
    var w = e ? e + "." + b : b;
    if (c) {
      var O = i.some(function(k) {
        return k instanceof RegExp ? k.test(w) : w === k;
      });
      if (O)
        return "continue";
    }
    if (!r(C))
      return { value: {
        keyPath: w,
        value: C
      } };
    if (typeof C == "object" && (o = Wf(C, w, r, n, i, s), o))
      return { value: o };
  }, d = 0, g = l; d < g.length; d++) {
    var p = g[d], y = p[0], v = p[1], _ = h(y, v);
    if (typeof _ == "object")
      return _.value;
  }
  return s && iy(t) && s.add(t), !1;
}
function iy(t) {
  if (!Object.isFrozen(t))
    return !1;
  for (var e = 0, r = Object.values(t); e < r.length; e++) {
    var n = r[e];
    if (!(typeof n != "object" || n === null) && !iy(n))
      return !1;
  }
  return !0;
}
function h2(t) {
  if (t === void 0 && (t = {}), process.env.NODE_ENV === "production")
    return function() {
      return function(O) {
        return function(k) {
          return O(k);
        };
      };
    };
  var e = t.isSerializable, r = e === void 0 ? ny : e, n = t.getEntries, i = t.ignoredActions, s = i === void 0 ? [] : i, o = t.ignoredActionPaths, l = o === void 0 ? ["meta.arg", "meta.baseQueryMeta"] : o, c = t.ignoredPaths, h = c === void 0 ? [] : c, d = t.warnAfter, g = d === void 0 ? 32 : d, p = t.ignoreState, y = p === void 0 ? !1 : p, v = t.ignoreActions, _ = v === void 0 ? !1 : v, b = t.disableCache, C = b === void 0 ? !1 : b, w = !C && WeakSet ? /* @__PURE__ */ new WeakSet() : void 0;
  return function(O) {
    return function(k) {
      return function(P) {
        var M = k(P), U = ey(g, "SerializableStateInvariantMiddleware");
        return !_ && !(s.length && s.indexOf(P.type) !== -1) && U.measureTime(function() {
          var V = Wf(P, "", r, n, l, w);
          if (V) {
            var Y = V.keyPath, X = V.value;
            console.error("A non-serializable value was detected in an action, in the path: `" + Y + "`. Value:", X, `
Take a look at the logic that dispatched this action: `, P, `
(See https://redux.js.org/faq/actions#why-should-type-be-a-string-or-at-least-serializable-why-should-my-action-types-be-constants)`, `
(To allow non-serializable values see: https://redux-toolkit.js.org/usage/usage-guide#working-with-non-serializable-data)`);
          }
        }), y || (U.measureTime(function() {
          var V = O.getState(), Y = Wf(V, "", r, n, h, w);
          if (Y) {
            var X = Y.keyPath, Z = Y.value;
            console.error("A non-serializable value was detected in the state, in the path: `" + X + "`. Value:", Z, `
Take a look at the reducer(s) handling this action type: ` + P.type + `.
(See https://redux.js.org/faq/organizing-state#can-i-put-functions-promises-or-other-non-serializable-items-in-my-store-state)`);
          }
        }), U.warnIfExceeded()), M;
      };
    };
  };
}
function gf(t) {
  return typeof t == "boolean";
}
function d2() {
  return function(e) {
    return g2(e);
  };
}
function g2(t) {
  t === void 0 && (t = {});
  var e = t.thunk, r = e === void 0 ? !0 : e, n = t.immutableCheck, i = n === void 0 ? !0 : n, s = t.serializableCheck, o = s === void 0 ? !0 : s, l = new a2();
  if (r && (gf(r) ? l.push(Op) : l.push(Op.withExtraArgument(r.extraArgument))), process.env.NODE_ENV !== "production") {
    if (i) {
      var c = {};
      gf(i) || (c = i), l.unshift(f2(c));
    }
    if (o) {
      var h = {};
      gf(o) || (h = o), l.push(h2(h));
    }
  }
  return l;
}
var pf = process.env.NODE_ENV === "production";
function p2(t) {
  var e = d2(), r = t || {}, n = r.reducer, i = n === void 0 ? void 0 : n, s = r.middleware, o = s === void 0 ? e() : s, l = r.devTools, c = l === void 0 ? !0 : l, h = r.preloadedState, d = h === void 0 ? void 0 : h, g = r.enhancers, p = g === void 0 ? void 0 : g, y;
  if (typeof i == "function")
    y = i;
  else if (Zv(i))
    y = Xv(i);
  else
    throw new Error('"reducer" is a required argument, and must be a function or an object of functions that can be passed to combineReducers');
  var v = o;
  if (typeof v == "function" && (v = v(e), !pf && !Array.isArray(v)))
    throw new Error("when using a middleware builder function, an array of middleware must be returned");
  if (!pf && v.some(function(O) {
    return typeof O != "function";
  }))
    throw new Error("each middleware provided to configureStore must be a function");
  var _ = Kx.apply(void 0, v), b = vl;
  c && (b = i2(wi({
    trace: !pf
  }, typeof c == "object" && c)));
  var C = [_];
  Array.isArray(p) ? C = yl([_], p) : typeof p == "function" && (C = p(C));
  var w = b.apply(void 0, C);
  return Ch(y, d, w);
}
function vt(t, e) {
  function r() {
    for (var n = [], i = 0; i < arguments.length; i++)
      n[i] = arguments[i];
    if (e) {
      var s = e.apply(void 0, n);
      if (!s)
        throw new Error("prepareAction did not return an object");
      return wi(wi({
        type: t,
        payload: s.payload
      }, "meta" in s && { meta: s.meta }), "error" in s && { error: s.error });
    }
    return { type: t, payload: n[0] };
  }
  return r.toString = function() {
    return "" + t;
  }, r.type = t, r.match = function(n) {
    return n.type === t;
  }, r;
}
function ay(t) {
  var e = {}, r = [], n, i = {
    addCase: function(s, o) {
      if (process.env.NODE_ENV !== "production") {
        if (r.length > 0)
          throw new Error("`builder.addCase` should only be called before calling `builder.addMatcher`");
        if (n)
          throw new Error("`builder.addCase` should only be called before calling `builder.addDefaultCase`");
      }
      var l = typeof s == "string" ? s : s.type;
      if (l in e)
        throw new Error("addCase cannot be called with two reducers for the same action type");
      return e[l] = o, i;
    },
    addMatcher: function(s, o) {
      if (process.env.NODE_ENV !== "production" && n)
        throw new Error("`builder.addMatcher` should only be called before calling `builder.addDefaultCase`");
      return r.push({ matcher: s, reducer: o }), i;
    },
    addDefaultCase: function(s) {
      if (process.env.NODE_ENV !== "production" && n)
        throw new Error("`builder.addDefaultCase` can only be called once");
      return n = s, i;
    }
  };
  return t(i), [e, r, n];
}
function m2(t) {
  return typeof t == "function";
}
var kp = !1;
function v2(t, e, r, n) {
  r === void 0 && (r = []), process.env.NODE_ENV !== "production" && typeof e == "object" && (kp || (kp = !0, console.warn("The object notation for `createReducer` is deprecated, and will be removed in RTK 2.0. Please use the 'builder callback' notation instead: https://redux-toolkit.js.org/api/createReducer")));
  var i = typeof e == "function" ? ay(e) : [e, r, n], s = i[0], o = i[1], l = i[2], c;
  if (m2(t))
    c = function() {
      return $f(t());
    };
  else {
    var h = $f(t);
    c = function() {
      return h;
    };
  }
  function d(g, p) {
    g === void 0 && (g = c());
    var y = yl([
      s[p.type]
    ], o.filter(function(v) {
      var _ = v.matcher;
      return _(p);
    }).map(function(v) {
      var _ = v.reducer;
      return _;
    }));
    return y.filter(function(v) {
      return !!v;
    }).length === 0 && (y = [l]), y.reduce(function(v, _) {
      if (_)
        if (Ci(v)) {
          var b = v, C = _(b, p);
          return C === void 0 ? v : C;
        } else {
          if (ei(v))
            return qv(v, function(w) {
              return _(w, p);
            });
          var C = _(v, p);
          if (C === void 0) {
            if (v === null)
              return v;
            throw Error("A case reducer on a non-draftable value must not return undefined");
          }
          return C;
        }
      return v;
    }, g);
  }
  return d.getInitialState = c, d;
}
var Ip = !1;
function y2(t, e) {
  return t + "/" + e;
}
function Qi(t) {
  var e = t.name;
  if (!e)
    throw new Error("`name` is a required option for createSlice");
  typeof process < "u" && process.env.NODE_ENV === "development" && t.initialState === void 0 && console.error("You must provide an `initialState` value that is not `undefined`. You may have misspelled `initialState`");
  var r = typeof t.initialState == "function" ? t.initialState : $f(t.initialState), n = t.reducers || {}, i = Object.keys(n), s = {}, o = {}, l = {};
  i.forEach(function(d) {
    var g = n[d], p = y2(e, d), y, v;
    "reducer" in g ? (y = g.reducer, v = g.prepare) : y = g, s[d] = y, o[p] = y, l[d] = v ? vt(p, v) : vt(p);
  });
  function c() {
    process.env.NODE_ENV !== "production" && typeof t.extraReducers == "object" && (Ip || (Ip = !0, console.warn("The object notation for `createSlice.extraReducers` is deprecated, and will be removed in RTK 2.0. Please use the 'builder callback' notation instead: https://redux-toolkit.js.org/api/createSlice")));
    var d = typeof t.extraReducers == "function" ? ay(t.extraReducers) : [t.extraReducers], g = d[0], p = g === void 0 ? {} : g, y = d[1], v = y === void 0 ? [] : y, _ = d[2], b = _ === void 0 ? void 0 : _, C = wi(wi({}, p), o);
    return v2(r, function(w) {
      for (var O in C)
        w.addCase(O, C[O]);
      for (var k = 0, P = v; k < P.length; k++) {
        var M = P[k];
        w.addMatcher(M.matcher, M.reducer);
      }
      b && w.addDefaultCase(b);
    });
  }
  var h;
  return {
    name: e,
    reducer: function(d, g) {
      return h || (h = c()), h(d, g);
    },
    actions: l,
    caseReducers: s,
    getInitialState: function() {
      return h || (h = c()), h.getInitialState();
    }
  };
}
var b2 = "ModuleSymbhasOwnPr-0123456789ABCDEFGHNRVfgctiUvz_KqYTJkLxpZXIjQW", _2 = function(t) {
  t === void 0 && (t = 21);
  for (var e = "", r = t; r--; )
    e += b2[Math.random() * 64 | 0];
  return e;
}, S2 = [
  "name",
  "message",
  "stack",
  "code"
], mf = (
  /** @class */
  function() {
    function t(e, r) {
      this.payload = e, this.meta = r;
    }
    return t;
  }()
), Lp = (
  /** @class */
  function() {
    function t(e, r) {
      this.payload = e, this.meta = r;
    }
    return t;
  }()
), w2 = function(t) {
  if (typeof t == "object" && t !== null) {
    for (var e = {}, r = 0, n = S2; r < n.length; r++) {
      var i = n[r];
      typeof t[i] == "string" && (e[i] = t[i]);
    }
    return e;
  }
  return { message: String(t) };
};
(function() {
  function t(e, r, n) {
    var i = vt(e + "/fulfilled", function(d, g, p, y) {
      return {
        payload: d,
        meta: df(wi({}, y || {}), {
          arg: p,
          requestId: g,
          requestStatus: "fulfilled"
        })
      };
    }), s = vt(e + "/pending", function(d, g, p) {
      return {
        payload: void 0,
        meta: df(wi({}, p || {}), {
          arg: g,
          requestId: d,
          requestStatus: "pending"
        })
      };
    }), o = vt(e + "/rejected", function(d, g, p, y, v) {
      return {
        payload: y,
        error: (n && n.serializeError || w2)(d || "Rejected"),
        meta: df(wi({}, v || {}), {
          arg: p,
          requestId: g,
          rejectedWithValue: !!y,
          requestStatus: "rejected",
          aborted: (d == null ? void 0 : d.name) === "AbortError",
          condition: (d == null ? void 0 : d.name) === "ConditionError"
        })
      };
    }), l = !1, c = typeof AbortController < "u" ? AbortController : (
      /** @class */
      function() {
        function d() {
          this.signal = {
            aborted: !1,
            addEventListener: function() {
            },
            dispatchEvent: function() {
              return !1;
            },
            onabort: function() {
            },
            removeEventListener: function() {
            },
            reason: void 0,
            throwIfAborted: function() {
            }
          };
        }
        return d.prototype.abort = function() {
          process.env.NODE_ENV !== "production" && (l || (l = !0, console.info("This platform does not implement AbortController. \nIf you want to use the AbortController to react to `abort` events, please consider importing a polyfill like 'abortcontroller-polyfill/dist/abortcontroller-polyfill-only'.")));
        }, d;
      }()
    );
    function h(d) {
      return function(g, p, y) {
        var v = n != null && n.idGenerator ? n.idGenerator(d) : _2(), _ = new c(), b;
        function C(O) {
          b = O, _.abort();
        }
        var w = function() {
          return n2(this, null, function() {
            var O, k, P, M, U, V, Y;
            return Jx(this, function(X) {
              switch (X.label) {
                case 0:
                  return X.trys.push([0, 4, , 5]), M = (O = n == null ? void 0 : n.condition) == null ? void 0 : O.call(n, d, { getState: p, extra: y }), C2(M) ? [4, M] : [3, 2];
                case 1:
                  M = X.sent(), X.label = 2;
                case 2:
                  if (M === !1 || _.signal.aborted)
                    throw {
                      name: "ConditionError",
                      message: "Aborted due to condition callback returning false."
                    };
                  return U = new Promise(function(Z, ce) {
                    return _.signal.addEventListener("abort", function() {
                      return ce({
                        name: "AbortError",
                        message: b || "Aborted"
                      });
                    });
                  }), g(s(v, d, (k = n == null ? void 0 : n.getPendingMeta) == null ? void 0 : k.call(n, { requestId: v, arg: d }, { getState: p, extra: y }))), [4, Promise.race([
                    U,
                    Promise.resolve(r(d, {
                      dispatch: g,
                      getState: p,
                      extra: y,
                      requestId: v,
                      signal: _.signal,
                      abort: C,
                      rejectWithValue: function(Z, ce) {
                        return new mf(Z, ce);
                      },
                      fulfillWithValue: function(Z, ce) {
                        return new Lp(Z, ce);
                      }
                    })).then(function(Z) {
                      if (Z instanceof mf)
                        throw Z;
                      return Z instanceof Lp ? i(Z.payload, v, d, Z.meta) : i(Z, v, d);
                    })
                  ])];
                case 3:
                  return P = X.sent(), [3, 5];
                case 4:
                  return V = X.sent(), P = V instanceof mf ? o(null, v, d, V.payload, V.meta) : o(V, v, d), [3, 5];
                case 5:
                  return Y = n && !n.dispatchConditionRejection && o.match(P) && P.meta.condition, Y || g(P), [2, P];
              }
            });
          });
        }();
        return Object.assign(w, {
          abort: C,
          requestId: v,
          arg: d,
          unwrap: function() {
            return w.then(E2);
          }
        });
      };
    }
    return Object.assign(h, {
      pending: s,
      rejected: o,
      fulfilled: i,
      typePrefix: e
    });
  }
  return t.withTypes = function() {
    return t;
  }, t;
})();
function E2(t) {
  if (t.meta && t.meta.rejectedWithValue)
    throw t.payload;
  if (t.error)
    throw t.error;
  return t.payload;
}
function C2(t) {
  return t !== null && typeof t == "object" && typeof t.then == "function";
}
var xh = "listenerMiddleware";
vt(xh + "/add");
vt(xh + "/removeAll");
vt(xh + "/remove");
var Dp;
typeof queueMicrotask == "function" && queueMicrotask.bind(typeof window < "u" ? window : typeof global < "u" ? global : globalThis);
Nx();
var Oh = "persist:", sy = "persist/FLUSH", Th = "persist/REHYDRATE", oy = "persist/PAUSE", uy = "persist/PERSIST", ly = "persist/PURGE", cy = "persist/REGISTER", x2 = -1;
function eo(t) {
  return typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? eo = function(r) {
    return typeof r;
  } : eo = function(r) {
    return r && typeof Symbol == "function" && r.constructor === Symbol && r !== Symbol.prototype ? "symbol" : typeof r;
  }, eo(t);
}
function Mp(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function O2(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Mp(r, !0).forEach(function(n) {
      T2(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Mp(r).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function T2(t, e, r) {
  return e in t ? Object.defineProperty(t, e, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = r, t;
}
function P2(t, e, r, n) {
  var i = n.debug, s = O2({}, r);
  return t && eo(t) === "object" && Object.keys(t).forEach(function(o) {
    if (o !== "_persist") {
      if (e[o] !== r[o]) {
        process.env.NODE_ENV !== "production" && i && console.log("redux-persist/stateReconciler: sub state for key `%s` modified, skipping.", o);
        return;
      }
      s[o] = t[o];
    }
  }), process.env.NODE_ENV !== "production" && i && t && eo(t) === "object" && console.log("redux-persist/stateReconciler: rehydrated keys '".concat(Object.keys(t).join(", "), "'")), s;
}
function R2(t) {
  var e = t.blacklist || null, r = t.whitelist || null, n = t.transforms || [], i = t.throttle || 0, s = t.debounce || 0, o = "".concat(t.keyPrefix !== void 0 ? t.keyPrefix : Oh).concat(t.key), l = t.storage, c;
  t.serialize === !1 ? c = function(U) {
    return U;
  } : typeof t.serialize == "function" ? c = t.serialize : c = A2;
  var h = t.writeFailHandler || null, d = {}, g = {}, p = [], y = null, v = null;
  let _ = 0;
  function b(M) {
    clearTimeout(_), _ = setTimeout(() => {
      Object.keys(M).forEach(function(U) {
        O(U) && d[U] !== M[U] && p.indexOf(U) === -1 && p.push(U);
      }), Object.keys(d).forEach(function(U) {
        M[U] === void 0 && O(U) && p.indexOf(U) === -1 && d[U] !== void 0 && p.push(U);
      }), y === null && (y = setInterval(C, i)), d = M;
    }, s);
  }
  function C() {
    if (p.length === 0) {
      y && clearInterval(y), y = null;
      return;
    }
    var M = p.shift(), U = n.reduce(function(V, Y) {
      return Y.in(V, M, d);
    }, d[M]);
    if (U !== void 0)
      try {
        g[M] = c(U);
      } catch (V) {
        console.error("redux-persist/createPersistoid: error serializing state", V);
      }
    else
      delete g[M];
    p.length === 0 && w();
  }
  function w() {
    Object.keys(g).forEach(function(M) {
      d[M] === void 0 && delete g[M];
    }), v = l.setItem(o, c(g)).catch(k);
  }
  function O(M) {
    return !(r && r.indexOf(M) === -1 && M !== "_persist" || e && e.indexOf(M) !== -1);
  }
  function k(M) {
    h && h(M), M && process.env.NODE_ENV !== "production" && console.error("Error storing data", M);
  }
  var P = function() {
    for (; p.length !== 0; )
      C();
    return v || Promise.resolve();
  };
  return {
    update: b,
    flush: P
  };
}
function A2(t) {
  return JSON.stringify(t);
}
function k2(t) {
  var e = t.transforms || [], r = "".concat(t.keyPrefix !== void 0 ? t.keyPrefix : Oh).concat(t.key), n = t.storage, i = t.debug, s;
  return t.deserialize === !1 ? s = function(l) {
    return l;
  } : typeof t.deserialize == "function" ? s = t.deserialize : s = I2, n.getItem(r).then(function(o) {
    if (o)
      try {
        var l = {}, c = s(o);
        return Object.keys(c).forEach(function(h) {
          l[h] = e.reduceRight(function(d, g) {
            return g.out(d, h, c);
          }, s(c[h]));
        }), l;
      } catch (h) {
        throw process.env.NODE_ENV !== "production" && i && console.log("redux-persist/getStoredState: Error restoring data ".concat(o), h), h;
      }
    else
      return;
  });
}
function I2(t) {
  return JSON.parse(t);
}
function L2(t) {
  var e = t.storage, r = "".concat(t.keyPrefix !== void 0 ? t.keyPrefix : Oh).concat(t.key);
  return e.removeItem(r, D2);
}
function D2(t) {
  t && process.env.NODE_ENV !== "production" && console.error("redux-persist/purgeStoredState: Error purging data stored state", t);
}
function Np(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Yn(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Np(r, !0).forEach(function(n) {
      M2(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Np(r).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function M2(t, e, r) {
  return e in t ? Object.defineProperty(t, e, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = r, t;
}
function N2(t, e) {
  if (t == null)
    return {};
  var r = B2(t, e), n, i;
  if (Object.getOwnPropertySymbols) {
    var s = Object.getOwnPropertySymbols(t);
    for (i = 0; i < s.length; i++)
      n = s[i], !(e.indexOf(n) >= 0) && Object.prototype.propertyIsEnumerable.call(t, n) && (r[n] = t[n]);
  }
  return r;
}
function B2(t, e) {
  if (t == null)
    return {};
  var r = {}, n = Object.keys(t), i, s;
  for (s = 0; s < n.length; s++)
    i = n[s], !(e.indexOf(i) >= 0) && (r[i] = t[i]);
  return r;
}
var F2 = 5e3;
function G2(t, e) {
  if (process.env.NODE_ENV !== "production") {
    if (!t)
      throw new Error("config is required for persistReducer");
    if (!t.key)
      throw new Error("key is required in persistor config");
    if (!t.storage)
      throw new Error("redux-persist: config.storage is required. Try using one of the provided storage engines `import storage from 'redux-persist/lib/storage'`");
  }
  var r = t.version !== void 0 ? t.version : x2;
  t.debug;
  var n = t.stateReconciler === void 0 ? P2 : t.stateReconciler, i = t.getStoredState || k2, s = t.timeout !== void 0 ? t.timeout : F2, o = null, l = !1, c = !0, h = function(g) {
    return g._persist.rehydrated && o && !c && o.update(g), g;
  };
  return function(d, g) {
    var p = d || {}, y = p._persist, v = N2(p, ["_persist"]), _ = v;
    if (g.type === uy) {
      var b = !1, C = function(V, Y) {
        process.env.NODE_ENV !== "production" && b && console.error('redux-persist: rehydrate for "'.concat(t.key, '" called after timeout.'), V, Y), b || (g.rehydrate(t.key, V, Y), b = !0);
      };
      if (s && setTimeout(function() {
        !b && C(void 0, new Error('redux-persist: persist timed out for persist key "'.concat(t.key, '"')));
      }, s), c = !1, o || (o = R2(t)), y)
        return Yn({}, e(_, g), {
          _persist: y
        });
      if (typeof g.rehydrate != "function" || typeof g.register != "function")
        throw new Error("redux-persist: either rehydrate or register is not a function on the PERSIST action. This can happen if the action is being replayed. This is an unexplored use case, please open an issue and we will figure out a resolution.");
      return g.register(t.key), i(t).then(function(U) {
        var V = t.migrate || function(Y, X) {
          return Promise.resolve(Y);
        };
        V(U, r).then(function(Y) {
          C(Y);
        }, function(Y) {
          process.env.NODE_ENV !== "production" && Y && console.error("redux-persist: migration error", Y), C(void 0, Y);
        });
      }, function(U) {
        C(void 0, U);
      }), Yn({}, e(_, g), {
        _persist: {
          version: r,
          rehydrated: !1
        }
      });
    } else {
      if (g.type === ly)
        return l = !0, g.result(L2(t)), Yn({}, e(_, g), {
          _persist: y
        });
      if (g.type === sy)
        return g.result(o && o.flush()), Yn({}, e(_, g), {
          _persist: y
        });
      if (g.type === oy)
        c = !0;
      else if (g.type === Th) {
        if (l)
          return Yn({}, _, {
            _persist: Yn({}, y, {
              rehydrated: !0
            })
            // @NOTE if key does not match, will continue to default else below
          });
        if (g.key === t.key) {
          var w = e(_, g), O = g.payload, k = n !== !1 && O !== void 0 ? n(O, d, w, t) : w, P = Yn({}, k, {
            _persist: Yn({}, y, {
              rehydrated: !0
            })
          });
          return h(P);
        }
      }
    }
    if (!y)
      return e(d, g);
    var M = e(_, g);
    return M === _ ? d : h(Yn({}, M, {
      _persist: y
    }));
  };
}
function Bp(t) {
  return W2(t) || $2(t) || j2();
}
function j2() {
  throw new TypeError("Invalid attempt to spread non-iterable instance");
}
function $2(t) {
  if (Symbol.iterator in Object(t) || Object.prototype.toString.call(t) === "[object Arguments]")
    return Array.from(t);
}
function W2(t) {
  if (Array.isArray(t)) {
    for (var e = 0, r = new Array(t.length); e < t.length; e++)
      r[e] = t[e];
    return r;
  }
}
function Fp(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Uf(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Fp(r, !0).forEach(function(n) {
      U2(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Fp(r).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function U2(t, e, r) {
  return e in t ? Object.defineProperty(t, e, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = r, t;
}
var fy = {
  registry: [],
  bootstrapped: !1
}, H2 = function() {
  var e = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : fy, r = arguments.length > 1 ? arguments[1] : void 0;
  switch (r.type) {
    case cy:
      return Uf({}, e, {
        registry: [].concat(Bp(e.registry), [r.key])
      });
    case Th:
      var n = e.registry.indexOf(r.key), i = Bp(e.registry);
      return i.splice(n, 1), Uf({}, e, {
        registry: i,
        bootstrapped: i.length === 0
      });
    default:
      return e;
  }
};
function z2(t, e, r) {
  if (process.env.NODE_ENV !== "production") {
    var n = e || {}, i = ["blacklist", "whitelist", "transforms", "storage", "keyPrefix", "migrate"];
    i.forEach(function(d) {
      n[d] && console.error('redux-persist: invalid option passed to persistStore: "'.concat(d, '". You may be incorrectly passing persistConfig into persistStore, whereas it should be passed into persistReducer.'));
    });
  }
  var s = r || !1, o = Ch(H2, fy, e && e.enhancer ? e.enhancer : void 0), l = function(g) {
    o.dispatch({
      type: cy,
      key: g
    });
  }, c = function(g, p, y) {
    var v = {
      type: Th,
      payload: p,
      err: y,
      key: g
      // dispatch to `store` to rehydrate and `persistor` to track result
    };
    t.dispatch(v), o.dispatch(v), s && h.getState().bootstrapped && (s(), s = !1);
  }, h = Uf({}, o, {
    purge: function() {
      var g = [];
      return t.dispatch({
        type: ly,
        result: function(y) {
          g.push(y);
        }
      }), Promise.all(g);
    },
    flush: function() {
      var g = [];
      return t.dispatch({
        type: sy,
        result: function(y) {
          g.push(y);
        }
      }), Promise.all(g);
    },
    pause: function() {
      t.dispatch({
        type: oy
      });
    },
    persist: function() {
      t.dispatch({
        type: uy,
        register: l,
        rehydrate: c
      });
    }
  });
  return e && e.manualPersist || h.persist(), h;
}
var Ph = {}, Rh = {};
Rh.__esModule = !0;
Rh.default = Y2;
function Qu(t) {
  return typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? Qu = function(r) {
    return typeof r;
  } : Qu = function(r) {
    return r && typeof Symbol == "function" && r.constructor === Symbol && r !== Symbol.prototype ? "symbol" : typeof r;
  }, Qu(t);
}
function vf() {
}
var V2 = {
  getItem: vf,
  setItem: vf,
  removeItem: vf
};
function q2(t) {
  if ((typeof self > "u" ? "undefined" : Qu(self)) !== "object" || !(t in self))
    return !1;
  try {
    var e = self[t], r = "redux-persist ".concat(t, " test");
    e.setItem(r, "test"), e.getItem(r), e.removeItem(r);
  } catch {
    return process.env.NODE_ENV !== "production" && console.warn("redux-persist ".concat(t, " test failed, persistence will be disabled.")), !1;
  }
  return !0;
}
function Y2(t) {
  var e = "".concat(t, "Storage");
  return q2(e) ? self[e] : (process.env.NODE_ENV !== "production" && console.error("redux-persist failed to create sync storage. falling back to noop storage."), V2);
}
Ph.__esModule = !0;
Ph.default = J2;
var K2 = X2(Rh);
function X2(t) {
  return t && t.__esModule ? t : { default: t };
}
function J2(t) {
  var e = (0, K2.default)(t);
  return {
    getItem: function(n) {
      return new Promise(function(i, s) {
        i(e.getItem(n));
      });
    },
    setItem: function(n, i) {
      return new Promise(function(s, o) {
        s(e.setItem(n, i));
      });
    },
    removeItem: function(n) {
      return new Promise(function(i, s) {
        i(e.removeItem(n));
      });
    }
  };
}
var hy = void 0, Q2 = Z2(Ph);
function Z2(t) {
  return t && t.__esModule ? t : { default: t };
}
var eO = (0, Q2.default)("local");
hy = eO;
var dy = {}, gy = {}, Ji = {};
Object.defineProperty(Ji, "__esModule", { value: !0 });
Ji.PLACEHOLDER_UNDEFINED = Ji.PACKAGE_NAME = void 0;
Ji.PACKAGE_NAME = "redux-deep-persist";
Ji.PLACEHOLDER_UNDEFINED = "@@placeholder/undefined";
var Ah = {};
(function(t) {
  Object.defineProperty(t, "__esModule", { value: !0 }), t.ConfigType = void 0, function(e) {
    e[e.WHITELIST = 0] = "WHITELIST", e[e.BLACKLIST = 1] = "BLACKLIST";
  }(t.ConfigType || (t.ConfigType = {}));
})(Ah);
(function(t) {
  Object.defineProperty(t, "__esModule", { value: !0 }), t.getRootKeysGroup = t.throwError = t.configValidator = t.transformsValidator = t.singleTransformValidator = t.findDuplicatesAndSubsets = t.unique = t.preserveUndefined = t.mergeDeep = t.dissocPath = t.assocPath = t.path = t.difference = t.cloneDeep = t._cloneDeep = t.getCircularPath = t.isEmpty = t.isDate = t.isString = t.isIntegerString = t.isPlainObject = t.isArray = t.isLength = t.isObjectLike = void 0;
  const e = Ji, r = Ah, n = function($) {
    return typeof $ == "object" && $ !== null;
  };
  t.isObjectLike = n;
  const i = function($) {
    return typeof $ == "number" && $ > -1 && $ % 1 == 0 && $ <= Number.MAX_SAFE_INTEGER;
  };
  t.isLength = i, t.isArray = Array.isArray || function($) {
    return (0, t.isLength)($ && $.length) && Object.prototype.toString.call($) === "[object Array]";
  };
  const s = function($) {
    return !!$ && typeof $ == "object" && !(0, t.isArray)($);
  };
  t.isPlainObject = s;
  const o = function($) {
    return String(~~$) === $ && Number($) >= 0;
  };
  t.isIntegerString = o;
  const l = function($) {
    return Object.prototype.toString.call($) === "[object String]";
  };
  t.isString = l;
  const c = function($) {
    return Object.prototype.toString.call($) === "[object Date]";
  };
  t.isDate = c;
  const h = function($) {
    return Object.keys($).length === 0;
  };
  t.isEmpty = h;
  const d = Object.prototype.hasOwnProperty, g = function($, R, F) {
    F || (F = /* @__PURE__ */ new Set([$])), R || (R = "");
    for (const H in $) {
      const re = R ? `${R}.${H}` : H, ae = $[H];
      if ((0, t.isObjectLike)(ae))
        return F.has(ae) ? `${R}.${H}:<Circular>` : (F.add(ae), (0, t.getCircularPath)(ae, re, F));
    }
    return null;
  };
  t.getCircularPath = g;
  const p = function($) {
    if (!(0, t.isObjectLike)($))
      return $;
    if ((0, t.isDate)($))
      return new Date(+$);
    const R = (0, t.isArray)($) ? [] : {};
    for (const F in $) {
      const H = $[F];
      R[F] = (0, t._cloneDeep)(H);
    }
    return R;
  };
  t._cloneDeep = p;
  const y = function($) {
    const R = (0, t.getCircularPath)($);
    if (R)
      throw new Error(`${e.PACKAGE_NAME}: circular dependency detected under the path '${R}' of object you're trying to persist: ${$}`);
    return (0, t._cloneDeep)($);
  };
  t.cloneDeep = y;
  const v = function($, R) {
    if ($ === R)
      return {};
    if (!(0, t.isObjectLike)($) || !(0, t.isObjectLike)(R))
      return R;
    const F = (0, t.cloneDeep)($), H = (0, t.cloneDeep)(R), re = Object.keys(F).reduce((ne, oe) => (d.call(H, oe) || (ne[oe] = void 0), ne), {});
    if ((0, t.isDate)(F) || (0, t.isDate)(H))
      return F.valueOf() === H.valueOf() ? {} : H;
    const ae = Object.keys(H).reduce((ne, oe) => {
      if (!d.call(F, oe))
        return ne[oe] = H[oe], ne;
      const de = (0, t.difference)(F[oe], H[oe]);
      return (0, t.isObjectLike)(de) && (0, t.isEmpty)(de) && !(0, t.isDate)(de) ? (0, t.isArray)(F) && !(0, t.isArray)(H) || !(0, t.isArray)(F) && (0, t.isArray)(H) ? H : ne : (ne[oe] = de, ne);
    }, re);
    return delete ae._persist, ae;
  };
  t.difference = v;
  const _ = function($, R) {
    return R.reduce((F, H) => {
      if (F) {
        const re = parseInt(H, 10), ae = (0, t.isIntegerString)(H) && re < 0 ? F.length + re : H;
        return (0, t.isString)(F) ? F.charAt(ae) : F[ae];
      }
    }, $);
  };
  t.path = _;
  const b = function($, R) {
    return [...$].reverse().reduce((re, ae, ne) => {
      const oe = (0, t.isIntegerString)(ae) ? [] : {};
      return oe[ae] = ne === 0 ? R : re, oe;
    }, {});
  };
  t.assocPath = b;
  const C = function($, R) {
    const F = (0, t.cloneDeep)($);
    return R.reduce((H, re, ae) => (ae === R.length - 1 && H && (0, t.isObjectLike)(H) && delete H[re], H && H[re]), F), F;
  };
  t.dissocPath = C;
  const w = function($, R, ...F) {
    if (!F || !F.length)
      return R;
    const H = F.shift(), { preservePlaceholder: re, preserveUndefined: ae } = $;
    if ((0, t.isObjectLike)(R) && (0, t.isObjectLike)(H))
      for (const ne in H)
        if ((0, t.isObjectLike)(H[ne]) && (0, t.isObjectLike)(R[ne]))
          R[ne] || (R[ne] = {}), w($, R[ne], H[ne]);
        else if ((0, t.isArray)(R)) {
          let oe = H[ne];
          const de = re ? e.PLACEHOLDER_UNDEFINED : void 0;
          ae || (oe = typeof oe < "u" ? oe : R[parseInt(ne, 10)]), oe = oe !== e.PLACEHOLDER_UNDEFINED ? oe : de, R[parseInt(ne, 10)] = oe;
        } else {
          const oe = H[ne] !== e.PLACEHOLDER_UNDEFINED ? H[ne] : void 0;
          R[ne] = oe;
        }
    return w($, R, ...F);
  }, O = function($, R, F) {
    return w({
      preservePlaceholder: F == null ? void 0 : F.preservePlaceholder,
      preserveUndefined: F == null ? void 0 : F.preserveUndefined
    }, (0, t.cloneDeep)($), (0, t.cloneDeep)(R));
  };
  t.mergeDeep = O;
  const k = function($, R = [], F, H, re) {
    if (!(0, t.isObjectLike)($))
      return $;
    for (const ae in $) {
      const ne = $[ae], oe = (0, t.isArray)($), de = H ? H + "." + ae : ae;
      ne === null && (F === r.ConfigType.WHITELIST && R.indexOf(de) === -1 || F === r.ConfigType.BLACKLIST && R.indexOf(de) !== -1) && oe && ($[parseInt(ae, 10)] = void 0), ne === void 0 && re && F === r.ConfigType.BLACKLIST && R.indexOf(de) === -1 && oe && ($[parseInt(ae, 10)] = e.PLACEHOLDER_UNDEFINED), k(ne, R, F, de, re);
    }
  }, P = function($, R, F, H) {
    const re = (0, t.cloneDeep)($);
    return k(re, R, F, "", H), re;
  };
  t.preserveUndefined = P;
  const M = function($, R, F) {
    return F.indexOf($) === R;
  };
  t.unique = M;
  const U = function($) {
    return $.reduce((R, F) => {
      const H = $.filter((Re) => Re === F), re = $.filter((Re) => (F + ".").indexOf(Re + ".") === 0), { duplicates: ae, subsets: ne } = R, oe = H.length > 1 && ae.indexOf(F) === -1, de = re.length > 1;
      return {
        duplicates: [...ae, ...oe ? H : []],
        subsets: [...ne, ...de ? re : []].filter(t.unique).sort()
      };
    }, {
      duplicates: [],
      subsets: []
    });
  };
  t.findDuplicatesAndSubsets = U;
  const V = function($, R, F) {
    const H = F === r.ConfigType.WHITELIST ? "whitelist" : "blacklist", re = `${e.PACKAGE_NAME}: incorrect ${H} configuration.`, ae = `Check your create${F === r.ConfigType.WHITELIST ? "White" : "Black"}list arguments.

`;
    if (!(0, t.isString)(R) || R.length < 1)
      throw new Error(`${re} Name (key) of reducer is required. ${ae}`);
    if (!$ || !$.length)
      return;
    const { duplicates: ne, subsets: oe } = (0, t.findDuplicatesAndSubsets)($);
    if (ne.length > 1)
      throw new Error(`${re} Duplicated paths.

 ${JSON.stringify(ne)}

 ${ae}`);
    if (oe.length > 1)
      throw new Error(`${re} You are trying to persist an entire property and also some of its subset.

${JSON.stringify(oe)}

 ${ae}`);
  };
  t.singleTransformValidator = V;
  const Y = function($) {
    if (!(0, t.isArray)($))
      return;
    const R = ($ == null ? void 0 : $.map((F) => F.deepPersistKey).filter((F) => F)) || [];
    if (R.length) {
      const F = R.filter((H, re) => R.indexOf(H) !== re);
      if (F.length)
        throw new Error(`${e.PACKAGE_NAME}: found duplicated keys in transforms creators. You can createWhitelist or createBlacklist for a specific root reducer key only once. Duplicated keys among createWhitelist and createBlacklist transforms are not allowed.

 Duplicates: ${JSON.stringify(F)}`);
    }
  };
  t.transformsValidator = Y;
  const X = function({ whitelist: $, blacklist: R }) {
    if ($ && $.length && R && R.length)
      throw new Error(`${e.PACKAGE_NAME}: you should not define a whitelist and blacklist in parallel. It is allowed to use only one of these lists per config.`);
    if ($) {
      const { duplicates: F, subsets: H } = (0, t.findDuplicatesAndSubsets)($);
      (0, t.throwError)({ duplicates: F, subsets: H }, "whitelist");
    }
    if (R) {
      const { duplicates: F, subsets: H } = (0, t.findDuplicatesAndSubsets)(R);
      (0, t.throwError)({ duplicates: F, subsets: H }, "blacklist");
    }
  };
  t.configValidator = X;
  const Z = function({ duplicates: $, subsets: R }, F) {
    if ($.length)
      throw new Error(`${e.PACKAGE_NAME}: duplicates of paths found in your ${F}.

 ${JSON.stringify($)}`);
    if (R.length)
      throw new Error(`${e.PACKAGE_NAME}: subsets of some parent keys found in your ${F}. You must decide if you want to persist an entire path or its specific subset.

 ${JSON.stringify(R)}`);
  };
  t.throwError = Z;
  const ce = function($) {
    return (0, t.isArray)($) ? $.filter(t.unique).reduce((R, F) => {
      const H = F.split("."), re = H[0], ae = H.slice(1).join(".") || void 0, ne = R.filter((de) => Object.keys(de)[0] === re)[0], oe = ne ? Object.values(ne)[0] : void 0;
      return ne || R.push({
        [re]: ae ? [ae] : void 0
      }), ne && !oe && ae && (ne[re] = [ae]), ne && oe && ae && oe.push(ae), R;
    }, []) : [];
  };
  t.getRootKeysGroup = ce;
})(gy);
(function(t) {
  var e = dr && dr.__rest || function(g, p) {
    var y = {};
    for (var v in g)
      Object.prototype.hasOwnProperty.call(g, v) && p.indexOf(v) < 0 && (y[v] = g[v]);
    if (g != null && typeof Object.getOwnPropertySymbols == "function")
      for (var _ = 0, v = Object.getOwnPropertySymbols(g); _ < v.length; _++)
        p.indexOf(v[_]) < 0 && Object.prototype.propertyIsEnumerable.call(g, v[_]) && (y[v[_]] = g[v[_]]);
    return y;
  };
  Object.defineProperty(t, "__esModule", { value: !0 }), t.getPersistConfig = t.getTransforms = t.createBlacklist = t.createWhitelist = t.autoMergeDeep = void 0;
  const r = gy, n = Ji, i = Ah, s = function(g, p, y = {}) {
    const v = y.whitelist || null, _ = y.blacklist || null;
    function b(C) {
      return !!(v && v.indexOf(C) === -1 || _ && _.indexOf(C) !== -1);
    }
    return {
      in: (C, w, O) => !b(w) && g ? g(C, w, O) : C,
      out: (C, w, O) => !b(w) && p ? p(C, w, O) : C,
      deepPersistKey: v && v[0]
    };
  }, o = (g, p, y, { debug: v, whitelist: _, blacklist: b, transforms: C }) => {
    if (_ || b)
      throw new Error("State reconciler autoMergeDeep uses custom transforms instead of old whitelist or blacklist config properties. Please use createWhitelist or createBlacklist transforms.");
    (0, r.transformsValidator)(C);
    const w = (0, r.cloneDeep)(y);
    let O = g;
    if (O && (0, r.isObjectLike)(O)) {
      const k = (0, r.difference)(p, y);
      (0, r.isEmpty)(k) || (O = (0, r.mergeDeep)(g, k, { preserveUndefined: !0 }), v && console.log(`${n.PACKAGE_NAME}/autoMergeDeep: sub state of your state was modified by reducer during rehydration. Values from reducer will be kept: ${JSON.stringify(k)}`)), Object.keys(O).forEach((P) => {
        if (P !== "_persist") {
          if ((0, r.isObjectLike)(w[P])) {
            w[P] = (0, r.mergeDeep)(w[P], O[P]);
            return;
          }
          w[P] = O[P];
        }
      });
    }
    return v && O && (0, r.isObjectLike)(O) && console.log(`${n.PACKAGE_NAME}/autoMergeDeep: rehydrated keys ${JSON.stringify(O)}`), w;
  };
  t.autoMergeDeep = o;
  const l = (g, p) => ((0, r.singleTransformValidator)(p, g, i.ConfigType.WHITELIST), s((y) => {
    if (!p || !p.length)
      return y;
    let v = null, _;
    return p.forEach((b) => {
      const C = b.split(".");
      _ = (0, r.path)(y, C), typeof _ > "u" && (0, r.isIntegerString)(C[C.length - 1]) && (_ = n.PLACEHOLDER_UNDEFINED);
      const w = (0, r.assocPath)(C, _), O = (0, r.isArray)(w) ? [] : {};
      v = (0, r.mergeDeep)(v || O, w, {
        preservePlaceholder: !0
      });
    }), v || y;
  }, (y) => (0, r.preserveUndefined)(y, p, i.ConfigType.WHITELIST), {
    whitelist: [g]
  }));
  t.createWhitelist = l;
  const c = (g, p) => ((0, r.singleTransformValidator)(p, g, i.ConfigType.BLACKLIST), s((y) => {
    if (!p || !p.length)
      return;
    const v = (0, r.preserveUndefined)(y, p, i.ConfigType.BLACKLIST, !0);
    return p.map((b) => b.split(".")).reduce((b, C) => (0, r.dissocPath)(b, C), v);
  }, (y) => (0, r.preserveUndefined)(y, p, i.ConfigType.BLACKLIST), {
    whitelist: [g]
  }));
  t.createBlacklist = c;
  const h = function(g, p) {
    return p.map((y) => {
      const v = Object.keys(y)[0], _ = y[v];
      return g === i.ConfigType.WHITELIST ? (0, t.createWhitelist)(v, _) : (0, t.createBlacklist)(v, _);
    });
  };
  t.getTransforms = h;
  const d = (g) => {
    var { key: p, whitelist: y, blacklist: v, storage: _, transforms: b, rootReducer: C } = g, w = e(g, ["key", "whitelist", "blacklist", "storage", "transforms", "rootReducer"]);
    (0, r.configValidator)({ whitelist: y, blacklist: v });
    const O = (0, r.getRootKeysGroup)(y), k = (0, r.getRootKeysGroup)(v), P = Object.keys(C(void 0, { type: "" })), M = O.map((ce) => Object.keys(ce)[0]), U = k.map((ce) => Object.keys(ce)[0]), V = P.filter((ce) => M.indexOf(ce) === -1 && U.indexOf(ce) === -1), Y = (0, t.getTransforms)(i.ConfigType.WHITELIST, O), X = (0, t.getTransforms)(i.ConfigType.BLACKLIST, k), Z = (0, r.isArray)(y) ? V.map((ce) => (0, t.createBlacklist)(ce)) : [];
    return Object.assign(Object.assign({}, w), {
      key: p,
      storage: _,
      transforms: [
        ...Y,
        ...X,
        ...Z,
        ...b || []
      ],
      stateReconciler: t.autoMergeDeep
    });
  };
  t.getPersistConfig = d;
})(dy);
const Tu = (t, e) => Math.floor(t / e) * e, Si = (t, e) => Math.round(t / e) * e;
var Ct = {}, tO = {
  get exports() {
    return Ct;
  },
  set exports(t) {
    Ct = t;
  }
};
/**
 * @license
 * Lodash <https://lodash.com/>
 * Copyright OpenJS Foundation and other contributors <https://openjsf.org/>
 * Released under MIT license <https://lodash.com/license>
 * Based on Underscore.js 1.8.3 <http://underscorejs.org/LICENSE>
 * Copyright Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 */
(function(t, e) {
  (function() {
    var r, n = "4.17.21", i = 200, s = "Unsupported core-js use. Try https://npms.io/search?q=ponyfill.", o = "Expected a function", l = "Invalid `variable` option passed into `_.template`", c = "__lodash_hash_undefined__", h = 500, d = "__lodash_placeholder__", g = 1, p = 2, y = 4, v = 1, _ = 2, b = 1, C = 2, w = 4, O = 8, k = 16, P = 32, M = 64, U = 128, V = 256, Y = 512, X = 30, Z = "...", ce = 800, $ = 16, R = 1, F = 2, H = 3, re = 1 / 0, ae = 9007199254740991, ne = 17976931348623157e292, oe = 0 / 0, de = 4294967295, Re = de - 1, Ce = de >>> 1, rt = [
      ["ary", U],
      ["bind", b],
      ["bindKey", C],
      ["curry", O],
      ["curryRight", k],
      ["flip", Y],
      ["partial", P],
      ["partialRight", M],
      ["rearg", V]
    ], xt = "[object Arguments]", be = "[object Array]", At = "[object AsyncFunction]", ot = "[object Boolean]", wt = "[object Date]", kt = "[object DOMException]", dt = "[object Error]", te = "[object Function]", $t = "[object GeneratorFunction]", fe = "[object Map]", Ot = "[object Number]", sr = "[object Null]", Dt = "[object Object]", dn = "[object Promise]", gn = "[object Proxy]", gr = "[object RegExp]", Mt = "[object Set]", qt = "[object String]", yt = "[object Symbol]", Dn = "[object Undefined]", Sr = "[object WeakMap]", Mn = "[object WeakSet]", er = "[object ArrayBuffer]", wr = "[object DataView]", Nn = "[object Float32Array]", Bn = "[object Float64Array]", pn = "[object Int8Array]", ii = "[object Int16Array]", ai = "[object Int32Array]", Ai = "[object Uint8Array]", ki = "[object Uint8ClampedArray]", Er = "[object Uint16Array]", mn = "[object Uint32Array]", ns = /\b__p \+= '';/g, si = /\b(__p \+=) '' \+/g, ia = /(__e\(.*?\)|\b__t\)) \+\n'';/g, Nr = /&(?:amp|lt|gt|quot|#39);/g, Br = /[&<>"']/g, Ii = RegExp(Nr.source), Fr = RegExp(Br.source), is = /<%-([\s\S]+?)%>/g, aa = /<%([\s\S]+?)%>/g, sa = /<%=([\s\S]+?)%>/g, oa = /\.|\[(?:[^[\]]*|(["'])(?:(?!\1)[^\\]|\\.)*?\1)\]/, ua = /^\w*$/, Cr = /[^.[\]]+|\[(?:(-?\d+(?:\.\d+)?)|(["'])((?:(?!\2)[^\\]|\\.)*?)\2)\]|(?=(?:\.|\[\])(?:\.|\[\]|$))/g, Li = /[\\^$.*+?()[\]{}|]/g, la = RegExp(Li.source), Di = /^\s+/, as = /\s/, ss = /\{(?:\n\/\* \[wrapped with .+\] \*\/)?\n?/, vn = /\{\n\/\* \[wrapped with (.+)\] \*/, L = /,? & /, ie = /[^\x00-\x2f\x3a-\x40\x5b-\x60\x7b-\x7f]+/g, he = /[()=,{}\[\]\/\s]/, Se = /\\(\\)?/g, je = /\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g, Ne = /\w*$/, Me = /^[-+]0x[0-9a-f]+$/i, Ae = /^0b[01]+$/i, Nt = /^\[object .+?Constructor\]$/, ft = /^0o[0-7]+$/i, gt = /^(?:0|[1-9]\d*)$/, tr = /[\xc0-\xd6\xd8-\xf6\xf8-\xff\u0100-\u017f]/g, or = /($^)/, ca = /['\n\r\u2028\u2029\\]/g, Bt = "\\ud800-\\udfff", os = "\\u0300-\\u036f", Hl = "\\ufe20-\\ufe2f", fa = "\\u20d0-\\u20ff", us = os + Hl + fa, So = "\\u2700-\\u27bf", wo = "a-z\\xdf-\\xf6\\xf8-\\xff", zl = "\\xac\\xb1\\xd7\\xf7", Eo = "\\x00-\\x2f\\x3a-\\x40\\x5b-\\x60\\x7b-\\xbf", Vl = "\\u2000-\\u206f", ql = " \\t\\x0b\\f\\xa0\\ufeff\\n\\r\\u2028\\u2029\\u1680\\u180e\\u2000\\u2001\\u2002\\u2003\\u2004\\u2005\\u2006\\u2007\\u2008\\u2009\\u200a\\u202f\\u205f\\u3000", Co = "A-Z\\xc0-\\xd6\\xd8-\\xde", xo = "\\ufe0e\\ufe0f", Oo = zl + Eo + Vl + ql, ls = "['’]", Yl = "[" + Bt + "]", To = "[" + Oo + "]", ha = "[" + us + "]", Po = "\\d+", da = "[" + So + "]", ga = "[" + wo + "]", Ro = "[^" + Bt + Oo + Po + So + wo + Co + "]", cs = "\\ud83c[\\udffb-\\udfff]", Ao = "(?:" + ha + "|" + cs + ")", ko = "[^" + Bt + "]", fs = "(?:\\ud83c[\\udde6-\\uddff]){2}", hs = "[\\ud800-\\udbff][\\udc00-\\udfff]", Fn = "[" + Co + "]", Io = "\\u200d", Lo = "(?:" + ga + "|" + Ro + ")", Kl = "(?:" + Fn + "|" + Ro + ")", pa = "(?:" + ls + "(?:d|ll|m|re|s|t|ve))?", Do = "(?:" + ls + "(?:D|LL|M|RE|S|T|VE))?", Mo = Ao + "?", No = "[" + xo + "]?", ma = "(?:" + Io + "(?:" + [ko, fs, hs].join("|") + ")" + No + Mo + ")*", ds = "\\d*(?:1st|2nd|3rd|(?![123])\\dth)(?=\\b|[A-Z_])", gs = "\\d*(?:1ST|2ND|3RD|(?![123])\\dTH)(?=\\b|[a-z_])", va = No + Mo + ma, Xl = "(?:" + [da, fs, hs].join("|") + ")" + va, Bo = "(?:" + [ko + ha + "?", ha, fs, hs, Yl].join("|") + ")", ps = RegExp(ls, "g"), Fo = RegExp(ha, "g"), Qr = RegExp(cs + "(?=" + cs + ")|" + Bo + va, "g"), Mi = RegExp([
      Fn + "?" + ga + "+" + pa + "(?=" + [To, Fn, "$"].join("|") + ")",
      Kl + "+" + Do + "(?=" + [To, Fn + Lo, "$"].join("|") + ")",
      Fn + "?" + Lo + "+" + pa,
      Fn + "+" + Do,
      gs,
      ds,
      Po,
      Xl
    ].join("|"), "g"), Jl = RegExp("[" + Io + Bt + us + xo + "]"), Go = /[a-z][A-Z]|[A-Z]{2}[a-z]|[0-9][a-zA-Z]|[a-zA-Z][0-9]|[^a-zA-Z0-9 ]/, Ql = [
      "Array",
      "Buffer",
      "DataView",
      "Date",
      "Error",
      "Float32Array",
      "Float64Array",
      "Function",
      "Int8Array",
      "Int16Array",
      "Int32Array",
      "Map",
      "Math",
      "Object",
      "Promise",
      "RegExp",
      "Set",
      "String",
      "Symbol",
      "TypeError",
      "Uint8Array",
      "Uint8ClampedArray",
      "Uint16Array",
      "Uint32Array",
      "WeakMap",
      "_",
      "clearTimeout",
      "isFinite",
      "parseInt",
      "setTimeout"
    ], jo = -1, ut = {};
    ut[Nn] = ut[Bn] = ut[pn] = ut[ii] = ut[ai] = ut[Ai] = ut[ki] = ut[Er] = ut[mn] = !0, ut[xt] = ut[be] = ut[er] = ut[ot] = ut[wr] = ut[wt] = ut[dt] = ut[te] = ut[fe] = ut[Ot] = ut[Dt] = ut[gr] = ut[Mt] = ut[qt] = ut[Sr] = !1;
    var at = {};
    at[xt] = at[be] = at[er] = at[wr] = at[ot] = at[wt] = at[Nn] = at[Bn] = at[pn] = at[ii] = at[ai] = at[fe] = at[Ot] = at[Dt] = at[gr] = at[Mt] = at[qt] = at[yt] = at[Ai] = at[ki] = at[Er] = at[mn] = !0, at[dt] = at[te] = at[Sr] = !1;
    var $o = {
      // Latin-1 Supplement block.
      À: "A",
      Á: "A",
      Â: "A",
      Ã: "A",
      Ä: "A",
      Å: "A",
      à: "a",
      á: "a",
      â: "a",
      ã: "a",
      ä: "a",
      å: "a",
      Ç: "C",
      ç: "c",
      Ð: "D",
      ð: "d",
      È: "E",
      É: "E",
      Ê: "E",
      Ë: "E",
      è: "e",
      é: "e",
      ê: "e",
      ë: "e",
      Ì: "I",
      Í: "I",
      Î: "I",
      Ï: "I",
      ì: "i",
      í: "i",
      î: "i",
      ï: "i",
      Ñ: "N",
      ñ: "n",
      Ò: "O",
      Ó: "O",
      Ô: "O",
      Õ: "O",
      Ö: "O",
      Ø: "O",
      ò: "o",
      ó: "o",
      ô: "o",
      õ: "o",
      ö: "o",
      ø: "o",
      Ù: "U",
      Ú: "U",
      Û: "U",
      Ü: "U",
      ù: "u",
      ú: "u",
      û: "u",
      ü: "u",
      Ý: "Y",
      ý: "y",
      ÿ: "y",
      Æ: "Ae",
      æ: "ae",
      Þ: "Th",
      þ: "th",
      ß: "ss",
      // Latin Extended-A block.
      Ā: "A",
      Ă: "A",
      Ą: "A",
      ā: "a",
      ă: "a",
      ą: "a",
      Ć: "C",
      Ĉ: "C",
      Ċ: "C",
      Č: "C",
      ć: "c",
      ĉ: "c",
      ċ: "c",
      č: "c",
      Ď: "D",
      Đ: "D",
      ď: "d",
      đ: "d",
      Ē: "E",
      Ĕ: "E",
      Ė: "E",
      Ę: "E",
      Ě: "E",
      ē: "e",
      ĕ: "e",
      ė: "e",
      ę: "e",
      ě: "e",
      Ĝ: "G",
      Ğ: "G",
      Ġ: "G",
      Ģ: "G",
      ĝ: "g",
      ğ: "g",
      ġ: "g",
      ģ: "g",
      Ĥ: "H",
      Ħ: "H",
      ĥ: "h",
      ħ: "h",
      Ĩ: "I",
      Ī: "I",
      Ĭ: "I",
      Į: "I",
      İ: "I",
      ĩ: "i",
      ī: "i",
      ĭ: "i",
      į: "i",
      ı: "i",
      Ĵ: "J",
      ĵ: "j",
      Ķ: "K",
      ķ: "k",
      ĸ: "k",
      Ĺ: "L",
      Ļ: "L",
      Ľ: "L",
      Ŀ: "L",
      Ł: "L",
      ĺ: "l",
      ļ: "l",
      ľ: "l",
      ŀ: "l",
      ł: "l",
      Ń: "N",
      Ņ: "N",
      Ň: "N",
      Ŋ: "N",
      ń: "n",
      ņ: "n",
      ň: "n",
      ŋ: "n",
      Ō: "O",
      Ŏ: "O",
      Ő: "O",
      ō: "o",
      ŏ: "o",
      ő: "o",
      Ŕ: "R",
      Ŗ: "R",
      Ř: "R",
      ŕ: "r",
      ŗ: "r",
      ř: "r",
      Ś: "S",
      Ŝ: "S",
      Ş: "S",
      Š: "S",
      ś: "s",
      ŝ: "s",
      ş: "s",
      š: "s",
      Ţ: "T",
      Ť: "T",
      Ŧ: "T",
      ţ: "t",
      ť: "t",
      ŧ: "t",
      Ũ: "U",
      Ū: "U",
      Ŭ: "U",
      Ů: "U",
      Ű: "U",
      Ų: "U",
      ũ: "u",
      ū: "u",
      ŭ: "u",
      ů: "u",
      ű: "u",
      ų: "u",
      Ŵ: "W",
      ŵ: "w",
      Ŷ: "Y",
      ŷ: "y",
      Ÿ: "Y",
      Ź: "Z",
      Ż: "Z",
      Ž: "Z",
      ź: "z",
      ż: "z",
      ž: "z",
      Ĳ: "IJ",
      ĳ: "ij",
      Œ: "Oe",
      œ: "oe",
      ŉ: "'n",
      ſ: "s"
    }, Zl = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }, T = {
      "&amp;": "&",
      "&lt;": "<",
      "&gt;": ">",
      "&quot;": '"',
      "&#39;": "'"
    }, N = {
      "\\": "\\",
      "'": "'",
      "\n": "n",
      "\r": "r",
      "\u2028": "u2028",
      "\u2029": "u2029"
    }, z = parseFloat, pe = parseInt, Be = typeof dr == "object" && dr && dr.Object === Object && dr, nt = typeof self == "object" && self && self.Object === Object && self, Te = Be || nt || Function("return this")(), Ie = e && !e.nodeType && e, He = Ie && !0 && t && !t.nodeType && t, ur = He && He.exports === Ie, Yt = ur && Be.process, Wt = function() {
      try {
        var G = He && He.require && He.require("util").types;
        return G || Yt && Yt.binding && Yt.binding("util");
      } catch {
      }
    }(), ya = Wt && Wt.isArrayBuffer, ba = Wt && Wt.isDate, ms = Wt && Wt.isMap, Qh = Wt && Wt.isRegExp, Zh = Wt && Wt.isSet, ed = Wt && Wt.isTypedArray;
    function xr(G, K, q) {
      switch (q.length) {
        case 0:
          return G.call(K);
        case 1:
          return G.call(K, q[0]);
        case 2:
          return G.call(K, q[0], q[1]);
        case 3:
          return G.call(K, q[0], q[1], q[2]);
      }
      return G.apply(K, q);
    }
    function pb(G, K, q, me) {
      for (var xe = -1, Ve = G == null ? 0 : G.length; ++xe < Ve; ) {
        var Ft = G[xe];
        K(me, Ft, q(Ft), G);
      }
      return me;
    }
    function Gr(G, K) {
      for (var q = -1, me = G == null ? 0 : G.length; ++q < me && K(G[q], q, G) !== !1; )
        ;
      return G;
    }
    function mb(G, K) {
      for (var q = G == null ? 0 : G.length; q-- && K(G[q], q, G) !== !1; )
        ;
      return G;
    }
    function td(G, K) {
      for (var q = -1, me = G == null ? 0 : G.length; ++q < me; )
        if (!K(G[q], q, G))
          return !1;
      return !0;
    }
    function oi(G, K) {
      for (var q = -1, me = G == null ? 0 : G.length, xe = 0, Ve = []; ++q < me; ) {
        var Ft = G[q];
        K(Ft, q, G) && (Ve[xe++] = Ft);
      }
      return Ve;
    }
    function Wo(G, K) {
      var q = G == null ? 0 : G.length;
      return !!q && _a(G, K, 0) > -1;
    }
    function ec(G, K, q) {
      for (var me = -1, xe = G == null ? 0 : G.length; ++me < xe; )
        if (q(K, G[me]))
          return !0;
      return !1;
    }
    function pt(G, K) {
      for (var q = -1, me = G == null ? 0 : G.length, xe = Array(me); ++q < me; )
        xe[q] = K(G[q], q, G);
      return xe;
    }
    function ui(G, K) {
      for (var q = -1, me = K.length, xe = G.length; ++q < me; )
        G[xe + q] = K[q];
      return G;
    }
    function tc(G, K, q, me) {
      var xe = -1, Ve = G == null ? 0 : G.length;
      for (me && Ve && (q = G[++xe]); ++xe < Ve; )
        q = K(q, G[xe], xe, G);
      return q;
    }
    function vb(G, K, q, me) {
      var xe = G == null ? 0 : G.length;
      for (me && xe && (q = G[--xe]); xe--; )
        q = K(q, G[xe], xe, G);
      return q;
    }
    function rc(G, K) {
      for (var q = -1, me = G == null ? 0 : G.length; ++q < me; )
        if (K(G[q], q, G))
          return !0;
      return !1;
    }
    var yb = nc("length");
    function bb(G) {
      return G.split("");
    }
    function _b(G) {
      return G.match(ie) || [];
    }
    function rd(G, K, q) {
      var me;
      return q(G, function(xe, Ve, Ft) {
        if (K(xe, Ve, Ft))
          return me = Ve, !1;
      }), me;
    }
    function Uo(G, K, q, me) {
      for (var xe = G.length, Ve = q + (me ? 1 : -1); me ? Ve-- : ++Ve < xe; )
        if (K(G[Ve], Ve, G))
          return Ve;
      return -1;
    }
    function _a(G, K, q) {
      return K === K ? Ib(G, K, q) : Uo(G, nd, q);
    }
    function Sb(G, K, q, me) {
      for (var xe = q - 1, Ve = G.length; ++xe < Ve; )
        if (me(G[xe], K))
          return xe;
      return -1;
    }
    function nd(G) {
      return G !== G;
    }
    function id(G, K) {
      var q = G == null ? 0 : G.length;
      return q ? ac(G, K) / q : oe;
    }
    function nc(G) {
      return function(K) {
        return K == null ? r : K[G];
      };
    }
    function ic(G) {
      return function(K) {
        return G == null ? r : G[K];
      };
    }
    function ad(G, K, q, me, xe) {
      return xe(G, function(Ve, Ft, lt) {
        q = me ? (me = !1, Ve) : K(q, Ve, Ft, lt);
      }), q;
    }
    function wb(G, K) {
      var q = G.length;
      for (G.sort(K); q--; )
        G[q] = G[q].value;
      return G;
    }
    function ac(G, K) {
      for (var q, me = -1, xe = G.length; ++me < xe; ) {
        var Ve = K(G[me]);
        Ve !== r && (q = q === r ? Ve : q + Ve);
      }
      return q;
    }
    function sc(G, K) {
      for (var q = -1, me = Array(G); ++q < G; )
        me[q] = K(q);
      return me;
    }
    function Eb(G, K) {
      return pt(K, function(q) {
        return [q, G[q]];
      });
    }
    function sd(G) {
      return G && G.slice(0, cd(G) + 1).replace(Di, "");
    }
    function Or(G) {
      return function(K) {
        return G(K);
      };
    }
    function oc(G, K) {
      return pt(K, function(q) {
        return G[q];
      });
    }
    function vs(G, K) {
      return G.has(K);
    }
    function od(G, K) {
      for (var q = -1, me = G.length; ++q < me && _a(K, G[q], 0) > -1; )
        ;
      return q;
    }
    function ud(G, K) {
      for (var q = G.length; q-- && _a(K, G[q], 0) > -1; )
        ;
      return q;
    }
    function Cb(G, K) {
      for (var q = G.length, me = 0; q--; )
        G[q] === K && ++me;
      return me;
    }
    var xb = ic($o), Ob = ic(Zl);
    function Tb(G) {
      return "\\" + N[G];
    }
    function Pb(G, K) {
      return G == null ? r : G[K];
    }
    function Sa(G) {
      return Jl.test(G);
    }
    function Rb(G) {
      return Go.test(G);
    }
    function Ab(G) {
      for (var K, q = []; !(K = G.next()).done; )
        q.push(K.value);
      return q;
    }
    function uc(G) {
      var K = -1, q = Array(G.size);
      return G.forEach(function(me, xe) {
        q[++K] = [xe, me];
      }), q;
    }
    function ld(G, K) {
      return function(q) {
        return G(K(q));
      };
    }
    function li(G, K) {
      for (var q = -1, me = G.length, xe = 0, Ve = []; ++q < me; ) {
        var Ft = G[q];
        (Ft === K || Ft === d) && (G[q] = d, Ve[xe++] = q);
      }
      return Ve;
    }
    function Ho(G) {
      var K = -1, q = Array(G.size);
      return G.forEach(function(me) {
        q[++K] = me;
      }), q;
    }
    function kb(G) {
      var K = -1, q = Array(G.size);
      return G.forEach(function(me) {
        q[++K] = [me, me];
      }), q;
    }
    function Ib(G, K, q) {
      for (var me = q - 1, xe = G.length; ++me < xe; )
        if (G[me] === K)
          return me;
      return -1;
    }
    function Lb(G, K, q) {
      for (var me = q + 1; me--; )
        if (G[me] === K)
          return me;
      return me;
    }
    function wa(G) {
      return Sa(G) ? Mb(G) : yb(G);
    }
    function Zr(G) {
      return Sa(G) ? Nb(G) : bb(G);
    }
    function cd(G) {
      for (var K = G.length; K-- && as.test(G.charAt(K)); )
        ;
      return K;
    }
    var Db = ic(T);
    function Mb(G) {
      for (var K = Qr.lastIndex = 0; Qr.test(G); )
        ++K;
      return K;
    }
    function Nb(G) {
      return G.match(Qr) || [];
    }
    function Bb(G) {
      return G.match(Mi) || [];
    }
    var Fb = function G(K) {
      K = K == null ? Te : Ea.defaults(Te.Object(), K, Ea.pick(Te, Ql));
      var q = K.Array, me = K.Date, xe = K.Error, Ve = K.Function, Ft = K.Math, lt = K.Object, lc = K.RegExp, Gb = K.String, jr = K.TypeError, zo = q.prototype, jb = Ve.prototype, Ca = lt.prototype, Vo = K["__core-js_shared__"], qo = jb.toString, it = Ca.hasOwnProperty, $b = 0, fd = function() {
        var a = /[^.]+$/.exec(Vo && Vo.keys && Vo.keys.IE_PROTO || "");
        return a ? "Symbol(src)_1." + a : "";
      }(), Yo = Ca.toString, Wb = qo.call(lt), Ub = Te._, Hb = lc(
        "^" + qo.call(it).replace(Li, "\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, "$1.*?") + "$"
      ), Ko = ur ? K.Buffer : r, ci = K.Symbol, Xo = K.Uint8Array, hd = Ko ? Ko.allocUnsafe : r, Jo = ld(lt.getPrototypeOf, lt), dd = lt.create, gd = Ca.propertyIsEnumerable, Qo = zo.splice, pd = ci ? ci.isConcatSpreadable : r, ys = ci ? ci.iterator : r, Ni = ci ? ci.toStringTag : r, Zo = function() {
        try {
          var a = $i(lt, "defineProperty");
          return a({}, "", {}), a;
        } catch {
        }
      }(), zb = K.clearTimeout !== Te.clearTimeout && K.clearTimeout, Vb = me && me.now !== Te.Date.now && me.now, qb = K.setTimeout !== Te.setTimeout && K.setTimeout, eu = Ft.ceil, tu = Ft.floor, cc = lt.getOwnPropertySymbols, Yb = Ko ? Ko.isBuffer : r, md = K.isFinite, Kb = zo.join, Xb = ld(lt.keys, lt), Gt = Ft.max, rr = Ft.min, Jb = me.now, Qb = K.parseInt, vd = Ft.random, Zb = zo.reverse, fc = $i(K, "DataView"), bs = $i(K, "Map"), hc = $i(K, "Promise"), xa = $i(K, "Set"), _s = $i(K, "WeakMap"), Ss = $i(lt, "create"), ru = _s && new _s(), Oa = {}, e_ = Wi(fc), t_ = Wi(bs), r_ = Wi(hc), n_ = Wi(xa), i_ = Wi(_s), nu = ci ? ci.prototype : r, ws = nu ? nu.valueOf : r, yd = nu ? nu.toString : r;
      function E(a) {
        if (Et(a) && !Oe(a) && !(a instanceof Ge)) {
          if (a instanceof $r)
            return a;
          if (it.call(a, "__wrapped__"))
            return bg(a);
        }
        return new $r(a);
      }
      var Ta = function() {
        function a() {
        }
        return function(u) {
          if (!bt(u))
            return {};
          if (dd)
            return dd(u);
          a.prototype = u;
          var f = new a();
          return a.prototype = r, f;
        };
      }();
      function iu() {
      }
      function $r(a, u) {
        this.__wrapped__ = a, this.__actions__ = [], this.__chain__ = !!u, this.__index__ = 0, this.__values__ = r;
      }
      E.templateSettings = {
        /**
         * Used to detect `data` property values to be HTML-escaped.
         *
         * @memberOf _.templateSettings
         * @type {RegExp}
         */
        escape: is,
        /**
         * Used to detect code to be evaluated.
         *
         * @memberOf _.templateSettings
         * @type {RegExp}
         */
        evaluate: aa,
        /**
         * Used to detect `data` property values to inject.
         *
         * @memberOf _.templateSettings
         * @type {RegExp}
         */
        interpolate: sa,
        /**
         * Used to reference the data object in the template text.
         *
         * @memberOf _.templateSettings
         * @type {string}
         */
        variable: "",
        /**
         * Used to import variables into the compiled template.
         *
         * @memberOf _.templateSettings
         * @type {Object}
         */
        imports: {
          /**
           * A reference to the `lodash` function.
           *
           * @memberOf _.templateSettings.imports
           * @type {Function}
           */
          _: E
        }
      }, E.prototype = iu.prototype, E.prototype.constructor = E, $r.prototype = Ta(iu.prototype), $r.prototype.constructor = $r;
      function Ge(a) {
        this.__wrapped__ = a, this.__actions__ = [], this.__dir__ = 1, this.__filtered__ = !1, this.__iteratees__ = [], this.__takeCount__ = de, this.__views__ = [];
      }
      function a_() {
        var a = new Ge(this.__wrapped__);
        return a.__actions__ = pr(this.__actions__), a.__dir__ = this.__dir__, a.__filtered__ = this.__filtered__, a.__iteratees__ = pr(this.__iteratees__), a.__takeCount__ = this.__takeCount__, a.__views__ = pr(this.__views__), a;
      }
      function s_() {
        if (this.__filtered__) {
          var a = new Ge(this);
          a.__dir__ = -1, a.__filtered__ = !0;
        } else
          a = this.clone(), a.__dir__ *= -1;
        return a;
      }
      function o_() {
        var a = this.__wrapped__.value(), u = this.__dir__, f = Oe(a), m = u < 0, S = f ? a.length : 0, x = bS(0, S, this.__views__), A = x.start, D = x.end, j = D - A, J = m ? D : A - 1, Q = this.__iteratees__, ee = Q.length, ge = 0, ye = rr(j, this.__takeCount__);
        if (!f || !m && S == j && ye == j)
          return Wd(a, this.__actions__);
        var we = [];
        e:
          for (; j-- && ge < ye; ) {
            J += u;
            for (var ke = -1, Ee = a[J]; ++ke < ee; ) {
              var Fe = Q[ke], $e = Fe.iteratee, Rr = Fe.type, fr = $e(Ee);
              if (Rr == F)
                Ee = fr;
              else if (!fr) {
                if (Rr == R)
                  continue e;
                break e;
              }
            }
            we[ge++] = Ee;
          }
        return we;
      }
      Ge.prototype = Ta(iu.prototype), Ge.prototype.constructor = Ge;
      function Bi(a) {
        var u = -1, f = a == null ? 0 : a.length;
        for (this.clear(); ++u < f; ) {
          var m = a[u];
          this.set(m[0], m[1]);
        }
      }
      function u_() {
        this.__data__ = Ss ? Ss(null) : {}, this.size = 0;
      }
      function l_(a) {
        var u = this.has(a) && delete this.__data__[a];
        return this.size -= u ? 1 : 0, u;
      }
      function c_(a) {
        var u = this.__data__;
        if (Ss) {
          var f = u[a];
          return f === c ? r : f;
        }
        return it.call(u, a) ? u[a] : r;
      }
      function f_(a) {
        var u = this.__data__;
        return Ss ? u[a] !== r : it.call(u, a);
      }
      function h_(a, u) {
        var f = this.__data__;
        return this.size += this.has(a) ? 0 : 1, f[a] = Ss && u === r ? c : u, this;
      }
      Bi.prototype.clear = u_, Bi.prototype.delete = l_, Bi.prototype.get = c_, Bi.prototype.has = f_, Bi.prototype.set = h_;
      function Gn(a) {
        var u = -1, f = a == null ? 0 : a.length;
        for (this.clear(); ++u < f; ) {
          var m = a[u];
          this.set(m[0], m[1]);
        }
      }
      function d_() {
        this.__data__ = [], this.size = 0;
      }
      function g_(a) {
        var u = this.__data__, f = au(u, a);
        if (f < 0)
          return !1;
        var m = u.length - 1;
        return f == m ? u.pop() : Qo.call(u, f, 1), --this.size, !0;
      }
      function p_(a) {
        var u = this.__data__, f = au(u, a);
        return f < 0 ? r : u[f][1];
      }
      function m_(a) {
        return au(this.__data__, a) > -1;
      }
      function v_(a, u) {
        var f = this.__data__, m = au(f, a);
        return m < 0 ? (++this.size, f.push([a, u])) : f[m][1] = u, this;
      }
      Gn.prototype.clear = d_, Gn.prototype.delete = g_, Gn.prototype.get = p_, Gn.prototype.has = m_, Gn.prototype.set = v_;
      function jn(a) {
        var u = -1, f = a == null ? 0 : a.length;
        for (this.clear(); ++u < f; ) {
          var m = a[u];
          this.set(m[0], m[1]);
        }
      }
      function y_() {
        this.size = 0, this.__data__ = {
          hash: new Bi(),
          map: new (bs || Gn)(),
          string: new Bi()
        };
      }
      function b_(a) {
        var u = vu(this, a).delete(a);
        return this.size -= u ? 1 : 0, u;
      }
      function __(a) {
        return vu(this, a).get(a);
      }
      function S_(a) {
        return vu(this, a).has(a);
      }
      function w_(a, u) {
        var f = vu(this, a), m = f.size;
        return f.set(a, u), this.size += f.size == m ? 0 : 1, this;
      }
      jn.prototype.clear = y_, jn.prototype.delete = b_, jn.prototype.get = __, jn.prototype.has = S_, jn.prototype.set = w_;
      function Fi(a) {
        var u = -1, f = a == null ? 0 : a.length;
        for (this.__data__ = new jn(); ++u < f; )
          this.add(a[u]);
      }
      function E_(a) {
        return this.__data__.set(a, c), this;
      }
      function C_(a) {
        return this.__data__.has(a);
      }
      Fi.prototype.add = Fi.prototype.push = E_, Fi.prototype.has = C_;
      function en(a) {
        var u = this.__data__ = new Gn(a);
        this.size = u.size;
      }
      function x_() {
        this.__data__ = new Gn(), this.size = 0;
      }
      function O_(a) {
        var u = this.__data__, f = u.delete(a);
        return this.size = u.size, f;
      }
      function T_(a) {
        return this.__data__.get(a);
      }
      function P_(a) {
        return this.__data__.has(a);
      }
      function R_(a, u) {
        var f = this.__data__;
        if (f instanceof Gn) {
          var m = f.__data__;
          if (!bs || m.length < i - 1)
            return m.push([a, u]), this.size = ++f.size, this;
          f = this.__data__ = new jn(m);
        }
        return f.set(a, u), this.size = f.size, this;
      }
      en.prototype.clear = x_, en.prototype.delete = O_, en.prototype.get = T_, en.prototype.has = P_, en.prototype.set = R_;
      function bd(a, u) {
        var f = Oe(a), m = !f && Ui(a), S = !f && !m && pi(a), x = !f && !m && !S && ka(a), A = f || m || S || x, D = A ? sc(a.length, Gb) : [], j = D.length;
        for (var J in a)
          (u || it.call(a, J)) && !(A && // Safari 9 has enumerable `arguments.length` in strict mode.
          (J == "length" || // Node.js 0.10 has enumerable non-index properties on buffers.
          S && (J == "offset" || J == "parent") || // PhantomJS 2 has enumerable non-index properties on typed arrays.
          x && (J == "buffer" || J == "byteLength" || J == "byteOffset") || // Skip index properties.
          Hn(J, j))) && D.push(J);
        return D;
      }
      function _d(a) {
        var u = a.length;
        return u ? a[Ec(0, u - 1)] : r;
      }
      function A_(a, u) {
        return yu(pr(a), Gi(u, 0, a.length));
      }
      function k_(a) {
        return yu(pr(a));
      }
      function dc(a, u, f) {
        (f !== r && !tn(a[u], f) || f === r && !(u in a)) && $n(a, u, f);
      }
      function Es(a, u, f) {
        var m = a[u];
        (!(it.call(a, u) && tn(m, f)) || f === r && !(u in a)) && $n(a, u, f);
      }
      function au(a, u) {
        for (var f = a.length; f--; )
          if (tn(a[f][0], u))
            return f;
        return -1;
      }
      function I_(a, u, f, m) {
        return fi(a, function(S, x, A) {
          u(m, S, f(S), A);
        }), m;
      }
      function Sd(a, u) {
        return a && bn(u, Ut(u), a);
      }
      function L_(a, u) {
        return a && bn(u, vr(u), a);
      }
      function $n(a, u, f) {
        u == "__proto__" && Zo ? Zo(a, u, {
          configurable: !0,
          enumerable: !0,
          value: f,
          writable: !0
        }) : a[u] = f;
      }
      function gc(a, u) {
        for (var f = -1, m = u.length, S = q(m), x = a == null; ++f < m; )
          S[f] = x ? r : Yc(a, u[f]);
        return S;
      }
      function Gi(a, u, f) {
        return a === a && (f !== r && (a = a <= f ? a : f), u !== r && (a = a >= u ? a : u)), a;
      }
      function Wr(a, u, f, m, S, x) {
        var A, D = u & g, j = u & p, J = u & y;
        if (f && (A = S ? f(a, m, S, x) : f(a)), A !== r)
          return A;
        if (!bt(a))
          return a;
        var Q = Oe(a);
        if (Q) {
          if (A = SS(a), !D)
            return pr(a, A);
        } else {
          var ee = nr(a), ge = ee == te || ee == $t;
          if (pi(a))
            return zd(a, D);
          if (ee == Dt || ee == xt || ge && !S) {
            if (A = j || ge ? {} : cg(a), !D)
              return j ? cS(a, L_(A, a)) : lS(a, Sd(A, a));
          } else {
            if (!at[ee])
              return S ? a : {};
            A = wS(a, ee, D);
          }
        }
        x || (x = new en());
        var ye = x.get(a);
        if (ye)
          return ye;
        x.set(a, A), Gg(a) ? a.forEach(function(Ee) {
          A.add(Wr(Ee, u, f, Ee, a, x));
        }) : Bg(a) && a.forEach(function(Ee, Fe) {
          A.set(Fe, Wr(Ee, u, f, Fe, a, x));
        });
        var we = J ? j ? Dc : Lc : j ? vr : Ut, ke = Q ? r : we(a);
        return Gr(ke || a, function(Ee, Fe) {
          ke && (Fe = Ee, Ee = a[Fe]), Es(A, Fe, Wr(Ee, u, f, Fe, a, x));
        }), A;
      }
      function D_(a) {
        var u = Ut(a);
        return function(f) {
          return wd(f, a, u);
        };
      }
      function wd(a, u, f) {
        var m = f.length;
        if (a == null)
          return !m;
        for (a = lt(a); m--; ) {
          var S = f[m], x = u[S], A = a[S];
          if (A === r && !(S in a) || !x(A))
            return !1;
        }
        return !0;
      }
      function Ed(a, u, f) {
        if (typeof a != "function")
          throw new jr(o);
        return As(function() {
          a.apply(r, f);
        }, u);
      }
      function Cs(a, u, f, m) {
        var S = -1, x = Wo, A = !0, D = a.length, j = [], J = u.length;
        if (!D)
          return j;
        f && (u = pt(u, Or(f))), m ? (x = ec, A = !1) : u.length >= i && (x = vs, A = !1, u = new Fi(u));
        e:
          for (; ++S < D; ) {
            var Q = a[S], ee = f == null ? Q : f(Q);
            if (Q = m || Q !== 0 ? Q : 0, A && ee === ee) {
              for (var ge = J; ge--; )
                if (u[ge] === ee)
                  continue e;
              j.push(Q);
            } else
              x(u, ee, m) || j.push(Q);
          }
        return j;
      }
      var fi = Xd(yn), Cd = Xd(mc, !0);
      function M_(a, u) {
        var f = !0;
        return fi(a, function(m, S, x) {
          return f = !!u(m, S, x), f;
        }), f;
      }
      function su(a, u, f) {
        for (var m = -1, S = a.length; ++m < S; ) {
          var x = a[m], A = u(x);
          if (A != null && (D === r ? A === A && !Pr(A) : f(A, D)))
            var D = A, j = x;
        }
        return j;
      }
      function N_(a, u, f, m) {
        var S = a.length;
        for (f = Pe(f), f < 0 && (f = -f > S ? 0 : S + f), m = m === r || m > S ? S : Pe(m), m < 0 && (m += S), m = f > m ? 0 : $g(m); f < m; )
          a[f++] = u;
        return a;
      }
      function xd(a, u) {
        var f = [];
        return fi(a, function(m, S, x) {
          u(m, S, x) && f.push(m);
        }), f;
      }
      function Kt(a, u, f, m, S) {
        var x = -1, A = a.length;
        for (f || (f = CS), S || (S = []); ++x < A; ) {
          var D = a[x];
          u > 0 && f(D) ? u > 1 ? Kt(D, u - 1, f, m, S) : ui(S, D) : m || (S[S.length] = D);
        }
        return S;
      }
      var pc = Jd(), Od = Jd(!0);
      function yn(a, u) {
        return a && pc(a, u, Ut);
      }
      function mc(a, u) {
        return a && Od(a, u, Ut);
      }
      function ou(a, u) {
        return oi(u, function(f) {
          return zn(a[f]);
        });
      }
      function ji(a, u) {
        u = di(u, a);
        for (var f = 0, m = u.length; a != null && f < m; )
          a = a[_n(u[f++])];
        return f && f == m ? a : r;
      }
      function Td(a, u, f) {
        var m = u(a);
        return Oe(a) ? m : ui(m, f(a));
      }
      function lr(a) {
        return a == null ? a === r ? Dn : sr : Ni && Ni in lt(a) ? yS(a) : kS(a);
      }
      function vc(a, u) {
        return a > u;
      }
      function B_(a, u) {
        return a != null && it.call(a, u);
      }
      function F_(a, u) {
        return a != null && u in lt(a);
      }
      function G_(a, u, f) {
        return a >= rr(u, f) && a < Gt(u, f);
      }
      function yc(a, u, f) {
        for (var m = f ? ec : Wo, S = a[0].length, x = a.length, A = x, D = q(x), j = 1 / 0, J = []; A--; ) {
          var Q = a[A];
          A && u && (Q = pt(Q, Or(u))), j = rr(Q.length, j), D[A] = !f && (u || S >= 120 && Q.length >= 120) ? new Fi(A && Q) : r;
        }
        Q = a[0];
        var ee = -1, ge = D[0];
        e:
          for (; ++ee < S && J.length < j; ) {
            var ye = Q[ee], we = u ? u(ye) : ye;
            if (ye = f || ye !== 0 ? ye : 0, !(ge ? vs(ge, we) : m(J, we, f))) {
              for (A = x; --A; ) {
                var ke = D[A];
                if (!(ke ? vs(ke, we) : m(a[A], we, f)))
                  continue e;
              }
              ge && ge.push(we), J.push(ye);
            }
          }
        return J;
      }
      function j_(a, u, f, m) {
        return yn(a, function(S, x, A) {
          u(m, f(S), x, A);
        }), m;
      }
      function xs(a, u, f) {
        u = di(u, a), a = gg(a, u);
        var m = a == null ? a : a[_n(Hr(u))];
        return m == null ? r : xr(m, a, f);
      }
      function Pd(a) {
        return Et(a) && lr(a) == xt;
      }
      function $_(a) {
        return Et(a) && lr(a) == er;
      }
      function W_(a) {
        return Et(a) && lr(a) == wt;
      }
      function Os(a, u, f, m, S) {
        return a === u ? !0 : a == null || u == null || !Et(a) && !Et(u) ? a !== a && u !== u : U_(a, u, f, m, Os, S);
      }
      function U_(a, u, f, m, S, x) {
        var A = Oe(a), D = Oe(u), j = A ? be : nr(a), J = D ? be : nr(u);
        j = j == xt ? Dt : j, J = J == xt ? Dt : J;
        var Q = j == Dt, ee = J == Dt, ge = j == J;
        if (ge && pi(a)) {
          if (!pi(u))
            return !1;
          A = !0, Q = !1;
        }
        if (ge && !Q)
          return x || (x = new en()), A || ka(a) ? og(a, u, f, m, S, x) : mS(a, u, j, f, m, S, x);
        if (!(f & v)) {
          var ye = Q && it.call(a, "__wrapped__"), we = ee && it.call(u, "__wrapped__");
          if (ye || we) {
            var ke = ye ? a.value() : a, Ee = we ? u.value() : u;
            return x || (x = new en()), S(ke, Ee, f, m, x);
          }
        }
        return ge ? (x || (x = new en()), vS(a, u, f, m, S, x)) : !1;
      }
      function H_(a) {
        return Et(a) && nr(a) == fe;
      }
      function bc(a, u, f, m) {
        var S = f.length, x = S, A = !m;
        if (a == null)
          return !x;
        for (a = lt(a); S--; ) {
          var D = f[S];
          if (A && D[2] ? D[1] !== a[D[0]] : !(D[0] in a))
            return !1;
        }
        for (; ++S < x; ) {
          D = f[S];
          var j = D[0], J = a[j], Q = D[1];
          if (A && D[2]) {
            if (J === r && !(j in a))
              return !1;
          } else {
            var ee = new en();
            if (m)
              var ge = m(J, Q, j, a, u, ee);
            if (!(ge === r ? Os(Q, J, v | _, m, ee) : ge))
              return !1;
          }
        }
        return !0;
      }
      function Rd(a) {
        if (!bt(a) || OS(a))
          return !1;
        var u = zn(a) ? Hb : Nt;
        return u.test(Wi(a));
      }
      function z_(a) {
        return Et(a) && lr(a) == gr;
      }
      function V_(a) {
        return Et(a) && nr(a) == Mt;
      }
      function q_(a) {
        return Et(a) && Cu(a.length) && !!ut[lr(a)];
      }
      function Ad(a) {
        return typeof a == "function" ? a : a == null ? yr : typeof a == "object" ? Oe(a) ? Ld(a[0], a[1]) : Id(a) : Qg(a);
      }
      function _c(a) {
        if (!Rs(a))
          return Xb(a);
        var u = [];
        for (var f in lt(a))
          it.call(a, f) && f != "constructor" && u.push(f);
        return u;
      }
      function Y_(a) {
        if (!bt(a))
          return AS(a);
        var u = Rs(a), f = [];
        for (var m in a)
          m == "constructor" && (u || !it.call(a, m)) || f.push(m);
        return f;
      }
      function Sc(a, u) {
        return a < u;
      }
      function kd(a, u) {
        var f = -1, m = mr(a) ? q(a.length) : [];
        return fi(a, function(S, x, A) {
          m[++f] = u(S, x, A);
        }), m;
      }
      function Id(a) {
        var u = Nc(a);
        return u.length == 1 && u[0][2] ? hg(u[0][0], u[0][1]) : function(f) {
          return f === a || bc(f, a, u);
        };
      }
      function Ld(a, u) {
        return Fc(a) && fg(u) ? hg(_n(a), u) : function(f) {
          var m = Yc(f, a);
          return m === r && m === u ? Kc(f, a) : Os(u, m, v | _);
        };
      }
      function uu(a, u, f, m, S) {
        a !== u && pc(u, function(x, A) {
          if (S || (S = new en()), bt(x))
            K_(a, u, A, f, uu, m, S);
          else {
            var D = m ? m(jc(a, A), x, A + "", a, u, S) : r;
            D === r && (D = x), dc(a, A, D);
          }
        }, vr);
      }
      function K_(a, u, f, m, S, x, A) {
        var D = jc(a, f), j = jc(u, f), J = A.get(j);
        if (J) {
          dc(a, f, J);
          return;
        }
        var Q = x ? x(D, j, f + "", a, u, A) : r, ee = Q === r;
        if (ee) {
          var ge = Oe(j), ye = !ge && pi(j), we = !ge && !ye && ka(j);
          Q = j, ge || ye || we ? Oe(D) ? Q = D : Tt(D) ? Q = pr(D) : ye ? (ee = !1, Q = zd(j, !0)) : we ? (ee = !1, Q = Vd(j, !0)) : Q = [] : ks(j) || Ui(j) ? (Q = D, Ui(D) ? Q = Wg(D) : (!bt(D) || zn(D)) && (Q = cg(j))) : ee = !1;
        }
        ee && (A.set(j, Q), S(Q, j, m, x, A), A.delete(j)), dc(a, f, Q);
      }
      function Dd(a, u) {
        var f = a.length;
        if (f)
          return u += u < 0 ? f : 0, Hn(u, f) ? a[u] : r;
      }
      function Md(a, u, f) {
        u.length ? u = pt(u, function(x) {
          return Oe(x) ? function(A) {
            return ji(A, x.length === 1 ? x[0] : x);
          } : x;
        }) : u = [yr];
        var m = -1;
        u = pt(u, Or(_e()));
        var S = kd(a, function(x, A, D) {
          var j = pt(u, function(J) {
            return J(x);
          });
          return { criteria: j, index: ++m, value: x };
        });
        return wb(S, function(x, A) {
          return uS(x, A, f);
        });
      }
      function X_(a, u) {
        return Nd(a, u, function(f, m) {
          return Kc(a, m);
        });
      }
      function Nd(a, u, f) {
        for (var m = -1, S = u.length, x = {}; ++m < S; ) {
          var A = u[m], D = ji(a, A);
          f(D, A) && Ts(x, di(A, a), D);
        }
        return x;
      }
      function J_(a) {
        return function(u) {
          return ji(u, a);
        };
      }
      function wc(a, u, f, m) {
        var S = m ? Sb : _a, x = -1, A = u.length, D = a;
        for (a === u && (u = pr(u)), f && (D = pt(a, Or(f))); ++x < A; )
          for (var j = 0, J = u[x], Q = f ? f(J) : J; (j = S(D, Q, j, m)) > -1; )
            D !== a && Qo.call(D, j, 1), Qo.call(a, j, 1);
        return a;
      }
      function Bd(a, u) {
        for (var f = a ? u.length : 0, m = f - 1; f--; ) {
          var S = u[f];
          if (f == m || S !== x) {
            var x = S;
            Hn(S) ? Qo.call(a, S, 1) : Oc(a, S);
          }
        }
        return a;
      }
      function Ec(a, u) {
        return a + tu(vd() * (u - a + 1));
      }
      function Q_(a, u, f, m) {
        for (var S = -1, x = Gt(eu((u - a) / (f || 1)), 0), A = q(x); x--; )
          A[m ? x : ++S] = a, a += f;
        return A;
      }
      function Cc(a, u) {
        var f = "";
        if (!a || u < 1 || u > ae)
          return f;
        do
          u % 2 && (f += a), u = tu(u / 2), u && (a += a);
        while (u);
        return f;
      }
      function Le(a, u) {
        return $c(dg(a, u, yr), a + "");
      }
      function Z_(a) {
        return _d(Ia(a));
      }
      function eS(a, u) {
        var f = Ia(a);
        return yu(f, Gi(u, 0, f.length));
      }
      function Ts(a, u, f, m) {
        if (!bt(a))
          return a;
        u = di(u, a);
        for (var S = -1, x = u.length, A = x - 1, D = a; D != null && ++S < x; ) {
          var j = _n(u[S]), J = f;
          if (j === "__proto__" || j === "constructor" || j === "prototype")
            return a;
          if (S != A) {
            var Q = D[j];
            J = m ? m(Q, j, D) : r, J === r && (J = bt(Q) ? Q : Hn(u[S + 1]) ? [] : {});
          }
          Es(D, j, J), D = D[j];
        }
        return a;
      }
      var Fd = ru ? function(a, u) {
        return ru.set(a, u), a;
      } : yr, tS = Zo ? function(a, u) {
        return Zo(a, "toString", {
          configurable: !0,
          enumerable: !1,
          value: Jc(u),
          writable: !0
        });
      } : yr;
      function rS(a) {
        return yu(Ia(a));
      }
      function Ur(a, u, f) {
        var m = -1, S = a.length;
        u < 0 && (u = -u > S ? 0 : S + u), f = f > S ? S : f, f < 0 && (f += S), S = u > f ? 0 : f - u >>> 0, u >>>= 0;
        for (var x = q(S); ++m < S; )
          x[m] = a[m + u];
        return x;
      }
      function nS(a, u) {
        var f;
        return fi(a, function(m, S, x) {
          return f = u(m, S, x), !f;
        }), !!f;
      }
      function lu(a, u, f) {
        var m = 0, S = a == null ? m : a.length;
        if (typeof u == "number" && u === u && S <= Ce) {
          for (; m < S; ) {
            var x = m + S >>> 1, A = a[x];
            A !== null && !Pr(A) && (f ? A <= u : A < u) ? m = x + 1 : S = x;
          }
          return S;
        }
        return xc(a, u, yr, f);
      }
      function xc(a, u, f, m) {
        var S = 0, x = a == null ? 0 : a.length;
        if (x === 0)
          return 0;
        u = f(u);
        for (var A = u !== u, D = u === null, j = Pr(u), J = u === r; S < x; ) {
          var Q = tu((S + x) / 2), ee = f(a[Q]), ge = ee !== r, ye = ee === null, we = ee === ee, ke = Pr(ee);
          if (A)
            var Ee = m || we;
          else
            J ? Ee = we && (m || ge) : D ? Ee = we && ge && (m || !ye) : j ? Ee = we && ge && !ye && (m || !ke) : ye || ke ? Ee = !1 : Ee = m ? ee <= u : ee < u;
          Ee ? S = Q + 1 : x = Q;
        }
        return rr(x, Re);
      }
      function Gd(a, u) {
        for (var f = -1, m = a.length, S = 0, x = []; ++f < m; ) {
          var A = a[f], D = u ? u(A) : A;
          if (!f || !tn(D, j)) {
            var j = D;
            x[S++] = A === 0 ? 0 : A;
          }
        }
        return x;
      }
      function jd(a) {
        return typeof a == "number" ? a : Pr(a) ? oe : +a;
      }
      function Tr(a) {
        if (typeof a == "string")
          return a;
        if (Oe(a))
          return pt(a, Tr) + "";
        if (Pr(a))
          return yd ? yd.call(a) : "";
        var u = a + "";
        return u == "0" && 1 / a == -re ? "-0" : u;
      }
      function hi(a, u, f) {
        var m = -1, S = Wo, x = a.length, A = !0, D = [], j = D;
        if (f)
          A = !1, S = ec;
        else if (x >= i) {
          var J = u ? null : gS(a);
          if (J)
            return Ho(J);
          A = !1, S = vs, j = new Fi();
        } else
          j = u ? [] : D;
        e:
          for (; ++m < x; ) {
            var Q = a[m], ee = u ? u(Q) : Q;
            if (Q = f || Q !== 0 ? Q : 0, A && ee === ee) {
              for (var ge = j.length; ge--; )
                if (j[ge] === ee)
                  continue e;
              u && j.push(ee), D.push(Q);
            } else
              S(j, ee, f) || (j !== D && j.push(ee), D.push(Q));
          }
        return D;
      }
      function Oc(a, u) {
        return u = di(u, a), a = gg(a, u), a == null || delete a[_n(Hr(u))];
      }
      function $d(a, u, f, m) {
        return Ts(a, u, f(ji(a, u)), m);
      }
      function cu(a, u, f, m) {
        for (var S = a.length, x = m ? S : -1; (m ? x-- : ++x < S) && u(a[x], x, a); )
          ;
        return f ? Ur(a, m ? 0 : x, m ? x + 1 : S) : Ur(a, m ? x + 1 : 0, m ? S : x);
      }
      function Wd(a, u) {
        var f = a;
        return f instanceof Ge && (f = f.value()), tc(u, function(m, S) {
          return S.func.apply(S.thisArg, ui([m], S.args));
        }, f);
      }
      function Tc(a, u, f) {
        var m = a.length;
        if (m < 2)
          return m ? hi(a[0]) : [];
        for (var S = -1, x = q(m); ++S < m; )
          for (var A = a[S], D = -1; ++D < m; )
            D != S && (x[S] = Cs(x[S] || A, a[D], u, f));
        return hi(Kt(x, 1), u, f);
      }
      function Ud(a, u, f) {
        for (var m = -1, S = a.length, x = u.length, A = {}; ++m < S; ) {
          var D = m < x ? u[m] : r;
          f(A, a[m], D);
        }
        return A;
      }
      function Pc(a) {
        return Tt(a) ? a : [];
      }
      function Rc(a) {
        return typeof a == "function" ? a : yr;
      }
      function di(a, u) {
        return Oe(a) ? a : Fc(a, u) ? [a] : yg(Ke(a));
      }
      var iS = Le;
      function gi(a, u, f) {
        var m = a.length;
        return f = f === r ? m : f, !u && f >= m ? a : Ur(a, u, f);
      }
      var Hd = zb || function(a) {
        return Te.clearTimeout(a);
      };
      function zd(a, u) {
        if (u)
          return a.slice();
        var f = a.length, m = hd ? hd(f) : new a.constructor(f);
        return a.copy(m), m;
      }
      function Ac(a) {
        var u = new a.constructor(a.byteLength);
        return new Xo(u).set(new Xo(a)), u;
      }
      function aS(a, u) {
        var f = u ? Ac(a.buffer) : a.buffer;
        return new a.constructor(f, a.byteOffset, a.byteLength);
      }
      function sS(a) {
        var u = new a.constructor(a.source, Ne.exec(a));
        return u.lastIndex = a.lastIndex, u;
      }
      function oS(a) {
        return ws ? lt(ws.call(a)) : {};
      }
      function Vd(a, u) {
        var f = u ? Ac(a.buffer) : a.buffer;
        return new a.constructor(f, a.byteOffset, a.length);
      }
      function qd(a, u) {
        if (a !== u) {
          var f = a !== r, m = a === null, S = a === a, x = Pr(a), A = u !== r, D = u === null, j = u === u, J = Pr(u);
          if (!D && !J && !x && a > u || x && A && j && !D && !J || m && A && j || !f && j || !S)
            return 1;
          if (!m && !x && !J && a < u || J && f && S && !m && !x || D && f && S || !A && S || !j)
            return -1;
        }
        return 0;
      }
      function uS(a, u, f) {
        for (var m = -1, S = a.criteria, x = u.criteria, A = S.length, D = f.length; ++m < A; ) {
          var j = qd(S[m], x[m]);
          if (j) {
            if (m >= D)
              return j;
            var J = f[m];
            return j * (J == "desc" ? -1 : 1);
          }
        }
        return a.index - u.index;
      }
      function Yd(a, u, f, m) {
        for (var S = -1, x = a.length, A = f.length, D = -1, j = u.length, J = Gt(x - A, 0), Q = q(j + J), ee = !m; ++D < j; )
          Q[D] = u[D];
        for (; ++S < A; )
          (ee || S < x) && (Q[f[S]] = a[S]);
        for (; J--; )
          Q[D++] = a[S++];
        return Q;
      }
      function Kd(a, u, f, m) {
        for (var S = -1, x = a.length, A = -1, D = f.length, j = -1, J = u.length, Q = Gt(x - D, 0), ee = q(Q + J), ge = !m; ++S < Q; )
          ee[S] = a[S];
        for (var ye = S; ++j < J; )
          ee[ye + j] = u[j];
        for (; ++A < D; )
          (ge || S < x) && (ee[ye + f[A]] = a[S++]);
        return ee;
      }
      function pr(a, u) {
        var f = -1, m = a.length;
        for (u || (u = q(m)); ++f < m; )
          u[f] = a[f];
        return u;
      }
      function bn(a, u, f, m) {
        var S = !f;
        f || (f = {});
        for (var x = -1, A = u.length; ++x < A; ) {
          var D = u[x], j = m ? m(f[D], a[D], D, f, a) : r;
          j === r && (j = a[D]), S ? $n(f, D, j) : Es(f, D, j);
        }
        return f;
      }
      function lS(a, u) {
        return bn(a, Bc(a), u);
      }
      function cS(a, u) {
        return bn(a, ug(a), u);
      }
      function fu(a, u) {
        return function(f, m) {
          var S = Oe(f) ? pb : I_, x = u ? u() : {};
          return S(f, a, _e(m, 2), x);
        };
      }
      function Pa(a) {
        return Le(function(u, f) {
          var m = -1, S = f.length, x = S > 1 ? f[S - 1] : r, A = S > 2 ? f[2] : r;
          for (x = a.length > 3 && typeof x == "function" ? (S--, x) : r, A && cr(f[0], f[1], A) && (x = S < 3 ? r : x, S = 1), u = lt(u); ++m < S; ) {
            var D = f[m];
            D && a(u, D, m, x);
          }
          return u;
        });
      }
      function Xd(a, u) {
        return function(f, m) {
          if (f == null)
            return f;
          if (!mr(f))
            return a(f, m);
          for (var S = f.length, x = u ? S : -1, A = lt(f); (u ? x-- : ++x < S) && m(A[x], x, A) !== !1; )
            ;
          return f;
        };
      }
      function Jd(a) {
        return function(u, f, m) {
          for (var S = -1, x = lt(u), A = m(u), D = A.length; D--; ) {
            var j = A[a ? D : ++S];
            if (f(x[j], j, x) === !1)
              break;
          }
          return u;
        };
      }
      function fS(a, u, f) {
        var m = u & b, S = Ps(a);
        function x() {
          var A = this && this !== Te && this instanceof x ? S : a;
          return A.apply(m ? f : this, arguments);
        }
        return x;
      }
      function Qd(a) {
        return function(u) {
          u = Ke(u);
          var f = Sa(u) ? Zr(u) : r, m = f ? f[0] : u.charAt(0), S = f ? gi(f, 1).join("") : u.slice(1);
          return m[a]() + S;
        };
      }
      function Ra(a) {
        return function(u) {
          return tc(Xg(Kg(u).replace(ps, "")), a, "");
        };
      }
      function Ps(a) {
        return function() {
          var u = arguments;
          switch (u.length) {
            case 0:
              return new a();
            case 1:
              return new a(u[0]);
            case 2:
              return new a(u[0], u[1]);
            case 3:
              return new a(u[0], u[1], u[2]);
            case 4:
              return new a(u[0], u[1], u[2], u[3]);
            case 5:
              return new a(u[0], u[1], u[2], u[3], u[4]);
            case 6:
              return new a(u[0], u[1], u[2], u[3], u[4], u[5]);
            case 7:
              return new a(u[0], u[1], u[2], u[3], u[4], u[5], u[6]);
          }
          var f = Ta(a.prototype), m = a.apply(f, u);
          return bt(m) ? m : f;
        };
      }
      function hS(a, u, f) {
        var m = Ps(a);
        function S() {
          for (var x = arguments.length, A = q(x), D = x, j = Aa(S); D--; )
            A[D] = arguments[D];
          var J = x < 3 && A[0] !== j && A[x - 1] !== j ? [] : li(A, j);
          if (x -= J.length, x < f)
            return ng(
              a,
              u,
              hu,
              S.placeholder,
              r,
              A,
              J,
              r,
              r,
              f - x
            );
          var Q = this && this !== Te && this instanceof S ? m : a;
          return xr(Q, this, A);
        }
        return S;
      }
      function Zd(a) {
        return function(u, f, m) {
          var S = lt(u);
          if (!mr(u)) {
            var x = _e(f, 3);
            u = Ut(u), f = function(D) {
              return x(S[D], D, S);
            };
          }
          var A = a(u, f, m);
          return A > -1 ? S[x ? u[A] : A] : r;
        };
      }
      function eg(a) {
        return Un(function(u) {
          var f = u.length, m = f, S = $r.prototype.thru;
          for (a && u.reverse(); m--; ) {
            var x = u[m];
            if (typeof x != "function")
              throw new jr(o);
            if (S && !A && mu(x) == "wrapper")
              var A = new $r([], !0);
          }
          for (m = A ? m : f; ++m < f; ) {
            x = u[m];
            var D = mu(x), j = D == "wrapper" ? Mc(x) : r;
            j && Gc(j[0]) && j[1] == (U | O | P | V) && !j[4].length && j[9] == 1 ? A = A[mu(j[0])].apply(A, j[3]) : A = x.length == 1 && Gc(x) ? A[D]() : A.thru(x);
          }
          return function() {
            var J = arguments, Q = J[0];
            if (A && J.length == 1 && Oe(Q))
              return A.plant(Q).value();
            for (var ee = 0, ge = f ? u[ee].apply(this, J) : Q; ++ee < f; )
              ge = u[ee].call(this, ge);
            return ge;
          };
        });
      }
      function hu(a, u, f, m, S, x, A, D, j, J) {
        var Q = u & U, ee = u & b, ge = u & C, ye = u & (O | k), we = u & Y, ke = ge ? r : Ps(a);
        function Ee() {
          for (var Fe = arguments.length, $e = q(Fe), Rr = Fe; Rr--; )
            $e[Rr] = arguments[Rr];
          if (ye)
            var fr = Aa(Ee), Ar = Cb($e, fr);
          if (m && ($e = Yd($e, m, S, ye)), x && ($e = Kd($e, x, A, ye)), Fe -= Ar, ye && Fe < J) {
            var Pt = li($e, fr);
            return ng(
              a,
              u,
              hu,
              Ee.placeholder,
              f,
              $e,
              Pt,
              D,
              j,
              J - Fe
            );
          }
          var rn = ee ? f : this, qn = ge ? rn[a] : a;
          return Fe = $e.length, D ? $e = IS($e, D) : we && Fe > 1 && $e.reverse(), Q && j < Fe && ($e.length = j), this && this !== Te && this instanceof Ee && (qn = ke || Ps(qn)), qn.apply(rn, $e);
        }
        return Ee;
      }
      function tg(a, u) {
        return function(f, m) {
          return j_(f, a, u(m), {});
        };
      }
      function du(a, u) {
        return function(f, m) {
          var S;
          if (f === r && m === r)
            return u;
          if (f !== r && (S = f), m !== r) {
            if (S === r)
              return m;
            typeof f == "string" || typeof m == "string" ? (f = Tr(f), m = Tr(m)) : (f = jd(f), m = jd(m)), S = a(f, m);
          }
          return S;
        };
      }
      function kc(a) {
        return Un(function(u) {
          return u = pt(u, Or(_e())), Le(function(f) {
            var m = this;
            return a(u, function(S) {
              return xr(S, m, f);
            });
          });
        });
      }
      function gu(a, u) {
        u = u === r ? " " : Tr(u);
        var f = u.length;
        if (f < 2)
          return f ? Cc(u, a) : u;
        var m = Cc(u, eu(a / wa(u)));
        return Sa(u) ? gi(Zr(m), 0, a).join("") : m.slice(0, a);
      }
      function dS(a, u, f, m) {
        var S = u & b, x = Ps(a);
        function A() {
          for (var D = -1, j = arguments.length, J = -1, Q = m.length, ee = q(Q + j), ge = this && this !== Te && this instanceof A ? x : a; ++J < Q; )
            ee[J] = m[J];
          for (; j--; )
            ee[J++] = arguments[++D];
          return xr(ge, S ? f : this, ee);
        }
        return A;
      }
      function rg(a) {
        return function(u, f, m) {
          return m && typeof m != "number" && cr(u, f, m) && (f = m = r), u = Vn(u), f === r ? (f = u, u = 0) : f = Vn(f), m = m === r ? u < f ? 1 : -1 : Vn(m), Q_(u, f, m, a);
        };
      }
      function pu(a) {
        return function(u, f) {
          return typeof u == "string" && typeof f == "string" || (u = zr(u), f = zr(f)), a(u, f);
        };
      }
      function ng(a, u, f, m, S, x, A, D, j, J) {
        var Q = u & O, ee = Q ? A : r, ge = Q ? r : A, ye = Q ? x : r, we = Q ? r : x;
        u |= Q ? P : M, u &= ~(Q ? M : P), u & w || (u &= ~(b | C));
        var ke = [
          a,
          u,
          S,
          ye,
          ee,
          we,
          ge,
          D,
          j,
          J
        ], Ee = f.apply(r, ke);
        return Gc(a) && pg(Ee, ke), Ee.placeholder = m, mg(Ee, a, u);
      }
      function Ic(a) {
        var u = Ft[a];
        return function(f, m) {
          if (f = zr(f), m = m == null ? 0 : rr(Pe(m), 292), m && md(f)) {
            var S = (Ke(f) + "e").split("e"), x = u(S[0] + "e" + (+S[1] + m));
            return S = (Ke(x) + "e").split("e"), +(S[0] + "e" + (+S[1] - m));
          }
          return u(f);
        };
      }
      var gS = xa && 1 / Ho(new xa([, -0]))[1] == re ? function(a) {
        return new xa(a);
      } : ef;
      function ig(a) {
        return function(u) {
          var f = nr(u);
          return f == fe ? uc(u) : f == Mt ? kb(u) : Eb(u, a(u));
        };
      }
      function Wn(a, u, f, m, S, x, A, D) {
        var j = u & C;
        if (!j && typeof a != "function")
          throw new jr(o);
        var J = m ? m.length : 0;
        if (J || (u &= ~(P | M), m = S = r), A = A === r ? A : Gt(Pe(A), 0), D = D === r ? D : Pe(D), J -= S ? S.length : 0, u & M) {
          var Q = m, ee = S;
          m = S = r;
        }
        var ge = j ? r : Mc(a), ye = [
          a,
          u,
          f,
          m,
          S,
          Q,
          ee,
          x,
          A,
          D
        ];
        if (ge && RS(ye, ge), a = ye[0], u = ye[1], f = ye[2], m = ye[3], S = ye[4], D = ye[9] = ye[9] === r ? j ? 0 : a.length : Gt(ye[9] - J, 0), !D && u & (O | k) && (u &= ~(O | k)), !u || u == b)
          var we = fS(a, u, f);
        else
          u == O || u == k ? we = hS(a, u, D) : (u == P || u == (b | P)) && !S.length ? we = dS(a, u, f, m) : we = hu.apply(r, ye);
        var ke = ge ? Fd : pg;
        return mg(ke(we, ye), a, u);
      }
      function ag(a, u, f, m) {
        return a === r || tn(a, Ca[f]) && !it.call(m, f) ? u : a;
      }
      function sg(a, u, f, m, S, x) {
        return bt(a) && bt(u) && (x.set(u, a), uu(a, u, r, sg, x), x.delete(u)), a;
      }
      function pS(a) {
        return ks(a) ? r : a;
      }
      function og(a, u, f, m, S, x) {
        var A = f & v, D = a.length, j = u.length;
        if (D != j && !(A && j > D))
          return !1;
        var J = x.get(a), Q = x.get(u);
        if (J && Q)
          return J == u && Q == a;
        var ee = -1, ge = !0, ye = f & _ ? new Fi() : r;
        for (x.set(a, u), x.set(u, a); ++ee < D; ) {
          var we = a[ee], ke = u[ee];
          if (m)
            var Ee = A ? m(ke, we, ee, u, a, x) : m(we, ke, ee, a, u, x);
          if (Ee !== r) {
            if (Ee)
              continue;
            ge = !1;
            break;
          }
          if (ye) {
            if (!rc(u, function(Fe, $e) {
              if (!vs(ye, $e) && (we === Fe || S(we, Fe, f, m, x)))
                return ye.push($e);
            })) {
              ge = !1;
              break;
            }
          } else if (!(we === ke || S(we, ke, f, m, x))) {
            ge = !1;
            break;
          }
        }
        return x.delete(a), x.delete(u), ge;
      }
      function mS(a, u, f, m, S, x, A) {
        switch (f) {
          case wr:
            if (a.byteLength != u.byteLength || a.byteOffset != u.byteOffset)
              return !1;
            a = a.buffer, u = u.buffer;
          case er:
            return !(a.byteLength != u.byteLength || !x(new Xo(a), new Xo(u)));
          case ot:
          case wt:
          case Ot:
            return tn(+a, +u);
          case dt:
            return a.name == u.name && a.message == u.message;
          case gr:
          case qt:
            return a == u + "";
          case fe:
            var D = uc;
          case Mt:
            var j = m & v;
            if (D || (D = Ho), a.size != u.size && !j)
              return !1;
            var J = A.get(a);
            if (J)
              return J == u;
            m |= _, A.set(a, u);
            var Q = og(D(a), D(u), m, S, x, A);
            return A.delete(a), Q;
          case yt:
            if (ws)
              return ws.call(a) == ws.call(u);
        }
        return !1;
      }
      function vS(a, u, f, m, S, x) {
        var A = f & v, D = Lc(a), j = D.length, J = Lc(u), Q = J.length;
        if (j != Q && !A)
          return !1;
        for (var ee = j; ee--; ) {
          var ge = D[ee];
          if (!(A ? ge in u : it.call(u, ge)))
            return !1;
        }
        var ye = x.get(a), we = x.get(u);
        if (ye && we)
          return ye == u && we == a;
        var ke = !0;
        x.set(a, u), x.set(u, a);
        for (var Ee = A; ++ee < j; ) {
          ge = D[ee];
          var Fe = a[ge], $e = u[ge];
          if (m)
            var Rr = A ? m($e, Fe, ge, u, a, x) : m(Fe, $e, ge, a, u, x);
          if (!(Rr === r ? Fe === $e || S(Fe, $e, f, m, x) : Rr)) {
            ke = !1;
            break;
          }
          Ee || (Ee = ge == "constructor");
        }
        if (ke && !Ee) {
          var fr = a.constructor, Ar = u.constructor;
          fr != Ar && "constructor" in a && "constructor" in u && !(typeof fr == "function" && fr instanceof fr && typeof Ar == "function" && Ar instanceof Ar) && (ke = !1);
        }
        return x.delete(a), x.delete(u), ke;
      }
      function Un(a) {
        return $c(dg(a, r, wg), a + "");
      }
      function Lc(a) {
        return Td(a, Ut, Bc);
      }
      function Dc(a) {
        return Td(a, vr, ug);
      }
      var Mc = ru ? function(a) {
        return ru.get(a);
      } : ef;
      function mu(a) {
        for (var u = a.name + "", f = Oa[u], m = it.call(Oa, u) ? f.length : 0; m--; ) {
          var S = f[m], x = S.func;
          if (x == null || x == a)
            return S.name;
        }
        return u;
      }
      function Aa(a) {
        var u = it.call(E, "placeholder") ? E : a;
        return u.placeholder;
      }
      function _e() {
        var a = E.iteratee || Qc;
        return a = a === Qc ? Ad : a, arguments.length ? a(arguments[0], arguments[1]) : a;
      }
      function vu(a, u) {
        var f = a.__data__;
        return xS(u) ? f[typeof u == "string" ? "string" : "hash"] : f.map;
      }
      function Nc(a) {
        for (var u = Ut(a), f = u.length; f--; ) {
          var m = u[f], S = a[m];
          u[f] = [m, S, fg(S)];
        }
        return u;
      }
      function $i(a, u) {
        var f = Pb(a, u);
        return Rd(f) ? f : r;
      }
      function yS(a) {
        var u = it.call(a, Ni), f = a[Ni];
        try {
          a[Ni] = r;
          var m = !0;
        } catch {
        }
        var S = Yo.call(a);
        return m && (u ? a[Ni] = f : delete a[Ni]), S;
      }
      var Bc = cc ? function(a) {
        return a == null ? [] : (a = lt(a), oi(cc(a), function(u) {
          return gd.call(a, u);
        }));
      } : tf, ug = cc ? function(a) {
        for (var u = []; a; )
          ui(u, Bc(a)), a = Jo(a);
        return u;
      } : tf, nr = lr;
      (fc && nr(new fc(new ArrayBuffer(1))) != wr || bs && nr(new bs()) != fe || hc && nr(hc.resolve()) != dn || xa && nr(new xa()) != Mt || _s && nr(new _s()) != Sr) && (nr = function(a) {
        var u = lr(a), f = u == Dt ? a.constructor : r, m = f ? Wi(f) : "";
        if (m)
          switch (m) {
            case e_:
              return wr;
            case t_:
              return fe;
            case r_:
              return dn;
            case n_:
              return Mt;
            case i_:
              return Sr;
          }
        return u;
      });
      function bS(a, u, f) {
        for (var m = -1, S = f.length; ++m < S; ) {
          var x = f[m], A = x.size;
          switch (x.type) {
            case "drop":
              a += A;
              break;
            case "dropRight":
              u -= A;
              break;
            case "take":
              u = rr(u, a + A);
              break;
            case "takeRight":
              a = Gt(a, u - A);
              break;
          }
        }
        return { start: a, end: u };
      }
      function _S(a) {
        var u = a.match(vn);
        return u ? u[1].split(L) : [];
      }
      function lg(a, u, f) {
        u = di(u, a);
        for (var m = -1, S = u.length, x = !1; ++m < S; ) {
          var A = _n(u[m]);
          if (!(x = a != null && f(a, A)))
            break;
          a = a[A];
        }
        return x || ++m != S ? x : (S = a == null ? 0 : a.length, !!S && Cu(S) && Hn(A, S) && (Oe(a) || Ui(a)));
      }
      function SS(a) {
        var u = a.length, f = new a.constructor(u);
        return u && typeof a[0] == "string" && it.call(a, "index") && (f.index = a.index, f.input = a.input), f;
      }
      function cg(a) {
        return typeof a.constructor == "function" && !Rs(a) ? Ta(Jo(a)) : {};
      }
      function wS(a, u, f) {
        var m = a.constructor;
        switch (u) {
          case er:
            return Ac(a);
          case ot:
          case wt:
            return new m(+a);
          case wr:
            return aS(a, f);
          case Nn:
          case Bn:
          case pn:
          case ii:
          case ai:
          case Ai:
          case ki:
          case Er:
          case mn:
            return Vd(a, f);
          case fe:
            return new m();
          case Ot:
          case qt:
            return new m(a);
          case gr:
            return sS(a);
          case Mt:
            return new m();
          case yt:
            return oS(a);
        }
      }
      function ES(a, u) {
        var f = u.length;
        if (!f)
          return a;
        var m = f - 1;
        return u[m] = (f > 1 ? "& " : "") + u[m], u = u.join(f > 2 ? ", " : " "), a.replace(ss, `{
/* [wrapped with ` + u + `] */
`);
      }
      function CS(a) {
        return Oe(a) || Ui(a) || !!(pd && a && a[pd]);
      }
      function Hn(a, u) {
        var f = typeof a;
        return u = u ?? ae, !!u && (f == "number" || f != "symbol" && gt.test(a)) && a > -1 && a % 1 == 0 && a < u;
      }
      function cr(a, u, f) {
        if (!bt(f))
          return !1;
        var m = typeof u;
        return (m == "number" ? mr(f) && Hn(u, f.length) : m == "string" && u in f) ? tn(f[u], a) : !1;
      }
      function Fc(a, u) {
        if (Oe(a))
          return !1;
        var f = typeof a;
        return f == "number" || f == "symbol" || f == "boolean" || a == null || Pr(a) ? !0 : ua.test(a) || !oa.test(a) || u != null && a in lt(u);
      }
      function xS(a) {
        var u = typeof a;
        return u == "string" || u == "number" || u == "symbol" || u == "boolean" ? a !== "__proto__" : a === null;
      }
      function Gc(a) {
        var u = mu(a), f = E[u];
        if (typeof f != "function" || !(u in Ge.prototype))
          return !1;
        if (a === f)
          return !0;
        var m = Mc(f);
        return !!m && a === m[0];
      }
      function OS(a) {
        return !!fd && fd in a;
      }
      var TS = Vo ? zn : rf;
      function Rs(a) {
        var u = a && a.constructor, f = typeof u == "function" && u.prototype || Ca;
        return a === f;
      }
      function fg(a) {
        return a === a && !bt(a);
      }
      function hg(a, u) {
        return function(f) {
          return f == null ? !1 : f[a] === u && (u !== r || a in lt(f));
        };
      }
      function PS(a) {
        var u = wu(a, function(m) {
          return f.size === h && f.clear(), m;
        }), f = u.cache;
        return u;
      }
      function RS(a, u) {
        var f = a[1], m = u[1], S = f | m, x = S < (b | C | U), A = m == U && f == O || m == U && f == V && a[7].length <= u[8] || m == (U | V) && u[7].length <= u[8] && f == O;
        if (!(x || A))
          return a;
        m & b && (a[2] = u[2], S |= f & b ? 0 : w);
        var D = u[3];
        if (D) {
          var j = a[3];
          a[3] = j ? Yd(j, D, u[4]) : D, a[4] = j ? li(a[3], d) : u[4];
        }
        return D = u[5], D && (j = a[5], a[5] = j ? Kd(j, D, u[6]) : D, a[6] = j ? li(a[5], d) : u[6]), D = u[7], D && (a[7] = D), m & U && (a[8] = a[8] == null ? u[8] : rr(a[8], u[8])), a[9] == null && (a[9] = u[9]), a[0] = u[0], a[1] = S, a;
      }
      function AS(a) {
        var u = [];
        if (a != null)
          for (var f in lt(a))
            u.push(f);
        return u;
      }
      function kS(a) {
        return Yo.call(a);
      }
      function dg(a, u, f) {
        return u = Gt(u === r ? a.length - 1 : u, 0), function() {
          for (var m = arguments, S = -1, x = Gt(m.length - u, 0), A = q(x); ++S < x; )
            A[S] = m[u + S];
          S = -1;
          for (var D = q(u + 1); ++S < u; )
            D[S] = m[S];
          return D[u] = f(A), xr(a, this, D);
        };
      }
      function gg(a, u) {
        return u.length < 2 ? a : ji(a, Ur(u, 0, -1));
      }
      function IS(a, u) {
        for (var f = a.length, m = rr(u.length, f), S = pr(a); m--; ) {
          var x = u[m];
          a[m] = Hn(x, f) ? S[x] : r;
        }
        return a;
      }
      function jc(a, u) {
        if (!(u === "constructor" && typeof a[u] == "function") && u != "__proto__")
          return a[u];
      }
      var pg = vg(Fd), As = qb || function(a, u) {
        return Te.setTimeout(a, u);
      }, $c = vg(tS);
      function mg(a, u, f) {
        var m = u + "";
        return $c(a, ES(m, LS(_S(m), f)));
      }
      function vg(a) {
        var u = 0, f = 0;
        return function() {
          var m = Jb(), S = $ - (m - f);
          if (f = m, S > 0) {
            if (++u >= ce)
              return arguments[0];
          } else
            u = 0;
          return a.apply(r, arguments);
        };
      }
      function yu(a, u) {
        var f = -1, m = a.length, S = m - 1;
        for (u = u === r ? m : u; ++f < u; ) {
          var x = Ec(f, S), A = a[x];
          a[x] = a[f], a[f] = A;
        }
        return a.length = u, a;
      }
      var yg = PS(function(a) {
        var u = [];
        return a.charCodeAt(0) === 46 && u.push(""), a.replace(Cr, function(f, m, S, x) {
          u.push(S ? x.replace(Se, "$1") : m || f);
        }), u;
      });
      function _n(a) {
        if (typeof a == "string" || Pr(a))
          return a;
        var u = a + "";
        return u == "0" && 1 / a == -re ? "-0" : u;
      }
      function Wi(a) {
        if (a != null) {
          try {
            return qo.call(a);
          } catch {
          }
          try {
            return a + "";
          } catch {
          }
        }
        return "";
      }
      function LS(a, u) {
        return Gr(rt, function(f) {
          var m = "_." + f[0];
          u & f[1] && !Wo(a, m) && a.push(m);
        }), a.sort();
      }
      function bg(a) {
        if (a instanceof Ge)
          return a.clone();
        var u = new $r(a.__wrapped__, a.__chain__);
        return u.__actions__ = pr(a.__actions__), u.__index__ = a.__index__, u.__values__ = a.__values__, u;
      }
      function DS(a, u, f) {
        (f ? cr(a, u, f) : u === r) ? u = 1 : u = Gt(Pe(u), 0);
        var m = a == null ? 0 : a.length;
        if (!m || u < 1)
          return [];
        for (var S = 0, x = 0, A = q(eu(m / u)); S < m; )
          A[x++] = Ur(a, S, S += u);
        return A;
      }
      function MS(a) {
        for (var u = -1, f = a == null ? 0 : a.length, m = 0, S = []; ++u < f; ) {
          var x = a[u];
          x && (S[m++] = x);
        }
        return S;
      }
      function NS() {
        var a = arguments.length;
        if (!a)
          return [];
        for (var u = q(a - 1), f = arguments[0], m = a; m--; )
          u[m - 1] = arguments[m];
        return ui(Oe(f) ? pr(f) : [f], Kt(u, 1));
      }
      var BS = Le(function(a, u) {
        return Tt(a) ? Cs(a, Kt(u, 1, Tt, !0)) : [];
      }), FS = Le(function(a, u) {
        var f = Hr(u);
        return Tt(f) && (f = r), Tt(a) ? Cs(a, Kt(u, 1, Tt, !0), _e(f, 2)) : [];
      }), GS = Le(function(a, u) {
        var f = Hr(u);
        return Tt(f) && (f = r), Tt(a) ? Cs(a, Kt(u, 1, Tt, !0), r, f) : [];
      });
      function jS(a, u, f) {
        var m = a == null ? 0 : a.length;
        return m ? (u = f || u === r ? 1 : Pe(u), Ur(a, u < 0 ? 0 : u, m)) : [];
      }
      function $S(a, u, f) {
        var m = a == null ? 0 : a.length;
        return m ? (u = f || u === r ? 1 : Pe(u), u = m - u, Ur(a, 0, u < 0 ? 0 : u)) : [];
      }
      function WS(a, u) {
        return a && a.length ? cu(a, _e(u, 3), !0, !0) : [];
      }
      function US(a, u) {
        return a && a.length ? cu(a, _e(u, 3), !0) : [];
      }
      function HS(a, u, f, m) {
        var S = a == null ? 0 : a.length;
        return S ? (f && typeof f != "number" && cr(a, u, f) && (f = 0, m = S), N_(a, u, f, m)) : [];
      }
      function _g(a, u, f) {
        var m = a == null ? 0 : a.length;
        if (!m)
          return -1;
        var S = f == null ? 0 : Pe(f);
        return S < 0 && (S = Gt(m + S, 0)), Uo(a, _e(u, 3), S);
      }
      function Sg(a, u, f) {
        var m = a == null ? 0 : a.length;
        if (!m)
          return -1;
        var S = m - 1;
        return f !== r && (S = Pe(f), S = f < 0 ? Gt(m + S, 0) : rr(S, m - 1)), Uo(a, _e(u, 3), S, !0);
      }
      function wg(a) {
        var u = a == null ? 0 : a.length;
        return u ? Kt(a, 1) : [];
      }
      function zS(a) {
        var u = a == null ? 0 : a.length;
        return u ? Kt(a, re) : [];
      }
      function VS(a, u) {
        var f = a == null ? 0 : a.length;
        return f ? (u = u === r ? 1 : Pe(u), Kt(a, u)) : [];
      }
      function qS(a) {
        for (var u = -1, f = a == null ? 0 : a.length, m = {}; ++u < f; ) {
          var S = a[u];
          m[S[0]] = S[1];
        }
        return m;
      }
      function Eg(a) {
        return a && a.length ? a[0] : r;
      }
      function YS(a, u, f) {
        var m = a == null ? 0 : a.length;
        if (!m)
          return -1;
        var S = f == null ? 0 : Pe(f);
        return S < 0 && (S = Gt(m + S, 0)), _a(a, u, S);
      }
      function KS(a) {
        var u = a == null ? 0 : a.length;
        return u ? Ur(a, 0, -1) : [];
      }
      var XS = Le(function(a) {
        var u = pt(a, Pc);
        return u.length && u[0] === a[0] ? yc(u) : [];
      }), JS = Le(function(a) {
        var u = Hr(a), f = pt(a, Pc);
        return u === Hr(f) ? u = r : f.pop(), f.length && f[0] === a[0] ? yc(f, _e(u, 2)) : [];
      }), QS = Le(function(a) {
        var u = Hr(a), f = pt(a, Pc);
        return u = typeof u == "function" ? u : r, u && f.pop(), f.length && f[0] === a[0] ? yc(f, r, u) : [];
      });
      function ZS(a, u) {
        return a == null ? "" : Kb.call(a, u);
      }
      function Hr(a) {
        var u = a == null ? 0 : a.length;
        return u ? a[u - 1] : r;
      }
      function e1(a, u, f) {
        var m = a == null ? 0 : a.length;
        if (!m)
          return -1;
        var S = m;
        return f !== r && (S = Pe(f), S = S < 0 ? Gt(m + S, 0) : rr(S, m - 1)), u === u ? Lb(a, u, S) : Uo(a, nd, S, !0);
      }
      function t1(a, u) {
        return a && a.length ? Dd(a, Pe(u)) : r;
      }
      var r1 = Le(Cg);
      function Cg(a, u) {
        return a && a.length && u && u.length ? wc(a, u) : a;
      }
      function n1(a, u, f) {
        return a && a.length && u && u.length ? wc(a, u, _e(f, 2)) : a;
      }
      function i1(a, u, f) {
        return a && a.length && u && u.length ? wc(a, u, r, f) : a;
      }
      var a1 = Un(function(a, u) {
        var f = a == null ? 0 : a.length, m = gc(a, u);
        return Bd(a, pt(u, function(S) {
          return Hn(S, f) ? +S : S;
        }).sort(qd)), m;
      });
      function s1(a, u) {
        var f = [];
        if (!(a && a.length))
          return f;
        var m = -1, S = [], x = a.length;
        for (u = _e(u, 3); ++m < x; ) {
          var A = a[m];
          u(A, m, a) && (f.push(A), S.push(m));
        }
        return Bd(a, S), f;
      }
      function Wc(a) {
        return a == null ? a : Zb.call(a);
      }
      function o1(a, u, f) {
        var m = a == null ? 0 : a.length;
        return m ? (f && typeof f != "number" && cr(a, u, f) ? (u = 0, f = m) : (u = u == null ? 0 : Pe(u), f = f === r ? m : Pe(f)), Ur(a, u, f)) : [];
      }
      function u1(a, u) {
        return lu(a, u);
      }
      function l1(a, u, f) {
        return xc(a, u, _e(f, 2));
      }
      function c1(a, u) {
        var f = a == null ? 0 : a.length;
        if (f) {
          var m = lu(a, u);
          if (m < f && tn(a[m], u))
            return m;
        }
        return -1;
      }
      function f1(a, u) {
        return lu(a, u, !0);
      }
      function h1(a, u, f) {
        return xc(a, u, _e(f, 2), !0);
      }
      function d1(a, u) {
        var f = a == null ? 0 : a.length;
        if (f) {
          var m = lu(a, u, !0) - 1;
          if (tn(a[m], u))
            return m;
        }
        return -1;
      }
      function g1(a) {
        return a && a.length ? Gd(a) : [];
      }
      function p1(a, u) {
        return a && a.length ? Gd(a, _e(u, 2)) : [];
      }
      function m1(a) {
        var u = a == null ? 0 : a.length;
        return u ? Ur(a, 1, u) : [];
      }
      function v1(a, u, f) {
        return a && a.length ? (u = f || u === r ? 1 : Pe(u), Ur(a, 0, u < 0 ? 0 : u)) : [];
      }
      function y1(a, u, f) {
        var m = a == null ? 0 : a.length;
        return m ? (u = f || u === r ? 1 : Pe(u), u = m - u, Ur(a, u < 0 ? 0 : u, m)) : [];
      }
      function b1(a, u) {
        return a && a.length ? cu(a, _e(u, 3), !1, !0) : [];
      }
      function _1(a, u) {
        return a && a.length ? cu(a, _e(u, 3)) : [];
      }
      var S1 = Le(function(a) {
        return hi(Kt(a, 1, Tt, !0));
      }), w1 = Le(function(a) {
        var u = Hr(a);
        return Tt(u) && (u = r), hi(Kt(a, 1, Tt, !0), _e(u, 2));
      }), E1 = Le(function(a) {
        var u = Hr(a);
        return u = typeof u == "function" ? u : r, hi(Kt(a, 1, Tt, !0), r, u);
      });
      function C1(a) {
        return a && a.length ? hi(a) : [];
      }
      function x1(a, u) {
        return a && a.length ? hi(a, _e(u, 2)) : [];
      }
      function O1(a, u) {
        return u = typeof u == "function" ? u : r, a && a.length ? hi(a, r, u) : [];
      }
      function Uc(a) {
        if (!(a && a.length))
          return [];
        var u = 0;
        return a = oi(a, function(f) {
          if (Tt(f))
            return u = Gt(f.length, u), !0;
        }), sc(u, function(f) {
          return pt(a, nc(f));
        });
      }
      function xg(a, u) {
        if (!(a && a.length))
          return [];
        var f = Uc(a);
        return u == null ? f : pt(f, function(m) {
          return xr(u, r, m);
        });
      }
      var T1 = Le(function(a, u) {
        return Tt(a) ? Cs(a, u) : [];
      }), P1 = Le(function(a) {
        return Tc(oi(a, Tt));
      }), R1 = Le(function(a) {
        var u = Hr(a);
        return Tt(u) && (u = r), Tc(oi(a, Tt), _e(u, 2));
      }), A1 = Le(function(a) {
        var u = Hr(a);
        return u = typeof u == "function" ? u : r, Tc(oi(a, Tt), r, u);
      }), k1 = Le(Uc);
      function I1(a, u) {
        return Ud(a || [], u || [], Es);
      }
      function L1(a, u) {
        return Ud(a || [], u || [], Ts);
      }
      var D1 = Le(function(a) {
        var u = a.length, f = u > 1 ? a[u - 1] : r;
        return f = typeof f == "function" ? (a.pop(), f) : r, xg(a, f);
      });
      function Og(a) {
        var u = E(a);
        return u.__chain__ = !0, u;
      }
      function M1(a, u) {
        return u(a), a;
      }
      function bu(a, u) {
        return u(a);
      }
      var N1 = Un(function(a) {
        var u = a.length, f = u ? a[0] : 0, m = this.__wrapped__, S = function(x) {
          return gc(x, a);
        };
        return u > 1 || this.__actions__.length || !(m instanceof Ge) || !Hn(f) ? this.thru(S) : (m = m.slice(f, +f + (u ? 1 : 0)), m.__actions__.push({
          func: bu,
          args: [S],
          thisArg: r
        }), new $r(m, this.__chain__).thru(function(x) {
          return u && !x.length && x.push(r), x;
        }));
      });
      function B1() {
        return Og(this);
      }
      function F1() {
        return new $r(this.value(), this.__chain__);
      }
      function G1() {
        this.__values__ === r && (this.__values__ = jg(this.value()));
        var a = this.__index__ >= this.__values__.length, u = a ? r : this.__values__[this.__index__++];
        return { done: a, value: u };
      }
      function j1() {
        return this;
      }
      function $1(a) {
        for (var u, f = this; f instanceof iu; ) {
          var m = bg(f);
          m.__index__ = 0, m.__values__ = r, u ? S.__wrapped__ = m : u = m;
          var S = m;
          f = f.__wrapped__;
        }
        return S.__wrapped__ = a, u;
      }
      function W1() {
        var a = this.__wrapped__;
        if (a instanceof Ge) {
          var u = a;
          return this.__actions__.length && (u = new Ge(this)), u = u.reverse(), u.__actions__.push({
            func: bu,
            args: [Wc],
            thisArg: r
          }), new $r(u, this.__chain__);
        }
        return this.thru(Wc);
      }
      function U1() {
        return Wd(this.__wrapped__, this.__actions__);
      }
      var H1 = fu(function(a, u, f) {
        it.call(a, f) ? ++a[f] : $n(a, f, 1);
      });
      function z1(a, u, f) {
        var m = Oe(a) ? td : M_;
        return f && cr(a, u, f) && (u = r), m(a, _e(u, 3));
      }
      function V1(a, u) {
        var f = Oe(a) ? oi : xd;
        return f(a, _e(u, 3));
      }
      var q1 = Zd(_g), Y1 = Zd(Sg);
      function K1(a, u) {
        return Kt(_u(a, u), 1);
      }
      function X1(a, u) {
        return Kt(_u(a, u), re);
      }
      function J1(a, u, f) {
        return f = f === r ? 1 : Pe(f), Kt(_u(a, u), f);
      }
      function Tg(a, u) {
        var f = Oe(a) ? Gr : fi;
        return f(a, _e(u, 3));
      }
      function Pg(a, u) {
        var f = Oe(a) ? mb : Cd;
        return f(a, _e(u, 3));
      }
      var Q1 = fu(function(a, u, f) {
        it.call(a, f) ? a[f].push(u) : $n(a, f, [u]);
      });
      function Z1(a, u, f, m) {
        a = mr(a) ? a : Ia(a), f = f && !m ? Pe(f) : 0;
        var S = a.length;
        return f < 0 && (f = Gt(S + f, 0)), xu(a) ? f <= S && a.indexOf(u, f) > -1 : !!S && _a(a, u, f) > -1;
      }
      var ew = Le(function(a, u, f) {
        var m = -1, S = typeof u == "function", x = mr(a) ? q(a.length) : [];
        return fi(a, function(A) {
          x[++m] = S ? xr(u, A, f) : xs(A, u, f);
        }), x;
      }), tw = fu(function(a, u, f) {
        $n(a, f, u);
      });
      function _u(a, u) {
        var f = Oe(a) ? pt : kd;
        return f(a, _e(u, 3));
      }
      function rw(a, u, f, m) {
        return a == null ? [] : (Oe(u) || (u = u == null ? [] : [u]), f = m ? r : f, Oe(f) || (f = f == null ? [] : [f]), Md(a, u, f));
      }
      var nw = fu(function(a, u, f) {
        a[f ? 0 : 1].push(u);
      }, function() {
        return [[], []];
      });
      function iw(a, u, f) {
        var m = Oe(a) ? tc : ad, S = arguments.length < 3;
        return m(a, _e(u, 4), f, S, fi);
      }
      function aw(a, u, f) {
        var m = Oe(a) ? vb : ad, S = arguments.length < 3;
        return m(a, _e(u, 4), f, S, Cd);
      }
      function sw(a, u) {
        var f = Oe(a) ? oi : xd;
        return f(a, Eu(_e(u, 3)));
      }
      function ow(a) {
        var u = Oe(a) ? _d : Z_;
        return u(a);
      }
      function uw(a, u, f) {
        (f ? cr(a, u, f) : u === r) ? u = 1 : u = Pe(u);
        var m = Oe(a) ? A_ : eS;
        return m(a, u);
      }
      function lw(a) {
        var u = Oe(a) ? k_ : rS;
        return u(a);
      }
      function cw(a) {
        if (a == null)
          return 0;
        if (mr(a))
          return xu(a) ? wa(a) : a.length;
        var u = nr(a);
        return u == fe || u == Mt ? a.size : _c(a).length;
      }
      function fw(a, u, f) {
        var m = Oe(a) ? rc : nS;
        return f && cr(a, u, f) && (u = r), m(a, _e(u, 3));
      }
      var hw = Le(function(a, u) {
        if (a == null)
          return [];
        var f = u.length;
        return f > 1 && cr(a, u[0], u[1]) ? u = [] : f > 2 && cr(u[0], u[1], u[2]) && (u = [u[0]]), Md(a, Kt(u, 1), []);
      }), Su = Vb || function() {
        return Te.Date.now();
      };
      function dw(a, u) {
        if (typeof u != "function")
          throw new jr(o);
        return a = Pe(a), function() {
          if (--a < 1)
            return u.apply(this, arguments);
        };
      }
      function Rg(a, u, f) {
        return u = f ? r : u, u = a && u == null ? a.length : u, Wn(a, U, r, r, r, r, u);
      }
      function Ag(a, u) {
        var f;
        if (typeof u != "function")
          throw new jr(o);
        return a = Pe(a), function() {
          return --a > 0 && (f = u.apply(this, arguments)), a <= 1 && (u = r), f;
        };
      }
      var Hc = Le(function(a, u, f) {
        var m = b;
        if (f.length) {
          var S = li(f, Aa(Hc));
          m |= P;
        }
        return Wn(a, m, u, f, S);
      }), kg = Le(function(a, u, f) {
        var m = b | C;
        if (f.length) {
          var S = li(f, Aa(kg));
          m |= P;
        }
        return Wn(u, m, a, f, S);
      });
      function Ig(a, u, f) {
        u = f ? r : u;
        var m = Wn(a, O, r, r, r, r, r, u);
        return m.placeholder = Ig.placeholder, m;
      }
      function Lg(a, u, f) {
        u = f ? r : u;
        var m = Wn(a, k, r, r, r, r, r, u);
        return m.placeholder = Lg.placeholder, m;
      }
      function Dg(a, u, f) {
        var m, S, x, A, D, j, J = 0, Q = !1, ee = !1, ge = !0;
        if (typeof a != "function")
          throw new jr(o);
        u = zr(u) || 0, bt(f) && (Q = !!f.leading, ee = "maxWait" in f, x = ee ? Gt(zr(f.maxWait) || 0, u) : x, ge = "trailing" in f ? !!f.trailing : ge);
        function ye(Pt) {
          var rn = m, qn = S;
          return m = S = r, J = Pt, A = a.apply(qn, rn), A;
        }
        function we(Pt) {
          return J = Pt, D = As(Fe, u), Q ? ye(Pt) : A;
        }
        function ke(Pt) {
          var rn = Pt - j, qn = Pt - J, Zg = u - rn;
          return ee ? rr(Zg, x - qn) : Zg;
        }
        function Ee(Pt) {
          var rn = Pt - j, qn = Pt - J;
          return j === r || rn >= u || rn < 0 || ee && qn >= x;
        }
        function Fe() {
          var Pt = Su();
          if (Ee(Pt))
            return $e(Pt);
          D = As(Fe, ke(Pt));
        }
        function $e(Pt) {
          return D = r, ge && m ? ye(Pt) : (m = S = r, A);
        }
        function Rr() {
          D !== r && Hd(D), J = 0, m = j = S = D = r;
        }
        function fr() {
          return D === r ? A : $e(Su());
        }
        function Ar() {
          var Pt = Su(), rn = Ee(Pt);
          if (m = arguments, S = this, j = Pt, rn) {
            if (D === r)
              return we(j);
            if (ee)
              return Hd(D), D = As(Fe, u), ye(j);
          }
          return D === r && (D = As(Fe, u)), A;
        }
        return Ar.cancel = Rr, Ar.flush = fr, Ar;
      }
      var gw = Le(function(a, u) {
        return Ed(a, 1, u);
      }), pw = Le(function(a, u, f) {
        return Ed(a, zr(u) || 0, f);
      });
      function mw(a) {
        return Wn(a, Y);
      }
      function wu(a, u) {
        if (typeof a != "function" || u != null && typeof u != "function")
          throw new jr(o);
        var f = function() {
          var m = arguments, S = u ? u.apply(this, m) : m[0], x = f.cache;
          if (x.has(S))
            return x.get(S);
          var A = a.apply(this, m);
          return f.cache = x.set(S, A) || x, A;
        };
        return f.cache = new (wu.Cache || jn)(), f;
      }
      wu.Cache = jn;
      function Eu(a) {
        if (typeof a != "function")
          throw new jr(o);
        return function() {
          var u = arguments;
          switch (u.length) {
            case 0:
              return !a.call(this);
            case 1:
              return !a.call(this, u[0]);
            case 2:
              return !a.call(this, u[0], u[1]);
            case 3:
              return !a.call(this, u[0], u[1], u[2]);
          }
          return !a.apply(this, u);
        };
      }
      function vw(a) {
        return Ag(2, a);
      }
      var yw = iS(function(a, u) {
        u = u.length == 1 && Oe(u[0]) ? pt(u[0], Or(_e())) : pt(Kt(u, 1), Or(_e()));
        var f = u.length;
        return Le(function(m) {
          for (var S = -1, x = rr(m.length, f); ++S < x; )
            m[S] = u[S].call(this, m[S]);
          return xr(a, this, m);
        });
      }), zc = Le(function(a, u) {
        var f = li(u, Aa(zc));
        return Wn(a, P, r, u, f);
      }), Mg = Le(function(a, u) {
        var f = li(u, Aa(Mg));
        return Wn(a, M, r, u, f);
      }), bw = Un(function(a, u) {
        return Wn(a, V, r, r, r, u);
      });
      function _w(a, u) {
        if (typeof a != "function")
          throw new jr(o);
        return u = u === r ? u : Pe(u), Le(a, u);
      }
      function Sw(a, u) {
        if (typeof a != "function")
          throw new jr(o);
        return u = u == null ? 0 : Gt(Pe(u), 0), Le(function(f) {
          var m = f[u], S = gi(f, 0, u);
          return m && ui(S, m), xr(a, this, S);
        });
      }
      function ww(a, u, f) {
        var m = !0, S = !0;
        if (typeof a != "function")
          throw new jr(o);
        return bt(f) && (m = "leading" in f ? !!f.leading : m, S = "trailing" in f ? !!f.trailing : S), Dg(a, u, {
          leading: m,
          maxWait: u,
          trailing: S
        });
      }
      function Ew(a) {
        return Rg(a, 1);
      }
      function Cw(a, u) {
        return zc(Rc(u), a);
      }
      function xw() {
        if (!arguments.length)
          return [];
        var a = arguments[0];
        return Oe(a) ? a : [a];
      }
      function Ow(a) {
        return Wr(a, y);
      }
      function Tw(a, u) {
        return u = typeof u == "function" ? u : r, Wr(a, y, u);
      }
      function Pw(a) {
        return Wr(a, g | y);
      }
      function Rw(a, u) {
        return u = typeof u == "function" ? u : r, Wr(a, g | y, u);
      }
      function Aw(a, u) {
        return u == null || wd(a, u, Ut(u));
      }
      function tn(a, u) {
        return a === u || a !== a && u !== u;
      }
      var kw = pu(vc), Iw = pu(function(a, u) {
        return a >= u;
      }), Ui = Pd(function() {
        return arguments;
      }()) ? Pd : function(a) {
        return Et(a) && it.call(a, "callee") && !gd.call(a, "callee");
      }, Oe = q.isArray, Lw = ya ? Or(ya) : $_;
      function mr(a) {
        return a != null && Cu(a.length) && !zn(a);
      }
      function Tt(a) {
        return Et(a) && mr(a);
      }
      function Dw(a) {
        return a === !0 || a === !1 || Et(a) && lr(a) == ot;
      }
      var pi = Yb || rf, Mw = ba ? Or(ba) : W_;
      function Nw(a) {
        return Et(a) && a.nodeType === 1 && !ks(a);
      }
      function Bw(a) {
        if (a == null)
          return !0;
        if (mr(a) && (Oe(a) || typeof a == "string" || typeof a.splice == "function" || pi(a) || ka(a) || Ui(a)))
          return !a.length;
        var u = nr(a);
        if (u == fe || u == Mt)
          return !a.size;
        if (Rs(a))
          return !_c(a).length;
        for (var f in a)
          if (it.call(a, f))
            return !1;
        return !0;
      }
      function Fw(a, u) {
        return Os(a, u);
      }
      function Gw(a, u, f) {
        f = typeof f == "function" ? f : r;
        var m = f ? f(a, u) : r;
        return m === r ? Os(a, u, r, f) : !!m;
      }
      function Vc(a) {
        if (!Et(a))
          return !1;
        var u = lr(a);
        return u == dt || u == kt || typeof a.message == "string" && typeof a.name == "string" && !ks(a);
      }
      function jw(a) {
        return typeof a == "number" && md(a);
      }
      function zn(a) {
        if (!bt(a))
          return !1;
        var u = lr(a);
        return u == te || u == $t || u == At || u == gn;
      }
      function Ng(a) {
        return typeof a == "number" && a == Pe(a);
      }
      function Cu(a) {
        return typeof a == "number" && a > -1 && a % 1 == 0 && a <= ae;
      }
      function bt(a) {
        var u = typeof a;
        return a != null && (u == "object" || u == "function");
      }
      function Et(a) {
        return a != null && typeof a == "object";
      }
      var Bg = ms ? Or(ms) : H_;
      function $w(a, u) {
        return a === u || bc(a, u, Nc(u));
      }
      function Ww(a, u, f) {
        return f = typeof f == "function" ? f : r, bc(a, u, Nc(u), f);
      }
      function Uw(a) {
        return Fg(a) && a != +a;
      }
      function Hw(a) {
        if (TS(a))
          throw new xe(s);
        return Rd(a);
      }
      function zw(a) {
        return a === null;
      }
      function Vw(a) {
        return a == null;
      }
      function Fg(a) {
        return typeof a == "number" || Et(a) && lr(a) == Ot;
      }
      function ks(a) {
        if (!Et(a) || lr(a) != Dt)
          return !1;
        var u = Jo(a);
        if (u === null)
          return !0;
        var f = it.call(u, "constructor") && u.constructor;
        return typeof f == "function" && f instanceof f && qo.call(f) == Wb;
      }
      var qc = Qh ? Or(Qh) : z_;
      function qw(a) {
        return Ng(a) && a >= -ae && a <= ae;
      }
      var Gg = Zh ? Or(Zh) : V_;
      function xu(a) {
        return typeof a == "string" || !Oe(a) && Et(a) && lr(a) == qt;
      }
      function Pr(a) {
        return typeof a == "symbol" || Et(a) && lr(a) == yt;
      }
      var ka = ed ? Or(ed) : q_;
      function Yw(a) {
        return a === r;
      }
      function Kw(a) {
        return Et(a) && nr(a) == Sr;
      }
      function Xw(a) {
        return Et(a) && lr(a) == Mn;
      }
      var Jw = pu(Sc), Qw = pu(function(a, u) {
        return a <= u;
      });
      function jg(a) {
        if (!a)
          return [];
        if (mr(a))
          return xu(a) ? Zr(a) : pr(a);
        if (ys && a[ys])
          return Ab(a[ys]());
        var u = nr(a), f = u == fe ? uc : u == Mt ? Ho : Ia;
        return f(a);
      }
      function Vn(a) {
        if (!a)
          return a === 0 ? a : 0;
        if (a = zr(a), a === re || a === -re) {
          var u = a < 0 ? -1 : 1;
          return u * ne;
        }
        return a === a ? a : 0;
      }
      function Pe(a) {
        var u = Vn(a), f = u % 1;
        return u === u ? f ? u - f : u : 0;
      }
      function $g(a) {
        return a ? Gi(Pe(a), 0, de) : 0;
      }
      function zr(a) {
        if (typeof a == "number")
          return a;
        if (Pr(a))
          return oe;
        if (bt(a)) {
          var u = typeof a.valueOf == "function" ? a.valueOf() : a;
          a = bt(u) ? u + "" : u;
        }
        if (typeof a != "string")
          return a === 0 ? a : +a;
        a = sd(a);
        var f = Ae.test(a);
        return f || ft.test(a) ? pe(a.slice(2), f ? 2 : 8) : Me.test(a) ? oe : +a;
      }
      function Wg(a) {
        return bn(a, vr(a));
      }
      function Zw(a) {
        return a ? Gi(Pe(a), -ae, ae) : a === 0 ? a : 0;
      }
      function Ke(a) {
        return a == null ? "" : Tr(a);
      }
      var eE = Pa(function(a, u) {
        if (Rs(u) || mr(u)) {
          bn(u, Ut(u), a);
          return;
        }
        for (var f in u)
          it.call(u, f) && Es(a, f, u[f]);
      }), Ug = Pa(function(a, u) {
        bn(u, vr(u), a);
      }), Ou = Pa(function(a, u, f, m) {
        bn(u, vr(u), a, m);
      }), tE = Pa(function(a, u, f, m) {
        bn(u, Ut(u), a, m);
      }), rE = Un(gc);
      function nE(a, u) {
        var f = Ta(a);
        return u == null ? f : Sd(f, u);
      }
      var iE = Le(function(a, u) {
        a = lt(a);
        var f = -1, m = u.length, S = m > 2 ? u[2] : r;
        for (S && cr(u[0], u[1], S) && (m = 1); ++f < m; )
          for (var x = u[f], A = vr(x), D = -1, j = A.length; ++D < j; ) {
            var J = A[D], Q = a[J];
            (Q === r || tn(Q, Ca[J]) && !it.call(a, J)) && (a[J] = x[J]);
          }
        return a;
      }), aE = Le(function(a) {
        return a.push(r, sg), xr(Hg, r, a);
      });
      function sE(a, u) {
        return rd(a, _e(u, 3), yn);
      }
      function oE(a, u) {
        return rd(a, _e(u, 3), mc);
      }
      function uE(a, u) {
        return a == null ? a : pc(a, _e(u, 3), vr);
      }
      function lE(a, u) {
        return a == null ? a : Od(a, _e(u, 3), vr);
      }
      function cE(a, u) {
        return a && yn(a, _e(u, 3));
      }
      function fE(a, u) {
        return a && mc(a, _e(u, 3));
      }
      function hE(a) {
        return a == null ? [] : ou(a, Ut(a));
      }
      function dE(a) {
        return a == null ? [] : ou(a, vr(a));
      }
      function Yc(a, u, f) {
        var m = a == null ? r : ji(a, u);
        return m === r ? f : m;
      }
      function gE(a, u) {
        return a != null && lg(a, u, B_);
      }
      function Kc(a, u) {
        return a != null && lg(a, u, F_);
      }
      var pE = tg(function(a, u, f) {
        u != null && typeof u.toString != "function" && (u = Yo.call(u)), a[u] = f;
      }, Jc(yr)), mE = tg(function(a, u, f) {
        u != null && typeof u.toString != "function" && (u = Yo.call(u)), it.call(a, u) ? a[u].push(f) : a[u] = [f];
      }, _e), vE = Le(xs);
      function Ut(a) {
        return mr(a) ? bd(a) : _c(a);
      }
      function vr(a) {
        return mr(a) ? bd(a, !0) : Y_(a);
      }
      function yE(a, u) {
        var f = {};
        return u = _e(u, 3), yn(a, function(m, S, x) {
          $n(f, u(m, S, x), m);
        }), f;
      }
      function bE(a, u) {
        var f = {};
        return u = _e(u, 3), yn(a, function(m, S, x) {
          $n(f, S, u(m, S, x));
        }), f;
      }
      var _E = Pa(function(a, u, f) {
        uu(a, u, f);
      }), Hg = Pa(function(a, u, f, m) {
        uu(a, u, f, m);
      }), SE = Un(function(a, u) {
        var f = {};
        if (a == null)
          return f;
        var m = !1;
        u = pt(u, function(x) {
          return x = di(x, a), m || (m = x.length > 1), x;
        }), bn(a, Dc(a), f), m && (f = Wr(f, g | p | y, pS));
        for (var S = u.length; S--; )
          Oc(f, u[S]);
        return f;
      });
      function wE(a, u) {
        return zg(a, Eu(_e(u)));
      }
      var EE = Un(function(a, u) {
        return a == null ? {} : X_(a, u);
      });
      function zg(a, u) {
        if (a == null)
          return {};
        var f = pt(Dc(a), function(m) {
          return [m];
        });
        return u = _e(u), Nd(a, f, function(m, S) {
          return u(m, S[0]);
        });
      }
      function CE(a, u, f) {
        u = di(u, a);
        var m = -1, S = u.length;
        for (S || (S = 1, a = r); ++m < S; ) {
          var x = a == null ? r : a[_n(u[m])];
          x === r && (m = S, x = f), a = zn(x) ? x.call(a) : x;
        }
        return a;
      }
      function xE(a, u, f) {
        return a == null ? a : Ts(a, u, f);
      }
      function OE(a, u, f, m) {
        return m = typeof m == "function" ? m : r, a == null ? a : Ts(a, u, f, m);
      }
      var Vg = ig(Ut), qg = ig(vr);
      function TE(a, u, f) {
        var m = Oe(a), S = m || pi(a) || ka(a);
        if (u = _e(u, 4), f == null) {
          var x = a && a.constructor;
          S ? f = m ? new x() : [] : bt(a) ? f = zn(x) ? Ta(Jo(a)) : {} : f = {};
        }
        return (S ? Gr : yn)(a, function(A, D, j) {
          return u(f, A, D, j);
        }), f;
      }
      function PE(a, u) {
        return a == null ? !0 : Oc(a, u);
      }
      function RE(a, u, f) {
        return a == null ? a : $d(a, u, Rc(f));
      }
      function AE(a, u, f, m) {
        return m = typeof m == "function" ? m : r, a == null ? a : $d(a, u, Rc(f), m);
      }
      function Ia(a) {
        return a == null ? [] : oc(a, Ut(a));
      }
      function kE(a) {
        return a == null ? [] : oc(a, vr(a));
      }
      function IE(a, u, f) {
        return f === r && (f = u, u = r), f !== r && (f = zr(f), f = f === f ? f : 0), u !== r && (u = zr(u), u = u === u ? u : 0), Gi(zr(a), u, f);
      }
      function LE(a, u, f) {
        return u = Vn(u), f === r ? (f = u, u = 0) : f = Vn(f), a = zr(a), G_(a, u, f);
      }
      function DE(a, u, f) {
        if (f && typeof f != "boolean" && cr(a, u, f) && (u = f = r), f === r && (typeof u == "boolean" ? (f = u, u = r) : typeof a == "boolean" && (f = a, a = r)), a === r && u === r ? (a = 0, u = 1) : (a = Vn(a), u === r ? (u = a, a = 0) : u = Vn(u)), a > u) {
          var m = a;
          a = u, u = m;
        }
        if (f || a % 1 || u % 1) {
          var S = vd();
          return rr(a + S * (u - a + z("1e-" + ((S + "").length - 1))), u);
        }
        return Ec(a, u);
      }
      var ME = Ra(function(a, u, f) {
        return u = u.toLowerCase(), a + (f ? Yg(u) : u);
      });
      function Yg(a) {
        return Xc(Ke(a).toLowerCase());
      }
      function Kg(a) {
        return a = Ke(a), a && a.replace(tr, xb).replace(Fo, "");
      }
      function NE(a, u, f) {
        a = Ke(a), u = Tr(u);
        var m = a.length;
        f = f === r ? m : Gi(Pe(f), 0, m);
        var S = f;
        return f -= u.length, f >= 0 && a.slice(f, S) == u;
      }
      function BE(a) {
        return a = Ke(a), a && Fr.test(a) ? a.replace(Br, Ob) : a;
      }
      function FE(a) {
        return a = Ke(a), a && la.test(a) ? a.replace(Li, "\\$&") : a;
      }
      var GE = Ra(function(a, u, f) {
        return a + (f ? "-" : "") + u.toLowerCase();
      }), jE = Ra(function(a, u, f) {
        return a + (f ? " " : "") + u.toLowerCase();
      }), $E = Qd("toLowerCase");
      function WE(a, u, f) {
        a = Ke(a), u = Pe(u);
        var m = u ? wa(a) : 0;
        if (!u || m >= u)
          return a;
        var S = (u - m) / 2;
        return gu(tu(S), f) + a + gu(eu(S), f);
      }
      function UE(a, u, f) {
        a = Ke(a), u = Pe(u);
        var m = u ? wa(a) : 0;
        return u && m < u ? a + gu(u - m, f) : a;
      }
      function HE(a, u, f) {
        a = Ke(a), u = Pe(u);
        var m = u ? wa(a) : 0;
        return u && m < u ? gu(u - m, f) + a : a;
      }
      function zE(a, u, f) {
        return f || u == null ? u = 0 : u && (u = +u), Qb(Ke(a).replace(Di, ""), u || 0);
      }
      function VE(a, u, f) {
        return (f ? cr(a, u, f) : u === r) ? u = 1 : u = Pe(u), Cc(Ke(a), u);
      }
      function qE() {
        var a = arguments, u = Ke(a[0]);
        return a.length < 3 ? u : u.replace(a[1], a[2]);
      }
      var YE = Ra(function(a, u, f) {
        return a + (f ? "_" : "") + u.toLowerCase();
      });
      function KE(a, u, f) {
        return f && typeof f != "number" && cr(a, u, f) && (u = f = r), f = f === r ? de : f >>> 0, f ? (a = Ke(a), a && (typeof u == "string" || u != null && !qc(u)) && (u = Tr(u), !u && Sa(a)) ? gi(Zr(a), 0, f) : a.split(u, f)) : [];
      }
      var XE = Ra(function(a, u, f) {
        return a + (f ? " " : "") + Xc(u);
      });
      function JE(a, u, f) {
        return a = Ke(a), f = f == null ? 0 : Gi(Pe(f), 0, a.length), u = Tr(u), a.slice(f, f + u.length) == u;
      }
      function QE(a, u, f) {
        var m = E.templateSettings;
        f && cr(a, u, f) && (u = r), a = Ke(a), u = Ou({}, u, m, ag);
        var S = Ou({}, u.imports, m.imports, ag), x = Ut(S), A = oc(S, x), D, j, J = 0, Q = u.interpolate || or, ee = "__p += '", ge = lc(
          (u.escape || or).source + "|" + Q.source + "|" + (Q === sa ? je : or).source + "|" + (u.evaluate || or).source + "|$",
          "g"
        ), ye = "//# sourceURL=" + (it.call(u, "sourceURL") ? (u.sourceURL + "").replace(/\s/g, " ") : "lodash.templateSources[" + ++jo + "]") + `
`;
        a.replace(ge, function(Ee, Fe, $e, Rr, fr, Ar) {
          return $e || ($e = Rr), ee += a.slice(J, Ar).replace(ca, Tb), Fe && (D = !0, ee += `' +
__e(` + Fe + `) +
'`), fr && (j = !0, ee += `';
` + fr + `;
__p += '`), $e && (ee += `' +
((__t = (` + $e + `)) == null ? '' : __t) +
'`), J = Ar + Ee.length, Ee;
        }), ee += `';
`;
        var we = it.call(u, "variable") && u.variable;
        if (!we)
          ee = `with (obj) {
` + ee + `
}
`;
        else if (he.test(we))
          throw new xe(l);
        ee = (j ? ee.replace(ns, "") : ee).replace(si, "$1").replace(ia, "$1;"), ee = "function(" + (we || "obj") + `) {
` + (we ? "" : `obj || (obj = {});
`) + "var __t, __p = ''" + (D ? ", __e = _.escape" : "") + (j ? `, __j = Array.prototype.join;
function print() { __p += __j.call(arguments, '') }
` : `;
`) + ee + `return __p
}`;
        var ke = Jg(function() {
          return Ve(x, ye + "return " + ee).apply(r, A);
        });
        if (ke.source = ee, Vc(ke))
          throw ke;
        return ke;
      }
      function ZE(a) {
        return Ke(a).toLowerCase();
      }
      function eC(a) {
        return Ke(a).toUpperCase();
      }
      function tC(a, u, f) {
        if (a = Ke(a), a && (f || u === r))
          return sd(a);
        if (!a || !(u = Tr(u)))
          return a;
        var m = Zr(a), S = Zr(u), x = od(m, S), A = ud(m, S) + 1;
        return gi(m, x, A).join("");
      }
      function rC(a, u, f) {
        if (a = Ke(a), a && (f || u === r))
          return a.slice(0, cd(a) + 1);
        if (!a || !(u = Tr(u)))
          return a;
        var m = Zr(a), S = ud(m, Zr(u)) + 1;
        return gi(m, 0, S).join("");
      }
      function nC(a, u, f) {
        if (a = Ke(a), a && (f || u === r))
          return a.replace(Di, "");
        if (!a || !(u = Tr(u)))
          return a;
        var m = Zr(a), S = od(m, Zr(u));
        return gi(m, S).join("");
      }
      function iC(a, u) {
        var f = X, m = Z;
        if (bt(u)) {
          var S = "separator" in u ? u.separator : S;
          f = "length" in u ? Pe(u.length) : f, m = "omission" in u ? Tr(u.omission) : m;
        }
        a = Ke(a);
        var x = a.length;
        if (Sa(a)) {
          var A = Zr(a);
          x = A.length;
        }
        if (f >= x)
          return a;
        var D = f - wa(m);
        if (D < 1)
          return m;
        var j = A ? gi(A, 0, D).join("") : a.slice(0, D);
        if (S === r)
          return j + m;
        if (A && (D += j.length - D), qc(S)) {
          if (a.slice(D).search(S)) {
            var J, Q = j;
            for (S.global || (S = lc(S.source, Ke(Ne.exec(S)) + "g")), S.lastIndex = 0; J = S.exec(Q); )
              var ee = J.index;
            j = j.slice(0, ee === r ? D : ee);
          }
        } else if (a.indexOf(Tr(S), D) != D) {
          var ge = j.lastIndexOf(S);
          ge > -1 && (j = j.slice(0, ge));
        }
        return j + m;
      }
      function aC(a) {
        return a = Ke(a), a && Ii.test(a) ? a.replace(Nr, Db) : a;
      }
      var sC = Ra(function(a, u, f) {
        return a + (f ? " " : "") + u.toUpperCase();
      }), Xc = Qd("toUpperCase");
      function Xg(a, u, f) {
        return a = Ke(a), u = f ? r : u, u === r ? Rb(a) ? Bb(a) : _b(a) : a.match(u) || [];
      }
      var Jg = Le(function(a, u) {
        try {
          return xr(a, r, u);
        } catch (f) {
          return Vc(f) ? f : new xe(f);
        }
      }), oC = Un(function(a, u) {
        return Gr(u, function(f) {
          f = _n(f), $n(a, f, Hc(a[f], a));
        }), a;
      });
      function uC(a) {
        var u = a == null ? 0 : a.length, f = _e();
        return a = u ? pt(a, function(m) {
          if (typeof m[1] != "function")
            throw new jr(o);
          return [f(m[0]), m[1]];
        }) : [], Le(function(m) {
          for (var S = -1; ++S < u; ) {
            var x = a[S];
            if (xr(x[0], this, m))
              return xr(x[1], this, m);
          }
        });
      }
      function lC(a) {
        return D_(Wr(a, g));
      }
      function Jc(a) {
        return function() {
          return a;
        };
      }
      function cC(a, u) {
        return a == null || a !== a ? u : a;
      }
      var fC = eg(), hC = eg(!0);
      function yr(a) {
        return a;
      }
      function Qc(a) {
        return Ad(typeof a == "function" ? a : Wr(a, g));
      }
      function dC(a) {
        return Id(Wr(a, g));
      }
      function gC(a, u) {
        return Ld(a, Wr(u, g));
      }
      var pC = Le(function(a, u) {
        return function(f) {
          return xs(f, a, u);
        };
      }), mC = Le(function(a, u) {
        return function(f) {
          return xs(a, f, u);
        };
      });
      function Zc(a, u, f) {
        var m = Ut(u), S = ou(u, m);
        f == null && !(bt(u) && (S.length || !m.length)) && (f = u, u = a, a = this, S = ou(u, Ut(u)));
        var x = !(bt(f) && "chain" in f) || !!f.chain, A = zn(a);
        return Gr(S, function(D) {
          var j = u[D];
          a[D] = j, A && (a.prototype[D] = function() {
            var J = this.__chain__;
            if (x || J) {
              var Q = a(this.__wrapped__), ee = Q.__actions__ = pr(this.__actions__);
              return ee.push({ func: j, args: arguments, thisArg: a }), Q.__chain__ = J, Q;
            }
            return j.apply(a, ui([this.value()], arguments));
          });
        }), a;
      }
      function vC() {
        return Te._ === this && (Te._ = Ub), this;
      }
      function ef() {
      }
      function yC(a) {
        return a = Pe(a), Le(function(u) {
          return Dd(u, a);
        });
      }
      var bC = kc(pt), _C = kc(td), SC = kc(rc);
      function Qg(a) {
        return Fc(a) ? nc(_n(a)) : J_(a);
      }
      function wC(a) {
        return function(u) {
          return a == null ? r : ji(a, u);
        };
      }
      var EC = rg(), CC = rg(!0);
      function tf() {
        return [];
      }
      function rf() {
        return !1;
      }
      function xC() {
        return {};
      }
      function OC() {
        return "";
      }
      function TC() {
        return !0;
      }
      function PC(a, u) {
        if (a = Pe(a), a < 1 || a > ae)
          return [];
        var f = de, m = rr(a, de);
        u = _e(u), a -= de;
        for (var S = sc(m, u); ++f < a; )
          u(f);
        return S;
      }
      function RC(a) {
        return Oe(a) ? pt(a, _n) : Pr(a) ? [a] : pr(yg(Ke(a)));
      }
      function AC(a) {
        var u = ++$b;
        return Ke(a) + u;
      }
      var kC = du(function(a, u) {
        return a + u;
      }, 0), IC = Ic("ceil"), LC = du(function(a, u) {
        return a / u;
      }, 1), DC = Ic("floor");
      function MC(a) {
        return a && a.length ? su(a, yr, vc) : r;
      }
      function NC(a, u) {
        return a && a.length ? su(a, _e(u, 2), vc) : r;
      }
      function BC(a) {
        return id(a, yr);
      }
      function FC(a, u) {
        return id(a, _e(u, 2));
      }
      function GC(a) {
        return a && a.length ? su(a, yr, Sc) : r;
      }
      function jC(a, u) {
        return a && a.length ? su(a, _e(u, 2), Sc) : r;
      }
      var $C = du(function(a, u) {
        return a * u;
      }, 1), WC = Ic("round"), UC = du(function(a, u) {
        return a - u;
      }, 0);
      function HC(a) {
        return a && a.length ? ac(a, yr) : 0;
      }
      function zC(a, u) {
        return a && a.length ? ac(a, _e(u, 2)) : 0;
      }
      return E.after = dw, E.ary = Rg, E.assign = eE, E.assignIn = Ug, E.assignInWith = Ou, E.assignWith = tE, E.at = rE, E.before = Ag, E.bind = Hc, E.bindAll = oC, E.bindKey = kg, E.castArray = xw, E.chain = Og, E.chunk = DS, E.compact = MS, E.concat = NS, E.cond = uC, E.conforms = lC, E.constant = Jc, E.countBy = H1, E.create = nE, E.curry = Ig, E.curryRight = Lg, E.debounce = Dg, E.defaults = iE, E.defaultsDeep = aE, E.defer = gw, E.delay = pw, E.difference = BS, E.differenceBy = FS, E.differenceWith = GS, E.drop = jS, E.dropRight = $S, E.dropRightWhile = WS, E.dropWhile = US, E.fill = HS, E.filter = V1, E.flatMap = K1, E.flatMapDeep = X1, E.flatMapDepth = J1, E.flatten = wg, E.flattenDeep = zS, E.flattenDepth = VS, E.flip = mw, E.flow = fC, E.flowRight = hC, E.fromPairs = qS, E.functions = hE, E.functionsIn = dE, E.groupBy = Q1, E.initial = KS, E.intersection = XS, E.intersectionBy = JS, E.intersectionWith = QS, E.invert = pE, E.invertBy = mE, E.invokeMap = ew, E.iteratee = Qc, E.keyBy = tw, E.keys = Ut, E.keysIn = vr, E.map = _u, E.mapKeys = yE, E.mapValues = bE, E.matches = dC, E.matchesProperty = gC, E.memoize = wu, E.merge = _E, E.mergeWith = Hg, E.method = pC, E.methodOf = mC, E.mixin = Zc, E.negate = Eu, E.nthArg = yC, E.omit = SE, E.omitBy = wE, E.once = vw, E.orderBy = rw, E.over = bC, E.overArgs = yw, E.overEvery = _C, E.overSome = SC, E.partial = zc, E.partialRight = Mg, E.partition = nw, E.pick = EE, E.pickBy = zg, E.property = Qg, E.propertyOf = wC, E.pull = r1, E.pullAll = Cg, E.pullAllBy = n1, E.pullAllWith = i1, E.pullAt = a1, E.range = EC, E.rangeRight = CC, E.rearg = bw, E.reject = sw, E.remove = s1, E.rest = _w, E.reverse = Wc, E.sampleSize = uw, E.set = xE, E.setWith = OE, E.shuffle = lw, E.slice = o1, E.sortBy = hw, E.sortedUniq = g1, E.sortedUniqBy = p1, E.split = KE, E.spread = Sw, E.tail = m1, E.take = v1, E.takeRight = y1, E.takeRightWhile = b1, E.takeWhile = _1, E.tap = M1, E.throttle = ww, E.thru = bu, E.toArray = jg, E.toPairs = Vg, E.toPairsIn = qg, E.toPath = RC, E.toPlainObject = Wg, E.transform = TE, E.unary = Ew, E.union = S1, E.unionBy = w1, E.unionWith = E1, E.uniq = C1, E.uniqBy = x1, E.uniqWith = O1, E.unset = PE, E.unzip = Uc, E.unzipWith = xg, E.update = RE, E.updateWith = AE, E.values = Ia, E.valuesIn = kE, E.without = T1, E.words = Xg, E.wrap = Cw, E.xor = P1, E.xorBy = R1, E.xorWith = A1, E.zip = k1, E.zipObject = I1, E.zipObjectDeep = L1, E.zipWith = D1, E.entries = Vg, E.entriesIn = qg, E.extend = Ug, E.extendWith = Ou, Zc(E, E), E.add = kC, E.attempt = Jg, E.camelCase = ME, E.capitalize = Yg, E.ceil = IC, E.clamp = IE, E.clone = Ow, E.cloneDeep = Pw, E.cloneDeepWith = Rw, E.cloneWith = Tw, E.conformsTo = Aw, E.deburr = Kg, E.defaultTo = cC, E.divide = LC, E.endsWith = NE, E.eq = tn, E.escape = BE, E.escapeRegExp = FE, E.every = z1, E.find = q1, E.findIndex = _g, E.findKey = sE, E.findLast = Y1, E.findLastIndex = Sg, E.findLastKey = oE, E.floor = DC, E.forEach = Tg, E.forEachRight = Pg, E.forIn = uE, E.forInRight = lE, E.forOwn = cE, E.forOwnRight = fE, E.get = Yc, E.gt = kw, E.gte = Iw, E.has = gE, E.hasIn = Kc, E.head = Eg, E.identity = yr, E.includes = Z1, E.indexOf = YS, E.inRange = LE, E.invoke = vE, E.isArguments = Ui, E.isArray = Oe, E.isArrayBuffer = Lw, E.isArrayLike = mr, E.isArrayLikeObject = Tt, E.isBoolean = Dw, E.isBuffer = pi, E.isDate = Mw, E.isElement = Nw, E.isEmpty = Bw, E.isEqual = Fw, E.isEqualWith = Gw, E.isError = Vc, E.isFinite = jw, E.isFunction = zn, E.isInteger = Ng, E.isLength = Cu, E.isMap = Bg, E.isMatch = $w, E.isMatchWith = Ww, E.isNaN = Uw, E.isNative = Hw, E.isNil = Vw, E.isNull = zw, E.isNumber = Fg, E.isObject = bt, E.isObjectLike = Et, E.isPlainObject = ks, E.isRegExp = qc, E.isSafeInteger = qw, E.isSet = Gg, E.isString = xu, E.isSymbol = Pr, E.isTypedArray = ka, E.isUndefined = Yw, E.isWeakMap = Kw, E.isWeakSet = Xw, E.join = ZS, E.kebabCase = GE, E.last = Hr, E.lastIndexOf = e1, E.lowerCase = jE, E.lowerFirst = $E, E.lt = Jw, E.lte = Qw, E.max = MC, E.maxBy = NC, E.mean = BC, E.meanBy = FC, E.min = GC, E.minBy = jC, E.stubArray = tf, E.stubFalse = rf, E.stubObject = xC, E.stubString = OC, E.stubTrue = TC, E.multiply = $C, E.nth = t1, E.noConflict = vC, E.noop = ef, E.now = Su, E.pad = WE, E.padEnd = UE, E.padStart = HE, E.parseInt = zE, E.random = DE, E.reduce = iw, E.reduceRight = aw, E.repeat = VE, E.replace = qE, E.result = CE, E.round = WC, E.runInContext = G, E.sample = ow, E.size = cw, E.snakeCase = YE, E.some = fw, E.sortedIndex = u1, E.sortedIndexBy = l1, E.sortedIndexOf = c1, E.sortedLastIndex = f1, E.sortedLastIndexBy = h1, E.sortedLastIndexOf = d1, E.startCase = XE, E.startsWith = JE, E.subtract = UC, E.sum = HC, E.sumBy = zC, E.template = QE, E.times = PC, E.toFinite = Vn, E.toInteger = Pe, E.toLength = $g, E.toLower = ZE, E.toNumber = zr, E.toSafeInteger = Zw, E.toString = Ke, E.toUpper = eC, E.trim = tC, E.trimEnd = rC, E.trimStart = nC, E.truncate = iC, E.unescape = aC, E.uniqueId = AC, E.upperCase = sC, E.upperFirst = Xc, E.each = Tg, E.eachRight = Pg, E.first = Eg, Zc(E, function() {
        var a = {};
        return yn(E, function(u, f) {
          it.call(E.prototype, f) || (a[f] = u);
        }), a;
      }(), { chain: !1 }), E.VERSION = n, Gr(["bind", "bindKey", "curry", "curryRight", "partial", "partialRight"], function(a) {
        E[a].placeholder = E;
      }), Gr(["drop", "take"], function(a, u) {
        Ge.prototype[a] = function(f) {
          f = f === r ? 1 : Gt(Pe(f), 0);
          var m = this.__filtered__ && !u ? new Ge(this) : this.clone();
          return m.__filtered__ ? m.__takeCount__ = rr(f, m.__takeCount__) : m.__views__.push({
            size: rr(f, de),
            type: a + (m.__dir__ < 0 ? "Right" : "")
          }), m;
        }, Ge.prototype[a + "Right"] = function(f) {
          return this.reverse()[a](f).reverse();
        };
      }), Gr(["filter", "map", "takeWhile"], function(a, u) {
        var f = u + 1, m = f == R || f == H;
        Ge.prototype[a] = function(S) {
          var x = this.clone();
          return x.__iteratees__.push({
            iteratee: _e(S, 3),
            type: f
          }), x.__filtered__ = x.__filtered__ || m, x;
        };
      }), Gr(["head", "last"], function(a, u) {
        var f = "take" + (u ? "Right" : "");
        Ge.prototype[a] = function() {
          return this[f](1).value()[0];
        };
      }), Gr(["initial", "tail"], function(a, u) {
        var f = "drop" + (u ? "" : "Right");
        Ge.prototype[a] = function() {
          return this.__filtered__ ? new Ge(this) : this[f](1);
        };
      }), Ge.prototype.compact = function() {
        return this.filter(yr);
      }, Ge.prototype.find = function(a) {
        return this.filter(a).head();
      }, Ge.prototype.findLast = function(a) {
        return this.reverse().find(a);
      }, Ge.prototype.invokeMap = Le(function(a, u) {
        return typeof a == "function" ? new Ge(this) : this.map(function(f) {
          return xs(f, a, u);
        });
      }), Ge.prototype.reject = function(a) {
        return this.filter(Eu(_e(a)));
      }, Ge.prototype.slice = function(a, u) {
        a = Pe(a);
        var f = this;
        return f.__filtered__ && (a > 0 || u < 0) ? new Ge(f) : (a < 0 ? f = f.takeRight(-a) : a && (f = f.drop(a)), u !== r && (u = Pe(u), f = u < 0 ? f.dropRight(-u) : f.take(u - a)), f);
      }, Ge.prototype.takeRightWhile = function(a) {
        return this.reverse().takeWhile(a).reverse();
      }, Ge.prototype.toArray = function() {
        return this.take(de);
      }, yn(Ge.prototype, function(a, u) {
        var f = /^(?:filter|find|map|reject)|While$/.test(u), m = /^(?:head|last)$/.test(u), S = E[m ? "take" + (u == "last" ? "Right" : "") : u], x = m || /^find/.test(u);
        S && (E.prototype[u] = function() {
          var A = this.__wrapped__, D = m ? [1] : arguments, j = A instanceof Ge, J = D[0], Q = j || Oe(A), ee = function(Fe) {
            var $e = S.apply(E, ui([Fe], D));
            return m && ge ? $e[0] : $e;
          };
          Q && f && typeof J == "function" && J.length != 1 && (j = Q = !1);
          var ge = this.__chain__, ye = !!this.__actions__.length, we = x && !ge, ke = j && !ye;
          if (!x && Q) {
            A = ke ? A : new Ge(this);
            var Ee = a.apply(A, D);
            return Ee.__actions__.push({ func: bu, args: [ee], thisArg: r }), new $r(Ee, ge);
          }
          return we && ke ? a.apply(this, D) : (Ee = this.thru(ee), we ? m ? Ee.value()[0] : Ee.value() : Ee);
        });
      }), Gr(["pop", "push", "shift", "sort", "splice", "unshift"], function(a) {
        var u = zo[a], f = /^(?:push|sort|unshift)$/.test(a) ? "tap" : "thru", m = /^(?:pop|shift)$/.test(a);
        E.prototype[a] = function() {
          var S = arguments;
          if (m && !this.__chain__) {
            var x = this.value();
            return u.apply(Oe(x) ? x : [], S);
          }
          return this[f](function(A) {
            return u.apply(Oe(A) ? A : [], S);
          });
        };
      }), yn(Ge.prototype, function(a, u) {
        var f = E[u];
        if (f) {
          var m = f.name + "";
          it.call(Oa, m) || (Oa[m] = []), Oa[m].push({ name: u, func: f });
        }
      }), Oa[hu(r, C).name] = [{
        name: "wrapper",
        func: r
      }], Ge.prototype.clone = a_, Ge.prototype.reverse = s_, Ge.prototype.value = o_, E.prototype.at = N1, E.prototype.chain = B1, E.prototype.commit = F1, E.prototype.next = G1, E.prototype.plant = $1, E.prototype.reverse = W1, E.prototype.toJSON = E.prototype.valueOf = E.prototype.value = U1, E.prototype.first = E.prototype.head, ys && (E.prototype[ys] = j1), E;
    }, Ea = Fb();
    He ? ((He.exports = Ea)._ = Ea, Ie._ = Ea) : Te._ = Ea;
  }).call(dr);
})(tO, Ct);
const La = (t, e, r, n, i, s, o) => {
  const l = t / 2 - (r + i / 2) * o, c = e / 2 - (n + s / 2) * o;
  return { x: l, y: c };
}, Da = (t, e, r, n, i = 0.95) => {
  const s = t * i / r, o = e * i / n;
  return Math.min(1, Math.min(s, o));
}, ND = 0.999, BD = 0.1, FD = 20, Ds = 0.95, GD = 30, jD = 10, Gp = (t) => ({
  x: Math.floor(t.x),
  y: Math.floor(t.y)
}), Hi = (t) => {
  const { width: e, height: r } = t, n = { width: e, height: r }, i = 512 * 512, s = e / r;
  let o = e * r, l = 448;
  for (; o < i; )
    if (l += 64, e === r) {
      n.width = 512, n.height = 512;
      break;
    } else
      s > 1 ? (n.width = l, n.height = Si(
        l / s,
        64
      )) : s < 1 && (n.height = l, n.width = Si(
        l * s,
        64
      )), o = n.width * n.height;
  return n;
}, rO = (t) => ({
  width: Si(t.width, 64),
  height: Si(t.height, 64)
}), $D = [
  { key: "Base", value: "base" },
  { key: "Mask", value: "mask" }
], WD = [
  { key: "Auto", value: "auto" },
  { key: "Manual", value: "manual" },
  { key: "None", value: "none" }
], py = (t) => t.kind === "line" && t.layer === "mask", UD = (t) => t.kind === "line" && t.layer === "base", jp = (t) => t.kind === "image" && t.layer === "base", HD = (t) => t.kind === "fillRect" && t.layer === "base", zD = (t) => t.kind === "eraseRect" && t.layer === "base", nO = (t) => t.kind === "line", qs = {
  objects: [],
  stagingArea: {
    images: [],
    selectedImageIndex: -1
  }
}, iO = {
  boundingBoxCoordinates: { x: 0, y: 0 },
  boundingBoxDimensions: { width: 512, height: 512 },
  boundingBoxPreviewFill: { r: 0, g: 0, b: 0, a: 0.5 },
  boundingBoxScaleMethod: "auto",
  brushColor: { r: 90, g: 90, b: 255, a: 1 },
  brushSize: 50,
  canvasContainerDimensions: { width: 0, height: 0 },
  colorPickerColor: { r: 90, g: 90, b: 255, a: 1 },
  cursorPosition: null,
  doesCanvasNeedScaling: !1,
  futureLayerStates: [],
  isCanvasInitialized: !1,
  isDrawing: !1,
  isMaskEnabled: !0,
  isMouseOverBoundingBox: !1,
  isMoveBoundingBoxKeyHeld: !1,
  isMoveStageKeyHeld: !1,
  isMovingBoundingBox: !1,
  isMovingStage: !1,
  isTransformingBoundingBox: !1,
  layer: "base",
  layerState: qs,
  maskColor: { r: 255, g: 90, b: 90, a: 1 },
  maxHistory: 128,
  minimumStageScale: 1,
  pastLayerStates: [],
  scaledBoundingBoxDimensions: { width: 512, height: 512 },
  shouldAutoSave: !1,
  shouldCropToBoundingBoxOnSave: !1,
  shouldDarkenOutsideBoundingBox: !1,
  shouldLockBoundingBox: !1,
  shouldPreserveMaskedArea: !1,
  shouldRestrictStrokesToBox: !0,
  shouldShowBoundingBox: !0,
  shouldShowBrush: !0,
  shouldShowBrushPreview: !1,
  shouldShowCanvasDebugInfo: !1,
  shouldShowCheckboardTransparency: !1,
  shouldShowGrid: !0,
  shouldShowIntermediates: !0,
  shouldShowStagingImage: !0,
  shouldShowStagingOutline: !0,
  shouldSnapToGrid: !0,
  stageCoordinates: { x: 0, y: 0 },
  stageDimensions: { width: 0, height: 0 },
  stageScale: 1,
  tool: "brush"
}, my = Qi({
  name: "canvas",
  initialState: iO,
  reducers: {
    setTool: (t, e) => {
      const r = e.payload;
      t.tool = e.payload, r !== "move" && (t.isTransformingBoundingBox = !1, t.isMouseOverBoundingBox = !1, t.isMovingBoundingBox = !1, t.isMovingStage = !1);
    },
    setLayer: (t, e) => {
      t.layer = e.payload;
    },
    toggleTool: (t) => {
      const e = t.tool;
      e !== "move" && (t.tool = e === "brush" ? "eraser" : "brush");
    },
    setMaskColor: (t, e) => {
      t.maskColor = e.payload;
    },
    setBrushColor: (t, e) => {
      t.brushColor = e.payload;
    },
    setBrushSize: (t, e) => {
      t.brushSize = e.payload;
    },
    clearMask: (t) => {
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.layerState.objects = t.layerState.objects.filter(
        (e) => !py(e)
      ), t.futureLayerStates = [], t.shouldPreserveMaskedArea = !1;
    },
    toggleShouldInvertMask: (t) => {
      t.shouldPreserveMaskedArea = !t.shouldPreserveMaskedArea;
    },
    toggleShouldShowMask: (t) => {
      t.isMaskEnabled = !t.isMaskEnabled;
    },
    setShouldPreserveMaskedArea: (t, e) => {
      t.shouldPreserveMaskedArea = e.payload;
    },
    setIsMaskEnabled: (t, e) => {
      t.isMaskEnabled = e.payload, t.layer = e.payload ? "mask" : "base";
    },
    setShouldShowCheckboardTransparency: (t, e) => {
      t.shouldShowCheckboardTransparency = e.payload;
    },
    setShouldShowBrushPreview: (t, e) => {
      t.shouldShowBrushPreview = e.payload;
    },
    setShouldShowBrush: (t, e) => {
      t.shouldShowBrush = e.payload;
    },
    setCursorPosition: (t, e) => {
      t.cursorPosition = e.payload;
    },
    setInitialCanvasImage: (t, e) => {
      const r = e.payload, { stageDimensions: n } = t, i = {
        width: Tu(Ct.clamp(r.width, 64, 512), 64),
        height: Tu(Ct.clamp(r.height, 64, 512), 64)
      }, s = {
        x: Si(
          r.width / 2 - i.width / 2,
          64
        ),
        y: Si(
          r.height / 2 - i.height / 2,
          64
        )
      };
      if (t.boundingBoxScaleMethod === "auto") {
        const c = Hi(
          i
        );
        t.scaledBoundingBoxDimensions = c;
      }
      t.boundingBoxDimensions = i, t.boundingBoxCoordinates = s, t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.layerState = {
        ...qs,
        objects: [
          {
            kind: "image",
            layer: "base",
            x: 0,
            y: 0,
            width: r.width,
            height: r.height,
            image: r
          }
        ]
      }, t.futureLayerStates = [], t.isCanvasInitialized = !1;
      const o = Da(
        n.width,
        n.height,
        r.width,
        r.height,
        Ds
      ), l = La(
        n.width,
        n.height,
        0,
        0,
        r.width,
        r.height,
        o
      );
      t.stageScale = o, t.stageCoordinates = l, t.doesCanvasNeedScaling = !0;
    },
    setBoundingBoxDimensions: (t, e) => {
      const r = rO(e.payload);
      if (t.boundingBoxDimensions = r, t.boundingBoxScaleMethod === "auto") {
        const n = Hi(r);
        t.scaledBoundingBoxDimensions = n;
      }
    },
    setBoundingBoxCoordinates: (t, e) => {
      t.boundingBoxCoordinates = Gp(e.payload);
    },
    setStageCoordinates: (t, e) => {
      t.stageCoordinates = e.payload;
    },
    setBoundingBoxPreviewFill: (t, e) => {
      t.boundingBoxPreviewFill = e.payload;
    },
    setDoesCanvasNeedScaling: (t, e) => {
      t.doesCanvasNeedScaling = e.payload;
    },
    setStageScale: (t, e) => {
      t.stageScale = e.payload;
    },
    setShouldDarkenOutsideBoundingBox: (t, e) => {
      t.shouldDarkenOutsideBoundingBox = e.payload;
    },
    setIsDrawing: (t, e) => {
      t.isDrawing = e.payload;
    },
    clearCanvasHistory: (t) => {
      t.pastLayerStates = [], t.futureLayerStates = [];
    },
    setShouldLockBoundingBox: (t, e) => {
      t.shouldLockBoundingBox = e.payload;
    },
    toggleShouldLockBoundingBox: (t) => {
      t.shouldLockBoundingBox = !t.shouldLockBoundingBox;
    },
    setShouldShowBoundingBox: (t, e) => {
      t.shouldShowBoundingBox = e.payload;
    },
    setIsTransformingBoundingBox: (t, e) => {
      t.isTransformingBoundingBox = e.payload;
    },
    setIsMovingBoundingBox: (t, e) => {
      t.isMovingBoundingBox = e.payload;
    },
    setIsMouseOverBoundingBox: (t, e) => {
      t.isMouseOverBoundingBox = e.payload;
    },
    setIsMoveBoundingBoxKeyHeld: (t, e) => {
      t.isMoveBoundingBoxKeyHeld = e.payload;
    },
    setIsMoveStageKeyHeld: (t, e) => {
      t.isMoveStageKeyHeld = e.payload;
    },
    addImageToStagingArea: (t, e) => {
      const { boundingBox: r, image: n } = e.payload;
      !r || !n || (t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift(), t.layerState.stagingArea.images.push({
        kind: "image",
        layer: "base",
        ...r,
        image: n
      }), t.layerState.stagingArea.selectedImageIndex = t.layerState.stagingArea.images.length - 1, t.futureLayerStates = []);
    },
    discardStagedImages: (t) => {
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift(), t.layerState.stagingArea = {
        ...qs.stagingArea
      }, t.futureLayerStates = [], t.shouldShowStagingOutline = !0, t.shouldShowStagingOutline = !0;
    },
    addFillRect: (t) => {
      const { boundingBoxCoordinates: e, boundingBoxDimensions: r, brushColor: n } = t;
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift(), t.layerState.objects.push({
        kind: "fillRect",
        layer: "base",
        ...e,
        ...r,
        color: n
      }), t.futureLayerStates = [];
    },
    addEraseRect: (t) => {
      const { boundingBoxCoordinates: e, boundingBoxDimensions: r } = t;
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift(), t.layerState.objects.push({
        kind: "eraseRect",
        layer: "base",
        ...e,
        ...r
      }), t.futureLayerStates = [];
    },
    addLine: (t, e) => {
      const { tool: r, layer: n, brushColor: i, brushSize: s, shouldRestrictStrokesToBox: o } = t;
      if (r === "move" || r === "colorPicker")
        return;
      const l = s / 2, c = n === "base" && r === "brush" ? { color: i } : {};
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift();
      const h = {
        kind: "line",
        layer: n,
        tool: r,
        strokeWidth: l,
        points: e.payload,
        ...c
      };
      o && (h.clip = {
        ...t.boundingBoxCoordinates,
        ...t.boundingBoxDimensions
      }), t.layerState.objects.push(h), t.futureLayerStates = [];
    },
    addPointToCurrentLine: (t, e) => {
      const r = t.layerState.objects.findLast(nO);
      r && r.points.push(...e.payload);
    },
    undo: (t) => {
      const e = t.pastLayerStates.pop();
      e && (t.futureLayerStates.unshift(Ct.cloneDeep(t.layerState)), t.futureLayerStates.length > t.maxHistory && t.futureLayerStates.pop(), t.layerState = e);
    },
    redo: (t) => {
      const e = t.futureLayerStates.shift();
      e && (t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift(), t.layerState = e);
    },
    setShouldShowGrid: (t, e) => {
      t.shouldShowGrid = e.payload;
    },
    setIsMovingStage: (t, e) => {
      t.isMovingStage = e.payload;
    },
    setShouldSnapToGrid: (t, e) => {
      t.shouldSnapToGrid = e.payload;
    },
    setShouldAutoSave: (t, e) => {
      t.shouldAutoSave = e.payload;
    },
    setShouldShowIntermediates: (t, e) => {
      t.shouldShowIntermediates = e.payload;
    },
    resetCanvas: (t) => {
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.layerState = qs, t.futureLayerStates = [];
    },
    setCanvasContainerDimensions: (t, e) => {
      t.canvasContainerDimensions = e.payload;
    },
    resizeAndScaleCanvas: (t) => {
      const { width: e, height: r } = t.canvasContainerDimensions, n = t.layerState.objects.find(jp), i = {
        width: Math.floor(e),
        height: Math.floor(r)
      };
      if (!n) {
        const d = Da(
          i.width,
          i.height,
          512,
          512,
          Ds
        ), g = La(
          i.width,
          i.height,
          0,
          0,
          512,
          512,
          d
        ), p = { width: 512, height: 512 };
        if (t.stageScale = d, t.stageCoordinates = g, t.stageDimensions = i, t.boundingBoxCoordinates = { x: 0, y: 0 }, t.boundingBoxDimensions = p, t.boundingBoxScaleMethod === "auto") {
          const y = Hi(
            p
          );
          t.scaledBoundingBoxDimensions = y;
        }
        return;
      }
      const { width: s, height: o } = n, c = Da(
        e,
        r,
        s,
        o,
        0.95
      ), h = La(
        i.width,
        i.height,
        0,
        0,
        s,
        o,
        c
      );
      t.minimumStageScale = c, t.stageScale = c, t.stageCoordinates = Gp(h), t.stageDimensions = i, t.isCanvasInitialized = !0;
    },
    resizeCanvas: (t) => {
      const { width: e, height: r } = t.canvasContainerDimensions, n = {
        width: Math.floor(e),
        height: Math.floor(r)
      };
      if (t.stageDimensions = n, !t.layerState.objects.find(jp)) {
        const i = Da(
          n.width,
          n.height,
          512,
          512,
          Ds
        ), s = La(
          n.width,
          n.height,
          0,
          0,
          512,
          512,
          i
        ), o = { width: 512, height: 512 };
        if (t.stageScale = i, t.stageCoordinates = s, t.boundingBoxCoordinates = { x: 0, y: 0 }, t.boundingBoxDimensions = o, t.boundingBoxScaleMethod === "auto") {
          const l = Hi(
            o
          );
          t.scaledBoundingBoxDimensions = l;
        }
      }
    },
    resetCanvasView: (t, e) => {
      const { contentRect: r, shouldScaleTo1: n } = e.payload, {
        stageDimensions: { width: i, height: s }
      } = t, { x: o, y: l, width: c, height: h } = r;
      if (c !== 0 && h !== 0) {
        const d = n ? 1 : Da(
          i,
          s,
          c,
          h,
          Ds
        ), g = La(
          i,
          s,
          o,
          l,
          c,
          h,
          d
        );
        t.stageScale = d, t.stageCoordinates = g;
      } else {
        const d = Da(
          i,
          s,
          512,
          512,
          Ds
        ), g = La(
          i,
          s,
          0,
          0,
          512,
          512,
          d
        ), p = { width: 512, height: 512 };
        if (t.stageScale = d, t.stageCoordinates = g, t.boundingBoxCoordinates = { x: 0, y: 0 }, t.boundingBoxDimensions = p, t.boundingBoxScaleMethod === "auto") {
          const y = Hi(
            p
          );
          t.scaledBoundingBoxDimensions = y;
        }
      }
    },
    nextStagingAreaImage: (t) => {
      const e = t.layerState.stagingArea.selectedImageIndex, r = t.layerState.stagingArea.images.length;
      t.layerState.stagingArea.selectedImageIndex = Math.min(
        e + 1,
        r - 1
      );
    },
    prevStagingAreaImage: (t) => {
      const e = t.layerState.stagingArea.selectedImageIndex;
      t.layerState.stagingArea.selectedImageIndex = Math.max(
        e - 1,
        0
      );
    },
    commitStagingAreaImage: (t) => {
      const { images: e, selectedImageIndex: r } = t.layerState.stagingArea;
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.pastLayerStates.length > t.maxHistory && t.pastLayerStates.shift(), t.layerState.objects.push({
        ...e[r]
      }), t.layerState.stagingArea = {
        ...qs.stagingArea
      }, t.futureLayerStates = [], t.shouldShowStagingOutline = !0, t.shouldShowStagingImage = !0;
    },
    fitBoundingBoxToStage: (t) => {
      const {
        boundingBoxDimensions: e,
        boundingBoxCoordinates: r,
        stageDimensions: n,
        stageScale: i
      } = t, s = n.width / i, o = n.height / i;
      if (r.x < 0 || r.x + e.width > s || r.y < 0 || r.y + e.height > o) {
        const l = {
          width: Tu(Ct.clamp(s, 64, 512), 64),
          height: Tu(Ct.clamp(o, 64, 512), 64)
        }, c = {
          x: Si(
            s / 2 - l.width / 2,
            64
          ),
          y: Si(
            o / 2 - l.height / 2,
            64
          )
        };
        if (t.boundingBoxDimensions = l, t.boundingBoxCoordinates = c, t.boundingBoxScaleMethod === "auto") {
          const h = Hi(
            l
          );
          t.scaledBoundingBoxDimensions = h;
        }
      }
    },
    setBoundingBoxScaleMethod: (t, e) => {
      if (t.boundingBoxScaleMethod = e.payload, e.payload === "auto") {
        const r = Hi(
          t.boundingBoxDimensions
        );
        t.scaledBoundingBoxDimensions = r;
      }
    },
    setScaledBoundingBoxDimensions: (t, e) => {
      t.scaledBoundingBoxDimensions = e.payload;
    },
    setShouldShowStagingImage: (t, e) => {
      t.shouldShowStagingImage = e.payload;
    },
    setShouldShowStagingOutline: (t, e) => {
      t.shouldShowStagingOutline = e.payload;
    },
    setShouldShowCanvasDebugInfo: (t, e) => {
      t.shouldShowCanvasDebugInfo = e.payload;
    },
    setShouldRestrictStrokesToBox: (t, e) => {
      t.shouldRestrictStrokesToBox = e.payload;
    },
    setShouldCropToBoundingBoxOnSave: (t, e) => {
      t.shouldCropToBoundingBoxOnSave = e.payload;
    },
    setColorPickerColor: (t, e) => {
      t.colorPickerColor = e.payload;
    },
    commitColorPickerColor: (t) => {
      t.brushColor = {
        ...t.colorPickerColor,
        a: t.brushColor.a
      }, t.tool = "brush";
    },
    setMergedCanvas: (t, e) => {
      t.pastLayerStates.push(Ct.cloneDeep(t.layerState)), t.futureLayerStates = [], t.layerState.objects = [e.payload];
    },
    resetCanvasInteractionState: (t) => {
      t.cursorPosition = null, t.isDrawing = !1, t.isMouseOverBoundingBox = !1, t.isMoveBoundingBoxKeyHeld = !1, t.isMoveStageKeyHeld = !1, t.isMovingBoundingBox = !1, t.isMovingStage = !1, t.isTransformingBoundingBox = !1;
    },
    mouseLeftCanvas: (t) => {
      t.cursorPosition = null, t.isDrawing = !1, t.isMouseOverBoundingBox = !1, t.isMovingBoundingBox = !1, t.isTransformingBoundingBox = !1;
    }
  }
}), {
  addEraseRect: VD,
  addFillRect: qD,
  addImageToStagingArea: aO,
  addLine: YD,
  addPointToCurrentLine: KD,
  clearCanvasHistory: XD,
  clearMask: JD,
  commitColorPickerColor: QD,
  commitStagingAreaImage: ZD,
  discardStagedImages: eM,
  fitBoundingBoxToStage: tM,
  mouseLeftCanvas: rM,
  nextStagingAreaImage: nM,
  prevStagingAreaImage: iM,
  redo: aM,
  resetCanvas: sM,
  resetCanvasInteractionState: oM,
  resetCanvasView: uM,
  resizeAndScaleCanvas: lM,
  resizeCanvas: cM,
  setBoundingBoxCoordinates: fM,
  setBoundingBoxDimensions: hM,
  setBoundingBoxPreviewFill: dM,
  setBoundingBoxScaleMethod: gM,
  setBrushColor: pM,
  setBrushSize: mM,
  setCanvasContainerDimensions: vM,
  setColorPickerColor: yM,
  setCursorPosition: bM,
  setDoesCanvasNeedScaling: _M,
  setInitialCanvasImage: SM,
  setIsDrawing: wM,
  setIsMaskEnabled: EM,
  setIsMouseOverBoundingBox: CM,
  setIsMoveBoundingBoxKeyHeld: xM,
  setIsMoveStageKeyHeld: OM,
  setIsMovingBoundingBox: TM,
  setIsMovingStage: PM,
  setIsTransformingBoundingBox: RM,
  setLayer: AM,
  setMaskColor: kM,
  setMergedCanvas: IM,
  setShouldAutoSave: LM,
  setShouldCropToBoundingBoxOnSave: DM,
  setShouldDarkenOutsideBoundingBox: MM,
  setShouldLockBoundingBox: NM,
  setShouldPreserveMaskedArea: BM,
  setShouldShowBoundingBox: FM,
  setShouldShowBrush: GM,
  setShouldShowBrushPreview: jM,
  setShouldShowCanvasDebugInfo: $M,
  setShouldShowCheckboardTransparency: WM,
  setShouldShowGrid: UM,
  setShouldShowIntermediates: HM,
  setShouldShowStagingImage: zM,
  setShouldShowStagingOutline: VM,
  setShouldSnapToGrid: qM,
  setStageCoordinates: YM,
  setStageScale: KM,
  setTool: XM,
  toggleShouldLockBoundingBox: JM,
  toggleTool: QM,
  undo: ZM,
  setScaledBoundingBoxDimensions: eN,
  setShouldRestrictStrokesToBox: tN
} = my.actions, sO = my.reducer, oO = {
  currentImageUuid: "",
  shouldPinGallery: !0,
  shouldShowGallery: !0,
  galleryImageMinimumWidth: 64,
  galleryImageObjectFit: "cover",
  shouldHoldGalleryOpen: !1,
  shouldAutoSwitchToNewImages: !0,
  currentCategory: "result",
  categories: {
    user: {
      images: [],
      latest_mtime: void 0,
      earliest_mtime: void 0,
      areMoreImagesAvailable: !0
    },
    result: {
      images: [],
      latest_mtime: void 0,
      earliest_mtime: void 0,
      areMoreImagesAvailable: !0
    }
  },
  galleryWidth: 300,
  shouldUseSingleGalleryColumn: !1
}, vy = Qi({
  name: "gallery",
  initialState: oO,
  reducers: {
    setCurrentImage: (t, e) => {
      t.currentImage = e.payload, t.currentImageUuid = e.payload.uuid;
    },
    removeImage: (t, e) => {
      const { uuid: r, category: n } = e.payload, i = t.categories[n].images, s = i.filter((o) => o.uuid !== r);
      if (r === t.currentImageUuid) {
        const o = i.findIndex(
          (c) => c.uuid === r
        ), l = Ct.clamp(
          o,
          0,
          s.length - 1
        );
        t.currentImage = s.length ? s[l] : void 0, t.currentImageUuid = s.length ? s[l].uuid : "";
      }
      t.categories[n].images = s;
    },
    addImage: (t, e) => {
      const { image: r, category: n } = e.payload, { uuid: i, url: s, mtime: o } = r, l = t.categories[n];
      l.images.find((c) => c.url === s && c.mtime === o) || (l.images.unshift(r), t.shouldAutoSwitchToNewImages && (t.currentImageUuid = i, t.currentImage = r, t.currentCategory = n), t.intermediateImage = void 0, l.latest_mtime = o);
    },
    setIntermediateImage: (t, e) => {
      t.intermediateImage = e.payload;
    },
    clearIntermediateImage: (t) => {
      t.intermediateImage = void 0;
    },
    selectNextImage: (t) => {
      const { currentImage: e } = t;
      if (!e)
        return;
      const r = t.categories[e.category].images;
      if (e) {
        const n = r.findIndex(
          (i) => i.uuid === e.uuid
        );
        if (n < r.length - 1) {
          const i = r[n + 1];
          t.currentImage = i, t.currentImageUuid = i.uuid;
        }
      }
    },
    selectPrevImage: (t) => {
      const { currentImage: e } = t;
      if (!e)
        return;
      const r = t.categories[e.category].images;
      if (e) {
        const n = r.findIndex(
          (i) => i.uuid === e.uuid
        );
        if (n > 0) {
          const i = r[n - 1];
          t.currentImage = i, t.currentImageUuid = i.uuid;
        }
      }
    },
    addGalleryImages: (t, e) => {
      const { images: r, areMoreImagesAvailable: n, category: i } = e.payload, s = t.categories[i].images;
      if (r.length > 0) {
        const o = r.filter(
          (l) => !s.find(
            (c) => c.url === l.url && c.mtime === l.mtime
          )
        );
        if (t.categories[i].images = s.concat(o).sort((l, c) => c.mtime - l.mtime), !t.currentImage) {
          const l = r[0];
          t.currentImage = l, t.currentImageUuid = l.uuid;
        }
        t.categories[i].latest_mtime = r[0].mtime, t.categories[i].earliest_mtime = r[r.length - 1].mtime;
      }
      n !== void 0 && (t.categories[i].areMoreImagesAvailable = n);
    },
    setShouldPinGallery: (t, e) => {
      t.shouldPinGallery = e.payload;
    },
    setShouldShowGallery: (t, e) => {
      t.shouldShowGallery = e.payload;
    },
    setGalleryImageMinimumWidth: (t, e) => {
      t.galleryImageMinimumWidth = e.payload;
    },
    setGalleryImageObjectFit: (t, e) => {
      t.galleryImageObjectFit = e.payload;
    },
    setShouldHoldGalleryOpen: (t, e) => {
      t.shouldHoldGalleryOpen = e.payload;
    },
    setShouldAutoSwitchToNewImages: (t, e) => {
      t.shouldAutoSwitchToNewImages = e.payload;
    },
    setCurrentCategory: (t, e) => {
      t.currentCategory = e.payload;
    },
    setGalleryWidth: (t, e) => {
      t.galleryWidth = e.payload;
    },
    setShouldUseSingleGalleryColumn: (t, e) => {
      t.shouldUseSingleGalleryColumn = e.payload;
    }
  }
}), {
  addImage: Pu,
  clearIntermediateImage: yf,
  removeImage: yy,
  setCurrentImage: rN,
  addGalleryImages: uO,
  setIntermediateImage: lO,
  selectNextImage: nN,
  selectPrevImage: iN,
  setShouldPinGallery: aN,
  setShouldShowGallery: sN,
  setGalleryImageMinimumWidth: oN,
  setGalleryImageObjectFit: uN,
  setShouldHoldGalleryOpen: lN,
  setShouldAutoSwitchToNewImages: cN,
  setCurrentCategory: fN,
  setGalleryWidth: hN,
  setShouldUseSingleGalleryColumn: dN
} = vy.actions, cO = vy.reducer, fO = {
  isLightboxOpen: !1
}, hO = fO, by = Qi({
  name: "lightbox",
  initialState: hO,
  reducers: {
    setIsLightboxOpen: (t, e) => {
      t.isLightboxOpen = e.payload;
    }
  }
}), { setIsLightboxOpen: gN } = by.actions, dO = by.reducer, Zu = (t) => typeof t == "string" ? t : t.length === 1 ? t[0].prompt : t.map(
  (e) => `${e.prompt}:${e.weight}`
).join(" ");
function gO(t) {
  let e = typeof t == "string" ? t : Zu(t), r = "";
  const n = new RegExp(/\[([^\][]*)]/, "gi"), i = [...e.matchAll(n)].map(
    (s) => s[1]
  );
  return i.length && (r = i.join(" "), i.forEach((s) => {
    e = e.replace(`[${s}]`, "").replaceAll("[]", "").trim();
  })), [e, r];
}
const pO = (t) => {
  const n = t.split(",").map((i) => i.split(":")).map((i) => ({ seed: Number(i[0]), weight: Number(i[1]) }));
  return mO(n) ? n : !1;
}, mO = (t) => Boolean(typeof t == "string" ? pO(t) : t.length && !t.some((e) => {
  const { seed: r, weight: n } = e, i = !isNaN(parseInt(r.toString(), 10)), s = !isNaN(parseInt(n.toString(), 10)) && n >= 0 && n <= 1;
  return !(i && s);
})), $p = (t) => t.reduce((e, r, n, i) => {
  const { seed: s, weight: o } = r;
  return e += `${s}:${o}`, n !== i.length - 1 && (e += ","), e;
}, ""), vO = (t) => t.split(",").map((n) => n.split(":")).map(
  (n) => [parseInt(n[0], 10), parseFloat(n[1])]
), _y = {
  cfgScale: 7.5,
  height: 512,
  img2imgStrength: 0.75,
  infillMethod: "patchmatch",
  iterations: 1,
  maskPath: "",
  perlin: 0,
  prompt: "",
  negativePrompt: "",
  sampler: "k_lms",
  seamBlur: 16,
  seamless: !1,
  seamSize: 96,
  seamSteps: 30,
  seamStrength: 0.7,
  seed: 0,
  seedWeights: "",
  shouldFitToWidthHeight: !0,
  shouldGenerateVariations: !1,
  shouldRandomizeSeed: !0,
  steps: 50,
  threshold: 0,
  tileSize: 32,
  variationAmount: 0.1,
  width: 512,
  shouldUseSymmetry: !1,
  horizontalSymmetrySteps: 0,
  verticalSymmetrySteps: 0
}, yO = _y, Sy = Qi({
  name: "generation",
  initialState: yO,
  reducers: {
    setPrompt: (t, e) => {
      const r = e.payload;
      typeof r == "string" ? t.prompt = r : t.prompt = Zu(r);
    },
    setNegativePrompt: (t, e) => {
      const r = e.payload;
      typeof r == "string" ? t.negativePrompt = r : t.negativePrompt = Zu(r);
    },
    setIterations: (t, e) => {
      t.iterations = e.payload;
    },
    setSteps: (t, e) => {
      t.steps = e.payload;
    },
    clampSymmetrySteps: (t) => {
      t.horizontalSymmetrySteps = Ct.clamp(
        t.horizontalSymmetrySteps,
        0,
        t.steps
      ), t.verticalSymmetrySteps = Ct.clamp(
        t.verticalSymmetrySteps,
        0,
        t.steps
      );
    },
    setCfgScale: (t, e) => {
      t.cfgScale = e.payload;
    },
    setThreshold: (t, e) => {
      t.threshold = e.payload;
    },
    setPerlin: (t, e) => {
      t.perlin = e.payload;
    },
    setHeight: (t, e) => {
      t.height = e.payload;
    },
    setWidth: (t, e) => {
      t.width = e.payload;
    },
    setSampler: (t, e) => {
      t.sampler = e.payload;
    },
    setSeed: (t, e) => {
      t.seed = e.payload, t.shouldRandomizeSeed = !1;
    },
    setImg2imgStrength: (t, e) => {
      t.img2imgStrength = e.payload;
    },
    setMaskPath: (t, e) => {
      t.maskPath = e.payload;
    },
    setSeamless: (t, e) => {
      t.seamless = e.payload;
    },
    setShouldFitToWidthHeight: (t, e) => {
      t.shouldFitToWidthHeight = e.payload;
    },
    resetSeed: (t) => {
      t.seed = -1;
    },
    setParameter: (t, e) => {
      const { key: r, value: n } = e.payload, i = { ...t, [r]: n };
      return r === "seed" && (i.shouldRandomizeSeed = !1), i;
    },
    setShouldGenerateVariations: (t, e) => {
      t.shouldGenerateVariations = e.payload;
    },
    setVariationAmount: (t, e) => {
      t.variationAmount = e.payload;
    },
    setSeedWeights: (t, e) => {
      t.seedWeights = e.payload, t.shouldGenerateVariations = !0, t.variationAmount = 0;
    },
    setAllTextToImageParameters: (t, e) => {
      const {
        sampler: r,
        prompt: n,
        seed: i,
        variations: s,
        steps: o,
        cfg_scale: l,
        threshold: c,
        perlin: h,
        seamless: d,
        _hires_fix: g,
        width: p,
        height: y
      } = e.payload.image;
      s && s.length > 0 ? (t.seedWeights = $p(s), t.shouldGenerateVariations = !0, t.variationAmount = 0) : t.shouldGenerateVariations = !1, i && (t.seed = i, t.shouldRandomizeSeed = !1), n && (t.prompt = Zu(n)), r && (t.sampler = r), o && (t.steps = o), l && (t.cfgScale = l), typeof c > "u" ? t.threshold = 0 : t.threshold = c, typeof h > "u" ? t.perlin = 0 : t.perlin = h, typeof d == "boolean" && (t.seamless = d), p && (t.width = p), y && (t.height = y);
    },
    setAllImageToImageParameters: (t, e) => {
      const { type: r, strength: n, fit: i, init_image_path: s, mask_image_path: o } = e.payload.image;
      r === "img2img" && (s && (t.initialImage = s), o && (t.maskPath = o), n && (t.img2imgStrength = n), typeof i == "boolean" && (t.shouldFitToWidthHeight = i));
    },
    setAllParameters: (t, e) => {
      const {
        type: r,
        sampler: n,
        prompt: i,
        seed: s,
        variations: o,
        steps: l,
        cfg_scale: c,
        threshold: h,
        perlin: d,
        seamless: g,
        _hires_fix: p,
        width: y,
        height: v,
        strength: _,
        fit: b,
        init_image_path: C,
        mask_image_path: w
      } = e.payload.image;
      if (r === "img2img" && (C && (t.initialImage = C), w && (t.maskPath = w), _ && (t.img2imgStrength = _), typeof b == "boolean" && (t.shouldFitToWidthHeight = b)), o && o.length > 0 ? (t.seedWeights = $p(o), t.shouldGenerateVariations = !0, t.variationAmount = 0) : t.shouldGenerateVariations = !1, s && (t.seed = s, t.shouldRandomizeSeed = !1), i) {
        const [O, k] = gO(i);
        O && (t.prompt = O), k ? t.negativePrompt = k : t.negativePrompt = "";
      }
      n && (t.sampler = n), l && (t.steps = l), c && (t.cfgScale = c), typeof h > "u" ? t.threshold = 0 : t.threshold = h, typeof d > "u" ? t.perlin = 0 : t.perlin = d, typeof g == "boolean" && (t.seamless = g), y && (t.width = y), v && (t.height = v);
    },
    resetParametersState: (t) => ({
      ...t,
      ..._y
    }),
    setShouldRandomizeSeed: (t, e) => {
      t.shouldRandomizeSeed = e.payload;
    },
    setInitialImage: (t, e) => {
      t.initialImage = e.payload;
    },
    clearInitialImage: (t) => {
      t.initialImage = void 0;
    },
    setSeamSize: (t, e) => {
      t.seamSize = e.payload;
    },
    setSeamBlur: (t, e) => {
      t.seamBlur = e.payload;
    },
    setSeamStrength: (t, e) => {
      t.seamStrength = e.payload;
    },
    setSeamSteps: (t, e) => {
      t.seamSteps = e.payload;
    },
    setTileSize: (t, e) => {
      t.tileSize = e.payload;
    },
    setInfillMethod: (t, e) => {
      t.infillMethod = e.payload;
    },
    setShouldUseSymmetry: (t, e) => {
      t.shouldUseSymmetry = e.payload;
    },
    setHorizontalSymmetrySteps: (t, e) => {
      t.horizontalSymmetrySteps = e.payload;
    },
    setVerticalSymmetrySteps: (t, e) => {
      t.verticalSymmetrySteps = e.payload;
    }
  }
}), {
  clampSymmetrySteps: pN,
  clearInitialImage: bO,
  resetParametersState: mN,
  resetSeed: vN,
  setAllImageToImageParameters: yN,
  setAllParameters: bN,
  setAllTextToImageParameters: _N,
  setCfgScale: SN,
  setHeight: wN,
  setImg2imgStrength: EN,
  setInfillMethod: _O,
  setInitialImage: SO,
  setIterations: CN,
  setMaskPath: wO,
  setParameter: xN,
  setPerlin: ON,
  setPrompt: TN,
  setNegativePrompt: PN,
  setSampler: RN,
  setSeamBlur: AN,
  setSeamless: kN,
  setSeamSize: IN,
  setSeamSteps: LN,
  setSeamStrength: DN,
  setSeed: MN,
  setSeedWeights: NN,
  setShouldFitToWidthHeight: BN,
  setShouldGenerateVariations: FN,
  setShouldRandomizeSeed: GN,
  setSteps: jN,
  setThreshold: $N,
  setTileSize: WN,
  setVariationAmount: UN,
  setWidth: HN,
  setShouldUseSymmetry: zN,
  setHorizontalSymmetrySteps: VN,
  setVerticalSymmetrySteps: qN
} = Sy.actions, EO = Sy.reducer, wy = {
  codeformerFidelity: 0.75,
  facetoolStrength: 0.75,
  facetoolType: "gfpgan",
  hiresFix: !1,
  hiresStrength: 0.75,
  shouldLoopback: !1,
  shouldRunESRGAN: !1,
  shouldRunFacetool: !1,
  upscalingLevel: 4,
  upscalingDenoising: 0.75,
  upscalingStrength: 0.75
}, CO = wy, Ey = Qi({
  name: "postprocessing",
  initialState: CO,
  reducers: {
    setFacetoolStrength: (t, e) => {
      t.facetoolStrength = e.payload;
    },
    setCodeformerFidelity: (t, e) => {
      t.codeformerFidelity = e.payload;
    },
    setUpscalingLevel: (t, e) => {
      t.upscalingLevel = e.payload;
    },
    setUpscalingDenoising: (t, e) => {
      t.upscalingDenoising = e.payload;
    },
    setUpscalingStrength: (t, e) => {
      t.upscalingStrength = e.payload;
    },
    setHiresFix: (t, e) => {
      t.hiresFix = e.payload;
    },
    setHiresStrength: (t, e) => {
      t.hiresStrength = e.payload;
    },
    resetPostprocessingState: (t) => ({
      ...t,
      ...wy
    }),
    setShouldRunFacetool: (t, e) => {
      t.shouldRunFacetool = e.payload;
    },
    setFacetoolType: (t, e) => {
      t.facetoolType = e.payload;
    },
    setShouldRunESRGAN: (t, e) => {
      t.shouldRunESRGAN = e.payload;
    },
    setShouldLoopback: (t, e) => {
      t.shouldLoopback = e.payload;
    }
  }
}), {
  resetPostprocessingState: YN,
  setCodeformerFidelity: KN,
  setFacetoolStrength: XN,
  setFacetoolType: JN,
  setHiresFix: QN,
  setHiresStrength: ZN,
  setShouldLoopback: eB,
  setShouldRunESRGAN: tB,
  setShouldRunFacetool: rB,
  setUpscalingLevel: nB,
  setUpscalingDenoising: iB,
  setUpscalingStrength: aB
} = Ey.actions, xO = Ey.reducer;
function Xr(t, e) {
  if (!(t instanceof e))
    throw new TypeError("Cannot call a class as a function");
}
function Wp(t, e) {
  for (var r = 0; r < e.length; r++) {
    var n = e[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(t, Yv(n.key), n);
  }
}
function Jr(t, e, r) {
  return e && Wp(t.prototype, e), r && Wp(t, r), Object.defineProperty(t, "prototype", {
    writable: !1
  }), t;
}
function Ei(t) {
  if (t === void 0)
    throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
  return t;
}
function Hf(t, e) {
  return Hf = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(n, i) {
    return n.__proto__ = i, n;
  }, Hf(t, e);
}
function Al(t, e) {
  if (typeof e != "function" && e !== null)
    throw new TypeError("Super expression must either be null or a function");
  t.prototype = Object.create(e && e.prototype, {
    constructor: {
      value: t,
      writable: !0,
      configurable: !0
    }
  }), Object.defineProperty(t, "prototype", {
    writable: !1
  }), e && Hf(t, e);
}
function yo(t, e) {
  if (e && (Kr(e) === "object" || typeof e == "function"))
    return e;
  if (e !== void 0)
    throw new TypeError("Derived constructors may only return object or undefined");
  return Ei(t);
}
function An(t) {
  return An = Object.setPrototypeOf ? Object.getPrototypeOf.bind() : function(r) {
    return r.__proto__ || Object.getPrototypeOf(r);
  }, An(t);
}
function Cy(t) {
  if (Array.isArray(t))
    return t;
}
function OO(t) {
  if (typeof Symbol < "u" && t[Symbol.iterator] != null || t["@@iterator"] != null)
    return Array.from(t);
}
function Up(t, e) {
  (e == null || e > t.length) && (e = t.length);
  for (var r = 0, n = new Array(e); r < e; r++)
    n[r] = t[r];
  return n;
}
function xy(t, e) {
  if (t) {
    if (typeof t == "string")
      return Up(t, e);
    var r = Object.prototype.toString.call(t).slice(8, -1);
    if (r === "Object" && t.constructor && (r = t.constructor.name), r === "Map" || r === "Set")
      return Array.from(t);
    if (r === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))
      return Up(t, e);
  }
}
function Oy() {
  throw new TypeError(`Invalid attempt to destructure non-iterable instance.
In order to be iterable, non-array objects must have a [Symbol.iterator]() method.`);
}
function TO(t) {
  return Cy(t) || OO(t) || xy(t) || Oy();
}
function Hp(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function zp(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Hp(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Hp(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
var PO = {
  type: "logger",
  log: function(e) {
    this.output("log", e);
  },
  warn: function(e) {
    this.output("warn", e);
  },
  error: function(e) {
    this.output("error", e);
  },
  output: function(e, r) {
    console && console[e] && console[e].apply(console, r);
  }
}, RO = function() {
  function t(e) {
    var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    Xr(this, t), this.init(e, r);
  }
  return Jr(t, [{
    key: "init",
    value: function(r) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
      this.prefix = n.prefix || "i18next:", this.logger = r || PO, this.options = n, this.debug = n.debug;
    }
  }, {
    key: "setDebug",
    value: function(r) {
      this.debug = r;
    }
  }, {
    key: "log",
    value: function() {
      for (var r = arguments.length, n = new Array(r), i = 0; i < r; i++)
        n[i] = arguments[i];
      return this.forward(n, "log", "", !0);
    }
  }, {
    key: "warn",
    value: function() {
      for (var r = arguments.length, n = new Array(r), i = 0; i < r; i++)
        n[i] = arguments[i];
      return this.forward(n, "warn", "", !0);
    }
  }, {
    key: "error",
    value: function() {
      for (var r = arguments.length, n = new Array(r), i = 0; i < r; i++)
        n[i] = arguments[i];
      return this.forward(n, "error", "");
    }
  }, {
    key: "deprecate",
    value: function() {
      for (var r = arguments.length, n = new Array(r), i = 0; i < r; i++)
        n[i] = arguments[i];
      return this.forward(n, "warn", "WARNING DEPRECATED: ", !0);
    }
  }, {
    key: "forward",
    value: function(r, n, i, s) {
      return s && !this.debug ? null : (typeof r[0] == "string" && (r[0] = "".concat(i).concat(this.prefix, " ").concat(r[0])), this.logger[n](r));
    }
  }, {
    key: "create",
    value: function(r) {
      return new t(this.logger, zp(zp({}, {
        prefix: "".concat(this.prefix, ":").concat(r, ":")
      }), this.options));
    }
  }, {
    key: "clone",
    value: function(r) {
      return r = r || this.options, r.prefix = r.prefix || this.prefix, new t(this.logger, r);
    }
  }]), t;
}(), xn = new RO(), xi = function() {
  function t() {
    Xr(this, t), this.observers = {};
  }
  return Jr(t, [{
    key: "on",
    value: function(r, n) {
      var i = this;
      return r.split(" ").forEach(function(s) {
        i.observers[s] = i.observers[s] || [], i.observers[s].push(n);
      }), this;
    }
  }, {
    key: "off",
    value: function(r, n) {
      if (this.observers[r]) {
        if (!n) {
          delete this.observers[r];
          return;
        }
        this.observers[r] = this.observers[r].filter(function(i) {
          return i !== n;
        });
      }
    }
  }, {
    key: "emit",
    value: function(r) {
      for (var n = arguments.length, i = new Array(n > 1 ? n - 1 : 0), s = 1; s < n; s++)
        i[s - 1] = arguments[s];
      if (this.observers[r]) {
        var o = [].concat(this.observers[r]);
        o.forEach(function(c) {
          c.apply(void 0, i);
        });
      }
      if (this.observers["*"]) {
        var l = [].concat(this.observers["*"]);
        l.forEach(function(c) {
          c.apply(c, [r].concat(i));
        });
      }
    }
  }]), t;
}();
function Ms() {
  var t, e, r = new Promise(function(n, i) {
    t = n, e = i;
  });
  return r.resolve = t, r.reject = e, r;
}
function Vp(t) {
  return t == null ? "" : "" + t;
}
function AO(t, e, r) {
  t.forEach(function(n) {
    e[n] && (r[n] = e[n]);
  });
}
function kh(t, e, r) {
  function n(l) {
    return l && l.indexOf("###") > -1 ? l.replace(/###/g, ".") : l;
  }
  function i() {
    return !t || typeof t == "string";
  }
  for (var s = typeof e != "string" ? [].concat(e) : e.split("."); s.length > 1; ) {
    if (i())
      return {};
    var o = n(s.shift());
    !t[o] && r && (t[o] = new r()), Object.prototype.hasOwnProperty.call(t, o) ? t = t[o] : t = {};
  }
  return i() ? {} : {
    obj: t,
    k: n(s.shift())
  };
}
function qp(t, e, r) {
  var n = kh(t, e, Object), i = n.obj, s = n.k;
  i[s] = r;
}
function kO(t, e, r, n) {
  var i = kh(t, e, Object), s = i.obj, o = i.k;
  s[o] = s[o] || [], n && (s[o] = s[o].concat(r)), n || s[o].push(r);
}
function bl(t, e) {
  var r = kh(t, e), n = r.obj, i = r.k;
  if (n)
    return n[i];
}
function Yp(t, e, r) {
  var n = bl(t, r);
  return n !== void 0 ? n : bl(e, r);
}
function Ty(t, e, r) {
  for (var n in e)
    n !== "__proto__" && n !== "constructor" && (n in t ? typeof t[n] == "string" || t[n] instanceof String || typeof e[n] == "string" || e[n] instanceof String ? r && (t[n] = e[n]) : Ty(t[n], e[n], r) : t[n] = e[n]);
  return t;
}
function Ma(t) {
  return t.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&");
}
var IO = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
  "/": "&#x2F;"
};
function LO(t) {
  return typeof t == "string" ? t.replace(/[&<>"'\/]/g, function(e) {
    return IO[e];
  }) : t;
}
var kl = typeof window < "u" && window.navigator && typeof window.navigator.userAgentData > "u" && window.navigator.userAgent && window.navigator.userAgent.indexOf("MSIE") > -1, DO = [" ", ",", "?", "!", ";"];
function MO(t, e, r) {
  e = e || "", r = r || "";
  var n = DO.filter(function(l) {
    return e.indexOf(l) < 0 && r.indexOf(l) < 0;
  });
  if (n.length === 0)
    return !0;
  var i = new RegExp("(".concat(n.map(function(l) {
    return l === "?" ? "\\?" : l;
  }).join("|"), ")")), s = !i.test(t);
  if (!s) {
    var o = t.indexOf(r);
    o > 0 && !i.test(t.substring(0, o)) && (s = !0);
  }
  return s;
}
function Kp(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Ru(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Kp(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Kp(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function NO(t) {
  var e = BO();
  return function() {
    var n = An(t), i;
    if (e) {
      var s = An(this).constructor;
      i = Reflect.construct(n, arguments, s);
    } else
      i = n.apply(this, arguments);
    return yo(this, i);
  };
}
function BO() {
  if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham)
    return !1;
  if (typeof Proxy == "function")
    return !0;
  try {
    return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    })), !0;
  } catch {
    return !1;
  }
}
function Py(t, e) {
  var r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : ".";
  if (t) {
    if (t[e])
      return t[e];
    for (var n = e.split(r), i = t, s = 0; s < n.length; ++s) {
      if (!i || typeof i[n[s]] == "string" && s + 1 < n.length)
        return;
      if (i[n[s]] === void 0) {
        for (var o = 2, l = n.slice(s, s + o).join(r), c = i[l]; c === void 0 && n.length > s + o; )
          o++, l = n.slice(s, s + o).join(r), c = i[l];
        if (c === void 0)
          return;
        if (c === null)
          return null;
        if (e.endsWith(l)) {
          if (typeof c == "string")
            return c;
          if (l && typeof c[l] == "string")
            return c[l];
        }
        var h = n.slice(s + o).join(r);
        return h ? Py(c, h, r) : void 0;
      }
      i = i[n[s]];
    }
    return i;
  }
}
var FO = function(t) {
  Al(r, t);
  var e = NO(r);
  function r(n) {
    var i, s = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {
      ns: ["translation"],
      defaultNS: "translation"
    };
    return Xr(this, r), i = e.call(this), kl && xi.call(Ei(i)), i.data = n || {}, i.options = s, i.options.keySeparator === void 0 && (i.options.keySeparator = "."), i.options.ignoreJSONStructure === void 0 && (i.options.ignoreJSONStructure = !0), i;
  }
  return Jr(r, [{
    key: "addNamespaces",
    value: function(i) {
      this.options.ns.indexOf(i) < 0 && this.options.ns.push(i);
    }
  }, {
    key: "removeNamespaces",
    value: function(i) {
      var s = this.options.ns.indexOf(i);
      s > -1 && this.options.ns.splice(s, 1);
    }
  }, {
    key: "getResource",
    value: function(i, s, o) {
      var l = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : {}, c = l.keySeparator !== void 0 ? l.keySeparator : this.options.keySeparator, h = l.ignoreJSONStructure !== void 0 ? l.ignoreJSONStructure : this.options.ignoreJSONStructure, d = [i, s];
      o && typeof o != "string" && (d = d.concat(o)), o && typeof o == "string" && (d = d.concat(c ? o.split(c) : o)), i.indexOf(".") > -1 && (d = i.split("."));
      var g = bl(this.data, d);
      return g || !h || typeof o != "string" ? g : Py(this.data && this.data[i] && this.data[i][s], o, c);
    }
  }, {
    key: "addResource",
    value: function(i, s, o, l) {
      var c = arguments.length > 4 && arguments[4] !== void 0 ? arguments[4] : {
        silent: !1
      }, h = this.options.keySeparator;
      h === void 0 && (h = ".");
      var d = [i, s];
      o && (d = d.concat(h ? o.split(h) : o)), i.indexOf(".") > -1 && (d = i.split("."), l = s, s = d[1]), this.addNamespaces(s), qp(this.data, d, l), c.silent || this.emit("added", i, s, o, l);
    }
  }, {
    key: "addResources",
    value: function(i, s, o) {
      var l = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : {
        silent: !1
      };
      for (var c in o)
        (typeof o[c] == "string" || Object.prototype.toString.apply(o[c]) === "[object Array]") && this.addResource(i, s, c, o[c], {
          silent: !0
        });
      l.silent || this.emit("added", i, s, o);
    }
  }, {
    key: "addResourceBundle",
    value: function(i, s, o, l, c) {
      var h = arguments.length > 5 && arguments[5] !== void 0 ? arguments[5] : {
        silent: !1
      }, d = [i, s];
      i.indexOf(".") > -1 && (d = i.split("."), l = o, o = s, s = d[1]), this.addNamespaces(s);
      var g = bl(this.data, d) || {};
      l ? Ty(g, o, c) : g = Ru(Ru({}, g), o), qp(this.data, d, g), h.silent || this.emit("added", i, s, o);
    }
  }, {
    key: "removeResourceBundle",
    value: function(i, s) {
      this.hasResourceBundle(i, s) && delete this.data[i][s], this.removeNamespaces(s), this.emit("removed", i, s);
    }
  }, {
    key: "hasResourceBundle",
    value: function(i, s) {
      return this.getResource(i, s) !== void 0;
    }
  }, {
    key: "getResourceBundle",
    value: function(i, s) {
      return s || (s = this.options.defaultNS), this.options.compatibilityAPI === "v1" ? Ru(Ru({}, {}), this.getResource(i, s)) : this.getResource(i, s);
    }
  }, {
    key: "getDataByLanguage",
    value: function(i) {
      return this.data[i];
    }
  }, {
    key: "hasLanguageSomeTranslations",
    value: function(i) {
      var s = this.getDataByLanguage(i), o = s && Object.keys(s) || [];
      return !!o.find(function(l) {
        return s[l] && Object.keys(s[l]).length > 0;
      });
    }
  }, {
    key: "toJSON",
    value: function() {
      return this.data;
    }
  }]), r;
}(xi), Ry = {
  processors: {},
  addPostProcessor: function(e) {
    this.processors[e.name] = e;
  },
  handle: function(e, r, n, i, s) {
    var o = this;
    return e.forEach(function(l) {
      o.processors[l] && (r = o.processors[l].process(r, n, i, s));
    }), r;
  }
};
function Xp(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function hr(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Xp(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Xp(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function GO(t) {
  var e = jO();
  return function() {
    var n = An(t), i;
    if (e) {
      var s = An(this).constructor;
      i = Reflect.construct(n, arguments, s);
    } else
      i = n.apply(this, arguments);
    return yo(this, i);
  };
}
function jO() {
  if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham)
    return !1;
  if (typeof Proxy == "function")
    return !0;
  try {
    return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    })), !0;
  } catch {
    return !1;
  }
}
var Jp = {}, Qp = function(t) {
  Al(r, t);
  var e = GO(r);
  function r(n) {
    var i, s = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    return Xr(this, r), i = e.call(this), kl && xi.call(Ei(i)), AO(["resourceStore", "languageUtils", "pluralResolver", "interpolator", "backendConnector", "i18nFormat", "utils"], n, Ei(i)), i.options = s, i.options.keySeparator === void 0 && (i.options.keySeparator = "."), i.logger = xn.create("translator"), i;
  }
  return Jr(r, [{
    key: "changeLanguage",
    value: function(i) {
      i && (this.language = i);
    }
  }, {
    key: "exists",
    value: function(i) {
      var s = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {
        interpolation: {}
      };
      if (i == null)
        return !1;
      var o = this.resolve(i, s);
      return o && o.res !== void 0;
    }
  }, {
    key: "extractFromKey",
    value: function(i, s) {
      var o = s.nsSeparator !== void 0 ? s.nsSeparator : this.options.nsSeparator;
      o === void 0 && (o = ":");
      var l = s.keySeparator !== void 0 ? s.keySeparator : this.options.keySeparator, c = s.ns || this.options.defaultNS || [], h = o && i.indexOf(o) > -1, d = !this.options.userDefinedKeySeparator && !s.keySeparator && !this.options.userDefinedNsSeparator && !s.nsSeparator && !MO(i, o, l);
      if (h && !d) {
        var g = i.match(this.interpolator.nestingRegexp);
        if (g && g.length > 0)
          return {
            key: i,
            namespaces: c
          };
        var p = i.split(o);
        (o !== l || o === l && this.options.ns.indexOf(p[0]) > -1) && (c = p.shift()), i = p.join(l);
      }
      return typeof c == "string" && (c = [c]), {
        key: i,
        namespaces: c
      };
    }
  }, {
    key: "translate",
    value: function(i, s, o) {
      var l = this;
      if (Kr(s) !== "object" && this.options.overloadTranslationOptionHandler && (s = this.options.overloadTranslationOptionHandler(arguments)), s || (s = {}), i == null)
        return "";
      Array.isArray(i) || (i = [String(i)]);
      var c = s.returnDetails !== void 0 ? s.returnDetails : this.options.returnDetails, h = s.keySeparator !== void 0 ? s.keySeparator : this.options.keySeparator, d = this.extractFromKey(i[i.length - 1], s), g = d.key, p = d.namespaces, y = p[p.length - 1], v = s.lng || this.language, _ = s.appendNamespaceToCIMode || this.options.appendNamespaceToCIMode;
      if (v && v.toLowerCase() === "cimode") {
        if (_) {
          var b = s.nsSeparator || this.options.nsSeparator;
          return c ? {
            res: "".concat(y).concat(b).concat(g),
            usedKey: g,
            exactUsedKey: g,
            usedLng: v,
            usedNS: y
          } : "".concat(y).concat(b).concat(g);
        }
        return c ? {
          res: g,
          usedKey: g,
          exactUsedKey: g,
          usedLng: v,
          usedNS: y
        } : g;
      }
      var C = this.resolve(i, s), w = C && C.res, O = C && C.usedKey || g, k = C && C.exactUsedKey || g, P = Object.prototype.toString.apply(w), M = ["[object Number]", "[object Function]", "[object RegExp]"], U = s.joinArrays !== void 0 ? s.joinArrays : this.options.joinArrays, V = !this.i18nFormat || this.i18nFormat.handleAsObject, Y = typeof w != "string" && typeof w != "boolean" && typeof w != "number";
      if (V && w && Y && M.indexOf(P) < 0 && !(typeof U == "string" && P === "[object Array]")) {
        if (!s.returnObjects && !this.options.returnObjects) {
          this.options.returnedObjectHandler || this.logger.warn("accessing an object - but returnObjects options is not enabled!");
          var X = this.options.returnedObjectHandler ? this.options.returnedObjectHandler(O, w, hr(hr({}, s), {}, {
            ns: p
          })) : "key '".concat(g, " (").concat(this.language, ")' returned an object instead of string.");
          return c ? (C.res = X, C) : X;
        }
        if (h) {
          var Z = P === "[object Array]", ce = Z ? [] : {}, $ = Z ? k : O;
          for (var R in w)
            if (Object.prototype.hasOwnProperty.call(w, R)) {
              var F = "".concat($).concat(h).concat(R);
              ce[R] = this.translate(F, hr(hr({}, s), {
                joinArrays: !1,
                ns: p
              })), ce[R] === F && (ce[R] = w[R]);
            }
          w = ce;
        }
      } else if (V && typeof U == "string" && P === "[object Array]")
        w = w.join(U), w && (w = this.extendTranslation(w, i, s, o));
      else {
        var H = !1, re = !1, ae = s.count !== void 0 && typeof s.count != "string", ne = r.hasDefaultValue(s), oe = ae ? this.pluralResolver.getSuffix(v, s.count, s) : "", de = s["defaultValue".concat(oe)] || s.defaultValue;
        !this.isValidLookup(w) && ne && (H = !0, w = de), this.isValidLookup(w) || (re = !0, w = g);
        var Re = s.missingKeyNoValueFallbackToKey || this.options.missingKeyNoValueFallbackToKey, Ce = Re && re ? void 0 : w, rt = ne && de !== w && this.options.updateMissing;
        if (re || H || rt) {
          if (this.logger.log(rt ? "updateKey" : "missingKey", v, y, g, rt ? de : w), h) {
            var xt = this.resolve(g, hr(hr({}, s), {}, {
              keySeparator: !1
            }));
            xt && xt.res && this.logger.warn("Seems the loaded translations were in flat JSON format instead of nested. Either set keySeparator: false on init or make sure your translations are published in nested format.");
          }
          var be = [], At = this.languageUtils.getFallbackCodes(this.options.fallbackLng, s.lng || this.language);
          if (this.options.saveMissingTo === "fallback" && At && At[0])
            for (var ot = 0; ot < At.length; ot++)
              be.push(At[ot]);
          else
            this.options.saveMissingTo === "all" ? be = this.languageUtils.toResolveHierarchy(s.lng || this.language) : be.push(s.lng || this.language);
          var wt = function(dt, te, $t) {
            var fe = ne && $t !== w ? $t : Ce;
            l.options.missingKeyHandler ? l.options.missingKeyHandler(dt, y, te, fe, rt, s) : l.backendConnector && l.backendConnector.saveMissing && l.backendConnector.saveMissing(dt, y, te, fe, rt, s), l.emit("missingKey", dt, y, te, w);
          };
          this.options.saveMissing && (this.options.saveMissingPlurals && ae ? be.forEach(function(kt) {
            l.pluralResolver.getSuffixes(kt, s).forEach(function(dt) {
              wt([kt], g + dt, s["defaultValue".concat(dt)] || de);
            });
          }) : wt(be, g, de));
        }
        w = this.extendTranslation(w, i, s, C, o), re && w === g && this.options.appendNamespaceToMissingKey && (w = "".concat(y, ":").concat(g)), (re || H) && this.options.parseMissingKeyHandler && (this.options.compatibilityAPI !== "v1" ? w = this.options.parseMissingKeyHandler(this.options.appendNamespaceToMissingKey ? "".concat(y, ":").concat(g) : g, H ? w : void 0) : w = this.options.parseMissingKeyHandler(w));
      }
      return c ? (C.res = w, C) : w;
    }
  }, {
    key: "extendTranslation",
    value: function(i, s, o, l, c) {
      var h = this;
      if (this.i18nFormat && this.i18nFormat.parse)
        i = this.i18nFormat.parse(i, hr(hr({}, this.options.interpolation.defaultVariables), o), l.usedLng, l.usedNS, l.usedKey, {
          resolved: l
        });
      else if (!o.skipInterpolation) {
        o.interpolation && this.interpolator.init(hr(hr({}, o), {
          interpolation: hr(hr({}, this.options.interpolation), o.interpolation)
        }));
        var d = typeof i == "string" && (o && o.interpolation && o.interpolation.skipOnVariables !== void 0 ? o.interpolation.skipOnVariables : this.options.interpolation.skipOnVariables), g;
        if (d) {
          var p = i.match(this.interpolator.nestingRegexp);
          g = p && p.length;
        }
        var y = o.replace && typeof o.replace != "string" ? o.replace : o;
        if (this.options.interpolation.defaultVariables && (y = hr(hr({}, this.options.interpolation.defaultVariables), y)), i = this.interpolator.interpolate(i, y, o.lng || this.language, o), d) {
          var v = i.match(this.interpolator.nestingRegexp), _ = v && v.length;
          g < _ && (o.nest = !1);
        }
        o.nest !== !1 && (i = this.interpolator.nest(i, function() {
          for (var w = arguments.length, O = new Array(w), k = 0; k < w; k++)
            O[k] = arguments[k];
          return c && c[0] === O[0] && !o.context ? (h.logger.warn("It seems you are nesting recursively key: ".concat(O[0], " in key: ").concat(s[0])), null) : h.translate.apply(h, O.concat([s]));
        }, o)), o.interpolation && this.interpolator.reset();
      }
      var b = o.postProcess || this.options.postProcess, C = typeof b == "string" ? [b] : b;
      return i != null && C && C.length && o.applyPostProcessor !== !1 && (i = Ry.handle(C, i, s, this.options && this.options.postProcessPassResolved ? hr({
        i18nResolved: l
      }, o) : o, this)), i;
    }
  }, {
    key: "resolve",
    value: function(i) {
      var s = this, o = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, l, c, h, d, g;
      return typeof i == "string" && (i = [i]), i.forEach(function(p) {
        if (!s.isValidLookup(l)) {
          var y = s.extractFromKey(p, o), v = y.key;
          c = v;
          var _ = y.namespaces;
          s.options.fallbackNS && (_ = _.concat(s.options.fallbackNS));
          var b = o.count !== void 0 && typeof o.count != "string", C = b && !o.ordinal && o.count === 0 && s.pluralResolver.shouldUseIntlApi(), w = o.context !== void 0 && (typeof o.context == "string" || typeof o.context == "number") && o.context !== "", O = o.lngs ? o.lngs : s.languageUtils.toResolveHierarchy(o.lng || s.language, o.fallbackLng);
          _.forEach(function(k) {
            s.isValidLookup(l) || (g = k, !Jp["".concat(O[0], "-").concat(k)] && s.utils && s.utils.hasLoadedNamespace && !s.utils.hasLoadedNamespace(g) && (Jp["".concat(O[0], "-").concat(k)] = !0, s.logger.warn('key "'.concat(c, '" for languages "').concat(O.join(", "), `" won't get resolved as namespace "`).concat(g, '" was not yet loaded'), "This means something IS WRONG in your setup. You access the t function before i18next.init / i18next.loadNamespace / i18next.changeLanguage was done. Wait for the callback or Promise to resolve before accessing it!!!")), O.forEach(function(P) {
              if (!s.isValidLookup(l)) {
                d = P;
                var M = [v];
                if (s.i18nFormat && s.i18nFormat.addLookupKeys)
                  s.i18nFormat.addLookupKeys(M, v, P, k, o);
                else {
                  var U;
                  b && (U = s.pluralResolver.getSuffix(P, o.count, o));
                  var V = "".concat(s.options.pluralSeparator, "zero");
                  if (b && (M.push(v + U), C && M.push(v + V)), w) {
                    var Y = "".concat(v).concat(s.options.contextSeparator).concat(o.context);
                    M.push(Y), b && (M.push(Y + U), C && M.push(Y + V));
                  }
                }
                for (var X; X = M.pop(); )
                  s.isValidLookup(l) || (h = X, l = s.getResource(P, k, X, o));
              }
            }));
          });
        }
      }), {
        res: l,
        usedKey: c,
        exactUsedKey: h,
        usedLng: d,
        usedNS: g
      };
    }
  }, {
    key: "isValidLookup",
    value: function(i) {
      return i !== void 0 && !(!this.options.returnNull && i === null) && !(!this.options.returnEmptyString && i === "");
    }
  }, {
    key: "getResource",
    value: function(i, s, o) {
      var l = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : {};
      return this.i18nFormat && this.i18nFormat.getResource ? this.i18nFormat.getResource(i, s, o, l) : this.resourceStore.getResource(i, s, o, l);
    }
  }], [{
    key: "hasDefaultValue",
    value: function(i) {
      var s = "defaultValue";
      for (var o in i)
        if (Object.prototype.hasOwnProperty.call(i, o) && s === o.substring(0, s.length) && i[o] !== void 0)
          return !0;
      return !1;
    }
  }]), r;
}(xi);
function bf(t) {
  return t.charAt(0).toUpperCase() + t.slice(1);
}
var Zp = function() {
  function t(e) {
    Xr(this, t), this.options = e, this.supportedLngs = this.options.supportedLngs || !1, this.logger = xn.create("languageUtils");
  }
  return Jr(t, [{
    key: "getScriptPartFromCode",
    value: function(r) {
      if (!r || r.indexOf("-") < 0)
        return null;
      var n = r.split("-");
      return n.length === 2 || (n.pop(), n[n.length - 1].toLowerCase() === "x") ? null : this.formatLanguageCode(n.join("-"));
    }
  }, {
    key: "getLanguagePartFromCode",
    value: function(r) {
      if (!r || r.indexOf("-") < 0)
        return r;
      var n = r.split("-");
      return this.formatLanguageCode(n[0]);
    }
  }, {
    key: "formatLanguageCode",
    value: function(r) {
      if (typeof r == "string" && r.indexOf("-") > -1) {
        var n = ["hans", "hant", "latn", "cyrl", "cans", "mong", "arab"], i = r.split("-");
        return this.options.lowerCaseLng ? i = i.map(function(s) {
          return s.toLowerCase();
        }) : i.length === 2 ? (i[0] = i[0].toLowerCase(), i[1] = i[1].toUpperCase(), n.indexOf(i[1].toLowerCase()) > -1 && (i[1] = bf(i[1].toLowerCase()))) : i.length === 3 && (i[0] = i[0].toLowerCase(), i[1].length === 2 && (i[1] = i[1].toUpperCase()), i[0] !== "sgn" && i[2].length === 2 && (i[2] = i[2].toUpperCase()), n.indexOf(i[1].toLowerCase()) > -1 && (i[1] = bf(i[1].toLowerCase())), n.indexOf(i[2].toLowerCase()) > -1 && (i[2] = bf(i[2].toLowerCase()))), i.join("-");
      }
      return this.options.cleanCode || this.options.lowerCaseLng ? r.toLowerCase() : r;
    }
  }, {
    key: "isSupportedCode",
    value: function(r) {
      return (this.options.load === "languageOnly" || this.options.nonExplicitSupportedLngs) && (r = this.getLanguagePartFromCode(r)), !this.supportedLngs || !this.supportedLngs.length || this.supportedLngs.indexOf(r) > -1;
    }
  }, {
    key: "getBestMatchFromCodes",
    value: function(r) {
      var n = this;
      if (!r)
        return null;
      var i;
      return r.forEach(function(s) {
        if (!i) {
          var o = n.formatLanguageCode(s);
          (!n.options.supportedLngs || n.isSupportedCode(o)) && (i = o);
        }
      }), !i && this.options.supportedLngs && r.forEach(function(s) {
        if (!i) {
          var o = n.getLanguagePartFromCode(s);
          if (n.isSupportedCode(o))
            return i = o;
          i = n.options.supportedLngs.find(function(l) {
            if (l.indexOf(o) === 0)
              return l;
          });
        }
      }), i || (i = this.getFallbackCodes(this.options.fallbackLng)[0]), i;
    }
  }, {
    key: "getFallbackCodes",
    value: function(r, n) {
      if (!r)
        return [];
      if (typeof r == "function" && (r = r(n)), typeof r == "string" && (r = [r]), Object.prototype.toString.apply(r) === "[object Array]")
        return r;
      if (!n)
        return r.default || [];
      var i = r[n];
      return i || (i = r[this.getScriptPartFromCode(n)]), i || (i = r[this.formatLanguageCode(n)]), i || (i = r[this.getLanguagePartFromCode(n)]), i || (i = r.default), i || [];
    }
  }, {
    key: "toResolveHierarchy",
    value: function(r, n) {
      var i = this, s = this.getFallbackCodes(n || this.options.fallbackLng || [], r), o = [], l = function(h) {
        h && (i.isSupportedCode(h) ? o.push(h) : i.logger.warn("rejecting language code not found in supportedLngs: ".concat(h)));
      };
      return typeof r == "string" && r.indexOf("-") > -1 ? (this.options.load !== "languageOnly" && l(this.formatLanguageCode(r)), this.options.load !== "languageOnly" && this.options.load !== "currentOnly" && l(this.getScriptPartFromCode(r)), this.options.load !== "currentOnly" && l(this.getLanguagePartFromCode(r))) : typeof r == "string" && l(this.formatLanguageCode(r)), s.forEach(function(c) {
        o.indexOf(c) < 0 && l(i.formatLanguageCode(c));
      }), o;
    }
  }]), t;
}(), $O = [{
  lngs: ["ach", "ak", "am", "arn", "br", "fil", "gun", "ln", "mfe", "mg", "mi", "oc", "pt", "pt-BR", "tg", "tl", "ti", "tr", "uz", "wa"],
  nr: [1, 2],
  fc: 1
}, {
  lngs: ["af", "an", "ast", "az", "bg", "bn", "ca", "da", "de", "dev", "el", "en", "eo", "es", "et", "eu", "fi", "fo", "fur", "fy", "gl", "gu", "ha", "hi", "hu", "hy", "ia", "it", "kk", "kn", "ku", "lb", "mai", "ml", "mn", "mr", "nah", "nap", "nb", "ne", "nl", "nn", "no", "nso", "pa", "pap", "pms", "ps", "pt-PT", "rm", "sco", "se", "si", "so", "son", "sq", "sv", "sw", "ta", "te", "tk", "ur", "yo"],
  nr: [1, 2],
  fc: 2
}, {
  lngs: ["ay", "bo", "cgg", "fa", "ht", "id", "ja", "jbo", "ka", "km", "ko", "ky", "lo", "ms", "sah", "su", "th", "tt", "ug", "vi", "wo", "zh"],
  nr: [1],
  fc: 3
}, {
  lngs: ["be", "bs", "cnr", "dz", "hr", "ru", "sr", "uk"],
  nr: [1, 2, 5],
  fc: 4
}, {
  lngs: ["ar"],
  nr: [0, 1, 2, 3, 11, 100],
  fc: 5
}, {
  lngs: ["cs", "sk"],
  nr: [1, 2, 5],
  fc: 6
}, {
  lngs: ["csb", "pl"],
  nr: [1, 2, 5],
  fc: 7
}, {
  lngs: ["cy"],
  nr: [1, 2, 3, 8],
  fc: 8
}, {
  lngs: ["fr"],
  nr: [1, 2],
  fc: 9
}, {
  lngs: ["ga"],
  nr: [1, 2, 3, 7, 11],
  fc: 10
}, {
  lngs: ["gd"],
  nr: [1, 2, 3, 20],
  fc: 11
}, {
  lngs: ["is"],
  nr: [1, 2],
  fc: 12
}, {
  lngs: ["jv"],
  nr: [0, 1],
  fc: 13
}, {
  lngs: ["kw"],
  nr: [1, 2, 3, 4],
  fc: 14
}, {
  lngs: ["lt"],
  nr: [1, 2, 10],
  fc: 15
}, {
  lngs: ["lv"],
  nr: [1, 2, 0],
  fc: 16
}, {
  lngs: ["mk"],
  nr: [1, 2],
  fc: 17
}, {
  lngs: ["mnk"],
  nr: [0, 1, 2],
  fc: 18
}, {
  lngs: ["mt"],
  nr: [1, 2, 11, 20],
  fc: 19
}, {
  lngs: ["or"],
  nr: [2, 1],
  fc: 2
}, {
  lngs: ["ro"],
  nr: [1, 2, 20],
  fc: 20
}, {
  lngs: ["sl"],
  nr: [5, 1, 2, 3],
  fc: 21
}, {
  lngs: ["he", "iw"],
  nr: [1, 2, 20, 21],
  fc: 22
}], WO = {
  1: function(e) {
    return Number(e > 1);
  },
  2: function(e) {
    return Number(e != 1);
  },
  3: function(e) {
    return 0;
  },
  4: function(e) {
    return Number(e % 10 == 1 && e % 100 != 11 ? 0 : e % 10 >= 2 && e % 10 <= 4 && (e % 100 < 10 || e % 100 >= 20) ? 1 : 2);
  },
  5: function(e) {
    return Number(e == 0 ? 0 : e == 1 ? 1 : e == 2 ? 2 : e % 100 >= 3 && e % 100 <= 10 ? 3 : e % 100 >= 11 ? 4 : 5);
  },
  6: function(e) {
    return Number(e == 1 ? 0 : e >= 2 && e <= 4 ? 1 : 2);
  },
  7: function(e) {
    return Number(e == 1 ? 0 : e % 10 >= 2 && e % 10 <= 4 && (e % 100 < 10 || e % 100 >= 20) ? 1 : 2);
  },
  8: function(e) {
    return Number(e == 1 ? 0 : e == 2 ? 1 : e != 8 && e != 11 ? 2 : 3);
  },
  9: function(e) {
    return Number(e >= 2);
  },
  10: function(e) {
    return Number(e == 1 ? 0 : e == 2 ? 1 : e < 7 ? 2 : e < 11 ? 3 : 4);
  },
  11: function(e) {
    return Number(e == 1 || e == 11 ? 0 : e == 2 || e == 12 ? 1 : e > 2 && e < 20 ? 2 : 3);
  },
  12: function(e) {
    return Number(e % 10 != 1 || e % 100 == 11);
  },
  13: function(e) {
    return Number(e !== 0);
  },
  14: function(e) {
    return Number(e == 1 ? 0 : e == 2 ? 1 : e == 3 ? 2 : 3);
  },
  15: function(e) {
    return Number(e % 10 == 1 && e % 100 != 11 ? 0 : e % 10 >= 2 && (e % 100 < 10 || e % 100 >= 20) ? 1 : 2);
  },
  16: function(e) {
    return Number(e % 10 == 1 && e % 100 != 11 ? 0 : e !== 0 ? 1 : 2);
  },
  17: function(e) {
    return Number(e == 1 || e % 10 == 1 && e % 100 != 11 ? 0 : 1);
  },
  18: function(e) {
    return Number(e == 0 ? 0 : e == 1 ? 1 : 2);
  },
  19: function(e) {
    return Number(e == 1 ? 0 : e == 0 || e % 100 > 1 && e % 100 < 11 ? 1 : e % 100 > 10 && e % 100 < 20 ? 2 : 3);
  },
  20: function(e) {
    return Number(e == 1 ? 0 : e == 0 || e % 100 > 0 && e % 100 < 20 ? 1 : 2);
  },
  21: function(e) {
    return Number(e % 100 == 1 ? 1 : e % 100 == 2 ? 2 : e % 100 == 3 || e % 100 == 4 ? 3 : 0);
  },
  22: function(e) {
    return Number(e == 1 ? 0 : e == 2 ? 1 : (e < 0 || e > 10) && e % 10 == 0 ? 2 : 3);
  }
}, UO = ["v1", "v2", "v3"], em = {
  zero: 0,
  one: 1,
  two: 2,
  few: 3,
  many: 4,
  other: 5
};
function HO() {
  var t = {};
  return $O.forEach(function(e) {
    e.lngs.forEach(function(r) {
      t[r] = {
        numbers: e.nr,
        plurals: WO[e.fc]
      };
    });
  }), t;
}
var zO = function() {
  function t(e) {
    var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    Xr(this, t), this.languageUtils = e, this.options = r, this.logger = xn.create("pluralResolver"), (!this.options.compatibilityJSON || this.options.compatibilityJSON === "v4") && (typeof Intl > "u" || !Intl.PluralRules) && (this.options.compatibilityJSON = "v3", this.logger.error("Your environment seems not to be Intl API compatible, use an Intl.PluralRules polyfill. Will fallback to the compatibilityJSON v3 format handling.")), this.rules = HO();
  }
  return Jr(t, [{
    key: "addRule",
    value: function(r, n) {
      this.rules[r] = n;
    }
  }, {
    key: "getRule",
    value: function(r) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
      if (this.shouldUseIntlApi())
        try {
          return new Intl.PluralRules(r, {
            type: n.ordinal ? "ordinal" : "cardinal"
          });
        } catch {
          return;
        }
      return this.rules[r] || this.rules[this.languageUtils.getLanguagePartFromCode(r)];
    }
  }, {
    key: "needsPlural",
    value: function(r) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, i = this.getRule(r, n);
      return this.shouldUseIntlApi() ? i && i.resolvedOptions().pluralCategories.length > 1 : i && i.numbers.length > 1;
    }
  }, {
    key: "getPluralFormsOfKey",
    value: function(r, n) {
      var i = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      return this.getSuffixes(r, i).map(function(s) {
        return "".concat(n).concat(s);
      });
    }
  }, {
    key: "getSuffixes",
    value: function(r) {
      var n = this, i = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, s = this.getRule(r, i);
      return s ? this.shouldUseIntlApi() ? s.resolvedOptions().pluralCategories.sort(function(o, l) {
        return em[o] - em[l];
      }).map(function(o) {
        return "".concat(n.options.prepend).concat(o);
      }) : s.numbers.map(function(o) {
        return n.getSuffix(r, o, i);
      }) : [];
    }
  }, {
    key: "getSuffix",
    value: function(r, n) {
      var i = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {}, s = this.getRule(r, i);
      return s ? this.shouldUseIntlApi() ? "".concat(this.options.prepend).concat(s.select(n)) : this.getSuffixRetroCompatible(s, n) : (this.logger.warn("no plural rule found for: ".concat(r)), "");
    }
  }, {
    key: "getSuffixRetroCompatible",
    value: function(r, n) {
      var i = this, s = r.noAbs ? r.plurals(n) : r.plurals(Math.abs(n)), o = r.numbers[s];
      this.options.simplifyPluralSuffix && r.numbers.length === 2 && r.numbers[0] === 1 && (o === 2 ? o = "plural" : o === 1 && (o = ""));
      var l = function() {
        return i.options.prepend && o.toString() ? i.options.prepend + o.toString() : o.toString();
      };
      return this.options.compatibilityJSON === "v1" ? o === 1 ? "" : typeof o == "number" ? "_plural_".concat(o.toString()) : l() : this.options.compatibilityJSON === "v2" || this.options.simplifyPluralSuffix && r.numbers.length === 2 && r.numbers[0] === 1 ? l() : this.options.prepend && s.toString() ? this.options.prepend + s.toString() : s.toString();
    }
  }, {
    key: "shouldUseIntlApi",
    value: function() {
      return !UO.includes(this.options.compatibilityJSON);
    }
  }]), t;
}();
function tm(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function nn(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? tm(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : tm(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
var VO = function() {
  function t() {
    var e = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    Xr(this, t), this.logger = xn.create("interpolator"), this.options = e, this.format = e.interpolation && e.interpolation.format || function(r) {
      return r;
    }, this.init(e);
  }
  return Jr(t, [{
    key: "init",
    value: function() {
      var r = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
      r.interpolation || (r.interpolation = {
        escapeValue: !0
      });
      var n = r.interpolation;
      this.escape = n.escape !== void 0 ? n.escape : LO, this.escapeValue = n.escapeValue !== void 0 ? n.escapeValue : !0, this.useRawValueToEscape = n.useRawValueToEscape !== void 0 ? n.useRawValueToEscape : !1, this.prefix = n.prefix ? Ma(n.prefix) : n.prefixEscaped || "{{", this.suffix = n.suffix ? Ma(n.suffix) : n.suffixEscaped || "}}", this.formatSeparator = n.formatSeparator ? n.formatSeparator : n.formatSeparator || ",", this.unescapePrefix = n.unescapeSuffix ? "" : n.unescapePrefix || "-", this.unescapeSuffix = this.unescapePrefix ? "" : n.unescapeSuffix || "", this.nestingPrefix = n.nestingPrefix ? Ma(n.nestingPrefix) : n.nestingPrefixEscaped || Ma("$t("), this.nestingSuffix = n.nestingSuffix ? Ma(n.nestingSuffix) : n.nestingSuffixEscaped || Ma(")"), this.nestingOptionsSeparator = n.nestingOptionsSeparator ? n.nestingOptionsSeparator : n.nestingOptionsSeparator || ",", this.maxReplaces = n.maxReplaces ? n.maxReplaces : 1e3, this.alwaysFormat = n.alwaysFormat !== void 0 ? n.alwaysFormat : !1, this.resetRegExp();
    }
  }, {
    key: "reset",
    value: function() {
      this.options && this.init(this.options);
    }
  }, {
    key: "resetRegExp",
    value: function() {
      var r = "".concat(this.prefix, "(.+?)").concat(this.suffix);
      this.regexp = new RegExp(r, "g");
      var n = "".concat(this.prefix).concat(this.unescapePrefix, "(.+?)").concat(this.unescapeSuffix).concat(this.suffix);
      this.regexpUnescape = new RegExp(n, "g");
      var i = "".concat(this.nestingPrefix, "(.+?)").concat(this.nestingSuffix);
      this.nestingRegexp = new RegExp(i, "g");
    }
  }, {
    key: "interpolate",
    value: function(r, n, i, s) {
      var o = this, l, c, h, d = this.options && this.options.interpolation && this.options.interpolation.defaultVariables || {};
      function g(b) {
        return b.replace(/\$/g, "$$$$");
      }
      var p = function(C) {
        if (C.indexOf(o.formatSeparator) < 0) {
          var w = Yp(n, d, C);
          return o.alwaysFormat ? o.format(w, void 0, i, nn(nn(nn({}, s), n), {}, {
            interpolationkey: C
          })) : w;
        }
        var O = C.split(o.formatSeparator), k = O.shift().trim(), P = O.join(o.formatSeparator).trim();
        return o.format(Yp(n, d, k), P, i, nn(nn(nn({}, s), n), {}, {
          interpolationkey: k
        }));
      };
      this.resetRegExp();
      var y = s && s.missingInterpolationHandler || this.options.missingInterpolationHandler, v = s && s.interpolation && s.interpolation.skipOnVariables !== void 0 ? s.interpolation.skipOnVariables : this.options.interpolation.skipOnVariables, _ = [{
        regex: this.regexpUnescape,
        safeValue: function(C) {
          return g(C);
        }
      }, {
        regex: this.regexp,
        safeValue: function(C) {
          return o.escapeValue ? g(o.escape(C)) : g(C);
        }
      }];
      return _.forEach(function(b) {
        for (h = 0; l = b.regex.exec(r); ) {
          var C = l[1].trim();
          if (c = p(C), c === void 0)
            if (typeof y == "function") {
              var w = y(r, l, s);
              c = typeof w == "string" ? w : "";
            } else if (s && Object.prototype.hasOwnProperty.call(s, C))
              c = "";
            else if (v) {
              c = l[0];
              continue;
            } else
              o.logger.warn("missed to pass in variable ".concat(C, " for interpolating ").concat(r)), c = "";
          else
            typeof c != "string" && !o.useRawValueToEscape && (c = Vp(c));
          var O = b.safeValue(c);
          if (r = r.replace(l[0], O), v ? (b.regex.lastIndex += c.length, b.regex.lastIndex -= l[0].length) : b.regex.lastIndex = 0, h++, h >= o.maxReplaces)
            break;
        }
      }), r;
    }
  }, {
    key: "nest",
    value: function(r, n) {
      var i = this, s = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {}, o, l, c;
      function h(y, v) {
        var _ = this.nestingOptionsSeparator;
        if (y.indexOf(_) < 0)
          return y;
        var b = y.split(new RegExp("".concat(_, "[ ]*{"))), C = "{".concat(b[1]);
        y = b[0], C = this.interpolate(C, c);
        var w = C.match(/'/g), O = C.match(/"/g);
        (w && w.length % 2 === 0 && !O || O.length % 2 !== 0) && (C = C.replace(/'/g, '"'));
        try {
          c = JSON.parse(C), v && (c = nn(nn({}, v), c));
        } catch (k) {
          return this.logger.warn("failed parsing options string in nesting for key ".concat(y), k), "".concat(y).concat(_).concat(C);
        }
        return delete c.defaultValue, y;
      }
      for (; o = this.nestingRegexp.exec(r); ) {
        var d = [];
        c = nn({}, s), c = c.replace && typeof c.replace != "string" ? c.replace : c, c.applyPostProcessor = !1, delete c.defaultValue;
        var g = !1;
        if (o[0].indexOf(this.formatSeparator) !== -1 && !/{.*}/.test(o[1])) {
          var p = o[1].split(this.formatSeparator).map(function(y) {
            return y.trim();
          });
          o[1] = p.shift(), d = p, g = !0;
        }
        if (l = n(h.call(this, o[1].trim(), c), c), l && o[0] === r && typeof l != "string")
          return l;
        typeof l != "string" && (l = Vp(l)), l || (this.logger.warn("missed to resolve ".concat(o[1], " for nesting ").concat(r)), l = ""), g && (l = d.reduce(function(y, v) {
          return i.format(y, v, s.lng, nn(nn({}, s), {}, {
            interpolationkey: o[1].trim()
          }));
        }, l.trim())), r = r.replace(o[0], l), this.regexp.lastIndex = 0;
      }
      return r;
    }
  }]), t;
}();
function rm(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Kn(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? rm(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : rm(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function qO(t) {
  var e = t.toLowerCase().trim(), r = {};
  if (t.indexOf("(") > -1) {
    var n = t.split("(");
    e = n[0].toLowerCase().trim();
    var i = n[1].substring(0, n[1].length - 1);
    if (e === "currency" && i.indexOf(":") < 0)
      r.currency || (r.currency = i.trim());
    else if (e === "relativetime" && i.indexOf(":") < 0)
      r.range || (r.range = i.trim());
    else {
      var s = i.split(";");
      s.forEach(function(o) {
        if (o) {
          var l = o.split(":"), c = TO(l), h = c[0], d = c.slice(1), g = d.join(":").trim().replace(/^'+|'+$/g, "");
          r[h.trim()] || (r[h.trim()] = g), g === "false" && (r[h.trim()] = !1), g === "true" && (r[h.trim()] = !0), isNaN(g) || (r[h.trim()] = parseInt(g, 10));
        }
      });
    }
  }
  return {
    formatName: e,
    formatOptions: r
  };
}
function Na(t) {
  var e = {};
  return function(n, i, s) {
    var o = i + JSON.stringify(s), l = e[o];
    return l || (l = t(i, s), e[o] = l), l(n);
  };
}
var YO = function() {
  function t() {
    var e = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
    Xr(this, t), this.logger = xn.create("formatter"), this.options = e, this.formats = {
      number: Na(function(r, n) {
        var i = new Intl.NumberFormat(r, Kn({}, n));
        return function(s) {
          return i.format(s);
        };
      }),
      currency: Na(function(r, n) {
        var i = new Intl.NumberFormat(r, Kn(Kn({}, n), {}, {
          style: "currency"
        }));
        return function(s) {
          return i.format(s);
        };
      }),
      datetime: Na(function(r, n) {
        var i = new Intl.DateTimeFormat(r, Kn({}, n));
        return function(s) {
          return i.format(s);
        };
      }),
      relativetime: Na(function(r, n) {
        var i = new Intl.RelativeTimeFormat(r, Kn({}, n));
        return function(s) {
          return i.format(s, n.range || "day");
        };
      }),
      list: Na(function(r, n) {
        var i = new Intl.ListFormat(r, Kn({}, n));
        return function(s) {
          return i.format(s);
        };
      })
    }, this.init(e);
  }
  return Jr(t, [{
    key: "init",
    value: function(r) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {
        interpolation: {}
      }, i = n.interpolation;
      this.formatSeparator = i.formatSeparator ? i.formatSeparator : i.formatSeparator || ",";
    }
  }, {
    key: "add",
    value: function(r, n) {
      this.formats[r.toLowerCase().trim()] = n;
    }
  }, {
    key: "addCached",
    value: function(r, n) {
      this.formats[r.toLowerCase().trim()] = Na(n);
    }
  }, {
    key: "format",
    value: function(r, n, i) {
      var s = this, o = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : {}, l = n.split(this.formatSeparator), c = l.reduce(function(h, d) {
        var g = qO(d), p = g.formatName, y = g.formatOptions;
        if (s.formats[p]) {
          var v = h;
          try {
            var _ = o && o.formatParams && o.formatParams[o.interpolationkey] || {}, b = _.locale || _.lng || o.locale || o.lng || i;
            v = s.formats[p](h, b, Kn(Kn(Kn({}, y), o), _));
          } catch (C) {
            s.logger.warn(C);
          }
          return v;
        } else
          s.logger.warn("there was no format function for ".concat(p));
        return h;
      }, r);
      return c;
    }
  }]), t;
}();
function nm(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function im(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? nm(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : nm(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function KO(t) {
  var e = XO();
  return function() {
    var n = An(t), i;
    if (e) {
      var s = An(this).constructor;
      i = Reflect.construct(n, arguments, s);
    } else
      i = n.apply(this, arguments);
    return yo(this, i);
  };
}
function XO() {
  if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham)
    return !1;
  if (typeof Proxy == "function")
    return !0;
  try {
    return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    })), !0;
  } catch {
    return !1;
  }
}
function JO(t, e) {
  t.pending[e] !== void 0 && (delete t.pending[e], t.pendingCount--);
}
var QO = function(t) {
  Al(r, t);
  var e = KO(r);
  function r(n, i, s) {
    var o, l = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : {};
    return Xr(this, r), o = e.call(this), kl && xi.call(Ei(o)), o.backend = n, o.store = i, o.services = s, o.languageUtils = s.languageUtils, o.options = l, o.logger = xn.create("backendConnector"), o.waitingReads = [], o.maxParallelReads = l.maxParallelReads || 10, o.readingCalls = 0, o.maxRetries = l.maxRetries >= 0 ? l.maxRetries : 5, o.retryTimeout = l.retryTimeout >= 1 ? l.retryTimeout : 350, o.state = {}, o.queue = [], o.backend && o.backend.init && o.backend.init(s, l.backend, l), o;
  }
  return Jr(r, [{
    key: "queueLoad",
    value: function(i, s, o, l) {
      var c = this, h = {}, d = {}, g = {}, p = {};
      return i.forEach(function(y) {
        var v = !0;
        s.forEach(function(_) {
          var b = "".concat(y, "|").concat(_);
          !o.reload && c.store.hasResourceBundle(y, _) ? c.state[b] = 2 : c.state[b] < 0 || (c.state[b] === 1 ? d[b] === void 0 && (d[b] = !0) : (c.state[b] = 1, v = !1, d[b] === void 0 && (d[b] = !0), h[b] === void 0 && (h[b] = !0), p[_] === void 0 && (p[_] = !0)));
        }), v || (g[y] = !0);
      }), (Object.keys(h).length || Object.keys(d).length) && this.queue.push({
        pending: d,
        pendingCount: Object.keys(d).length,
        loaded: {},
        errors: [],
        callback: l
      }), {
        toLoad: Object.keys(h),
        pending: Object.keys(d),
        toLoadLanguages: Object.keys(g),
        toLoadNamespaces: Object.keys(p)
      };
    }
  }, {
    key: "loaded",
    value: function(i, s, o) {
      var l = i.split("|"), c = l[0], h = l[1];
      s && this.emit("failedLoading", c, h, s), o && this.store.addResourceBundle(c, h, o), this.state[i] = s ? -1 : 2;
      var d = {};
      this.queue.forEach(function(g) {
        kO(g.loaded, [c], h), JO(g, i), s && g.errors.push(s), g.pendingCount === 0 && !g.done && (Object.keys(g.loaded).forEach(function(p) {
          d[p] || (d[p] = {});
          var y = g.loaded[p];
          y.length && y.forEach(function(v) {
            d[p][v] === void 0 && (d[p][v] = !0);
          });
        }), g.done = !0, g.errors.length ? g.callback(g.errors) : g.callback());
      }), this.emit("loaded", d), this.queue = this.queue.filter(function(g) {
        return !g.done;
      });
    }
  }, {
    key: "read",
    value: function(i, s, o) {
      var l = this, c = arguments.length > 3 && arguments[3] !== void 0 ? arguments[3] : 0, h = arguments.length > 4 && arguments[4] !== void 0 ? arguments[4] : this.retryTimeout, d = arguments.length > 5 ? arguments[5] : void 0;
      if (!i.length)
        return d(null, {});
      if (this.readingCalls >= this.maxParallelReads) {
        this.waitingReads.push({
          lng: i,
          ns: s,
          fcName: o,
          tried: c,
          wait: h,
          callback: d
        });
        return;
      }
      this.readingCalls++;
      var g = function(_, b) {
        if (l.readingCalls--, l.waitingReads.length > 0) {
          var C = l.waitingReads.shift();
          l.read(C.lng, C.ns, C.fcName, C.tried, C.wait, C.callback);
        }
        if (_ && b && c < l.maxRetries) {
          setTimeout(function() {
            l.read.call(l, i, s, o, c + 1, h * 2, d);
          }, h);
          return;
        }
        d(_, b);
      }, p = this.backend[o].bind(this.backend);
      if (p.length === 2) {
        try {
          var y = p(i, s);
          y && typeof y.then == "function" ? y.then(function(v) {
            return g(null, v);
          }).catch(g) : g(null, y);
        } catch (v) {
          g(v);
        }
        return;
      }
      return p(i, s, g);
    }
  }, {
    key: "prepareLoading",
    value: function(i, s) {
      var o = this, l = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {}, c = arguments.length > 3 ? arguments[3] : void 0;
      if (!this.backend)
        return this.logger.warn("No backend was added via i18next.use. Will not load resources."), c && c();
      typeof i == "string" && (i = this.languageUtils.toResolveHierarchy(i)), typeof s == "string" && (s = [s]);
      var h = this.queueLoad(i, s, l, c);
      if (!h.toLoad.length)
        return h.pending.length || c(), null;
      h.toLoad.forEach(function(d) {
        o.loadOne(d);
      });
    }
  }, {
    key: "load",
    value: function(i, s, o) {
      this.prepareLoading(i, s, {}, o);
    }
  }, {
    key: "reload",
    value: function(i, s, o) {
      this.prepareLoading(i, s, {
        reload: !0
      }, o);
    }
  }, {
    key: "loadOne",
    value: function(i) {
      var s = this, o = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : "", l = i.split("|"), c = l[0], h = l[1];
      this.read(c, h, "read", void 0, void 0, function(d, g) {
        d && s.logger.warn("".concat(o, "loading namespace ").concat(h, " for language ").concat(c, " failed"), d), !d && g && s.logger.log("".concat(o, "loaded namespace ").concat(h, " for language ").concat(c), g), s.loaded(i, d, g);
      });
    }
  }, {
    key: "saveMissing",
    value: function(i, s, o, l, c) {
      var h = arguments.length > 5 && arguments[5] !== void 0 ? arguments[5] : {}, d = arguments.length > 6 && arguments[6] !== void 0 ? arguments[6] : function() {
      };
      if (this.services.utils && this.services.utils.hasLoadedNamespace && !this.services.utils.hasLoadedNamespace(s)) {
        this.logger.warn('did not save key "'.concat(o, '" as the namespace "').concat(s, '" was not yet loaded'), "This means something IS WRONG in your setup. You access the t function before i18next.init / i18next.loadNamespace / i18next.changeLanguage was done. Wait for the callback or Promise to resolve before accessing it!!!");
        return;
      }
      if (!(o == null || o === "")) {
        if (this.backend && this.backend.create) {
          var g = im(im({}, h), {}, {
            isUpdate: c
          }), p = this.backend.create.bind(this.backend);
          if (p.length < 6)
            try {
              var y;
              p.length === 5 ? y = p(i, s, o, l, g) : y = p(i, s, o, l), y && typeof y.then == "function" ? y.then(function(v) {
                return d(null, v);
              }).catch(d) : d(null, y);
            } catch (v) {
              d(v);
            }
          else
            p(i, s, o, l, d, g);
        }
        !i || !i[0] || this.store.addResource(i[0], s, o, l);
      }
    }
  }]), r;
}(xi);
function am() {
  return {
    debug: !1,
    initImmediate: !0,
    ns: ["translation"],
    defaultNS: ["translation"],
    fallbackLng: ["dev"],
    fallbackNS: !1,
    supportedLngs: !1,
    nonExplicitSupportedLngs: !1,
    load: "all",
    preload: !1,
    simplifyPluralSuffix: !0,
    keySeparator: ".",
    nsSeparator: ":",
    pluralSeparator: "_",
    contextSeparator: "_",
    partialBundledLanguages: !1,
    saveMissing: !1,
    updateMissing: !1,
    saveMissingTo: "fallback",
    saveMissingPlurals: !0,
    missingKeyHandler: !1,
    missingInterpolationHandler: !1,
    postProcess: !1,
    postProcessPassResolved: !1,
    returnNull: !0,
    returnEmptyString: !0,
    returnObjects: !1,
    joinArrays: !1,
    returnedObjectHandler: !1,
    parseMissingKeyHandler: !1,
    appendNamespaceToMissingKey: !1,
    appendNamespaceToCIMode: !1,
    overloadTranslationOptionHandler: function(e) {
      var r = {};
      if (Kr(e[1]) === "object" && (r = e[1]), typeof e[1] == "string" && (r.defaultValue = e[1]), typeof e[2] == "string" && (r.tDescription = e[2]), Kr(e[2]) === "object" || Kr(e[3]) === "object") {
        var n = e[3] || e[2];
        Object.keys(n).forEach(function(i) {
          r[i] = n[i];
        });
      }
      return r;
    },
    interpolation: {
      escapeValue: !0,
      format: function(e, r, n, i) {
        return e;
      },
      prefix: "{{",
      suffix: "}}",
      formatSeparator: ",",
      unescapePrefix: "-",
      nestingPrefix: "$t(",
      nestingSuffix: ")",
      nestingOptionsSeparator: ",",
      maxReplaces: 1e3,
      skipOnVariables: !0
    }
  };
}
function sm(t) {
  return typeof t.ns == "string" && (t.ns = [t.ns]), typeof t.fallbackLng == "string" && (t.fallbackLng = [t.fallbackLng]), typeof t.fallbackNS == "string" && (t.fallbackNS = [t.fallbackNS]), t.supportedLngs && t.supportedLngs.indexOf("cimode") < 0 && (t.supportedLngs = t.supportedLngs.concat(["cimode"])), t;
}
function om(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function Sn(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? om(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : om(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
function ZO(t) {
  var e = eT();
  return function() {
    var n = An(t), i;
    if (e) {
      var s = An(this).constructor;
      i = Reflect.construct(n, arguments, s);
    } else
      i = n.apply(this, arguments);
    return yo(this, i);
  };
}
function eT() {
  if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham)
    return !1;
  if (typeof Proxy == "function")
    return !0;
  try {
    return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function() {
    })), !0;
  } catch {
    return !1;
  }
}
function Au() {
}
function tT(t) {
  var e = Object.getOwnPropertyNames(Object.getPrototypeOf(t));
  e.forEach(function(r) {
    typeof t[r] == "function" && (t[r] = t[r].bind(t));
  });
}
var _l = function(t) {
  Al(r, t);
  var e = ZO(r);
  function r() {
    var n, i = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, s = arguments.length > 1 ? arguments[1] : void 0;
    if (Xr(this, r), n = e.call(this), kl && xi.call(Ei(n)), n.options = sm(i), n.services = {}, n.logger = xn, n.modules = {
      external: []
    }, tT(Ei(n)), s && !n.isInitialized && !i.isClone) {
      if (!n.options.initImmediate)
        return n.init(i, s), yo(n, Ei(n));
      setTimeout(function() {
        n.init(i, s);
      }, 0);
    }
    return n;
  }
  return Jr(r, [{
    key: "init",
    value: function() {
      var i = this, s = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, o = arguments.length > 1 ? arguments[1] : void 0;
      typeof s == "function" && (o = s, s = {}), !s.defaultNS && s.defaultNS !== !1 && s.ns && (typeof s.ns == "string" ? s.defaultNS = s.ns : s.ns.indexOf("translation") < 0 && (s.defaultNS = s.ns[0]));
      var l = am();
      this.options = Sn(Sn(Sn({}, l), this.options), sm(s)), this.options.compatibilityAPI !== "v1" && (this.options.interpolation = Sn(Sn({}, l.interpolation), this.options.interpolation)), s.keySeparator !== void 0 && (this.options.userDefinedKeySeparator = s.keySeparator), s.nsSeparator !== void 0 && (this.options.userDefinedNsSeparator = s.nsSeparator);
      function c(C) {
        return C ? typeof C == "function" ? new C() : C : null;
      }
      if (!this.options.isClone) {
        this.modules.logger ? xn.init(c(this.modules.logger), this.options) : xn.init(null, this.options);
        var h;
        this.modules.formatter ? h = this.modules.formatter : typeof Intl < "u" && (h = YO);
        var d = new Zp(this.options);
        this.store = new FO(this.options.resources, this.options);
        var g = this.services;
        g.logger = xn, g.resourceStore = this.store, g.languageUtils = d, g.pluralResolver = new zO(d, {
          prepend: this.options.pluralSeparator,
          compatibilityJSON: this.options.compatibilityJSON,
          simplifyPluralSuffix: this.options.simplifyPluralSuffix
        }), h && (!this.options.interpolation.format || this.options.interpolation.format === l.interpolation.format) && (g.formatter = c(h), g.formatter.init(g, this.options), this.options.interpolation.format = g.formatter.format.bind(g.formatter)), g.interpolator = new VO(this.options), g.utils = {
          hasLoadedNamespace: this.hasLoadedNamespace.bind(this)
        }, g.backendConnector = new QO(c(this.modules.backend), g.resourceStore, g, this.options), g.backendConnector.on("*", function(C) {
          for (var w = arguments.length, O = new Array(w > 1 ? w - 1 : 0), k = 1; k < w; k++)
            O[k - 1] = arguments[k];
          i.emit.apply(i, [C].concat(O));
        }), this.modules.languageDetector && (g.languageDetector = c(this.modules.languageDetector), g.languageDetector.init && g.languageDetector.init(g, this.options.detection, this.options)), this.modules.i18nFormat && (g.i18nFormat = c(this.modules.i18nFormat), g.i18nFormat.init && g.i18nFormat.init(this)), this.translator = new Qp(this.services, this.options), this.translator.on("*", function(C) {
          for (var w = arguments.length, O = new Array(w > 1 ? w - 1 : 0), k = 1; k < w; k++)
            O[k - 1] = arguments[k];
          i.emit.apply(i, [C].concat(O));
        }), this.modules.external.forEach(function(C) {
          C.init && C.init(i);
        });
      }
      if (this.format = this.options.interpolation.format, o || (o = Au), this.options.fallbackLng && !this.services.languageDetector && !this.options.lng) {
        var p = this.services.languageUtils.getFallbackCodes(this.options.fallbackLng);
        p.length > 0 && p[0] !== "dev" && (this.options.lng = p[0]);
      }
      !this.services.languageDetector && !this.options.lng && this.logger.warn("init: no languageDetector is used and no lng is defined");
      var y = ["getResource", "hasResourceBundle", "getResourceBundle", "getDataByLanguage"];
      y.forEach(function(C) {
        i[C] = function() {
          var w;
          return (w = i.store)[C].apply(w, arguments);
        };
      });
      var v = ["addResource", "addResources", "addResourceBundle", "removeResourceBundle"];
      v.forEach(function(C) {
        i[C] = function() {
          var w;
          return (w = i.store)[C].apply(w, arguments), i;
        };
      });
      var _ = Ms(), b = function() {
        var w = function(k, P) {
          i.isInitialized && !i.initializedStoreOnce && i.logger.warn("init: i18next is already initialized. You should call init just once!"), i.isInitialized = !0, i.options.isClone || i.logger.log("initialized", i.options), i.emit("initialized", i.options), _.resolve(P), o(k, P);
        };
        if (i.languages && i.options.compatibilityAPI !== "v1" && !i.isInitialized)
          return w(null, i.t.bind(i));
        i.changeLanguage(i.options.lng, w);
      };
      return this.options.resources || !this.options.initImmediate ? b() : setTimeout(b, 0), _;
    }
  }, {
    key: "loadResources",
    value: function(i) {
      var s = this, o = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : Au, l = o, c = typeof i == "string" ? i : this.language;
      if (typeof i == "function" && (l = i), !this.options.resources || this.options.partialBundledLanguages) {
        if (c && c.toLowerCase() === "cimode")
          return l();
        var h = [], d = function(y) {
          if (y) {
            var v = s.services.languageUtils.toResolveHierarchy(y);
            v.forEach(function(_) {
              h.indexOf(_) < 0 && h.push(_);
            });
          }
        };
        if (c)
          d(c);
        else {
          var g = this.services.languageUtils.getFallbackCodes(this.options.fallbackLng);
          g.forEach(function(p) {
            return d(p);
          });
        }
        this.options.preload && this.options.preload.forEach(function(p) {
          return d(p);
        }), this.services.backendConnector.load(h, this.options.ns, function(p) {
          !p && !s.resolvedLanguage && s.language && s.setResolvedLanguage(s.language), l(p);
        });
      } else
        l(null);
    }
  }, {
    key: "reloadResources",
    value: function(i, s, o) {
      var l = Ms();
      return i || (i = this.languages), s || (s = this.options.ns), o || (o = Au), this.services.backendConnector.reload(i, s, function(c) {
        l.resolve(), o(c);
      }), l;
    }
  }, {
    key: "use",
    value: function(i) {
      if (!i)
        throw new Error("You are passing an undefined module! Please check the object you are passing to i18next.use()");
      if (!i.type)
        throw new Error("You are passing a wrong module! Please check the object you are passing to i18next.use()");
      return i.type === "backend" && (this.modules.backend = i), (i.type === "logger" || i.log && i.warn && i.error) && (this.modules.logger = i), i.type === "languageDetector" && (this.modules.languageDetector = i), i.type === "i18nFormat" && (this.modules.i18nFormat = i), i.type === "postProcessor" && Ry.addPostProcessor(i), i.type === "formatter" && (this.modules.formatter = i), i.type === "3rdParty" && this.modules.external.push(i), this;
    }
  }, {
    key: "setResolvedLanguage",
    value: function(i) {
      if (!(!i || !this.languages) && !(["cimode", "dev"].indexOf(i) > -1))
        for (var s = 0; s < this.languages.length; s++) {
          var o = this.languages[s];
          if (!(["cimode", "dev"].indexOf(o) > -1) && this.store.hasLanguageSomeTranslations(o)) {
            this.resolvedLanguage = o;
            break;
          }
        }
    }
  }, {
    key: "changeLanguage",
    value: function(i, s) {
      var o = this;
      this.isLanguageChangingTo = i;
      var l = Ms();
      this.emit("languageChanging", i);
      var c = function(p) {
        o.language = p, o.languages = o.services.languageUtils.toResolveHierarchy(p), o.resolvedLanguage = void 0, o.setResolvedLanguage(p);
      }, h = function(p, y) {
        y ? (c(y), o.translator.changeLanguage(y), o.isLanguageChangingTo = void 0, o.emit("languageChanged", y), o.logger.log("languageChanged", y)) : o.isLanguageChangingTo = void 0, l.resolve(function() {
          return o.t.apply(o, arguments);
        }), s && s(p, function() {
          return o.t.apply(o, arguments);
        });
      }, d = function(p) {
        !i && !p && o.services.languageDetector && (p = []);
        var y = typeof p == "string" ? p : o.services.languageUtils.getBestMatchFromCodes(p);
        y && (o.language || c(y), o.translator.language || o.translator.changeLanguage(y), o.services.languageDetector && o.services.languageDetector.cacheUserLanguage && o.services.languageDetector.cacheUserLanguage(y)), o.loadResources(y, function(v) {
          h(v, y);
        });
      };
      return !i && this.services.languageDetector && !this.services.languageDetector.async ? d(this.services.languageDetector.detect()) : !i && this.services.languageDetector && this.services.languageDetector.async ? this.services.languageDetector.detect.length === 0 ? this.services.languageDetector.detect().then(d) : this.services.languageDetector.detect(d) : d(i), l;
    }
  }, {
    key: "getFixedT",
    value: function(i, s, o) {
      var l = this, c = function h(d, g) {
        var p;
        if (Kr(g) !== "object") {
          for (var y = arguments.length, v = new Array(y > 2 ? y - 2 : 0), _ = 2; _ < y; _++)
            v[_ - 2] = arguments[_];
          p = l.options.overloadTranslationOptionHandler([d, g].concat(v));
        } else
          p = Sn({}, g);
        p.lng = p.lng || h.lng, p.lngs = p.lngs || h.lngs, p.ns = p.ns || h.ns, p.keyPrefix = p.keyPrefix || o || h.keyPrefix;
        var b = l.options.keySeparator || ".", C;
        return p.keyPrefix && Array.isArray(d) ? C = d.map(function(w) {
          return "".concat(p.keyPrefix).concat(b).concat(w);
        }) : C = p.keyPrefix ? "".concat(p.keyPrefix).concat(b).concat(d) : d, l.t(C, p);
      };
      return typeof i == "string" ? c.lng = i : c.lngs = i, c.ns = s, c.keyPrefix = o, c;
    }
  }, {
    key: "t",
    value: function() {
      var i;
      return this.translator && (i = this.translator).translate.apply(i, arguments);
    }
  }, {
    key: "exists",
    value: function() {
      var i;
      return this.translator && (i = this.translator).exists.apply(i, arguments);
    }
  }, {
    key: "setDefaultNamespace",
    value: function(i) {
      this.options.defaultNS = i;
    }
  }, {
    key: "hasLoadedNamespace",
    value: function(i) {
      var s = this, o = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
      if (!this.isInitialized)
        return this.logger.warn("hasLoadedNamespace: i18next was not initialized", this.languages), !1;
      if (!this.languages || !this.languages.length)
        return this.logger.warn("hasLoadedNamespace: i18n.languages were undefined or empty", this.languages), !1;
      var l = this.resolvedLanguage || this.languages[0], c = this.options ? this.options.fallbackLng : !1, h = this.languages[this.languages.length - 1];
      if (l.toLowerCase() === "cimode")
        return !0;
      var d = function(y, v) {
        var _ = s.services.backendConnector.state["".concat(y, "|").concat(v)];
        return _ === -1 || _ === 2;
      };
      if (o.precheck) {
        var g = o.precheck(this, d);
        if (g !== void 0)
          return g;
      }
      return !!(this.hasResourceBundle(l, i) || !this.services.backendConnector.backend || this.options.resources && !this.options.partialBundledLanguages || d(l, i) && (!c || d(h, i)));
    }
  }, {
    key: "loadNamespaces",
    value: function(i, s) {
      var o = this, l = Ms();
      return this.options.ns ? (typeof i == "string" && (i = [i]), i.forEach(function(c) {
        o.options.ns.indexOf(c) < 0 && o.options.ns.push(c);
      }), this.loadResources(function(c) {
        l.resolve(), s && s(c);
      }), l) : (s && s(), Promise.resolve());
    }
  }, {
    key: "loadLanguages",
    value: function(i, s) {
      var o = Ms();
      typeof i == "string" && (i = [i]);
      var l = this.options.preload || [], c = i.filter(function(h) {
        return l.indexOf(h) < 0;
      });
      return c.length ? (this.options.preload = l.concat(c), this.loadResources(function(h) {
        o.resolve(), s && s(h);
      }), o) : (s && s(), Promise.resolve());
    }
  }, {
    key: "dir",
    value: function(i) {
      if (i || (i = this.resolvedLanguage || (this.languages && this.languages.length > 0 ? this.languages[0] : this.language)), !i)
        return "rtl";
      var s = ["ar", "shu", "sqr", "ssh", "xaa", "yhd", "yud", "aao", "abh", "abv", "acm", "acq", "acw", "acx", "acy", "adf", "ads", "aeb", "aec", "afb", "ajp", "apc", "apd", "arb", "arq", "ars", "ary", "arz", "auz", "avl", "ayh", "ayl", "ayn", "ayp", "bbz", "pga", "he", "iw", "ps", "pbt", "pbu", "pst", "prp", "prd", "ug", "ur", "ydd", "yds", "yih", "ji", "yi", "hbo", "men", "xmn", "fa", "jpr", "peo", "pes", "prs", "dv", "sam", "ckb"], o = this.services && this.services.languageUtils || new Zp(am());
      return s.indexOf(o.getLanguagePartFromCode(i)) > -1 || i.toLowerCase().indexOf("-arab") > 1 ? "rtl" : "ltr";
    }
  }, {
    key: "cloneInstance",
    value: function() {
      var i = this, s = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, o = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : Au, l = Sn(Sn(Sn({}, this.options), s), {
        isClone: !0
      }), c = new r(l);
      (s.debug !== void 0 || s.prefix !== void 0) && (c.logger = c.logger.clone(s));
      var h = ["store", "services", "language"];
      return h.forEach(function(d) {
        c[d] = i[d];
      }), c.services = Sn({}, this.services), c.services.utils = {
        hasLoadedNamespace: c.hasLoadedNamespace.bind(c)
      }, c.translator = new Qp(c.services, c.options), c.translator.on("*", function(d) {
        for (var g = arguments.length, p = new Array(g > 1 ? g - 1 : 0), y = 1; y < g; y++)
          p[y - 1] = arguments[y];
        c.emit.apply(c, [d].concat(p));
      }), c.init(l, o), c.translator.options = c.options, c.translator.backendConnector.services.utils = {
        hasLoadedNamespace: c.hasLoadedNamespace.bind(c)
      }, c;
    }
  }, {
    key: "toJSON",
    value: function() {
      return {
        options: this.options,
        store: this.store,
        language: this.language,
        languages: this.languages,
        resolvedLanguage: this.resolvedLanguage
      };
    }
  }]), r;
}(xi);
fn(_l, "createInstance", function() {
  var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {}, e = arguments.length > 1 ? arguments[1] : void 0;
  return new _l(t, e);
});
var De = _l.createInstance();
De.createInstance = _l.createInstance;
De.createInstance;
De.dir;
De.init;
De.loadResources;
De.reloadResources;
De.use;
De.changeLanguage;
De.getFixedT;
De.t;
De.exists;
De.setDefaultNamespace;
De.hasLoadedNamespace;
De.loadNamespaces;
De.loadLanguages;
var Ay = [], rT = Ay.forEach, nT = Ay.slice;
function iT(t) {
  return rT.call(nT.call(arguments, 1), function(e) {
    if (e)
      for (var r in e)
        t[r] === void 0 && (t[r] = e[r]);
  }), t;
}
var um = /^[\u0009\u0020-\u007e\u0080-\u00ff]+$/, aT = function(e, r, n) {
  var i = n || {};
  i.path = i.path || "/";
  var s = encodeURIComponent(r), o = "".concat(e, "=").concat(s);
  if (i.maxAge > 0) {
    var l = i.maxAge - 0;
    if (Number.isNaN(l))
      throw new Error("maxAge should be a Number");
    o += "; Max-Age=".concat(Math.floor(l));
  }
  if (i.domain) {
    if (!um.test(i.domain))
      throw new TypeError("option domain is invalid");
    o += "; Domain=".concat(i.domain);
  }
  if (i.path) {
    if (!um.test(i.path))
      throw new TypeError("option path is invalid");
    o += "; Path=".concat(i.path);
  }
  if (i.expires) {
    if (typeof i.expires.toUTCString != "function")
      throw new TypeError("option expires is invalid");
    o += "; Expires=".concat(i.expires.toUTCString());
  }
  if (i.httpOnly && (o += "; HttpOnly"), i.secure && (o += "; Secure"), i.sameSite) {
    var c = typeof i.sameSite == "string" ? i.sameSite.toLowerCase() : i.sameSite;
    switch (c) {
      case !0:
        o += "; SameSite=Strict";
        break;
      case "lax":
        o += "; SameSite=Lax";
        break;
      case "strict":
        o += "; SameSite=Strict";
        break;
      case "none":
        o += "; SameSite=None";
        break;
      default:
        throw new TypeError("option sameSite is invalid");
    }
  }
  return o;
}, lm = {
  create: function(e, r, n, i) {
    var s = arguments.length > 4 && arguments[4] !== void 0 ? arguments[4] : {
      path: "/",
      sameSite: "strict"
    };
    n && (s.expires = new Date(), s.expires.setTime(s.expires.getTime() + n * 60 * 1e3)), i && (s.domain = i), document.cookie = aT(e, encodeURIComponent(r), s);
  },
  read: function(e) {
    for (var r = "".concat(e, "="), n = document.cookie.split(";"), i = 0; i < n.length; i++) {
      for (var s = n[i]; s.charAt(0) === " "; )
        s = s.substring(1, s.length);
      if (s.indexOf(r) === 0)
        return s.substring(r.length, s.length);
    }
    return null;
  },
  remove: function(e) {
    this.create(e, "", -1);
  }
}, sT = {
  name: "cookie",
  lookup: function(e) {
    var r;
    if (e.lookupCookie && typeof document < "u") {
      var n = lm.read(e.lookupCookie);
      n && (r = n);
    }
    return r;
  },
  cacheUserLanguage: function(e, r) {
    r.lookupCookie && typeof document < "u" && lm.create(r.lookupCookie, e, r.cookieMinutes, r.cookieDomain, r.cookieOptions);
  }
}, oT = {
  name: "querystring",
  lookup: function(e) {
    var r;
    if (typeof window < "u") {
      var n = window.location.search;
      !window.location.search && window.location.hash && window.location.hash.indexOf("?") > -1 && (n = window.location.hash.substring(window.location.hash.indexOf("?")));
      for (var i = n.substring(1), s = i.split("&"), o = 0; o < s.length; o++) {
        var l = s[o].indexOf("=");
        if (l > 0) {
          var c = s[o].substring(0, l);
          c === e.lookupQuerystring && (r = s[o].substring(l + 1));
        }
      }
    }
    return r;
  }
}, Ns = null, cm = function() {
  if (Ns !== null)
    return Ns;
  try {
    Ns = window !== "undefined" && window.localStorage !== null;
    var e = "i18next.translate.boo";
    window.localStorage.setItem(e, "foo"), window.localStorage.removeItem(e);
  } catch {
    Ns = !1;
  }
  return Ns;
}, uT = {
  name: "localStorage",
  lookup: function(e) {
    var r;
    if (e.lookupLocalStorage && cm()) {
      var n = window.localStorage.getItem(e.lookupLocalStorage);
      n && (r = n);
    }
    return r;
  },
  cacheUserLanguage: function(e, r) {
    r.lookupLocalStorage && cm() && window.localStorage.setItem(r.lookupLocalStorage, e);
  }
}, Bs = null, fm = function() {
  if (Bs !== null)
    return Bs;
  try {
    Bs = window !== "undefined" && window.sessionStorage !== null;
    var e = "i18next.translate.boo";
    window.sessionStorage.setItem(e, "foo"), window.sessionStorage.removeItem(e);
  } catch {
    Bs = !1;
  }
  return Bs;
}, lT = {
  name: "sessionStorage",
  lookup: function(e) {
    var r;
    if (e.lookupSessionStorage && fm()) {
      var n = window.sessionStorage.getItem(e.lookupSessionStorage);
      n && (r = n);
    }
    return r;
  },
  cacheUserLanguage: function(e, r) {
    r.lookupSessionStorage && fm() && window.sessionStorage.setItem(r.lookupSessionStorage, e);
  }
}, cT = {
  name: "navigator",
  lookup: function(e) {
    var r = [];
    if (typeof navigator < "u") {
      if (navigator.languages)
        for (var n = 0; n < navigator.languages.length; n++)
          r.push(navigator.languages[n]);
      navigator.userLanguage && r.push(navigator.userLanguage), navigator.language && r.push(navigator.language);
    }
    return r.length > 0 ? r : void 0;
  }
}, fT = {
  name: "htmlTag",
  lookup: function(e) {
    var r, n = e.htmlTag || (typeof document < "u" ? document.documentElement : null);
    return n && typeof n.getAttribute == "function" && (r = n.getAttribute("lang")), r;
  }
}, hT = {
  name: "path",
  lookup: function(e) {
    var r;
    if (typeof window < "u") {
      var n = window.location.pathname.match(/\/([a-zA-Z-]*)/g);
      if (n instanceof Array)
        if (typeof e.lookupFromPathIndex == "number") {
          if (typeof n[e.lookupFromPathIndex] != "string")
            return;
          r = n[e.lookupFromPathIndex].replace("/", "");
        } else
          r = n[0].replace("/", "");
    }
    return r;
  }
}, dT = {
  name: "subdomain",
  lookup: function(e) {
    var r = typeof e.lookupFromSubdomainIndex == "number" ? e.lookupFromSubdomainIndex + 1 : 1, n = typeof window < "u" && window.location && window.location.hostname && window.location.hostname.match(/^(\w{2,5})\.(([a-z0-9-]{1,63}\.[a-z]{2,6})|localhost)/i);
    if (n)
      return n[r];
  }
};
function gT() {
  return {
    order: ["querystring", "cookie", "localStorage", "sessionStorage", "navigator", "htmlTag"],
    lookupQuerystring: "lng",
    lookupCookie: "i18next",
    lookupLocalStorage: "i18nextLng",
    lookupSessionStorage: "i18nextLng",
    // cache user language
    caches: ["localStorage"],
    excludeCacheFor: ["cimode"]
    // cookieMinutes: 10,
    // cookieDomain: 'myDomain'
  };
}
var ky = /* @__PURE__ */ function() {
  function t(e) {
    var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
    Xr(this, t), this.type = "languageDetector", this.detectors = {}, this.init(e, r);
  }
  return Jr(t, [{
    key: "init",
    value: function(r) {
      var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, i = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      this.services = r, this.options = iT(n, this.options || {}, gT()), this.options.lookupFromUrlIndex && (this.options.lookupFromPathIndex = this.options.lookupFromUrlIndex), this.i18nOptions = i, this.addDetector(sT), this.addDetector(oT), this.addDetector(uT), this.addDetector(lT), this.addDetector(cT), this.addDetector(fT), this.addDetector(hT), this.addDetector(dT);
    }
  }, {
    key: "addDetector",
    value: function(r) {
      this.detectors[r.name] = r;
    }
  }, {
    key: "detect",
    value: function(r) {
      var n = this;
      r || (r = this.options.order);
      var i = [];
      return r.forEach(function(s) {
        if (n.detectors[s]) {
          var o = n.detectors[s].lookup(n.options);
          o && typeof o == "string" && (o = [o]), o && (i = i.concat(o));
        }
      }), this.services.languageUtils.getBestMatchFromCodes ? i : i.length > 0 ? i[0] : null;
    }
  }, {
    key: "cacheUserLanguage",
    value: function(r, n) {
      var i = this;
      n || (n = this.options.caches), n && (this.options.excludeCacheFor && this.options.excludeCacheFor.indexOf(r) > -1 || n.forEach(function(s) {
        i.detectors[s] && i.detectors[s].cacheUserLanguage(r, i.options);
      }));
    }
  }]), t;
}();
ky.type = "languageDetector";
function zf(t) {
  return zf = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(e) {
    return typeof e;
  } : function(e) {
    return e && typeof Symbol == "function" && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e;
  }, zf(t);
}
var Iy = [], pT = Iy.forEach, mT = Iy.slice;
function Vf(t) {
  return pT.call(mT.call(arguments, 1), function(e) {
    if (e)
      for (var r in e)
        t[r] === void 0 && (t[r] = e[r]);
  }), t;
}
function Ly() {
  return typeof XMLHttpRequest == "function" || (typeof XMLHttpRequest > "u" ? "undefined" : zf(XMLHttpRequest)) === "object";
}
function vT(t) {
  return !!t && typeof t.then == "function";
}
function yT(t) {
  return vT(t) ? t : Promise.resolve(t);
}
function bT(t) {
  throw new Error('Could not dynamically require "' + t + '". Please configure the dynamicRequireTargets or/and ignoreDynamicRequires option of @rollup/plugin-commonjs appropriately for this require call to work.');
}
var lo = {}, _T = {
  get exports() {
    return lo;
  },
  set exports(t) {
    lo = t;
  }
}, to = {}, ST = {
  get exports() {
    return to;
  },
  set exports(t) {
    to = t;
  }
}, hm;
function wT() {
  return hm || (hm = 1, function(t, e) {
    var r = typeof self < "u" ? self : dr, n = function() {
      function s() {
        this.fetch = !1, this.DOMException = r.DOMException;
      }
      return s.prototype = r, new s();
    }();
    (function(s) {
      (function(o) {
        var l = {
          searchParams: "URLSearchParams" in s,
          iterable: "Symbol" in s && "iterator" in Symbol,
          blob: "FileReader" in s && "Blob" in s && function() {
            try {
              return new Blob(), !0;
            } catch {
              return !1;
            }
          }(),
          formData: "FormData" in s,
          arrayBuffer: "ArrayBuffer" in s
        };
        function c(R) {
          return R && DataView.prototype.isPrototypeOf(R);
        }
        if (l.arrayBuffer)
          var h = [
            "[object Int8Array]",
            "[object Uint8Array]",
            "[object Uint8ClampedArray]",
            "[object Int16Array]",
            "[object Uint16Array]",
            "[object Int32Array]",
            "[object Uint32Array]",
            "[object Float32Array]",
            "[object Float64Array]"
          ], d = ArrayBuffer.isView || function(R) {
            return R && h.indexOf(Object.prototype.toString.call(R)) > -1;
          };
        function g(R) {
          if (typeof R != "string" && (R = String(R)), /[^a-z0-9\-#$%&'*+.^_`|~]/i.test(R))
            throw new TypeError("Invalid character in header field name");
          return R.toLowerCase();
        }
        function p(R) {
          return typeof R != "string" && (R = String(R)), R;
        }
        function y(R) {
          var F = {
            next: function() {
              var H = R.shift();
              return { done: H === void 0, value: H };
            }
          };
          return l.iterable && (F[Symbol.iterator] = function() {
            return F;
          }), F;
        }
        function v(R) {
          this.map = {}, R instanceof v ? R.forEach(function(F, H) {
            this.append(H, F);
          }, this) : Array.isArray(R) ? R.forEach(function(F) {
            this.append(F[0], F[1]);
          }, this) : R && Object.getOwnPropertyNames(R).forEach(function(F) {
            this.append(F, R[F]);
          }, this);
        }
        v.prototype.append = function(R, F) {
          R = g(R), F = p(F);
          var H = this.map[R];
          this.map[R] = H ? H + ", " + F : F;
        }, v.prototype.delete = function(R) {
          delete this.map[g(R)];
        }, v.prototype.get = function(R) {
          return R = g(R), this.has(R) ? this.map[R] : null;
        }, v.prototype.has = function(R) {
          return this.map.hasOwnProperty(g(R));
        }, v.prototype.set = function(R, F) {
          this.map[g(R)] = p(F);
        }, v.prototype.forEach = function(R, F) {
          for (var H in this.map)
            this.map.hasOwnProperty(H) && R.call(F, this.map[H], H, this);
        }, v.prototype.keys = function() {
          var R = [];
          return this.forEach(function(F, H) {
            R.push(H);
          }), y(R);
        }, v.prototype.values = function() {
          var R = [];
          return this.forEach(function(F) {
            R.push(F);
          }), y(R);
        }, v.prototype.entries = function() {
          var R = [];
          return this.forEach(function(F, H) {
            R.push([H, F]);
          }), y(R);
        }, l.iterable && (v.prototype[Symbol.iterator] = v.prototype.entries);
        function _(R) {
          if (R.bodyUsed)
            return Promise.reject(new TypeError("Already read"));
          R.bodyUsed = !0;
        }
        function b(R) {
          return new Promise(function(F, H) {
            R.onload = function() {
              F(R.result);
            }, R.onerror = function() {
              H(R.error);
            };
          });
        }
        function C(R) {
          var F = new FileReader(), H = b(F);
          return F.readAsArrayBuffer(R), H;
        }
        function w(R) {
          var F = new FileReader(), H = b(F);
          return F.readAsText(R), H;
        }
        function O(R) {
          for (var F = new Uint8Array(R), H = new Array(F.length), re = 0; re < F.length; re++)
            H[re] = String.fromCharCode(F[re]);
          return H.join("");
        }
        function k(R) {
          if (R.slice)
            return R.slice(0);
          var F = new Uint8Array(R.byteLength);
          return F.set(new Uint8Array(R)), F.buffer;
        }
        function P() {
          return this.bodyUsed = !1, this._initBody = function(R) {
            this._bodyInit = R, R ? typeof R == "string" ? this._bodyText = R : l.blob && Blob.prototype.isPrototypeOf(R) ? this._bodyBlob = R : l.formData && FormData.prototype.isPrototypeOf(R) ? this._bodyFormData = R : l.searchParams && URLSearchParams.prototype.isPrototypeOf(R) ? this._bodyText = R.toString() : l.arrayBuffer && l.blob && c(R) ? (this._bodyArrayBuffer = k(R.buffer), this._bodyInit = new Blob([this._bodyArrayBuffer])) : l.arrayBuffer && (ArrayBuffer.prototype.isPrototypeOf(R) || d(R)) ? this._bodyArrayBuffer = k(R) : this._bodyText = R = Object.prototype.toString.call(R) : this._bodyText = "", this.headers.get("content-type") || (typeof R == "string" ? this.headers.set("content-type", "text/plain;charset=UTF-8") : this._bodyBlob && this._bodyBlob.type ? this.headers.set("content-type", this._bodyBlob.type) : l.searchParams && URLSearchParams.prototype.isPrototypeOf(R) && this.headers.set("content-type", "application/x-www-form-urlencoded;charset=UTF-8"));
          }, l.blob && (this.blob = function() {
            var R = _(this);
            if (R)
              return R;
            if (this._bodyBlob)
              return Promise.resolve(this._bodyBlob);
            if (this._bodyArrayBuffer)
              return Promise.resolve(new Blob([this._bodyArrayBuffer]));
            if (this._bodyFormData)
              throw new Error("could not read FormData body as blob");
            return Promise.resolve(new Blob([this._bodyText]));
          }, this.arrayBuffer = function() {
            return this._bodyArrayBuffer ? _(this) || Promise.resolve(this._bodyArrayBuffer) : this.blob().then(C);
          }), this.text = function() {
            var R = _(this);
            if (R)
              return R;
            if (this._bodyBlob)
              return w(this._bodyBlob);
            if (this._bodyArrayBuffer)
              return Promise.resolve(O(this._bodyArrayBuffer));
            if (this._bodyFormData)
              throw new Error("could not read FormData body as text");
            return Promise.resolve(this._bodyText);
          }, l.formData && (this.formData = function() {
            return this.text().then(Y);
          }), this.json = function() {
            return this.text().then(JSON.parse);
          }, this;
        }
        var M = ["DELETE", "GET", "HEAD", "OPTIONS", "POST", "PUT"];
        function U(R) {
          var F = R.toUpperCase();
          return M.indexOf(F) > -1 ? F : R;
        }
        function V(R, F) {
          F = F || {};
          var H = F.body;
          if (R instanceof V) {
            if (R.bodyUsed)
              throw new TypeError("Already read");
            this.url = R.url, this.credentials = R.credentials, F.headers || (this.headers = new v(R.headers)), this.method = R.method, this.mode = R.mode, this.signal = R.signal, !H && R._bodyInit != null && (H = R._bodyInit, R.bodyUsed = !0);
          } else
            this.url = String(R);
          if (this.credentials = F.credentials || this.credentials || "same-origin", (F.headers || !this.headers) && (this.headers = new v(F.headers)), this.method = U(F.method || this.method || "GET"), this.mode = F.mode || this.mode || null, this.signal = F.signal || this.signal, this.referrer = null, (this.method === "GET" || this.method === "HEAD") && H)
            throw new TypeError("Body not allowed for GET or HEAD requests");
          this._initBody(H);
        }
        V.prototype.clone = function() {
          return new V(this, { body: this._bodyInit });
        };
        function Y(R) {
          var F = new FormData();
          return R.trim().split("&").forEach(function(H) {
            if (H) {
              var re = H.split("="), ae = re.shift().replace(/\+/g, " "), ne = re.join("=").replace(/\+/g, " ");
              F.append(decodeURIComponent(ae), decodeURIComponent(ne));
            }
          }), F;
        }
        function X(R) {
          var F = new v(), H = R.replace(/\r?\n[\t ]+/g, " ");
          return H.split(/\r?\n/).forEach(function(re) {
            var ae = re.split(":"), ne = ae.shift().trim();
            if (ne) {
              var oe = ae.join(":").trim();
              F.append(ne, oe);
            }
          }), F;
        }
        P.call(V.prototype);
        function Z(R, F) {
          F || (F = {}), this.type = "default", this.status = F.status === void 0 ? 200 : F.status, this.ok = this.status >= 200 && this.status < 300, this.statusText = "statusText" in F ? F.statusText : "OK", this.headers = new v(F.headers), this.url = F.url || "", this._initBody(R);
        }
        P.call(Z.prototype), Z.prototype.clone = function() {
          return new Z(this._bodyInit, {
            status: this.status,
            statusText: this.statusText,
            headers: new v(this.headers),
            url: this.url
          });
        }, Z.error = function() {
          var R = new Z(null, { status: 0, statusText: "" });
          return R.type = "error", R;
        };
        var ce = [301, 302, 303, 307, 308];
        Z.redirect = function(R, F) {
          if (ce.indexOf(F) === -1)
            throw new RangeError("Invalid status code");
          return new Z(null, { status: F, headers: { location: R } });
        }, o.DOMException = s.DOMException;
        try {
          new o.DOMException();
        } catch {
          o.DOMException = function(F, H) {
            this.message = F, this.name = H;
            var re = Error(F);
            this.stack = re.stack;
          }, o.DOMException.prototype = Object.create(Error.prototype), o.DOMException.prototype.constructor = o.DOMException;
        }
        function $(R, F) {
          return new Promise(function(H, re) {
            var ae = new V(R, F);
            if (ae.signal && ae.signal.aborted)
              return re(new o.DOMException("Aborted", "AbortError"));
            var ne = new XMLHttpRequest();
            function oe() {
              ne.abort();
            }
            ne.onload = function() {
              var de = {
                status: ne.status,
                statusText: ne.statusText,
                headers: X(ne.getAllResponseHeaders() || "")
              };
              de.url = "responseURL" in ne ? ne.responseURL : de.headers.get("X-Request-URL");
              var Re = "response" in ne ? ne.response : ne.responseText;
              H(new Z(Re, de));
            }, ne.onerror = function() {
              re(new TypeError("Network request failed"));
            }, ne.ontimeout = function() {
              re(new TypeError("Network request failed"));
            }, ne.onabort = function() {
              re(new o.DOMException("Aborted", "AbortError"));
            }, ne.open(ae.method, ae.url, !0), ae.credentials === "include" ? ne.withCredentials = !0 : ae.credentials === "omit" && (ne.withCredentials = !1), "responseType" in ne && l.blob && (ne.responseType = "blob"), ae.headers.forEach(function(de, Re) {
              ne.setRequestHeader(Re, de);
            }), ae.signal && (ae.signal.addEventListener("abort", oe), ne.onreadystatechange = function() {
              ne.readyState === 4 && ae.signal.removeEventListener("abort", oe);
            }), ne.send(typeof ae._bodyInit > "u" ? null : ae._bodyInit);
          });
        }
        return $.polyfill = !0, s.fetch || (s.fetch = $, s.Headers = v, s.Request = V, s.Response = Z), o.Headers = v, o.Request = V, o.Response = Z, o.fetch = $, Object.defineProperty(o, "__esModule", { value: !0 }), o;
      })({});
    })(n), n.fetch.ponyfill = !0, delete n.fetch.polyfill;
    var i = n;
    e = i.fetch, e.default = i.fetch, e.fetch = i.fetch, e.Headers = i.Headers, e.Request = i.Request, e.Response = i.Response, t.exports = e;
  }(ST, to)), to;
}
(function(t, e) {
  var r;
  if (typeof fetch == "function" && (typeof dr < "u" && dr.fetch ? r = dr.fetch : typeof window < "u" && window.fetch ? r = window.fetch : r = fetch), typeof bT < "u" && (typeof window > "u" || typeof window.document > "u")) {
    var n = r || wT();
    n.default && (n = n.default), e.default = n, t.exports = e.default;
  }
})(_T, lo);
const Dy = lo, dm = /* @__PURE__ */ XC({
  __proto__: null,
  default: Dy
}, [lo]);
function Sl(t) {
  return Sl = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(e) {
    return typeof e;
  } : function(e) {
    return e && typeof Symbol == "function" && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e;
  }, Sl(t);
}
var Zn;
typeof fetch == "function" && (typeof global < "u" && global.fetch ? Zn = global.fetch : typeof window < "u" && window.fetch ? Zn = window.fetch : Zn = fetch);
var co;
Ly() && (typeof global < "u" && global.XMLHttpRequest ? co = global.XMLHttpRequest : typeof window < "u" && window.XMLHttpRequest && (co = window.XMLHttpRequest));
var wl;
typeof ActiveXObject == "function" && (typeof global < "u" && global.ActiveXObject ? wl = global.ActiveXObject : typeof window < "u" && window.ActiveXObject && (wl = window.ActiveXObject));
!Zn && dm && !co && !wl && (Zn = Dy || dm);
typeof Zn != "function" && (Zn = void 0);
var qf = function(e, r) {
  if (r && Sl(r) === "object") {
    var n = "";
    for (var i in r)
      n += "&" + encodeURIComponent(i) + "=" + encodeURIComponent(r[i]);
    if (!n)
      return e;
    e = e + (e.indexOf("?") !== -1 ? "&" : "?") + n.slice(1);
  }
  return e;
}, gm = function(e, r, n) {
  Zn(e, r).then(function(i) {
    if (!i.ok)
      return n(i.statusText || "Error", {
        status: i.status
      });
    i.text().then(function(s) {
      n(null, {
        status: i.status,
        data: s
      });
    }).catch(n);
  }).catch(n);
}, pm = !1, ET = function(e, r, n, i) {
  e.queryStringParams && (r = qf(r, e.queryStringParams));
  var s = Vf({}, typeof e.customHeaders == "function" ? e.customHeaders() : e.customHeaders);
  n && (s["Content-Type"] = "application/json");
  var o = typeof e.requestOptions == "function" ? e.requestOptions(n) : e.requestOptions, l = Vf({
    method: n ? "POST" : "GET",
    body: n ? e.stringify(n) : void 0,
    headers: s
  }, pm ? {} : o);
  try {
    gm(r, l, i);
  } catch (c) {
    if (!o || Object.keys(o).length === 0 || !c.message || c.message.indexOf("not implemented") < 0)
      return i(c);
    try {
      Object.keys(o).forEach(function(h) {
        delete l[h];
      }), gm(r, l, i), pm = !0;
    } catch (h) {
      i(h);
    }
  }
}, CT = function(e, r, n, i) {
  n && Sl(n) === "object" && (n = qf("", n).slice(1)), e.queryStringParams && (r = qf(r, e.queryStringParams));
  try {
    var s;
    co ? s = new co() : s = new wl("MSXML2.XMLHTTP.3.0"), s.open(n ? "POST" : "GET", r, 1), e.crossDomain || s.setRequestHeader("X-Requested-With", "XMLHttpRequest"), s.withCredentials = !!e.withCredentials, n && s.setRequestHeader("Content-Type", "application/x-www-form-urlencoded"), s.overrideMimeType && s.overrideMimeType("application/json");
    var o = e.customHeaders;
    if (o = typeof o == "function" ? o() : o, o)
      for (var l in o)
        s.setRequestHeader(l, o[l]);
    s.onreadystatechange = function() {
      s.readyState > 3 && i(s.status >= 400 ? s.statusText : null, {
        status: s.status,
        data: s.responseText
      });
    }, s.send(n);
  } catch (c) {
    console && console.log(c);
  }
}, xT = function(e, r, n, i) {
  if (typeof n == "function" && (i = n, n = void 0), i = i || function() {
  }, Zn && r.indexOf("file:") !== 0)
    return ET(e, r, n, i);
  if (Ly() || typeof ActiveXObject == "function")
    return CT(e, r, n, i);
  i(new Error("No fetch and no xhr implementation found!"));
};
function fo(t) {
  return fo = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(e) {
    return typeof e;
  } : function(e) {
    return e && typeof Symbol == "function" && e.constructor === Symbol && e !== Symbol.prototype ? "symbol" : typeof e;
  }, fo(t);
}
function OT(t, e) {
  if (!(t instanceof e))
    throw new TypeError("Cannot call a class as a function");
}
function mm(t, e) {
  for (var r = 0; r < e.length; r++) {
    var n = e[r];
    n.enumerable = n.enumerable || !1, n.configurable = !0, "value" in n && (n.writable = !0), Object.defineProperty(t, My(n.key), n);
  }
}
function TT(t, e, r) {
  return e && mm(t.prototype, e), r && mm(t, r), Object.defineProperty(t, "prototype", { writable: !1 }), t;
}
function PT(t, e, r) {
  return e = My(e), e in t ? Object.defineProperty(t, e, { value: r, enumerable: !0, configurable: !0, writable: !0 }) : t[e] = r, t;
}
function My(t) {
  var e = RT(t, "string");
  return fo(e) === "symbol" ? e : String(e);
}
function RT(t, e) {
  if (fo(t) !== "object" || t === null)
    return t;
  var r = t[Symbol.toPrimitive];
  if (r !== void 0) {
    var n = r.call(t, e || "default");
    if (fo(n) !== "object")
      return n;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (e === "string" ? String : Number)(t);
}
var AT = function() {
  return {
    loadPath: "/locales/{{lng}}/{{ns}}.json",
    addPath: "/locales/add/{{lng}}/{{ns}}",
    allowMultiLoading: !1,
    parse: function(r) {
      return JSON.parse(r);
    },
    stringify: JSON.stringify,
    parsePayload: function(r, n, i) {
      return PT({}, n, i || "");
    },
    request: xT,
    reloadInterval: typeof window < "u" ? !1 : 60 * 60 * 1e3,
    customHeaders: {},
    queryStringParams: {},
    crossDomain: !1,
    withCredentials: !1,
    overrideMimeType: !1,
    requestOptions: {
      mode: "cors",
      credentials: "same-origin",
      cache: "default"
    }
  };
}, Ny = function() {
  function t(e) {
    var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, n = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
    OT(this, t), this.services = e, this.options = r, this.allOptions = n, this.type = "backend", this.init(e, r, n);
  }
  return TT(t, [{
    key: "init",
    value: function(r) {
      var n = this, i = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, s = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
      this.services = r, this.options = Vf(i, this.options || {}, AT()), this.allOptions = s, this.services && this.options.reloadInterval && setInterval(function() {
        return n.reload();
      }, this.options.reloadInterval);
    }
  }, {
    key: "readMulti",
    value: function(r, n, i) {
      this._readAny(r, r, n, n, i);
    }
  }, {
    key: "read",
    value: function(r, n, i) {
      this._readAny([r], r, [n], n, i);
    }
  }, {
    key: "_readAny",
    value: function(r, n, i, s, o) {
      var l = this, c = this.options.loadPath;
      typeof this.options.loadPath == "function" && (c = this.options.loadPath(r, i)), c = yT(c), c.then(function(h) {
        if (!h)
          return o(null, {});
        var d = l.services.interpolator.interpolate(h, {
          lng: r.join("+"),
          ns: i.join("+")
        });
        l.loadUrl(d, o, n, s);
      });
    }
  }, {
    key: "loadUrl",
    value: function(r, n, i, s) {
      var o = this;
      this.options.request(this.options, r, void 0, function(l, c) {
        if (c && (c.status >= 500 && c.status < 600 || !c.status))
          return n("failed loading " + r + "; status code: " + c.status, !0);
        if (c && c.status >= 400 && c.status < 500)
          return n("failed loading " + r + "; status code: " + c.status, !1);
        if (!c && l && l.message && l.message.indexOf("Failed to fetch") > -1)
          return n("failed loading " + r + ": " + l.message, !0);
        if (l)
          return n(l, !1);
        var h, d;
        try {
          typeof c.data == "string" ? h = o.options.parse(c.data, i, s) : h = c.data;
        } catch {
          d = "failed parsing " + r + " to json";
        }
        if (d)
          return n(d, !1);
        n(null, h);
      });
    }
  }, {
    key: "create",
    value: function(r, n, i, s, o) {
      var l = this;
      if (this.options.addPath) {
        typeof r == "string" && (r = [r]);
        var c = this.options.parsePayload(n, i, s), h = 0, d = [], g = [];
        r.forEach(function(p) {
          var y = l.options.addPath;
          typeof l.options.addPath == "function" && (y = l.options.addPath(p, n));
          var v = l.services.interpolator.interpolate(y, {
            lng: p,
            ns: n
          });
          l.options.request(l.options, v, c, function(_, b) {
            h += 1, d.push(_), g.push(b), h === r.length && typeof o == "function" && o(d, g);
          });
        });
      }
    }
  }, {
    key: "reload",
    value: function() {
      var r = this, n = this.services, i = n.backendConnector, s = n.languageUtils, o = n.logger, l = i.language;
      if (!(l && l.toLowerCase() === "cimode")) {
        var c = [], h = function(g) {
          var p = s.toResolveHierarchy(g);
          p.forEach(function(y) {
            c.indexOf(y) < 0 && c.push(y);
          });
        };
        h(l), this.allOptions.preload && this.allOptions.preload.forEach(function(d) {
          return h(d);
        }), c.forEach(function(d) {
          r.allOptions.ns.forEach(function(g) {
            i.read(d, g, "read", null, null, function(p, y) {
              p && o.warn("loading namespace ".concat(g, " for language ").concat(d, " failed"), p), !p && y && o.log("loaded namespace ".concat(g, " for language ").concat(d), y), i.loaded("".concat(d, "|").concat(g), p, y);
            });
          });
        });
      }
    }
  }]), t;
}();
Ny.type = "backend";
function kT() {
  if (console && console.warn) {
    for (var t, e = arguments.length, r = new Array(e), n = 0; n < e; n++)
      r[n] = arguments[n];
    typeof r[0] == "string" && (r[0] = "react-i18next:: ".concat(r[0])), (t = console).warn.apply(t, r);
  }
}
var vm = {};
function Yf() {
  for (var t = arguments.length, e = new Array(t), r = 0; r < t; r++)
    e[r] = arguments[r];
  typeof e[0] == "string" && vm[e[0]] || (typeof e[0] == "string" && (vm[e[0]] = new Date()), kT.apply(void 0, e));
}
function ym(t, e, r) {
  t.loadNamespaces(e, function() {
    if (t.isInitialized)
      r();
    else {
      var n = function i() {
        setTimeout(function() {
          t.off("initialized", i);
        }, 0), r();
      };
      t.on("initialized", n);
    }
  });
}
function IT(t, e) {
  var r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {}, n = e.languages[0], i = e.options ? e.options.fallbackLng : !1, s = e.languages[e.languages.length - 1];
  if (n.toLowerCase() === "cimode")
    return !0;
  var o = function(c, h) {
    var d = e.services.backendConnector.state["".concat(c, "|").concat(h)];
    return d === -1 || d === 2;
  };
  return r.bindI18n && r.bindI18n.indexOf("languageChanging") > -1 && e.services.backendConnector.backend && e.isLanguageChangingTo && !o(e.isLanguageChangingTo, t) ? !1 : !!(e.hasResourceBundle(n, t) || !e.services.backendConnector.backend || e.options.resources && !e.options.partialBundledLanguages || o(n, t) && (!i || o(s, t)));
}
function LT(t, e) {
  var r = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : {};
  if (!e.languages || !e.languages.length)
    return Yf("i18n.languages were undefined or empty", e.languages), !0;
  var n = e.options.ignoreJSONStructure !== void 0;
  return n ? e.hasLoadedNamespace(t, {
    precheck: function(s, o) {
      if (r.bindI18n && r.bindI18n.indexOf("languageChanging") > -1 && s.services.backendConnector.backend && s.isLanguageChangingTo && !o(s.isLanguageChangingTo, t))
        return !1;
    }
  }) : IT(t, e, r);
}
var DT = /&(?:amp|#38|lt|#60|gt|#62|apos|#39|quot|#34|nbsp|#160|copy|#169|reg|#174|hellip|#8230|#x2F|#47);/g, MT = {
  "&amp;": "&",
  "&#38;": "&",
  "&lt;": "<",
  "&#60;": "<",
  "&gt;": ">",
  "&#62;": ">",
  "&apos;": "'",
  "&#39;": "'",
  "&quot;": '"',
  "&#34;": '"',
  "&nbsp;": " ",
  "&#160;": " ",
  "&copy;": "©",
  "&#169;": "©",
  "&reg;": "®",
  "&#174;": "®",
  "&hellip;": "…",
  "&#8230;": "…",
  "&#x2F;": "/",
  "&#47;": "/"
}, NT = function(e) {
  return MT[e];
}, BT = function(e) {
  return e.replace(DT, NT);
};
function bm(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function _m(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? bm(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : bm(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
var Kf = {
  bindI18n: "languageChanged",
  bindI18nStore: "",
  transEmptyNodeValue: "",
  transSupportBasicHtmlNodes: !0,
  transWrapTextNodes: "",
  transKeepBasicHtmlNodesFor: ["br", "strong", "i", "p"],
  useSuspense: !0,
  unescape: BT
};
function FT() {
  var t = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : {};
  Kf = _m(_m({}, Kf), t);
}
function GT() {
  return Kf;
}
var By;
function jT(t) {
  By = t;
}
function $T() {
  return By;
}
var WT = {
  type: "3rdParty",
  init: function(e) {
    FT(e.options.react), jT(e);
  }
}, UT = vo(), HT = function() {
  function t() {
    Xr(this, t), this.usedNamespaces = {};
  }
  return Jr(t, [{
    key: "addUsedNamespaces",
    value: function(r) {
      var n = this;
      r.forEach(function(i) {
        n.usedNamespaces[i] || (n.usedNamespaces[i] = !0);
      });
    }
  }, {
    key: "getUsedNamespaces",
    value: function() {
      return Object.keys(this.usedNamespaces);
    }
  }]), t;
}();
function zT(t, e) {
  var r = t == null ? null : typeof Symbol < "u" && t[Symbol.iterator] || t["@@iterator"];
  if (r != null) {
    var n, i, s, o, l = [], c = !0, h = !1;
    try {
      if (s = (r = r.call(t)).next, e === 0) {
        if (Object(r) !== r)
          return;
        c = !1;
      } else
        for (; !(c = (n = s.call(r)).done) && (l.push(n.value), l.length !== e); c = !0)
          ;
    } catch (d) {
      h = !0, i = d;
    } finally {
      try {
        if (!c && r.return != null && (o = r.return(), Object(o) !== o))
          return;
      } finally {
        if (h)
          throw i;
      }
    }
    return l;
  }
}
function VT(t, e) {
  return Cy(t) || zT(t, e) || xy(t, e) || Oy();
}
function Sm(t, e) {
  var r = Object.keys(t);
  if (Object.getOwnPropertySymbols) {
    var n = Object.getOwnPropertySymbols(t);
    e && (n = n.filter(function(i) {
      return Object.getOwnPropertyDescriptor(t, i).enumerable;
    })), r.push.apply(r, n);
  }
  return r;
}
function _f(t) {
  for (var e = 1; e < arguments.length; e++) {
    var r = arguments[e] != null ? arguments[e] : {};
    e % 2 ? Sm(Object(r), !0).forEach(function(n) {
      fn(t, n, r[n]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(t, Object.getOwnPropertyDescriptors(r)) : Sm(Object(r)).forEach(function(n) {
      Object.defineProperty(t, n, Object.getOwnPropertyDescriptor(r, n));
    });
  }
  return t;
}
var qT = function(e, r) {
  var n = so();
  return fl(function() {
    n.current = r ? n.current : e;
  }, [e, r]), n.current;
};
function YT(t) {
  var e = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {}, r = e.i18n, n = cn(UT) || {}, i = n.i18n, s = n.defaultNS, o = r || i || $T();
  if (o && !o.reportNamespaces && (o.reportNamespaces = new HT()), !o) {
    Yf("You will need to pass in an i18next instance by using initReactI18next");
    var l = function(Y, X) {
      return typeof X == "string" ? X : X && Kr(X) === "object" && typeof X.defaultValue == "string" ? X.defaultValue : Array.isArray(Y) ? Y[Y.length - 1] : Y;
    }, c = [l, {}, !1];
    return c.t = l, c.i18n = {}, c.ready = !1, c;
  }
  o.options.react && o.options.react.wait !== void 0 && Yf("It seems you are still using the old wait option, you may migrate to the new useSuspense behaviour.");
  var h = _f(_f(_f({}, GT()), o.options.react), e), d = h.useSuspense, g = h.keyPrefix, p = t || s || o.options && o.options.defaultNS;
  p = typeof p == "string" ? [p] : p || ["translation"], o.reportNamespaces.addUsedNamespaces && o.reportNamespaces.addUsedNamespaces(p);
  var y = (o.isInitialized || o.initializedStoreOnce) && p.every(function(V) {
    return LT(V, o, h);
  });
  function v() {
    return o.getFixedT(null, h.nsMode === "fallback" ? p : p[0], g);
  }
  var _ = YC(v), b = VT(_, 2), C = b[0], w = b[1], O = p.join(), k = qT(O), P = so(!0);
  fl(function() {
    var V = h.bindI18n, Y = h.bindI18nStore;
    P.current = !0, !y && !d && ym(o, p, function() {
      P.current && w(v);
    }), y && k && k !== O && P.current && w(v);
    function X() {
      P.current && w(v);
    }
    return V && o && o.on(V, X), Y && o && o.store.on(Y, X), function() {
      P.current = !1, V && o && V.split(" ").forEach(function(Z) {
        return o.off(Z, X);
      }), Y && o && Y.split(" ").forEach(function(Z) {
        return o.store.off(Z, X);
      });
    };
  }, [o, O]);
  var M = so(!0);
  fl(function() {
    P.current && !M.current && w(v), M.current = !1;
  }, [o, g]);
  var U = [C, o, y];
  if (U.t = C, U.i18n = o, U.ready = y, y || !y && !d)
    return U;
  throw new Promise(function(V) {
    ym(o, p, function() {
      V();
    });
  });
}
De.use(Ny).use(ky).use(WT).init({
  fallbackLng: "en",
  debug: !1,
  backend: {
    loadPath: "/locales/{{lng}}.json"
  },
  interpolation: {
    escapeValue: !1
  },
  returnNull: !1
});
const KT = {
  isConnected: !1,
  isProcessing: !1,
  log: [],
  shouldShowLogViewer: !1,
  shouldDisplayInProgressType: "latents",
  shouldDisplayGuides: !0,
  isGFPGANAvailable: !0,
  isESRGANAvailable: !0,
  socketId: "",
  shouldConfirmOnDelete: !0,
  openAccordions: [0],
  currentStep: 0,
  totalSteps: 0,
  currentIteration: 0,
  totalIterations: 0,
  currentStatus: De.isInitialized ? De.t("common.statusDisconnected") : "Disconnected",
  currentStatusHasSteps: !1,
  model: "",
  model_id: "",
  model_hash: "",
  app_id: "",
  app_version: "",
  model_list: {},
  infill_methods: [],
  hasError: !1,
  wasErrorSeen: !0,
  isCancelable: !0,
  saveIntermediatesInterval: 5,
  enableImageDebugging: !1,
  toastQueue: [],
  searchFolder: null,
  foundModels: null,
  openModel: null,
  cancelOptions: {
    cancelType: "immediate",
    cancelAfter: null
  }
}, Fy = Qi({
  name: "system",
  initialState: KT,
  reducers: {
    setShouldDisplayInProgressType: (t, e) => {
      t.shouldDisplayInProgressType = e.payload;
    },
    setIsProcessing: (t, e) => {
      t.isProcessing = e.payload;
    },
    setCurrentStatus: (t, e) => {
      t.currentStatus = e.payload;
    },
    setSystemStatus: (t, e) => ({ ...t, ...e.payload }),
    errorOccurred: (t) => {
      t.hasError = !0, t.isProcessing = !1, t.isCancelable = !0, t.currentStep = 0, t.totalSteps = 0, t.currentIteration = 0, t.totalIterations = 0, t.currentStatusHasSteps = !1, t.currentStatus = De.t("common.statusError"), t.wasErrorSeen = !1;
    },
    errorSeen: (t) => {
      t.hasError = !1, t.wasErrorSeen = !0, t.currentStatus = t.isConnected ? De.t("common.statusConnected") : De.t("common.statusDisconnected");
    },
    addLogEntry: (t, e) => {
      const { timestamp: r, message: n, level: i } = e.payload, o = {
        timestamp: r,
        message: n,
        level: i || "info"
      };
      t.log.push(o);
    },
    setShouldShowLogViewer: (t, e) => {
      t.shouldShowLogViewer = e.payload;
    },
    setIsConnected: (t, e) => {
      t.isConnected = e.payload, t.isProcessing = !1, t.isCancelable = !0, t.currentStep = 0, t.totalSteps = 0, t.currentIteration = 0, t.totalIterations = 0, t.currentStatusHasSteps = !1, t.hasError = !1;
    },
    setSocketId: (t, e) => {
      t.socketId = e.payload;
    },
    setShouldConfirmOnDelete: (t, e) => {
      t.shouldConfirmOnDelete = e.payload;
    },
    setOpenAccordions: (t, e) => {
      t.openAccordions = e.payload;
    },
    setSystemConfig: (t, e) => ({
      ...t,
      ...e.payload
    }),
    setShouldDisplayGuides: (t, e) => {
      t.shouldDisplayGuides = e.payload;
    },
    processingCanceled: (t) => {
      t.isProcessing = !1, t.isCancelable = !0, t.currentStep = 0, t.totalSteps = 0, t.currentIteration = 0, t.totalIterations = 0, t.currentStatusHasSteps = !1, t.currentStatus = De.t("common.statusProcessingCanceled");
    },
    generationRequested: (t) => {
      t.isProcessing = !0, t.isCancelable = !0, t.currentStep = 0, t.totalSteps = 0, t.currentIteration = 0, t.totalIterations = 0, t.currentStatusHasSteps = !1, t.currentStatus = De.t("common.statusPreparing");
    },
    setModelList: (t, e) => {
      t.model_list = e.payload;
    },
    setIsCancelable: (t, e) => {
      t.isCancelable = e.payload;
    },
    modelChangeRequested: (t) => {
      t.currentStatus = De.t("common.statusLoadingModel"), t.isCancelable = !1, t.isProcessing = !0, t.currentStatusHasSteps = !1;
    },
    modelConvertRequested: (t) => {
      t.currentStatus = De.t("common.statusConvertingModel"), t.isCancelable = !1, t.isProcessing = !0, t.currentStatusHasSteps = !1;
    },
    modelMergingRequested: (t) => {
      t.currentStatus = De.t("common.statusMergingModels"), t.isCancelable = !1, t.isProcessing = !0, t.currentStatusHasSteps = !1;
    },
    setSaveIntermediatesInterval: (t, e) => {
      t.saveIntermediatesInterval = e.payload;
    },
    setEnableImageDebugging: (t, e) => {
      t.enableImageDebugging = e.payload;
    },
    addToast: (t, e) => {
      t.toastQueue.push(e.payload);
    },
    clearToastQueue: (t) => {
      t.toastQueue = [];
    },
    setProcessingIndeterminateTask: (t, e) => {
      t.isProcessing = !0, t.currentStatus = e.payload, t.currentStatusHasSteps = !1;
    },
    setSearchFolder: (t, e) => {
      t.searchFolder = e.payload;
    },
    setFoundModels: (t, e) => {
      t.foundModels = e.payload;
    },
    setOpenModel: (t, e) => {
      t.openModel = e.payload;
    },
    setCancelType: (t, e) => {
      t.cancelOptions.cancelType = e.payload;
    },
    setCancelAfter: (t, e) => {
      t.cancelOptions.cancelAfter = e.payload;
    }
  }
}), {
  setShouldDisplayInProgressType: sB,
  setIsProcessing: En,
  addLogEntry: Ht,
  setShouldShowLogViewer: oB,
  setIsConnected: wm,
  setSocketId: uB,
  setShouldConfirmOnDelete: lB,
  setOpenAccordions: cB,
  setSystemStatus: XT,
  setCurrentStatus: Ba,
  setSystemConfig: JT,
  setShouldDisplayGuides: fB,
  processingCanceled: QT,
  errorOccurred: Em,
  errorSeen: hB,
  setModelList: Fa,
  setIsCancelable: ku,
  modelChangeRequested: ZT,
  modelConvertRequested: eP,
  modelMergingRequested: tP,
  setSaveIntermediatesInterval: dB,
  setEnableImageDebugging: gB,
  generationRequested: rP,
  addToast: Fs,
  clearToastQueue: pB,
  setProcessingIndeterminateTask: mB,
  setSearchFolder: nP,
  setFoundModels: iP,
  setOpenModel: vB,
  setCancelType: yB,
  setCancelAfter: bB
} = Fy.actions, aP = Fy.reducer, Gy = [
  "txt2img",
  "img2img",
  "unifiedCanvas",
  "nodes",
  "postprocess",
  "training"
], sP = {
  activeTab: 0,
  currentTheme: "dark",
  parametersPanelScrollPosition: 0,
  shouldHoldParametersPanelOpen: !1,
  shouldPinParametersPanel: !0,
  shouldShowParametersPanel: !0,
  shouldShowImageDetails: !1,
  shouldUseCanvasBetaLayout: !1,
  shouldShowExistingModelsInSearch: !1,
  shouldUseSliders: !1,
  addNewModelUIOption: null
}, oP = sP, jy = Qi({
  name: "ui",
  initialState: oP,
  reducers: {
    setActiveTab: (t, e) => {
      typeof e.payload == "number" ? t.activeTab = e.payload : t.activeTab = Gy.indexOf(e.payload);
    },
    setCurrentTheme: (t, e) => {
      t.currentTheme = e.payload;
    },
    setParametersPanelScrollPosition: (t, e) => {
      t.parametersPanelScrollPosition = e.payload;
    },
    setShouldPinParametersPanel: (t, e) => {
      t.shouldPinParametersPanel = e.payload;
    },
    setShouldShowParametersPanel: (t, e) => {
      t.shouldShowParametersPanel = e.payload;
    },
    setShouldHoldParametersPanelOpen: (t, e) => {
      t.shouldHoldParametersPanelOpen = e.payload;
    },
    setShouldShowImageDetails: (t, e) => {
      t.shouldShowImageDetails = e.payload;
    },
    setShouldUseCanvasBetaLayout: (t, e) => {
      t.shouldUseCanvasBetaLayout = e.payload;
    },
    setShouldShowExistingModelsInSearch: (t, e) => {
      t.shouldShowExistingModelsInSearch = e.payload;
    },
    setShouldUseSliders: (t, e) => {
      t.shouldUseSliders = e.payload;
    },
    setAddNewModelUIOption: (t, e) => {
      t.addNewModelUIOption = e.payload;
    }
  }
}), {
  setActiveTab: _B,
  setCurrentTheme: SB,
  setParametersPanelScrollPosition: wB,
  setShouldHoldParametersPanelOpen: EB,
  setShouldPinParametersPanel: CB,
  setShouldShowParametersPanel: xB,
  setShouldShowImageDetails: OB,
  setShouldUseCanvasBetaLayout: TB,
  setShouldShowExistingModelsInSearch: PB,
  setShouldUseSliders: RB,
  setAddNewModelUIOption: AB
} = jy.actions, uP = jy.reducer, kn = /* @__PURE__ */ Object.create(null);
kn.open = "0";
kn.close = "1";
kn.ping = "2";
kn.pong = "3";
kn.message = "4";
kn.upgrade = "5";
kn.noop = "6";
const el = /* @__PURE__ */ Object.create(null);
Object.keys(kn).forEach((t) => {
  el[kn[t]] = t;
});
const lP = { type: "error", data: "parser error" }, cP = typeof Blob == "function" || typeof Blob < "u" && Object.prototype.toString.call(Blob) === "[object BlobConstructor]", fP = typeof ArrayBuffer == "function", hP = (t) => typeof ArrayBuffer.isView == "function" ? ArrayBuffer.isView(t) : t && t.buffer instanceof ArrayBuffer, $y = ({ type: t, data: e }, r, n) => cP && e instanceof Blob ? r ? n(e) : Cm(e, n) : fP && (e instanceof ArrayBuffer || hP(e)) ? r ? n(e) : Cm(new Blob([e]), n) : n(kn[t] + (e || "")), Cm = (t, e) => {
  const r = new FileReader();
  return r.onload = function() {
    const n = r.result.split(",")[1];
    e("b" + (n || ""));
  }, r.readAsDataURL(t);
}, xm = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", Ys = typeof Uint8Array > "u" ? [] : new Uint8Array(256);
for (let t = 0; t < xm.length; t++)
  Ys[xm.charCodeAt(t)] = t;
const dP = (t) => {
  let e = t.length * 0.75, r = t.length, n, i = 0, s, o, l, c;
  t[t.length - 1] === "=" && (e--, t[t.length - 2] === "=" && e--);
  const h = new ArrayBuffer(e), d = new Uint8Array(h);
  for (n = 0; n < r; n += 4)
    s = Ys[t.charCodeAt(n)], o = Ys[t.charCodeAt(n + 1)], l = Ys[t.charCodeAt(n + 2)], c = Ys[t.charCodeAt(n + 3)], d[i++] = s << 2 | o >> 4, d[i++] = (o & 15) << 4 | l >> 2, d[i++] = (l & 3) << 6 | c & 63;
  return h;
}, gP = typeof ArrayBuffer == "function", Wy = (t, e) => {
  if (typeof t != "string")
    return {
      type: "message",
      data: Uy(t, e)
    };
  const r = t.charAt(0);
  return r === "b" ? {
    type: "message",
    data: pP(t.substring(1), e)
  } : el[r] ? t.length > 1 ? {
    type: el[r],
    data: t.substring(1)
  } : {
    type: el[r]
  } : lP;
}, pP = (t, e) => {
  if (gP) {
    const r = dP(t);
    return Uy(r, e);
  } else
    return { base64: !0, data: t };
}, Uy = (t, e) => {
  switch (e) {
    case "blob":
      return t instanceof ArrayBuffer ? new Blob([t]) : t;
    case "arraybuffer":
    default:
      return t;
  }
}, Hy = String.fromCharCode(30), mP = (t, e) => {
  const r = t.length, n = new Array(r);
  let i = 0;
  t.forEach((s, o) => {
    $y(s, !1, (l) => {
      n[o] = l, ++i === r && e(n.join(Hy));
    });
  });
}, vP = (t, e) => {
  const r = t.split(Hy), n = [];
  for (let i = 0; i < r.length; i++) {
    const s = Wy(r[i], e);
    if (n.push(s), s.type === "error")
      break;
  }
  return n;
}, zy = 4;
function Lt(t) {
  if (t)
    return yP(t);
}
function yP(t) {
  for (var e in Lt.prototype)
    t[e] = Lt.prototype[e];
  return t;
}
Lt.prototype.on = Lt.prototype.addEventListener = function(t, e) {
  return this._callbacks = this._callbacks || {}, (this._callbacks["$" + t] = this._callbacks["$" + t] || []).push(e), this;
};
Lt.prototype.once = function(t, e) {
  function r() {
    this.off(t, r), e.apply(this, arguments);
  }
  return r.fn = e, this.on(t, r), this;
};
Lt.prototype.off = Lt.prototype.removeListener = Lt.prototype.removeAllListeners = Lt.prototype.removeEventListener = function(t, e) {
  if (this._callbacks = this._callbacks || {}, arguments.length == 0)
    return this._callbacks = {}, this;
  var r = this._callbacks["$" + t];
  if (!r)
    return this;
  if (arguments.length == 1)
    return delete this._callbacks["$" + t], this;
  for (var n, i = 0; i < r.length; i++)
    if (n = r[i], n === e || n.fn === e) {
      r.splice(i, 1);
      break;
    }
  return r.length === 0 && delete this._callbacks["$" + t], this;
};
Lt.prototype.emit = function(t) {
  this._callbacks = this._callbacks || {};
  for (var e = new Array(arguments.length - 1), r = this._callbacks["$" + t], n = 1; n < arguments.length; n++)
    e[n - 1] = arguments[n];
  if (r) {
    r = r.slice(0);
    for (var n = 0, i = r.length; n < i; ++n)
      r[n].apply(this, e);
  }
  return this;
};
Lt.prototype.emitReserved = Lt.prototype.emit;
Lt.prototype.listeners = function(t) {
  return this._callbacks = this._callbacks || {}, this._callbacks["$" + t] || [];
};
Lt.prototype.hasListeners = function(t) {
  return !!this.listeners(t).length;
};
const qr = (() => typeof self < "u" ? self : typeof window < "u" ? window : Function("return this")())();
function Vy(t, ...e) {
  return e.reduce((r, n) => (t.hasOwnProperty(n) && (r[n] = t[n]), r), {});
}
const bP = qr.setTimeout, _P = qr.clearTimeout;
function Il(t, e) {
  e.useNativeTimers ? (t.setTimeoutFn = bP.bind(qr), t.clearTimeoutFn = _P.bind(qr)) : (t.setTimeoutFn = qr.setTimeout.bind(qr), t.clearTimeoutFn = qr.clearTimeout.bind(qr));
}
const SP = 1.33;
function wP(t) {
  return typeof t == "string" ? EP(t) : Math.ceil((t.byteLength || t.size) * SP);
}
function EP(t) {
  let e = 0, r = 0;
  for (let n = 0, i = t.length; n < i; n++)
    e = t.charCodeAt(n), e < 128 ? r += 1 : e < 2048 ? r += 2 : e < 55296 || e >= 57344 ? r += 3 : (n++, r += 4);
  return r;
}
class CP extends Error {
  constructor(e, r, n) {
    super(e), this.description = r, this.context = n, this.type = "TransportError";
  }
}
class qy extends Lt {
  /**
   * Transport abstract constructor.
   *
   * @param {Object} opts - options
   * @protected
   */
  constructor(e) {
    super(), this.writable = !1, Il(this, e), this.opts = e, this.query = e.query, this.socket = e.socket;
  }
  /**
   * Emits an error.
   *
   * @param {String} reason
   * @param description
   * @param context - the error context
   * @return {Transport} for chaining
   * @protected
   */
  onError(e, r, n) {
    return super.emitReserved("error", new CP(e, r, n)), this;
  }
  /**
   * Opens the transport.
   */
  open() {
    return this.readyState = "opening", this.doOpen(), this;
  }
  /**
   * Closes the transport.
   */
  close() {
    return (this.readyState === "opening" || this.readyState === "open") && (this.doClose(), this.onClose()), this;
  }
  /**
   * Sends multiple packets.
   *
   * @param {Array} packets
   */
  send(e) {
    this.readyState === "open" && this.write(e);
  }
  /**
   * Called upon open
   *
   * @protected
   */
  onOpen() {
    this.readyState = "open", this.writable = !0, super.emitReserved("open");
  }
  /**
   * Called with data.
   *
   * @param {String} data
   * @protected
   */
  onData(e) {
    const r = Wy(e, this.socket.binaryType);
    this.onPacket(r);
  }
  /**
   * Called with a decoded packet.
   *
   * @protected
   */
  onPacket(e) {
    super.emitReserved("packet", e);
  }
  /**
   * Called upon close.
   *
   * @protected
   */
  onClose(e) {
    this.readyState = "closed", super.emitReserved("close", e);
  }
  /**
   * Pauses the transport, in order not to lose packets during an upgrade.
   *
   * @param onPause
   */
  pause(e) {
  }
}
const Yy = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_".split(""), Xf = 64, xP = {};
let Om = 0, Iu = 0, Tm;
function Pm(t) {
  let e = "";
  do
    e = Yy[t % Xf] + e, t = Math.floor(t / Xf);
  while (t > 0);
  return e;
}
function Ky() {
  const t = Pm(+new Date());
  return t !== Tm ? (Om = 0, Tm = t) : t + "." + Pm(Om++);
}
for (; Iu < Xf; Iu++)
  xP[Yy[Iu]] = Iu;
function Xy(t) {
  let e = "";
  for (let r in t)
    t.hasOwnProperty(r) && (e.length && (e += "&"), e += encodeURIComponent(r) + "=" + encodeURIComponent(t[r]));
  return e;
}
function OP(t) {
  let e = {}, r = t.split("&");
  for (let n = 0, i = r.length; n < i; n++) {
    let s = r[n].split("=");
    e[decodeURIComponent(s[0])] = decodeURIComponent(s[1]);
  }
  return e;
}
let Jy = !1;
try {
  Jy = typeof XMLHttpRequest < "u" && "withCredentials" in new XMLHttpRequest();
} catch {
}
const TP = Jy;
function Qy(t) {
  const e = t.xdomain;
  try {
    if (typeof XMLHttpRequest < "u" && (!e || TP))
      return new XMLHttpRequest();
  } catch {
  }
  if (!e)
    try {
      return new qr[["Active"].concat("Object").join("X")]("Microsoft.XMLHTTP");
    } catch {
    }
}
function PP() {
}
const RP = function() {
  return new Qy({
    xdomain: !1
  }).responseType != null;
}();
class AP extends qy {
  /**
   * XHR Polling constructor.
   *
   * @param {Object} opts
   * @package
   */
  constructor(e) {
    if (super(e), this.polling = !1, typeof location < "u") {
      const n = location.protocol === "https:";
      let i = location.port;
      i || (i = n ? "443" : "80"), this.xd = typeof location < "u" && e.hostname !== location.hostname || i !== e.port, this.xs = e.secure !== n;
    }
    const r = e && e.forceBase64;
    this.supportsBinary = RP && !r;
  }
  get name() {
    return "polling";
  }
  /**
   * Opens the socket (triggers polling). We write a PING message to determine
   * when the transport is open.
   *
   * @protected
   */
  doOpen() {
    this.poll();
  }
  /**
   * Pauses polling.
   *
   * @param {Function} onPause - callback upon buffers are flushed and transport is paused
   * @package
   */
  pause(e) {
    this.readyState = "pausing";
    const r = () => {
      this.readyState = "paused", e();
    };
    if (this.polling || !this.writable) {
      let n = 0;
      this.polling && (n++, this.once("pollComplete", function() {
        --n || r();
      })), this.writable || (n++, this.once("drain", function() {
        --n || r();
      }));
    } else
      r();
  }
  /**
   * Starts polling cycle.
   *
   * @private
   */
  poll() {
    this.polling = !0, this.doPoll(), this.emitReserved("poll");
  }
  /**
   * Overloads onData to detect payloads.
   *
   * @protected
   */
  onData(e) {
    const r = (n) => {
      if (this.readyState === "opening" && n.type === "open" && this.onOpen(), n.type === "close")
        return this.onClose({ description: "transport closed by the server" }), !1;
      this.onPacket(n);
    };
    vP(e, this.socket.binaryType).forEach(r), this.readyState !== "closed" && (this.polling = !1, this.emitReserved("pollComplete"), this.readyState === "open" && this.poll());
  }
  /**
   * For polling, send a close packet.
   *
   * @protected
   */
  doClose() {
    const e = () => {
      this.write([{ type: "close" }]);
    };
    this.readyState === "open" ? e() : this.once("open", e);
  }
  /**
   * Writes a packets payload.
   *
   * @param {Array} packets - data packets
   * @protected
   */
  write(e) {
    this.writable = !1, mP(e, (r) => {
      this.doWrite(r, () => {
        this.writable = !0, this.emitReserved("drain");
      });
    });
  }
  /**
   * Generates uri for connection.
   *
   * @private
   */
  uri() {
    let e = this.query || {};
    const r = this.opts.secure ? "https" : "http";
    let n = "";
    this.opts.timestampRequests !== !1 && (e[this.opts.timestampParam] = Ky()), !this.supportsBinary && !e.sid && (e.b64 = 1), this.opts.port && (r === "https" && Number(this.opts.port) !== 443 || r === "http" && Number(this.opts.port) !== 80) && (n = ":" + this.opts.port);
    const i = Xy(e), s = this.opts.hostname.indexOf(":") !== -1;
    return r + "://" + (s ? "[" + this.opts.hostname + "]" : this.opts.hostname) + n + this.opts.path + (i.length ? "?" + i : "");
  }
  /**
   * Creates a request.
   *
   * @param {String} method
   * @private
   */
  request(e = {}) {
    return Object.assign(e, { xd: this.xd, xs: this.xs }, this.opts), new Pn(this.uri(), e);
  }
  /**
   * Sends data.
   *
   * @param {String} data to send.
   * @param {Function} called upon flush.
   * @private
   */
  doWrite(e, r) {
    const n = this.request({
      method: "POST",
      data: e
    });
    n.on("success", r), n.on("error", (i, s) => {
      this.onError("xhr post error", i, s);
    });
  }
  /**
   * Starts a poll cycle.
   *
   * @private
   */
  doPoll() {
    const e = this.request();
    e.on("data", this.onData.bind(this)), e.on("error", (r, n) => {
      this.onError("xhr poll error", r, n);
    }), this.pollXhr = e;
  }
}
class Pn extends Lt {
  /**
   * Request constructor
   *
   * @param {Object} options
   * @package
   */
  constructor(e, r) {
    super(), Il(this, r), this.opts = r, this.method = r.method || "GET", this.uri = e, this.async = r.async !== !1, this.data = r.data !== void 0 ? r.data : null, this.create();
  }
  /**
   * Creates the XHR object and sends the request.
   *
   * @private
   */
  create() {
    const e = Vy(this.opts, "agent", "pfx", "key", "passphrase", "cert", "ca", "ciphers", "rejectUnauthorized", "autoUnref");
    e.xdomain = !!this.opts.xd, e.xscheme = !!this.opts.xs;
    const r = this.xhr = new Qy(e);
    try {
      r.open(this.method, this.uri, this.async);
      try {
        if (this.opts.extraHeaders) {
          r.setDisableHeaderCheck && r.setDisableHeaderCheck(!0);
          for (let n in this.opts.extraHeaders)
            this.opts.extraHeaders.hasOwnProperty(n) && r.setRequestHeader(n, this.opts.extraHeaders[n]);
        }
      } catch {
      }
      if (this.method === "POST")
        try {
          r.setRequestHeader("Content-type", "text/plain;charset=UTF-8");
        } catch {
        }
      try {
        r.setRequestHeader("Accept", "*/*");
      } catch {
      }
      "withCredentials" in r && (r.withCredentials = this.opts.withCredentials), this.opts.requestTimeout && (r.timeout = this.opts.requestTimeout), r.onreadystatechange = () => {
        r.readyState === 4 && (r.status === 200 || r.status === 1223 ? this.onLoad() : this.setTimeoutFn(() => {
          this.onError(typeof r.status == "number" ? r.status : 0);
        }, 0));
      }, r.send(this.data);
    } catch (n) {
      this.setTimeoutFn(() => {
        this.onError(n);
      }, 0);
      return;
    }
    typeof document < "u" && (this.index = Pn.requestsCount++, Pn.requests[this.index] = this);
  }
  /**
   * Called upon error.
   *
   * @private
   */
  onError(e) {
    this.emitReserved("error", e, this.xhr), this.cleanup(!0);
  }
  /**
   * Cleans up house.
   *
   * @private
   */
  cleanup(e) {
    if (!(typeof this.xhr > "u" || this.xhr === null)) {
      if (this.xhr.onreadystatechange = PP, e)
        try {
          this.xhr.abort();
        } catch {
        }
      typeof document < "u" && delete Pn.requests[this.index], this.xhr = null;
    }
  }
  /**
   * Called upon load.
   *
   * @private
   */
  onLoad() {
    const e = this.xhr.responseText;
    e !== null && (this.emitReserved("data", e), this.emitReserved("success"), this.cleanup());
  }
  /**
   * Aborts the request.
   *
   * @package
   */
  abort() {
    this.cleanup();
  }
}
Pn.requestsCount = 0;
Pn.requests = {};
if (typeof document < "u") {
  if (typeof attachEvent == "function")
    attachEvent("onunload", Rm);
  else if (typeof addEventListener == "function") {
    const t = "onpagehide" in qr ? "pagehide" : "unload";
    addEventListener(t, Rm, !1);
  }
}
function Rm() {
  for (let t in Pn.requests)
    Pn.requests.hasOwnProperty(t) && Pn.requests[t].abort();
}
const Zy = (() => typeof Promise == "function" && typeof Promise.resolve == "function" ? (e) => Promise.resolve().then(e) : (e, r) => r(e, 0))(), Lu = qr.WebSocket || qr.MozWebSocket, Am = !0, kP = "arraybuffer", km = typeof navigator < "u" && typeof navigator.product == "string" && navigator.product.toLowerCase() === "reactnative";
class IP extends qy {
  /**
   * WebSocket transport constructor.
   *
   * @param {Object} opts - connection options
   * @protected
   */
  constructor(e) {
    super(e), this.supportsBinary = !e.forceBase64;
  }
  get name() {
    return "websocket";
  }
  doOpen() {
    if (!this.check())
      return;
    const e = this.uri(), r = this.opts.protocols, n = km ? {} : Vy(this.opts, "agent", "perMessageDeflate", "pfx", "key", "passphrase", "cert", "ca", "ciphers", "rejectUnauthorized", "localAddress", "protocolVersion", "origin", "maxPayload", "family", "checkServerIdentity");
    this.opts.extraHeaders && (n.headers = this.opts.extraHeaders);
    try {
      this.ws = Am && !km ? r ? new Lu(e, r) : new Lu(e) : new Lu(e, r, n);
    } catch (i) {
      return this.emitReserved("error", i);
    }
    this.ws.binaryType = this.socket.binaryType || kP, this.addEventListeners();
  }
  /**
   * Adds event listeners to the socket
   *
   * @private
   */
  addEventListeners() {
    this.ws.onopen = () => {
      this.opts.autoUnref && this.ws._socket.unref(), this.onOpen();
    }, this.ws.onclose = (e) => this.onClose({
      description: "websocket connection closed",
      context: e
    }), this.ws.onmessage = (e) => this.onData(e.data), this.ws.onerror = (e) => this.onError("websocket error", e);
  }
  write(e) {
    this.writable = !1;
    for (let r = 0; r < e.length; r++) {
      const n = e[r], i = r === e.length - 1;
      $y(n, this.supportsBinary, (s) => {
        const o = {};
        try {
          Am && this.ws.send(s);
        } catch {
        }
        i && Zy(() => {
          this.writable = !0, this.emitReserved("drain");
        }, this.setTimeoutFn);
      });
    }
  }
  doClose() {
    typeof this.ws < "u" && (this.ws.close(), this.ws = null);
  }
  /**
   * Generates uri for connection.
   *
   * @private
   */
  uri() {
    let e = this.query || {};
    const r = this.opts.secure ? "wss" : "ws";
    let n = "";
    this.opts.port && (r === "wss" && Number(this.opts.port) !== 443 || r === "ws" && Number(this.opts.port) !== 80) && (n = ":" + this.opts.port), this.opts.timestampRequests && (e[this.opts.timestampParam] = Ky()), this.supportsBinary || (e.b64 = 1);
    const i = Xy(e), s = this.opts.hostname.indexOf(":") !== -1;
    return r + "://" + (s ? "[" + this.opts.hostname + "]" : this.opts.hostname) + n + this.opts.path + (i.length ? "?" + i : "");
  }
  /**
   * Feature detection for WebSocket.
   *
   * @return {Boolean} whether this transport is available.
   * @private
   */
  check() {
    return !!Lu;
  }
}
const LP = {
  websocket: IP,
  polling: AP
}, DP = /^(?:(?![^:@\/?#]+:[^:@\/]*@)(http|https|ws|wss):\/\/)?((?:(([^:@\/?#]*)(?::([^:@\/?#]*))?)?@)?((?:[a-f0-9]{0,4}:){2,7}[a-f0-9]{0,4}|[^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/, MP = [
  "source",
  "protocol",
  "authority",
  "userInfo",
  "user",
  "password",
  "host",
  "port",
  "relative",
  "path",
  "directory",
  "file",
  "query",
  "anchor"
];
function Jf(t) {
  const e = t, r = t.indexOf("["), n = t.indexOf("]");
  r != -1 && n != -1 && (t = t.substring(0, r) + t.substring(r, n).replace(/:/g, ";") + t.substring(n, t.length));
  let i = DP.exec(t || ""), s = {}, o = 14;
  for (; o--; )
    s[MP[o]] = i[o] || "";
  return r != -1 && n != -1 && (s.source = e, s.host = s.host.substring(1, s.host.length - 1).replace(/;/g, ":"), s.authority = s.authority.replace("[", "").replace("]", "").replace(/;/g, ":"), s.ipv6uri = !0), s.pathNames = NP(s, s.path), s.queryKey = BP(s, s.query), s;
}
function NP(t, e) {
  const r = /\/{2,9}/g, n = e.replace(r, "/").split("/");
  return (e.slice(0, 1) == "/" || e.length === 0) && n.splice(0, 1), e.slice(-1) == "/" && n.splice(n.length - 1, 1), n;
}
function BP(t, e) {
  const r = {};
  return e.replace(/(?:^|&)([^&=]*)=?([^&]*)/g, function(n, i, s) {
    i && (r[i] = s);
  }), r;
}
let e0 = class $a extends Lt {
  /**
   * Socket constructor.
   *
   * @param {String|Object} uri - uri or options
   * @param {Object} opts - options
   */
  constructor(e, r = {}) {
    super(), this.writeBuffer = [], e && typeof e == "object" && (r = e, e = null), e ? (e = Jf(e), r.hostname = e.host, r.secure = e.protocol === "https" || e.protocol === "wss", r.port = e.port, e.query && (r.query = e.query)) : r.host && (r.hostname = Jf(r.host).host), Il(this, r), this.secure = r.secure != null ? r.secure : typeof location < "u" && location.protocol === "https:", r.hostname && !r.port && (r.port = this.secure ? "443" : "80"), this.hostname = r.hostname || (typeof location < "u" ? location.hostname : "localhost"), this.port = r.port || (typeof location < "u" && location.port ? location.port : this.secure ? "443" : "80"), this.transports = r.transports || ["polling", "websocket"], this.writeBuffer = [], this.prevBufferLen = 0, this.opts = Object.assign({
      path: "/engine.io",
      agent: !1,
      withCredentials: !1,
      upgrade: !0,
      timestampParam: "t",
      rememberUpgrade: !1,
      addTrailingSlash: !0,
      rejectUnauthorized: !0,
      perMessageDeflate: {
        threshold: 1024
      },
      transportOptions: {},
      closeOnBeforeunload: !0
    }, r), this.opts.path = this.opts.path.replace(/\/$/, "") + (this.opts.addTrailingSlash ? "/" : ""), typeof this.opts.query == "string" && (this.opts.query = OP(this.opts.query)), this.id = null, this.upgrades = null, this.pingInterval = null, this.pingTimeout = null, this.pingTimeoutTimer = null, typeof addEventListener == "function" && (this.opts.closeOnBeforeunload && (this.beforeunloadEventListener = () => {
      this.transport && (this.transport.removeAllListeners(), this.transport.close());
    }, addEventListener("beforeunload", this.beforeunloadEventListener, !1)), this.hostname !== "localhost" && (this.offlineEventListener = () => {
      this.onClose("transport close", {
        description: "network connection lost"
      });
    }, addEventListener("offline", this.offlineEventListener, !1))), this.open();
  }
  /**
   * Creates transport of the given type.
   *
   * @param {String} name - transport name
   * @return {Transport}
   * @private
   */
  createTransport(e) {
    const r = Object.assign({}, this.opts.query);
    r.EIO = zy, r.transport = e, this.id && (r.sid = this.id);
    const n = Object.assign({}, this.opts.transportOptions[e], this.opts, {
      query: r,
      socket: this,
      hostname: this.hostname,
      secure: this.secure,
      port: this.port
    });
    return new LP[e](n);
  }
  /**
   * Initializes transport to use and starts probe.
   *
   * @private
   */
  open() {
    let e;
    if (this.opts.rememberUpgrade && $a.priorWebsocketSuccess && this.transports.indexOf("websocket") !== -1)
      e = "websocket";
    else if (this.transports.length === 0) {
      this.setTimeoutFn(() => {
        this.emitReserved("error", "No transports available");
      }, 0);
      return;
    } else
      e = this.transports[0];
    this.readyState = "opening";
    try {
      e = this.createTransport(e);
    } catch {
      this.transports.shift(), this.open();
      return;
    }
    e.open(), this.setTransport(e);
  }
  /**
   * Sets the current transport. Disables the existing one (if any).
   *
   * @private
   */
  setTransport(e) {
    this.transport && this.transport.removeAllListeners(), this.transport = e, e.on("drain", this.onDrain.bind(this)).on("packet", this.onPacket.bind(this)).on("error", this.onError.bind(this)).on("close", (r) => this.onClose("transport close", r));
  }
  /**
   * Probes a transport.
   *
   * @param {String} name - transport name
   * @private
   */
  probe(e) {
    let r = this.createTransport(e), n = !1;
    $a.priorWebsocketSuccess = !1;
    const i = () => {
      n || (r.send([{ type: "ping", data: "probe" }]), r.once("packet", (g) => {
        if (!n)
          if (g.type === "pong" && g.data === "probe") {
            if (this.upgrading = !0, this.emitReserved("upgrading", r), !r)
              return;
            $a.priorWebsocketSuccess = r.name === "websocket", this.transport.pause(() => {
              n || this.readyState !== "closed" && (d(), this.setTransport(r), r.send([{ type: "upgrade" }]), this.emitReserved("upgrade", r), r = null, this.upgrading = !1, this.flush());
            });
          } else {
            const p = new Error("probe error");
            p.transport = r.name, this.emitReserved("upgradeError", p);
          }
      }));
    };
    function s() {
      n || (n = !0, d(), r.close(), r = null);
    }
    const o = (g) => {
      const p = new Error("probe error: " + g);
      p.transport = r.name, s(), this.emitReserved("upgradeError", p);
    };
    function l() {
      o("transport closed");
    }
    function c() {
      o("socket closed");
    }
    function h(g) {
      r && g.name !== r.name && s();
    }
    const d = () => {
      r.removeListener("open", i), r.removeListener("error", o), r.removeListener("close", l), this.off("close", c), this.off("upgrading", h);
    };
    r.once("open", i), r.once("error", o), r.once("close", l), this.once("close", c), this.once("upgrading", h), r.open();
  }
  /**
   * Called when connection is deemed open.
   *
   * @private
   */
  onOpen() {
    if (this.readyState = "open", $a.priorWebsocketSuccess = this.transport.name === "websocket", this.emitReserved("open"), this.flush(), this.readyState === "open" && this.opts.upgrade) {
      let e = 0;
      const r = this.upgrades.length;
      for (; e < r; e++)
        this.probe(this.upgrades[e]);
    }
  }
  /**
   * Handles a packet.
   *
   * @private
   */
  onPacket(e) {
    if (this.readyState === "opening" || this.readyState === "open" || this.readyState === "closing")
      switch (this.emitReserved("packet", e), this.emitReserved("heartbeat"), e.type) {
        case "open":
          this.onHandshake(JSON.parse(e.data));
          break;
        case "ping":
          this.resetPingTimeout(), this.sendPacket("pong"), this.emitReserved("ping"), this.emitReserved("pong");
          break;
        case "error":
          const r = new Error("server error");
          r.code = e.data, this.onError(r);
          break;
        case "message":
          this.emitReserved("data", e.data), this.emitReserved("message", e.data);
          break;
      }
  }
  /**
   * Called upon handshake completion.
   *
   * @param {Object} data - handshake obj
   * @private
   */
  onHandshake(e) {
    this.emitReserved("handshake", e), this.id = e.sid, this.transport.query.sid = e.sid, this.upgrades = this.filterUpgrades(e.upgrades), this.pingInterval = e.pingInterval, this.pingTimeout = e.pingTimeout, this.maxPayload = e.maxPayload, this.onOpen(), this.readyState !== "closed" && this.resetPingTimeout();
  }
  /**
   * Sets and resets ping timeout timer based on server pings.
   *
   * @private
   */
  resetPingTimeout() {
    this.clearTimeoutFn(this.pingTimeoutTimer), this.pingTimeoutTimer = this.setTimeoutFn(() => {
      this.onClose("ping timeout");
    }, this.pingInterval + this.pingTimeout), this.opts.autoUnref && this.pingTimeoutTimer.unref();
  }
  /**
   * Called on `drain` event
   *
   * @private
   */
  onDrain() {
    this.writeBuffer.splice(0, this.prevBufferLen), this.prevBufferLen = 0, this.writeBuffer.length === 0 ? this.emitReserved("drain") : this.flush();
  }
  /**
   * Flush write buffers.
   *
   * @private
   */
  flush() {
    if (this.readyState !== "closed" && this.transport.writable && !this.upgrading && this.writeBuffer.length) {
      const e = this.getWritablePackets();
      this.transport.send(e), this.prevBufferLen = e.length, this.emitReserved("flush");
    }
  }
  /**
   * Ensure the encoded size of the writeBuffer is below the maxPayload value sent by the server (only for HTTP
   * long-polling)
   *
   * @private
   */
  getWritablePackets() {
    if (!(this.maxPayload && this.transport.name === "polling" && this.writeBuffer.length > 1))
      return this.writeBuffer;
    let r = 1;
    for (let n = 0; n < this.writeBuffer.length; n++) {
      const i = this.writeBuffer[n].data;
      if (i && (r += wP(i)), n > 0 && r > this.maxPayload)
        return this.writeBuffer.slice(0, n);
      r += 2;
    }
    return this.writeBuffer;
  }
  /**
   * Sends a message.
   *
   * @param {String} msg - message.
   * @param {Object} options.
   * @param {Function} callback function.
   * @return {Socket} for chaining.
   */
  write(e, r, n) {
    return this.sendPacket("message", e, r, n), this;
  }
  send(e, r, n) {
    return this.sendPacket("message", e, r, n), this;
  }
  /**
   * Sends a packet.
   *
   * @param {String} type: packet type.
   * @param {String} data.
   * @param {Object} options.
   * @param {Function} fn - callback function.
   * @private
   */
  sendPacket(e, r, n, i) {
    if (typeof r == "function" && (i = r, r = void 0), typeof n == "function" && (i = n, n = null), this.readyState === "closing" || this.readyState === "closed")
      return;
    n = n || {}, n.compress = n.compress !== !1;
    const s = {
      type: e,
      data: r,
      options: n
    };
    this.emitReserved("packetCreate", s), this.writeBuffer.push(s), i && this.once("flush", i), this.flush();
  }
  /**
   * Closes the connection.
   */
  close() {
    const e = () => {
      this.onClose("forced close"), this.transport.close();
    }, r = () => {
      this.off("upgrade", r), this.off("upgradeError", r), e();
    }, n = () => {
      this.once("upgrade", r), this.once("upgradeError", r);
    };
    return (this.readyState === "opening" || this.readyState === "open") && (this.readyState = "closing", this.writeBuffer.length ? this.once("drain", () => {
      this.upgrading ? n() : e();
    }) : this.upgrading ? n() : e()), this;
  }
  /**
   * Called upon transport error
   *
   * @private
   */
  onError(e) {
    $a.priorWebsocketSuccess = !1, this.emitReserved("error", e), this.onClose("transport error", e);
  }
  /**
   * Called upon transport close.
   *
   * @private
   */
  onClose(e, r) {
    (this.readyState === "opening" || this.readyState === "open" || this.readyState === "closing") && (this.clearTimeoutFn(this.pingTimeoutTimer), this.transport.removeAllListeners("close"), this.transport.close(), this.transport.removeAllListeners(), typeof removeEventListener == "function" && (removeEventListener("beforeunload", this.beforeunloadEventListener, !1), removeEventListener("offline", this.offlineEventListener, !1)), this.readyState = "closed", this.id = null, this.emitReserved("close", e, r), this.writeBuffer = [], this.prevBufferLen = 0);
  }
  /**
   * Filters upgrades, returning only those matching client transports.
   *
   * @param {Array} upgrades - server upgrades
   * @private
   */
  filterUpgrades(e) {
    const r = [];
    let n = 0;
    const i = e.length;
    for (; n < i; n++)
      ~this.transports.indexOf(e[n]) && r.push(e[n]);
    return r;
  }
};
e0.protocol = zy;
function FP(t, e = "", r) {
  let n = t;
  r = r || typeof location < "u" && location, t == null && (t = r.protocol + "//" + r.host), typeof t == "string" && (t.charAt(0) === "/" && (t.charAt(1) === "/" ? t = r.protocol + t : t = r.host + t), /^(https?|wss?):\/\//.test(t) || (typeof r < "u" ? t = r.protocol + "//" + t : t = "https://" + t), n = Jf(t)), n.port || (/^(http|ws)$/.test(n.protocol) ? n.port = "80" : /^(http|ws)s$/.test(n.protocol) && (n.port = "443")), n.path = n.path || "/";
  const s = n.host.indexOf(":") !== -1 ? "[" + n.host + "]" : n.host;
  return n.id = n.protocol + "://" + s + ":" + n.port + e, n.href = n.protocol + "://" + s + (r && r.port === n.port ? "" : ":" + n.port), n;
}
const GP = typeof ArrayBuffer == "function", jP = (t) => typeof ArrayBuffer.isView == "function" ? ArrayBuffer.isView(t) : t.buffer instanceof ArrayBuffer, t0 = Object.prototype.toString, $P = typeof Blob == "function" || typeof Blob < "u" && t0.call(Blob) === "[object BlobConstructor]", WP = typeof File == "function" || typeof File < "u" && t0.call(File) === "[object FileConstructor]";
function Ih(t) {
  return GP && (t instanceof ArrayBuffer || jP(t)) || $P && t instanceof Blob || WP && t instanceof File;
}
function tl(t, e) {
  if (!t || typeof t != "object")
    return !1;
  if (Array.isArray(t)) {
    for (let r = 0, n = t.length; r < n; r++)
      if (tl(t[r]))
        return !0;
    return !1;
  }
  if (Ih(t))
    return !0;
  if (t.toJSON && typeof t.toJSON == "function" && arguments.length === 1)
    return tl(t.toJSON(), !0);
  for (const r in t)
    if (Object.prototype.hasOwnProperty.call(t, r) && tl(t[r]))
      return !0;
  return !1;
}
function UP(t) {
  const e = [], r = t.data, n = t;
  return n.data = Qf(r, e), n.attachments = e.length, { packet: n, buffers: e };
}
function Qf(t, e) {
  if (!t)
    return t;
  if (Ih(t)) {
    const r = { _placeholder: !0, num: e.length };
    return e.push(t), r;
  } else if (Array.isArray(t)) {
    const r = new Array(t.length);
    for (let n = 0; n < t.length; n++)
      r[n] = Qf(t[n], e);
    return r;
  } else if (typeof t == "object" && !(t instanceof Date)) {
    const r = {};
    for (const n in t)
      Object.prototype.hasOwnProperty.call(t, n) && (r[n] = Qf(t[n], e));
    return r;
  }
  return t;
}
function HP(t, e) {
  return t.data = Zf(t.data, e), delete t.attachments, t;
}
function Zf(t, e) {
  if (!t)
    return t;
  if (t && t._placeholder === !0) {
    if (typeof t.num == "number" && t.num >= 0 && t.num < e.length)
      return e[t.num];
    throw new Error("illegal attachments");
  } else if (Array.isArray(t))
    for (let r = 0; r < t.length; r++)
      t[r] = Zf(t[r], e);
  else if (typeof t == "object")
    for (const r in t)
      Object.prototype.hasOwnProperty.call(t, r) && (t[r] = Zf(t[r], e));
  return t;
}
const zP = 5;
var We;
(function(t) {
  t[t.CONNECT = 0] = "CONNECT", t[t.DISCONNECT = 1] = "DISCONNECT", t[t.EVENT = 2] = "EVENT", t[t.ACK = 3] = "ACK", t[t.CONNECT_ERROR = 4] = "CONNECT_ERROR", t[t.BINARY_EVENT = 5] = "BINARY_EVENT", t[t.BINARY_ACK = 6] = "BINARY_ACK";
})(We || (We = {}));
class VP {
  /**
   * Encoder constructor
   *
   * @param {function} replacer - custom replacer to pass down to JSON.parse
   */
  constructor(e) {
    this.replacer = e;
  }
  /**
   * Encode a packet as a single string if non-binary, or as a
   * buffer sequence, depending on packet type.
   *
   * @param {Object} obj - packet object
   */
  encode(e) {
    return (e.type === We.EVENT || e.type === We.ACK) && tl(e) ? this.encodeAsBinary({
      type: e.type === We.EVENT ? We.BINARY_EVENT : We.BINARY_ACK,
      nsp: e.nsp,
      data: e.data,
      id: e.id
    }) : [this.encodeAsString(e)];
  }
  /**
   * Encode packet as string.
   */
  encodeAsString(e) {
    let r = "" + e.type;
    return (e.type === We.BINARY_EVENT || e.type === We.BINARY_ACK) && (r += e.attachments + "-"), e.nsp && e.nsp !== "/" && (r += e.nsp + ","), e.id != null && (r += e.id), e.data != null && (r += JSON.stringify(e.data, this.replacer)), r;
  }
  /**
   * Encode packet as 'buffer sequence' by removing blobs, and
   * deconstructing packet into object with placeholders and
   * a list of buffers.
   */
  encodeAsBinary(e) {
    const r = UP(e), n = this.encodeAsString(r.packet), i = r.buffers;
    return i.unshift(n), i;
  }
}
class Lh extends Lt {
  /**
   * Decoder constructor
   *
   * @param {function} reviver - custom reviver to pass down to JSON.stringify
   */
  constructor(e) {
    super(), this.reviver = e;
  }
  /**
   * Decodes an encoded packet string into packet JSON.
   *
   * @param {String} obj - encoded packet
   */
  add(e) {
    let r;
    if (typeof e == "string") {
      if (this.reconstructor)
        throw new Error("got plaintext data when reconstructing a packet");
      r = this.decodeString(e);
      const n = r.type === We.BINARY_EVENT;
      n || r.type === We.BINARY_ACK ? (r.type = n ? We.EVENT : We.ACK, this.reconstructor = new qP(r), r.attachments === 0 && super.emitReserved("decoded", r)) : super.emitReserved("decoded", r);
    } else if (Ih(e) || e.base64)
      if (this.reconstructor)
        r = this.reconstructor.takeBinaryData(e), r && (this.reconstructor = null, super.emitReserved("decoded", r));
      else
        throw new Error("got binary data when not reconstructing a packet");
    else
      throw new Error("Unknown type: " + e);
  }
  /**
   * Decode a packet String (JSON data)
   *
   * @param {String} str
   * @return {Object} packet
   */
  decodeString(e) {
    let r = 0;
    const n = {
      type: Number(e.charAt(0))
    };
    if (We[n.type] === void 0)
      throw new Error("unknown packet type " + n.type);
    if (n.type === We.BINARY_EVENT || n.type === We.BINARY_ACK) {
      const s = r + 1;
      for (; e.charAt(++r) !== "-" && r != e.length; )
        ;
      const o = e.substring(s, r);
      if (o != Number(o) || e.charAt(r) !== "-")
        throw new Error("Illegal attachments");
      n.attachments = Number(o);
    }
    if (e.charAt(r + 1) === "/") {
      const s = r + 1;
      for (; ++r && !(e.charAt(r) === "," || r === e.length); )
        ;
      n.nsp = e.substring(s, r);
    } else
      n.nsp = "/";
    const i = e.charAt(r + 1);
    if (i !== "" && Number(i) == i) {
      const s = r + 1;
      for (; ++r; ) {
        const o = e.charAt(r);
        if (o == null || Number(o) != o) {
          --r;
          break;
        }
        if (r === e.length)
          break;
      }
      n.id = Number(e.substring(s, r + 1));
    }
    if (e.charAt(++r)) {
      const s = this.tryParse(e.substr(r));
      if (Lh.isPayloadValid(n.type, s))
        n.data = s;
      else
        throw new Error("invalid payload");
    }
    return n;
  }
  tryParse(e) {
    try {
      return JSON.parse(e, this.reviver);
    } catch {
      return !1;
    }
  }
  static isPayloadValid(e, r) {
    switch (e) {
      case We.CONNECT:
        return typeof r == "object";
      case We.DISCONNECT:
        return r === void 0;
      case We.CONNECT_ERROR:
        return typeof r == "string" || typeof r == "object";
      case We.EVENT:
      case We.BINARY_EVENT:
        return Array.isArray(r) && r.length > 0;
      case We.ACK:
      case We.BINARY_ACK:
        return Array.isArray(r);
    }
  }
  /**
   * Deallocates a parser's resources
   */
  destroy() {
    this.reconstructor && (this.reconstructor.finishedReconstruction(), this.reconstructor = null);
  }
}
class qP {
  constructor(e) {
    this.packet = e, this.buffers = [], this.reconPack = e;
  }
  /**
   * Method to be called when binary data received from connection
   * after a BINARY_EVENT packet.
   *
   * @param {Buffer | ArrayBuffer} binData - the raw binary data received
   * @return {null | Object} returns null if more binary data is expected or
   *   a reconstructed packet object if all buffers have been received.
   */
  takeBinaryData(e) {
    if (this.buffers.push(e), this.buffers.length === this.reconPack.attachments) {
      const r = HP(this.reconPack, this.buffers);
      return this.finishedReconstruction(), r;
    }
    return null;
  }
  /**
   * Cleans up binary packet reconstruction variables.
   */
  finishedReconstruction() {
    this.reconPack = null, this.buffers = [];
  }
}
const YP = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  Decoder: Lh,
  Encoder: VP,
  get PacketType() {
    return We;
  },
  protocol: zP
}, Symbol.toStringTag, { value: "Module" }));
function sn(t, e, r) {
  return t.on(e, r), function() {
    t.off(e, r);
  };
}
const KP = Object.freeze({
  connect: 1,
  connect_error: 1,
  disconnect: 1,
  disconnecting: 1,
  // EventEmitter reserved events: https://nodejs.org/api/events.html#events_event_newlistener
  newListener: 1,
  removeListener: 1
});
class r0 extends Lt {
  /**
   * `Socket` constructor.
   */
  constructor(e, r, n) {
    super(), this.connected = !1, this.recovered = !1, this.receiveBuffer = [], this.sendBuffer = [], this._queue = [], this._queueSeq = 0, this.ids = 0, this.acks = {}, this.flags = {}, this.io = e, this.nsp = r, n && n.auth && (this.auth = n.auth), this._opts = Object.assign({}, n), this.io._autoConnect && this.open();
  }
  /**
   * Whether the socket is currently disconnected
   *
   * @example
   * const socket = io();
   *
   * socket.on("connect", () => {
   *   console.log(socket.disconnected); // false
   * });
   *
   * socket.on("disconnect", () => {
   *   console.log(socket.disconnected); // true
   * });
   */
  get disconnected() {
    return !this.connected;
  }
  /**
   * Subscribe to open, close and packet events
   *
   * @private
   */
  subEvents() {
    if (this.subs)
      return;
    const e = this.io;
    this.subs = [
      sn(e, "open", this.onopen.bind(this)),
      sn(e, "packet", this.onpacket.bind(this)),
      sn(e, "error", this.onerror.bind(this)),
      sn(e, "close", this.onclose.bind(this))
    ];
  }
  /**
   * Whether the Socket will try to reconnect when its Manager connects or reconnects.
   *
   * @example
   * const socket = io();
   *
   * console.log(socket.active); // true
   *
   * socket.on("disconnect", (reason) => {
   *   if (reason === "io server disconnect") {
   *     // the disconnection was initiated by the server, you need to manually reconnect
   *     console.log(socket.active); // false
   *   }
   *   // else the socket will automatically try to reconnect
   *   console.log(socket.active); // true
   * });
   */
  get active() {
    return !!this.subs;
  }
  /**
   * "Opens" the socket.
   *
   * @example
   * const socket = io({
   *   autoConnect: false
   * });
   *
   * socket.connect();
   */
  connect() {
    return this.connected ? this : (this.subEvents(), this.io._reconnecting || this.io.open(), this.io._readyState === "open" && this.onopen(), this);
  }
  /**
   * Alias for {@link connect()}.
   */
  open() {
    return this.connect();
  }
  /**
   * Sends a `message` event.
   *
   * This method mimics the WebSocket.send() method.
   *
   * @see https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/send
   *
   * @example
   * socket.send("hello");
   *
   * // this is equivalent to
   * socket.emit("message", "hello");
   *
   * @return self
   */
  send(...e) {
    return e.unshift("message"), this.emit.apply(this, e), this;
  }
  /**
   * Override `emit`.
   * If the event is in `events`, it's emitted normally.
   *
   * @example
   * socket.emit("hello", "world");
   *
   * // all serializable datastructures are supported (no need to call JSON.stringify)
   * socket.emit("hello", 1, "2", { 3: ["4"], 5: Uint8Array.from([6]) });
   *
   * // with an acknowledgement from the server
   * socket.emit("hello", "world", (val) => {
   *   // ...
   * });
   *
   * @return self
   */
  emit(e, ...r) {
    if (KP.hasOwnProperty(e))
      throw new Error('"' + e.toString() + '" is a reserved event name');
    if (r.unshift(e), this._opts.retries && !this.flags.fromQueue && !this.flags.volatile)
      return this._addToQueue(r), this;
    const n = {
      type: We.EVENT,
      data: r
    };
    if (n.options = {}, n.options.compress = this.flags.compress !== !1, typeof r[r.length - 1] == "function") {
      const o = this.ids++, l = r.pop();
      this._registerAckCallback(o, l), n.id = o;
    }
    const i = this.io.engine && this.io.engine.transport && this.io.engine.transport.writable;
    return this.flags.volatile && (!i || !this.connected) || (this.connected ? (this.notifyOutgoingListeners(n), this.packet(n)) : this.sendBuffer.push(n)), this.flags = {}, this;
  }
  /**
   * @private
   */
  _registerAckCallback(e, r) {
    var n;
    const i = (n = this.flags.timeout) !== null && n !== void 0 ? n : this._opts.ackTimeout;
    if (i === void 0) {
      this.acks[e] = r;
      return;
    }
    const s = this.io.setTimeoutFn(() => {
      delete this.acks[e];
      for (let o = 0; o < this.sendBuffer.length; o++)
        this.sendBuffer[o].id === e && this.sendBuffer.splice(o, 1);
      r.call(this, new Error("operation has timed out"));
    }, i);
    this.acks[e] = (...o) => {
      this.io.clearTimeoutFn(s), r.apply(this, [null, ...o]);
    };
  }
  /**
   * Emits an event and waits for an acknowledgement
   *
   * @example
   * // without timeout
   * const response = await socket.emitWithAck("hello", "world");
   *
   * // with a specific timeout
   * try {
   *   const response = await socket.timeout(1000).emitWithAck("hello", "world");
   * } catch (err) {
   *   // the server did not acknowledge the event in the given delay
   * }
   *
   * @return a Promise that will be fulfilled when the server acknowledges the event
   */
  emitWithAck(e, ...r) {
    const n = this.flags.timeout !== void 0 || this._opts.ackTimeout !== void 0;
    return new Promise((i, s) => {
      r.push((o, l) => n ? o ? s(o) : i(l) : i(o)), this.emit(e, ...r);
    });
  }
  /**
   * Add the packet to the queue.
   * @param args
   * @private
   */
  _addToQueue(e) {
    let r;
    typeof e[e.length - 1] == "function" && (r = e.pop());
    const n = {
      id: this._queueSeq++,
      tryCount: 0,
      pending: !1,
      args: e,
      flags: Object.assign({ fromQueue: !0 }, this.flags)
    };
    e.push((i, ...s) => n !== this._queue[0] ? void 0 : (i !== null ? n.tryCount > this._opts.retries && (this._queue.shift(), r && r(i)) : (this._queue.shift(), r && r(null, ...s)), n.pending = !1, this._drainQueue())), this._queue.push(n), this._drainQueue();
  }
  /**
   * Send the first packet of the queue, and wait for an acknowledgement from the server.
   * @param force - whether to resend a packet that has not been acknowledged yet
   *
   * @private
   */
  _drainQueue(e = !1) {
    if (!this.connected || this._queue.length === 0)
      return;
    const r = this._queue[0];
    r.pending && !e || (r.pending = !0, r.tryCount++, this.flags = r.flags, this.emit.apply(this, r.args));
  }
  /**
   * Sends a packet.
   *
   * @param packet
   * @private
   */
  packet(e) {
    e.nsp = this.nsp, this.io._packet(e);
  }
  /**
   * Called upon engine `open`.
   *
   * @private
   */
  onopen() {
    typeof this.auth == "function" ? this.auth((e) => {
      this._sendConnectPacket(e);
    }) : this._sendConnectPacket(this.auth);
  }
  /**
   * Sends a CONNECT packet to initiate the Socket.IO session.
   *
   * @param data
   * @private
   */
  _sendConnectPacket(e) {
    this.packet({
      type: We.CONNECT,
      data: this._pid ? Object.assign({ pid: this._pid, offset: this._lastOffset }, e) : e
    });
  }
  /**
   * Called upon engine or manager `error`.
   *
   * @param err
   * @private
   */
  onerror(e) {
    this.connected || this.emitReserved("connect_error", e);
  }
  /**
   * Called upon engine `close`.
   *
   * @param reason
   * @param description
   * @private
   */
  onclose(e, r) {
    this.connected = !1, delete this.id, this.emitReserved("disconnect", e, r);
  }
  /**
   * Called with socket packet.
   *
   * @param packet
   * @private
   */
  onpacket(e) {
    if (e.nsp === this.nsp)
      switch (e.type) {
        case We.CONNECT:
          e.data && e.data.sid ? this.onconnect(e.data.sid, e.data.pid) : this.emitReserved("connect_error", new Error("It seems you are trying to reach a Socket.IO server in v2.x with a v3.x client, but they are not compatible (more information here: https://socket.io/docs/v3/migrating-from-2-x-to-3-0/)"));
          break;
        case We.EVENT:
        case We.BINARY_EVENT:
          this.onevent(e);
          break;
        case We.ACK:
        case We.BINARY_ACK:
          this.onack(e);
          break;
        case We.DISCONNECT:
          this.ondisconnect();
          break;
        case We.CONNECT_ERROR:
          this.destroy();
          const n = new Error(e.data.message);
          n.data = e.data.data, this.emitReserved("connect_error", n);
          break;
      }
  }
  /**
   * Called upon a server event.
   *
   * @param packet
   * @private
   */
  onevent(e) {
    const r = e.data || [];
    e.id != null && r.push(this.ack(e.id)), this.connected ? this.emitEvent(r) : this.receiveBuffer.push(Object.freeze(r));
  }
  emitEvent(e) {
    if (this._anyListeners && this._anyListeners.length) {
      const r = this._anyListeners.slice();
      for (const n of r)
        n.apply(this, e);
    }
    super.emit.apply(this, e), this._pid && e.length && typeof e[e.length - 1] == "string" && (this._lastOffset = e[e.length - 1]);
  }
  /**
   * Produces an ack callback to emit with an event.
   *
   * @private
   */
  ack(e) {
    const r = this;
    let n = !1;
    return function(...i) {
      n || (n = !0, r.packet({
        type: We.ACK,
        id: e,
        data: i
      }));
    };
  }
  /**
   * Called upon a server acknowlegement.
   *
   * @param packet
   * @private
   */
  onack(e) {
    const r = this.acks[e.id];
    typeof r == "function" && (r.apply(this, e.data), delete this.acks[e.id]);
  }
  /**
   * Called upon server connect.
   *
   * @private
   */
  onconnect(e, r) {
    this.id = e, this.recovered = r && this._pid === r, this._pid = r, this.connected = !0, this.emitBuffered(), this.emitReserved("connect"), this._drainQueue(!0);
  }
  /**
   * Emit buffered events (received and emitted).
   *
   * @private
   */
  emitBuffered() {
    this.receiveBuffer.forEach((e) => this.emitEvent(e)), this.receiveBuffer = [], this.sendBuffer.forEach((e) => {
      this.notifyOutgoingListeners(e), this.packet(e);
    }), this.sendBuffer = [];
  }
  /**
   * Called upon server disconnect.
   *
   * @private
   */
  ondisconnect() {
    this.destroy(), this.onclose("io server disconnect");
  }
  /**
   * Called upon forced client/server side disconnections,
   * this method ensures the manager stops tracking us and
   * that reconnections don't get triggered for this.
   *
   * @private
   */
  destroy() {
    this.subs && (this.subs.forEach((e) => e()), this.subs = void 0), this.io._destroy(this);
  }
  /**
   * Disconnects the socket manually. In that case, the socket will not try to reconnect.
   *
   * If this is the last active Socket instance of the {@link Manager}, the low-level connection will be closed.
   *
   * @example
   * const socket = io();
   *
   * socket.on("disconnect", (reason) => {
   *   // console.log(reason); prints "io client disconnect"
   * });
   *
   * socket.disconnect();
   *
   * @return self
   */
  disconnect() {
    return this.connected && this.packet({ type: We.DISCONNECT }), this.destroy(), this.connected && this.onclose("io client disconnect"), this;
  }
  /**
   * Alias for {@link disconnect()}.
   *
   * @return self
   */
  close() {
    return this.disconnect();
  }
  /**
   * Sets the compress flag.
   *
   * @example
   * socket.compress(false).emit("hello");
   *
   * @param compress - if `true`, compresses the sending data
   * @return self
   */
  compress(e) {
    return this.flags.compress = e, this;
  }
  /**
   * Sets a modifier for a subsequent event emission that the event message will be dropped when this socket is not
   * ready to send messages.
   *
   * @example
   * socket.volatile.emit("hello"); // the server may or may not receive it
   *
   * @returns self
   */
  get volatile() {
    return this.flags.volatile = !0, this;
  }
  /**
   * Sets a modifier for a subsequent event emission that the callback will be called with an error when the
   * given number of milliseconds have elapsed without an acknowledgement from the server:
   *
   * @example
   * socket.timeout(5000).emit("my-event", (err) => {
   *   if (err) {
   *     // the server did not acknowledge the event in the given delay
   *   }
   * });
   *
   * @returns self
   */
  timeout(e) {
    return this.flags.timeout = e, this;
  }
  /**
   * Adds a listener that will be fired when any event is emitted. The event name is passed as the first argument to the
   * callback.
   *
   * @example
   * socket.onAny((event, ...args) => {
   *   console.log(`got ${event}`);
   * });
   *
   * @param listener
   */
  onAny(e) {
    return this._anyListeners = this._anyListeners || [], this._anyListeners.push(e), this;
  }
  /**
   * Adds a listener that will be fired when any event is emitted. The event name is passed as the first argument to the
   * callback. The listener is added to the beginning of the listeners array.
   *
   * @example
   * socket.prependAny((event, ...args) => {
   *   console.log(`got event ${event}`);
   * });
   *
   * @param listener
   */
  prependAny(e) {
    return this._anyListeners = this._anyListeners || [], this._anyListeners.unshift(e), this;
  }
  /**
   * Removes the listener that will be fired when any event is emitted.
   *
   * @example
   * const catchAllListener = (event, ...args) => {
   *   console.log(`got event ${event}`);
   * }
   *
   * socket.onAny(catchAllListener);
   *
   * // remove a specific listener
   * socket.offAny(catchAllListener);
   *
   * // or remove all listeners
   * socket.offAny();
   *
   * @param listener
   */
  offAny(e) {
    if (!this._anyListeners)
      return this;
    if (e) {
      const r = this._anyListeners;
      for (let n = 0; n < r.length; n++)
        if (e === r[n])
          return r.splice(n, 1), this;
    } else
      this._anyListeners = [];
    return this;
  }
  /**
   * Returns an array of listeners that are listening for any event that is specified. This array can be manipulated,
   * e.g. to remove listeners.
   */
  listenersAny() {
    return this._anyListeners || [];
  }
  /**
   * Adds a listener that will be fired when any event is emitted. The event name is passed as the first argument to the
   * callback.
   *
   * Note: acknowledgements sent to the server are not included.
   *
   * @example
   * socket.onAnyOutgoing((event, ...args) => {
   *   console.log(`sent event ${event}`);
   * });
   *
   * @param listener
   */
  onAnyOutgoing(e) {
    return this._anyOutgoingListeners = this._anyOutgoingListeners || [], this._anyOutgoingListeners.push(e), this;
  }
  /**
   * Adds a listener that will be fired when any event is emitted. The event name is passed as the first argument to the
   * callback. The listener is added to the beginning of the listeners array.
   *
   * Note: acknowledgements sent to the server are not included.
   *
   * @example
   * socket.prependAnyOutgoing((event, ...args) => {
   *   console.log(`sent event ${event}`);
   * });
   *
   * @param listener
   */
  prependAnyOutgoing(e) {
    return this._anyOutgoingListeners = this._anyOutgoingListeners || [], this._anyOutgoingListeners.unshift(e), this;
  }
  /**
   * Removes the listener that will be fired when any event is emitted.
   *
   * @example
   * const catchAllListener = (event, ...args) => {
   *   console.log(`sent event ${event}`);
   * }
   *
   * socket.onAnyOutgoing(catchAllListener);
   *
   * // remove a specific listener
   * socket.offAnyOutgoing(catchAllListener);
   *
   * // or remove all listeners
   * socket.offAnyOutgoing();
   *
   * @param [listener] - the catch-all listener (optional)
   */
  offAnyOutgoing(e) {
    if (!this._anyOutgoingListeners)
      return this;
    if (e) {
      const r = this._anyOutgoingListeners;
      for (let n = 0; n < r.length; n++)
        if (e === r[n])
          return r.splice(n, 1), this;
    } else
      this._anyOutgoingListeners = [];
    return this;
  }
  /**
   * Returns an array of listeners that are listening for any event that is specified. This array can be manipulated,
   * e.g. to remove listeners.
   */
  listenersAnyOutgoing() {
    return this._anyOutgoingListeners || [];
  }
  /**
   * Notify the listeners for each packet sent
   *
   * @param packet
   *
   * @private
   */
  notifyOutgoingListeners(e) {
    if (this._anyOutgoingListeners && this._anyOutgoingListeners.length) {
      const r = this._anyOutgoingListeners.slice();
      for (const n of r)
        n.apply(this, e.data);
    }
  }
}
function Za(t) {
  t = t || {}, this.ms = t.min || 100, this.max = t.max || 1e4, this.factor = t.factor || 2, this.jitter = t.jitter > 0 && t.jitter <= 1 ? t.jitter : 0, this.attempts = 0;
}
Za.prototype.duration = function() {
  var t = this.ms * Math.pow(this.factor, this.attempts++);
  if (this.jitter) {
    var e = Math.random(), r = Math.floor(e * this.jitter * t);
    t = Math.floor(e * 10) & 1 ? t + r : t - r;
  }
  return Math.min(t, this.max) | 0;
};
Za.prototype.reset = function() {
  this.attempts = 0;
};
Za.prototype.setMin = function(t) {
  this.ms = t;
};
Za.prototype.setMax = function(t) {
  this.max = t;
};
Za.prototype.setJitter = function(t) {
  this.jitter = t;
};
class eh extends Lt {
  constructor(e, r) {
    var n;
    super(), this.nsps = {}, this.subs = [], e && typeof e == "object" && (r = e, e = void 0), r = r || {}, r.path = r.path || "/socket.io", this.opts = r, Il(this, r), this.reconnection(r.reconnection !== !1), this.reconnectionAttempts(r.reconnectionAttempts || 1 / 0), this.reconnectionDelay(r.reconnectionDelay || 1e3), this.reconnectionDelayMax(r.reconnectionDelayMax || 5e3), this.randomizationFactor((n = r.randomizationFactor) !== null && n !== void 0 ? n : 0.5), this.backoff = new Za({
      min: this.reconnectionDelay(),
      max: this.reconnectionDelayMax(),
      jitter: this.randomizationFactor()
    }), this.timeout(r.timeout == null ? 2e4 : r.timeout), this._readyState = "closed", this.uri = e;
    const i = r.parser || YP;
    this.encoder = new i.Encoder(), this.decoder = new i.Decoder(), this._autoConnect = r.autoConnect !== !1, this._autoConnect && this.open();
  }
  reconnection(e) {
    return arguments.length ? (this._reconnection = !!e, this) : this._reconnection;
  }
  reconnectionAttempts(e) {
    return e === void 0 ? this._reconnectionAttempts : (this._reconnectionAttempts = e, this);
  }
  reconnectionDelay(e) {
    var r;
    return e === void 0 ? this._reconnectionDelay : (this._reconnectionDelay = e, (r = this.backoff) === null || r === void 0 || r.setMin(e), this);
  }
  randomizationFactor(e) {
    var r;
    return e === void 0 ? this._randomizationFactor : (this._randomizationFactor = e, (r = this.backoff) === null || r === void 0 || r.setJitter(e), this);
  }
  reconnectionDelayMax(e) {
    var r;
    return e === void 0 ? this._reconnectionDelayMax : (this._reconnectionDelayMax = e, (r = this.backoff) === null || r === void 0 || r.setMax(e), this);
  }
  timeout(e) {
    return arguments.length ? (this._timeout = e, this) : this._timeout;
  }
  /**
   * Starts trying to reconnect if reconnection is enabled and we have not
   * started reconnecting yet
   *
   * @private
   */
  maybeReconnectOnOpen() {
    !this._reconnecting && this._reconnection && this.backoff.attempts === 0 && this.reconnect();
  }
  /**
   * Sets the current transport `socket`.
   *
   * @param {Function} fn - optional, callback
   * @return self
   * @public
   */
  open(e) {
    if (~this._readyState.indexOf("open"))
      return this;
    this.engine = new e0(this.uri, this.opts);
    const r = this.engine, n = this;
    this._readyState = "opening", this.skipReconnect = !1;
    const i = sn(r, "open", function() {
      n.onopen(), e && e();
    }), s = sn(r, "error", (o) => {
      n.cleanup(), n._readyState = "closed", this.emitReserved("error", o), e ? e(o) : n.maybeReconnectOnOpen();
    });
    if (this._timeout !== !1) {
      const o = this._timeout;
      o === 0 && i();
      const l = this.setTimeoutFn(() => {
        i(), r.close(), r.emit("error", new Error("timeout"));
      }, o);
      this.opts.autoUnref && l.unref(), this.subs.push(function() {
        clearTimeout(l);
      });
    }
    return this.subs.push(i), this.subs.push(s), this;
  }
  /**
   * Alias for open()
   *
   * @return self
   * @public
   */
  connect(e) {
    return this.open(e);
  }
  /**
   * Called upon transport open.
   *
   * @private
   */
  onopen() {
    this.cleanup(), this._readyState = "open", this.emitReserved("open");
    const e = this.engine;
    this.subs.push(sn(e, "ping", this.onping.bind(this)), sn(e, "data", this.ondata.bind(this)), sn(e, "error", this.onerror.bind(this)), sn(e, "close", this.onclose.bind(this)), sn(this.decoder, "decoded", this.ondecoded.bind(this)));
  }
  /**
   * Called upon a ping.
   *
   * @private
   */
  onping() {
    this.emitReserved("ping");
  }
  /**
   * Called with data.
   *
   * @private
   */
  ondata(e) {
    try {
      this.decoder.add(e);
    } catch (r) {
      this.onclose("parse error", r);
    }
  }
  /**
   * Called when parser fully decodes a packet.
   *
   * @private
   */
  ondecoded(e) {
    Zy(() => {
      this.emitReserved("packet", e);
    }, this.setTimeoutFn);
  }
  /**
   * Called upon socket error.
   *
   * @private
   */
  onerror(e) {
    this.emitReserved("error", e);
  }
  /**
   * Creates a new socket for the given `nsp`.
   *
   * @return {Socket}
   * @public
   */
  socket(e, r) {
    let n = this.nsps[e];
    return n ? this._autoConnect && !n.active && n.connect() : (n = new r0(this, e, r), this.nsps[e] = n), n;
  }
  /**
   * Called upon a socket close.
   *
   * @param socket
   * @private
   */
  _destroy(e) {
    const r = Object.keys(this.nsps);
    for (const n of r)
      if (this.nsps[n].active)
        return;
    this._close();
  }
  /**
   * Writes a packet.
   *
   * @param packet
   * @private
   */
  _packet(e) {
    const r = this.encoder.encode(e);
    for (let n = 0; n < r.length; n++)
      this.engine.write(r[n], e.options);
  }
  /**
   * Clean up transport subscriptions and packet buffer.
   *
   * @private
   */
  cleanup() {
    this.subs.forEach((e) => e()), this.subs.length = 0, this.decoder.destroy();
  }
  /**
   * Close the current socket.
   *
   * @private
   */
  _close() {
    this.skipReconnect = !0, this._reconnecting = !1, this.onclose("forced close"), this.engine && this.engine.close();
  }
  /**
   * Alias for close()
   *
   * @private
   */
  disconnect() {
    return this._close();
  }
  /**
   * Called upon engine close.
   *
   * @private
   */
  onclose(e, r) {
    this.cleanup(), this.backoff.reset(), this._readyState = "closed", this.emitReserved("close", e, r), this._reconnection && !this.skipReconnect && this.reconnect();
  }
  /**
   * Attempt a reconnection.
   *
   * @private
   */
  reconnect() {
    if (this._reconnecting || this.skipReconnect)
      return this;
    const e = this;
    if (this.backoff.attempts >= this._reconnectionAttempts)
      this.backoff.reset(), this.emitReserved("reconnect_failed"), this._reconnecting = !1;
    else {
      const r = this.backoff.duration();
      this._reconnecting = !0;
      const n = this.setTimeoutFn(() => {
        e.skipReconnect || (this.emitReserved("reconnect_attempt", e.backoff.attempts), !e.skipReconnect && e.open((i) => {
          i ? (e._reconnecting = !1, e.reconnect(), this.emitReserved("reconnect_error", i)) : e.onreconnect();
        }));
      }, r);
      this.opts.autoUnref && n.unref(), this.subs.push(function() {
        clearTimeout(n);
      });
    }
  }
  /**
   * Called upon successful reconnect.
   *
   * @private
   */
  onreconnect() {
    const e = this.backoff.attempts;
    this._reconnecting = !1, this.backoff.reset(), this.emitReserved("reconnect", e);
  }
}
const Gs = {};
function rl(t, e) {
  typeof t == "object" && (e = t, t = void 0), e = e || {};
  const r = FP(t, e.path || "/socket.io"), n = r.source, i = r.id, s = r.path, o = Gs[i] && s in Gs[i].nsps, l = e.forceNew || e["force new connection"] || e.multiplex === !1 || o;
  let c;
  return l ? c = new eh(n, e) : (Gs[i] || (Gs[i] = new eh(n, e)), c = Gs[i]), r.query && !e.query && (e.query = r.queryKey), c.socket(r.path, e);
}
Object.assign(rl, {
  Manager: eh,
  Socket: r0,
  io: rl,
  connect: rl
});
const kB = [
  "ddim",
  "plms",
  "k_lms",
  "k_dpm_2",
  "k_dpm_2_a",
  "k_dpmpp_2",
  "k_dpmpp_2_a",
  "k_euler",
  "k_euler_a",
  "k_heun"
], IB = [
  "ddim",
  "plms",
  "k_lms",
  "dpmpp_2",
  "k_dpm_2",
  "k_dpm_2_a",
  "k_dpmpp_2",
  "k_euler",
  "k_euler_a",
  "k_heun"
], LB = [
  64,
  128,
  192,
  256,
  320,
  384,
  448,
  512,
  576,
  640,
  704,
  768,
  832,
  896,
  960,
  1024,
  1088,
  1152,
  1216,
  1280,
  1344,
  1408,
  1472,
  1536,
  1600,
  1664,
  1728,
  1792,
  1856,
  1920,
  1984,
  2048
], DB = [
  64,
  128,
  192,
  256,
  320,
  384,
  448,
  512,
  576,
  640,
  704,
  768,
  832,
  896,
  960,
  1024,
  1088,
  1152,
  1216,
  1280,
  1344,
  1408,
  1472,
  1536,
  1600,
  1664,
  1728,
  1792,
  1856,
  1920,
  1984,
  2048
], MB = [
  { key: "2x", value: 2 },
  { key: "4x", value: 4 }
], XP = 0, JP = 4294967295, NB = ["gfpgan", "codeformer"], BB = [
  { key: "None", value: "none" },
  { key: "Fast", value: "latents" },
  { key: "Accurate", value: "full-res" }
];
var QP = Math.PI / 180;
function ZP() {
  return typeof window < "u" && ({}.toString.call(window) === "[object Window]" || {}.toString.call(window) === "[object global]");
}
const za = typeof global < "u" ? global : typeof window < "u" ? window : typeof WorkerGlobalScope < "u" ? self : {}, ve = {
  _global: za,
  version: "8.4.2",
  isBrowser: ZP(),
  isUnminified: /param/.test(function(t) {
  }.toString()),
  dblClickWindow: 400,
  getAngle(t) {
    return ve.angleDeg ? t * QP : t;
  },
  enableTrace: !1,
  pointerEventsEnabled: !0,
  autoDrawEnabled: !0,
  hitOnDragEnabled: !1,
  capturePointerEventsEnabled: !1,
  _mouseListenClick: !1,
  _touchListenClick: !1,
  _pointerListenClick: !1,
  _mouseInDblClickWindow: !1,
  _touchInDblClickWindow: !1,
  _pointerInDblClickWindow: !1,
  _mouseDblClickPointerId: null,
  _touchDblClickPointerId: null,
  _pointerDblClickPointerId: null,
  pixelRatio: typeof window < "u" && window.devicePixelRatio || 1,
  dragDistance: 3,
  angleDeg: !0,
  showWarnings: !0,
  dragButtons: [0, 1],
  isDragging() {
    return ve.DD.isDragging;
  },
  isDragReady() {
    return !!ve.DD.node;
  },
  releaseCanvasOnDestroy: !0,
  document: za.document,
  _injectGlobal(t) {
    za.Konva = t;
  }
}, St = (t) => {
  ve[t.prototype.getClassName()] = t;
};
ve._injectGlobal(ve);
class Ir {
  constructor(e = [1, 0, 0, 1, 0, 0]) {
    this.dirty = !1, this.m = e && e.slice() || [1, 0, 0, 1, 0, 0];
  }
  reset() {
    this.m[0] = 1, this.m[1] = 0, this.m[2] = 0, this.m[3] = 1, this.m[4] = 0, this.m[5] = 0;
  }
  copy() {
    return new Ir(this.m);
  }
  copyInto(e) {
    e.m[0] = this.m[0], e.m[1] = this.m[1], e.m[2] = this.m[2], e.m[3] = this.m[3], e.m[4] = this.m[4], e.m[5] = this.m[5];
  }
  point(e) {
    var r = this.m;
    return {
      x: r[0] * e.x + r[2] * e.y + r[4],
      y: r[1] * e.x + r[3] * e.y + r[5]
    };
  }
  translate(e, r) {
    return this.m[4] += this.m[0] * e + this.m[2] * r, this.m[5] += this.m[1] * e + this.m[3] * r, this;
  }
  scale(e, r) {
    return this.m[0] *= e, this.m[1] *= e, this.m[2] *= r, this.m[3] *= r, this;
  }
  rotate(e) {
    var r = Math.cos(e), n = Math.sin(e), i = this.m[0] * r + this.m[2] * n, s = this.m[1] * r + this.m[3] * n, o = this.m[0] * -n + this.m[2] * r, l = this.m[1] * -n + this.m[3] * r;
    return this.m[0] = i, this.m[1] = s, this.m[2] = o, this.m[3] = l, this;
  }
  getTranslation() {
    return {
      x: this.m[4],
      y: this.m[5]
    };
  }
  skew(e, r) {
    var n = this.m[0] + this.m[2] * r, i = this.m[1] + this.m[3] * r, s = this.m[2] + this.m[0] * e, o = this.m[3] + this.m[1] * e;
    return this.m[0] = n, this.m[1] = i, this.m[2] = s, this.m[3] = o, this;
  }
  multiply(e) {
    var r = this.m[0] * e.m[0] + this.m[2] * e.m[1], n = this.m[1] * e.m[0] + this.m[3] * e.m[1], i = this.m[0] * e.m[2] + this.m[2] * e.m[3], s = this.m[1] * e.m[2] + this.m[3] * e.m[3], o = this.m[0] * e.m[4] + this.m[2] * e.m[5] + this.m[4], l = this.m[1] * e.m[4] + this.m[3] * e.m[5] + this.m[5];
    return this.m[0] = r, this.m[1] = n, this.m[2] = i, this.m[3] = s, this.m[4] = o, this.m[5] = l, this;
  }
  invert() {
    var e = 1 / (this.m[0] * this.m[3] - this.m[1] * this.m[2]), r = this.m[3] * e, n = -this.m[1] * e, i = -this.m[2] * e, s = this.m[0] * e, o = e * (this.m[2] * this.m[5] - this.m[3] * this.m[4]), l = e * (this.m[1] * this.m[4] - this.m[0] * this.m[5]);
    return this.m[0] = r, this.m[1] = n, this.m[2] = i, this.m[3] = s, this.m[4] = o, this.m[5] = l, this;
  }
  getMatrix() {
    return this.m;
  }
  decompose() {
    var e = this.m[0], r = this.m[1], n = this.m[2], i = this.m[3], s = this.m[4], o = this.m[5], l = e * i - r * n;
    let c = {
      x: s,
      y: o,
      rotation: 0,
      scaleX: 0,
      scaleY: 0,
      skewX: 0,
      skewY: 0
    };
    if (e != 0 || r != 0) {
      var h = Math.sqrt(e * e + r * r);
      c.rotation = r > 0 ? Math.acos(e / h) : -Math.acos(e / h), c.scaleX = h, c.scaleY = l / h, c.skewX = (e * n + r * i) / l, c.skewY = 0;
    } else if (n != 0 || i != 0) {
      var d = Math.sqrt(n * n + i * i);
      c.rotation = Math.PI / 2 - (i > 0 ? Math.acos(-n / d) : -Math.acos(n / d)), c.scaleX = l / d, c.scaleY = d, c.skewX = 0, c.skewY = (e * n + r * i) / l;
    }
    return c.rotation = W._getRotation(c.rotation), c;
  }
}
var eR = "[object Array]", tR = "[object Number]", rR = "[object String]", nR = "[object Boolean]", iR = Math.PI / 180, aR = 180 / Math.PI, Sf = "#", sR = "", oR = "0", uR = "Konva warning: ", Im = "Konva error: ", lR = "rgb(", wf = {
  aliceblue: [240, 248, 255],
  antiquewhite: [250, 235, 215],
  aqua: [0, 255, 255],
  aquamarine: [127, 255, 212],
  azure: [240, 255, 255],
  beige: [245, 245, 220],
  bisque: [255, 228, 196],
  black: [0, 0, 0],
  blanchedalmond: [255, 235, 205],
  blue: [0, 0, 255],
  blueviolet: [138, 43, 226],
  brown: [165, 42, 42],
  burlywood: [222, 184, 135],
  cadetblue: [95, 158, 160],
  chartreuse: [127, 255, 0],
  chocolate: [210, 105, 30],
  coral: [255, 127, 80],
  cornflowerblue: [100, 149, 237],
  cornsilk: [255, 248, 220],
  crimson: [220, 20, 60],
  cyan: [0, 255, 255],
  darkblue: [0, 0, 139],
  darkcyan: [0, 139, 139],
  darkgoldenrod: [184, 132, 11],
  darkgray: [169, 169, 169],
  darkgreen: [0, 100, 0],
  darkgrey: [169, 169, 169],
  darkkhaki: [189, 183, 107],
  darkmagenta: [139, 0, 139],
  darkolivegreen: [85, 107, 47],
  darkorange: [255, 140, 0],
  darkorchid: [153, 50, 204],
  darkred: [139, 0, 0],
  darksalmon: [233, 150, 122],
  darkseagreen: [143, 188, 143],
  darkslateblue: [72, 61, 139],
  darkslategray: [47, 79, 79],
  darkslategrey: [47, 79, 79],
  darkturquoise: [0, 206, 209],
  darkviolet: [148, 0, 211],
  deeppink: [255, 20, 147],
  deepskyblue: [0, 191, 255],
  dimgray: [105, 105, 105],
  dimgrey: [105, 105, 105],
  dodgerblue: [30, 144, 255],
  firebrick: [178, 34, 34],
  floralwhite: [255, 255, 240],
  forestgreen: [34, 139, 34],
  fuchsia: [255, 0, 255],
  gainsboro: [220, 220, 220],
  ghostwhite: [248, 248, 255],
  gold: [255, 215, 0],
  goldenrod: [218, 165, 32],
  gray: [128, 128, 128],
  green: [0, 128, 0],
  greenyellow: [173, 255, 47],
  grey: [128, 128, 128],
  honeydew: [240, 255, 240],
  hotpink: [255, 105, 180],
  indianred: [205, 92, 92],
  indigo: [75, 0, 130],
  ivory: [255, 255, 240],
  khaki: [240, 230, 140],
  lavender: [230, 230, 250],
  lavenderblush: [255, 240, 245],
  lawngreen: [124, 252, 0],
  lemonchiffon: [255, 250, 205],
  lightblue: [173, 216, 230],
  lightcoral: [240, 128, 128],
  lightcyan: [224, 255, 255],
  lightgoldenrodyellow: [250, 250, 210],
  lightgray: [211, 211, 211],
  lightgreen: [144, 238, 144],
  lightgrey: [211, 211, 211],
  lightpink: [255, 182, 193],
  lightsalmon: [255, 160, 122],
  lightseagreen: [32, 178, 170],
  lightskyblue: [135, 206, 250],
  lightslategray: [119, 136, 153],
  lightslategrey: [119, 136, 153],
  lightsteelblue: [176, 196, 222],
  lightyellow: [255, 255, 224],
  lime: [0, 255, 0],
  limegreen: [50, 205, 50],
  linen: [250, 240, 230],
  magenta: [255, 0, 255],
  maroon: [128, 0, 0],
  mediumaquamarine: [102, 205, 170],
  mediumblue: [0, 0, 205],
  mediumorchid: [186, 85, 211],
  mediumpurple: [147, 112, 219],
  mediumseagreen: [60, 179, 113],
  mediumslateblue: [123, 104, 238],
  mediumspringgreen: [0, 250, 154],
  mediumturquoise: [72, 209, 204],
  mediumvioletred: [199, 21, 133],
  midnightblue: [25, 25, 112],
  mintcream: [245, 255, 250],
  mistyrose: [255, 228, 225],
  moccasin: [255, 228, 181],
  navajowhite: [255, 222, 173],
  navy: [0, 0, 128],
  oldlace: [253, 245, 230],
  olive: [128, 128, 0],
  olivedrab: [107, 142, 35],
  orange: [255, 165, 0],
  orangered: [255, 69, 0],
  orchid: [218, 112, 214],
  palegoldenrod: [238, 232, 170],
  palegreen: [152, 251, 152],
  paleturquoise: [175, 238, 238],
  palevioletred: [219, 112, 147],
  papayawhip: [255, 239, 213],
  peachpuff: [255, 218, 185],
  peru: [205, 133, 63],
  pink: [255, 192, 203],
  plum: [221, 160, 203],
  powderblue: [176, 224, 230],
  purple: [128, 0, 128],
  rebeccapurple: [102, 51, 153],
  red: [255, 0, 0],
  rosybrown: [188, 143, 143],
  royalblue: [65, 105, 225],
  saddlebrown: [139, 69, 19],
  salmon: [250, 128, 114],
  sandybrown: [244, 164, 96],
  seagreen: [46, 139, 87],
  seashell: [255, 245, 238],
  sienna: [160, 82, 45],
  silver: [192, 192, 192],
  skyblue: [135, 206, 235],
  slateblue: [106, 90, 205],
  slategray: [119, 128, 144],
  slategrey: [119, 128, 144],
  snow: [255, 255, 250],
  springgreen: [0, 255, 127],
  steelblue: [70, 130, 180],
  tan: [210, 180, 140],
  teal: [0, 128, 128],
  thistle: [216, 191, 216],
  transparent: [255, 255, 255, 0],
  tomato: [255, 99, 71],
  turquoise: [64, 224, 208],
  violet: [238, 130, 238],
  wheat: [245, 222, 179],
  white: [255, 255, 255],
  whitesmoke: [245, 245, 245],
  yellow: [255, 255, 0],
  yellowgreen: [154, 205, 5]
}, cR = /rgb\((\d{1,3}),(\d{1,3}),(\d{1,3})\)/, Du = [];
const fR = typeof requestAnimationFrame < "u" && requestAnimationFrame || function(t) {
  setTimeout(t, 60);
}, W = {
  _isElement(t) {
    return !!(t && t.nodeType == 1);
  },
  _isFunction(t) {
    return !!(t && t.constructor && t.call && t.apply);
  },
  _isPlainObject(t) {
    return !!t && t.constructor === Object;
  },
  _isArray(t) {
    return Object.prototype.toString.call(t) === eR;
  },
  _isNumber(t) {
    return Object.prototype.toString.call(t) === tR && !isNaN(t) && isFinite(t);
  },
  _isString(t) {
    return Object.prototype.toString.call(t) === rR;
  },
  _isBoolean(t) {
    return Object.prototype.toString.call(t) === nR;
  },
  isObject(t) {
    return t instanceof Object;
  },
  isValidSelector(t) {
    if (typeof t != "string")
      return !1;
    var e = t[0];
    return e === "#" || e === "." || e === e.toUpperCase();
  },
  _sign(t) {
    return t === 0 || t > 0 ? 1 : -1;
  },
  requestAnimFrame(t) {
    Du.push(t), Du.length === 1 && fR(function() {
      const e = Du;
      Du = [], e.forEach(function(r) {
        r();
      });
    });
  },
  createCanvasElement() {
    var t = document.createElement("canvas");
    try {
      t.style = t.style || {};
    } catch {
    }
    return t;
  },
  createImageElement() {
    return document.createElement("img");
  },
  _isInDocument(t) {
    for (; t = t.parentNode; )
      if (t == document)
        return !0;
    return !1;
  },
  _urlToImage(t, e) {
    var r = W.createImageElement();
    r.onload = function() {
      e(r);
    }, r.src = t;
  },
  _rgbToHex(t, e, r) {
    return ((1 << 24) + (t << 16) + (e << 8) + r).toString(16).slice(1);
  },
  _hexToRgb(t) {
    t = t.replace(Sf, sR);
    var e = parseInt(t, 16);
    return {
      r: e >> 16 & 255,
      g: e >> 8 & 255,
      b: e & 255
    };
  },
  getRandomColor() {
    for (var t = (Math.random() * 16777215 << 0).toString(16); t.length < 6; )
      t = oR + t;
    return Sf + t;
  },
  getRGB(t) {
    var e;
    return t in wf ? (e = wf[t], {
      r: e[0],
      g: e[1],
      b: e[2]
    }) : t[0] === Sf ? this._hexToRgb(t.substring(1)) : t.substr(0, 4) === lR ? (e = cR.exec(t.replace(/ /g, "")), {
      r: parseInt(e[1], 10),
      g: parseInt(e[2], 10),
      b: parseInt(e[3], 10)
    }) : {
      r: 0,
      g: 0,
      b: 0
    };
  },
  colorToRGBA(t) {
    return t = t || "black", W._namedColorToRBA(t) || W._hex3ColorToRGBA(t) || W._hex4ColorToRGBA(t) || W._hex6ColorToRGBA(t) || W._hex8ColorToRGBA(t) || W._rgbColorToRGBA(t) || W._rgbaColorToRGBA(t) || W._hslColorToRGBA(t);
  },
  _namedColorToRBA(t) {
    var e = wf[t.toLowerCase()];
    return e ? {
      r: e[0],
      g: e[1],
      b: e[2],
      a: 1
    } : null;
  },
  _rgbColorToRGBA(t) {
    if (t.indexOf("rgb(") === 0) {
      t = t.match(/rgb\(([^)]+)\)/)[1];
      var e = t.split(/ *, */).map(Number);
      return {
        r: e[0],
        g: e[1],
        b: e[2],
        a: 1
      };
    }
  },
  _rgbaColorToRGBA(t) {
    if (t.indexOf("rgba(") === 0) {
      t = t.match(/rgba\(([^)]+)\)/)[1];
      var e = t.split(/ *, */).map((r, n) => r.slice(-1) === "%" ? n === 3 ? parseInt(r) / 100 : parseInt(r) / 100 * 255 : Number(r));
      return {
        r: e[0],
        g: e[1],
        b: e[2],
        a: e[3]
      };
    }
  },
  _hex8ColorToRGBA(t) {
    if (t[0] === "#" && t.length === 9)
      return {
        r: parseInt(t.slice(1, 3), 16),
        g: parseInt(t.slice(3, 5), 16),
        b: parseInt(t.slice(5, 7), 16),
        a: parseInt(t.slice(7, 9), 16) / 255
      };
  },
  _hex6ColorToRGBA(t) {
    if (t[0] === "#" && t.length === 7)
      return {
        r: parseInt(t.slice(1, 3), 16),
        g: parseInt(t.slice(3, 5), 16),
        b: parseInt(t.slice(5, 7), 16),
        a: 1
      };
  },
  _hex4ColorToRGBA(t) {
    if (t[0] === "#" && t.length === 5)
      return {
        r: parseInt(t[1] + t[1], 16),
        g: parseInt(t[2] + t[2], 16),
        b: parseInt(t[3] + t[3], 16),
        a: parseInt(t[4] + t[4], 16) / 255
      };
  },
  _hex3ColorToRGBA(t) {
    if (t[0] === "#" && t.length === 4)
      return {
        r: parseInt(t[1] + t[1], 16),
        g: parseInt(t[2] + t[2], 16),
        b: parseInt(t[3] + t[3], 16),
        a: 1
      };
  },
  _hslColorToRGBA(t) {
    if (/hsl\((\d+),\s*([\d.]+)%,\s*([\d.]+)%\)/g.test(t)) {
      const [e, ...r] = /hsl\((\d+),\s*([\d.]+)%,\s*([\d.]+)%\)/g.exec(t), n = Number(r[0]) / 360, i = Number(r[1]) / 100, s = Number(r[2]) / 100;
      let o, l, c;
      if (i === 0)
        return c = s * 255, {
          r: Math.round(c),
          g: Math.round(c),
          b: Math.round(c),
          a: 1
        };
      s < 0.5 ? o = s * (1 + i) : o = s + i - s * i;
      const h = 2 * s - o, d = [0, 0, 0];
      for (let g = 0; g < 3; g++)
        l = n + 1 / 3 * -(g - 1), l < 0 && l++, l > 1 && l--, 6 * l < 1 ? c = h + (o - h) * 6 * l : 2 * l < 1 ? c = o : 3 * l < 2 ? c = h + (o - h) * (2 / 3 - l) * 6 : c = h, d[g] = c * 255;
      return {
        r: Math.round(d[0]),
        g: Math.round(d[1]),
        b: Math.round(d[2]),
        a: 1
      };
    }
  },
  haveIntersection(t, e) {
    return !(e.x > t.x + t.width || e.x + e.width < t.x || e.y > t.y + t.height || e.y + e.height < t.y);
  },
  cloneObject(t) {
    var e = {};
    for (var r in t)
      this._isPlainObject(t[r]) ? e[r] = this.cloneObject(t[r]) : this._isArray(t[r]) ? e[r] = this.cloneArray(t[r]) : e[r] = t[r];
    return e;
  },
  cloneArray(t) {
    return t.slice(0);
  },
  degToRad(t) {
    return t * iR;
  },
  radToDeg(t) {
    return t * aR;
  },
  _degToRad(t) {
    return W.warn("Util._degToRad is removed. Please use public Util.degToRad instead."), W.degToRad(t);
  },
  _radToDeg(t) {
    return W.warn("Util._radToDeg is removed. Please use public Util.radToDeg instead."), W.radToDeg(t);
  },
  _getRotation(t) {
    return ve.angleDeg ? W.radToDeg(t) : t;
  },
  _capitalize(t) {
    return t.charAt(0).toUpperCase() + t.slice(1);
  },
  throw(t) {
    throw new Error(Im + t);
  },
  error(t) {
    console.error(Im + t);
  },
  warn(t) {
    ve.showWarnings && console.warn(uR + t);
  },
  each(t, e) {
    for (var r in t)
      e(r, t[r]);
  },
  _inRange(t, e, r) {
    return e <= t && t < r;
  },
  _getProjectionToSegment(t, e, r, n, i, s) {
    var o, l, c, h = (t - r) * (t - r) + (e - n) * (e - n);
    if (h == 0)
      o = t, l = e, c = (i - r) * (i - r) + (s - n) * (s - n);
    else {
      var d = ((i - t) * (r - t) + (s - e) * (n - e)) / h;
      d < 0 ? (o = t, l = e, c = (t - i) * (t - i) + (e - s) * (e - s)) : d > 1 ? (o = r, l = n, c = (r - i) * (r - i) + (n - s) * (n - s)) : (o = t + d * (r - t), l = e + d * (n - e), c = (o - i) * (o - i) + (l - s) * (l - s));
    }
    return [o, l, c];
  },
  _getProjectionToLine(t, e, r) {
    var n = W.cloneObject(t), i = Number.MAX_VALUE;
    return e.forEach(function(s, o) {
      if (!(!r && o === e.length - 1)) {
        var l = e[(o + 1) % e.length], c = W._getProjectionToSegment(s.x, s.y, l.x, l.y, t.x, t.y), h = c[0], d = c[1], g = c[2];
        g < i && (n.x = h, n.y = d, i = g);
      }
    }), n;
  },
  _prepareArrayForTween(t, e, r) {
    var n, i = [], s = [];
    if (t.length > e.length) {
      var o = e;
      e = t, t = o;
    }
    for (n = 0; n < t.length; n += 2)
      i.push({
        x: t[n],
        y: t[n + 1]
      });
    for (n = 0; n < e.length; n += 2)
      s.push({
        x: e[n],
        y: e[n + 1]
      });
    var l = [];
    return s.forEach(function(c) {
      var h = W._getProjectionToLine(c, i, r);
      l.push(h.x), l.push(h.y);
    }), l;
  },
  _prepareToStringify(t) {
    var e;
    t.visitedByCircularReferenceRemoval = !0;
    for (var r in t)
      if (t.hasOwnProperty(r) && t[r] && typeof t[r] == "object") {
        if (e = Object.getOwnPropertyDescriptor(t, r), t[r].visitedByCircularReferenceRemoval || W._isElement(t[r]))
          if (e.configurable)
            delete t[r];
          else
            return null;
        else if (W._prepareToStringify(t[r]) === null)
          if (e.configurable)
            delete t[r];
          else
            return null;
      }
    return delete t.visitedByCircularReferenceRemoval, t;
  },
  _assign(t, e) {
    for (var r in e)
      t[r] = e[r];
    return t;
  },
  _getFirstPointerId(t) {
    return t.touches ? t.changedTouches[0].identifier : t.pointerId || 999;
  },
  releaseCanvas(...t) {
    ve.releaseCanvasOnDestroy && t.forEach((e) => {
      e.width = 0, e.height = 0;
    });
  },
  drawRoundedRectPath(t, e, r, n) {
    let i = 0, s = 0, o = 0, l = 0;
    typeof n == "number" ? i = s = o = l = Math.min(n, e / 2, r / 2) : (i = Math.min(n[0] || 0, e / 2, r / 2), s = Math.min(n[1] || 0, e / 2, r / 2), l = Math.min(n[2] || 0, e / 2, r / 2), o = Math.min(n[3] || 0, e / 2, r / 2)), t.moveTo(i, 0), t.lineTo(e - s, 0), t.arc(e - s, s, s, Math.PI * 3 / 2, 0, !1), t.lineTo(e, r - l), t.arc(e - l, r - l, l, 0, Math.PI / 2, !1), t.lineTo(o, r), t.arc(o, r - o, o, Math.PI / 2, Math.PI, !1), t.lineTo(0, i), t.arc(i, i, i, Math.PI, Math.PI * 3 / 2, !1);
  }
};
function Ti(t) {
  return W._isString(t) ? '"' + t + '"' : Object.prototype.toString.call(t) === "[object Number]" || W._isBoolean(t) ? t : Object.prototype.toString.call(t);
}
function n0(t) {
  return t > 255 ? 255 : t < 0 ? 0 : Math.round(t);
}
function le() {
  if (ve.isUnminified)
    return function(t, e) {
      return W._isNumber(t) || W.warn(Ti(t) + ' is a not valid value for "' + e + '" attribute. The value should be a number.'), t;
    };
}
function Dh(t) {
  if (ve.isUnminified)
    return function(e, r) {
      let n = W._isNumber(e), i = W._isArray(e) && e.length == t;
      return !n && !i && W.warn(Ti(e) + ' is a not valid value for "' + r + '" attribute. The value should be a number or Array<number>(' + t + ")"), e;
    };
}
function Mh() {
  if (ve.isUnminified)
    return function(t, e) {
      var r = W._isNumber(t), n = t === "auto";
      return r || n || W.warn(Ti(t) + ' is a not valid value for "' + e + '" attribute. The value should be a number or "auto".'), t;
    };
}
function es() {
  if (ve.isUnminified)
    return function(t, e) {
      return W._isString(t) || W.warn(Ti(t) + ' is a not valid value for "' + e + '" attribute. The value should be a string.'), t;
    };
}
function i0() {
  if (ve.isUnminified)
    return function(t, e) {
      const r = W._isString(t), n = Object.prototype.toString.call(t) === "[object CanvasGradient]" || t && t.addColorStop;
      return r || n || W.warn(Ti(t) + ' is a not valid value for "' + e + '" attribute. The value should be a string or a native gradient.'), t;
    };
}
function hR() {
  if (ve.isUnminified)
    return function(t, e) {
      const r = Int8Array ? Object.getPrototypeOf(Int8Array) : null;
      return r && t instanceof r || (W._isArray(t) ? t.forEach(function(n) {
        W._isNumber(n) || W.warn('"' + e + '" attribute has non numeric element ' + n + ". Make sure that all elements are numbers.");
      }) : W.warn(Ti(t) + ' is a not valid value for "' + e + '" attribute. The value should be a array of numbers.')), t;
    };
}
function hn() {
  if (ve.isUnminified)
    return function(t, e) {
      var r = t === !0 || t === !1;
      return r || W.warn(Ti(t) + ' is a not valid value for "' + e + '" attribute. The value should be a boolean.'), t;
    };
}
function dR(t) {
  if (ve.isUnminified)
    return function(e, r) {
      return e == null || W.isObject(e) || W.warn(Ti(e) + ' is a not valid value for "' + r + '" attribute. The value should be an object with properties ' + t), e;
    };
}
var js = "get", $s = "set";
const I = {
  addGetterSetter(t, e, r, n, i) {
    I.addGetter(t, e, r), I.addSetter(t, e, n, i), I.addOverloadedGetterSetter(t, e);
  },
  addGetter(t, e, r) {
    var n = js + W._capitalize(e);
    t.prototype[n] = t.prototype[n] || function() {
      var i = this.attrs[e];
      return i === void 0 ? r : i;
    };
  },
  addSetter(t, e, r, n) {
    var i = $s + W._capitalize(e);
    t.prototype[i] || I.overWriteSetter(t, e, r, n);
  },
  overWriteSetter(t, e, r, n) {
    var i = $s + W._capitalize(e);
    t.prototype[i] = function(s) {
      return r && s !== void 0 && s !== null && (s = r.call(this, s, e)), this._setAttr(e, s), n && n.call(this), this;
    };
  },
  addComponentsGetterSetter(t, e, r, n, i) {
    var s = r.length, o = W._capitalize, l = js + o(e), c = $s + o(e), h, d;
    t.prototype[l] = function() {
      var p = {};
      for (h = 0; h < s; h++)
        d = r[h], p[d] = this.getAttr(e + o(d));
      return p;
    };
    var g = dR(r);
    t.prototype[c] = function(p) {
      var y = this.attrs[e], v;
      n && (p = n.call(this, p)), g && g.call(this, p, e);
      for (v in p)
        p.hasOwnProperty(v) && this._setAttr(e + o(v), p[v]);
      return p || r.forEach((_) => {
        this._setAttr(e + o(_), void 0);
      }), this._fireChangeEvent(e, y, p), i && i.call(this), this;
    }, I.addOverloadedGetterSetter(t, e);
  },
  addOverloadedGetterSetter(t, e) {
    var r = W._capitalize(e), n = $s + r, i = js + r;
    t.prototype[e] = function() {
      return arguments.length ? (this[n](arguments[0]), this) : this[i]();
    };
  },
  addDeprecatedGetterSetter(t, e, r, n) {
    W.error("Adding deprecated " + e);
    var i = js + W._capitalize(e), s = e + " property is deprecated and will be removed soon. Look at Konva change log for more information.";
    t.prototype[i] = function() {
      W.error(s);
      var o = this.attrs[e];
      return o === void 0 ? r : o;
    }, I.addSetter(t, e, n, function() {
      W.error(s);
    }), I.addOverloadedGetterSetter(t, e);
  },
  backCompat(t, e) {
    W.each(e, function(r, n) {
      var i = t.prototype[n], s = js + W._capitalize(r), o = $s + W._capitalize(r);
      function l() {
        i.apply(this, arguments), W.error('"' + r + '" method is deprecated and will be removed soon. Use ""' + n + '" instead.');
      }
      t.prototype[r] = l, t.prototype[s] = l, t.prototype[o] = l;
    });
  },
  afterSetFilter() {
    this._filterUpToDate = !1;
  }
};
function gR(t) {
  var e = [], r = t.length, n = W, i, s;
  for (i = 0; i < r; i++)
    s = t[i], n._isNumber(s) ? s = Math.round(s * 1e3) / 1e3 : n._isString(s) || (s = s + ""), e.push(s);
  return e;
}
var Lm = ",", pR = "(", mR = ")", vR = "([", yR = "])", bR = ";", _R = "()", SR = "=", Dm = [
  "arc",
  "arcTo",
  "beginPath",
  "bezierCurveTo",
  "clearRect",
  "clip",
  "closePath",
  "createLinearGradient",
  "createPattern",
  "createRadialGradient",
  "drawImage",
  "ellipse",
  "fill",
  "fillText",
  "getImageData",
  "createImageData",
  "lineTo",
  "moveTo",
  "putImageData",
  "quadraticCurveTo",
  "rect",
  "restore",
  "rotate",
  "save",
  "scale",
  "setLineDash",
  "setTransform",
  "stroke",
  "strokeText",
  "transform",
  "translate"
], wR = [
  "fillStyle",
  "strokeStyle",
  "shadowColor",
  "shadowBlur",
  "shadowOffsetX",
  "shadowOffsetY",
  "lineCap",
  "lineDashOffset",
  "lineJoin",
  "lineWidth",
  "miterLimit",
  "font",
  "textAlign",
  "textBaseline",
  "globalAlpha",
  "globalCompositeOperation",
  "imageSmoothingEnabled"
];
const ER = 100;
class Ll {
  constructor(e) {
    this.canvas = e, ve.enableTrace && (this.traceArr = [], this._enableTrace());
  }
  fillShape(e) {
    e.fillEnabled() && this._fill(e);
  }
  _fill(e) {
  }
  strokeShape(e) {
    e.hasStroke() && this._stroke(e);
  }
  _stroke(e) {
  }
  fillStrokeShape(e) {
    e.attrs.fillAfterStrokeEnabled ? (this.strokeShape(e), this.fillShape(e)) : (this.fillShape(e), this.strokeShape(e));
  }
  getTrace(e, r) {
    var n = this.traceArr, i = n.length, s = "", o, l, c, h;
    for (o = 0; o < i; o++)
      l = n[o], c = l.method, c ? (h = l.args, s += c, e ? s += _R : W._isArray(h[0]) ? s += vR + h.join(Lm) + yR : (r && (h = h.map((d) => typeof d == "number" ? Math.floor(d) : d)), s += pR + h.join(Lm) + mR)) : (s += l.property, e || (s += SR + l.val)), s += bR;
    return s;
  }
  clearTrace() {
    this.traceArr = [];
  }
  _trace(e) {
    var r = this.traceArr, n;
    r.push(e), n = r.length, n >= ER && r.shift();
  }
  reset() {
    var e = this.getCanvas().getPixelRatio();
    this.setTransform(1 * e, 0, 0, 1 * e, 0, 0);
  }
  getCanvas() {
    return this.canvas;
  }
  clear(e) {
    var r = this.getCanvas();
    e ? this.clearRect(e.x || 0, e.y || 0, e.width || 0, e.height || 0) : this.clearRect(0, 0, r.getWidth() / r.pixelRatio, r.getHeight() / r.pixelRatio);
  }
  _applyLineCap(e) {
    var r = e.getLineCap();
    r && this.setAttr("lineCap", r);
  }
  _applyOpacity(e) {
    var r = e.getAbsoluteOpacity();
    r !== 1 && this.setAttr("globalAlpha", r);
  }
  _applyLineJoin(e) {
    var r = e.attrs.lineJoin;
    r && this.setAttr("lineJoin", r);
  }
  setAttr(e, r) {
    this._context[e] = r;
  }
  arc(e, r, n, i, s, o) {
    this._context.arc(e, r, n, i, s, o);
  }
  arcTo(e, r, n, i, s) {
    this._context.arcTo(e, r, n, i, s);
  }
  beginPath() {
    this._context.beginPath();
  }
  bezierCurveTo(e, r, n, i, s, o) {
    this._context.bezierCurveTo(e, r, n, i, s, o);
  }
  clearRect(e, r, n, i) {
    this._context.clearRect(e, r, n, i);
  }
  clip() {
    this._context.clip();
  }
  closePath() {
    this._context.closePath();
  }
  createImageData(e, r) {
    var n = arguments;
    if (n.length === 2)
      return this._context.createImageData(e, r);
    if (n.length === 1)
      return this._context.createImageData(e);
  }
  createLinearGradient(e, r, n, i) {
    return this._context.createLinearGradient(e, r, n, i);
  }
  createPattern(e, r) {
    return this._context.createPattern(e, r);
  }
  createRadialGradient(e, r, n, i, s, o) {
    return this._context.createRadialGradient(e, r, n, i, s, o);
  }
  drawImage(e, r, n, i, s, o, l, c, h) {
    var d = arguments, g = this._context;
    d.length === 3 ? g.drawImage(e, r, n) : d.length === 5 ? g.drawImage(e, r, n, i, s) : d.length === 9 && g.drawImage(e, r, n, i, s, o, l, c, h);
  }
  ellipse(e, r, n, i, s, o, l, c) {
    this._context.ellipse(e, r, n, i, s, o, l, c);
  }
  isPointInPath(e, r, n, i) {
    return n ? this._context.isPointInPath(n, e, r, i) : this._context.isPointInPath(e, r, i);
  }
  fill(e) {
    e ? this._context.fill(e) : this._context.fill();
  }
  fillRect(e, r, n, i) {
    this._context.fillRect(e, r, n, i);
  }
  strokeRect(e, r, n, i) {
    this._context.strokeRect(e, r, n, i);
  }
  fillText(e, r, n, i) {
    i ? this._context.fillText(e, r, n, i) : this._context.fillText(e, r, n);
  }
  measureText(e) {
    return this._context.measureText(e);
  }
  getImageData(e, r, n, i) {
    return this._context.getImageData(e, r, n, i);
  }
  lineTo(e, r) {
    this._context.lineTo(e, r);
  }
  moveTo(e, r) {
    this._context.moveTo(e, r);
  }
  rect(e, r, n, i) {
    this._context.rect(e, r, n, i);
  }
  putImageData(e, r, n) {
    this._context.putImageData(e, r, n);
  }
  quadraticCurveTo(e, r, n, i) {
    this._context.quadraticCurveTo(e, r, n, i);
  }
  restore() {
    this._context.restore();
  }
  rotate(e) {
    this._context.rotate(e);
  }
  save() {
    this._context.save();
  }
  scale(e, r) {
    this._context.scale(e, r);
  }
  setLineDash(e) {
    this._context.setLineDash ? this._context.setLineDash(e) : "mozDash" in this._context ? this._context.mozDash = e : "webkitLineDash" in this._context && (this._context.webkitLineDash = e);
  }
  getLineDash() {
    return this._context.getLineDash();
  }
  setTransform(e, r, n, i, s, o) {
    this._context.setTransform(e, r, n, i, s, o);
  }
  stroke(e) {
    e ? this._context.stroke(e) : this._context.stroke();
  }
  strokeText(e, r, n, i) {
    this._context.strokeText(e, r, n, i);
  }
  transform(e, r, n, i, s, o) {
    this._context.transform(e, r, n, i, s, o);
  }
  translate(e, r) {
    this._context.translate(e, r);
  }
  _enableTrace() {
    var e = this, r = Dm.length, n = this.setAttr, i, s, o = function(l) {
      var c = e[l], h;
      e[l] = function() {
        return s = gR(Array.prototype.slice.call(arguments, 0)), h = c.apply(e, arguments), e._trace({
          method: l,
          args: s
        }), h;
      };
    };
    for (i = 0; i < r; i++)
      o(Dm[i]);
    e.setAttr = function() {
      n.apply(e, arguments);
      var l = arguments[0], c = arguments[1];
      (l === "shadowOffsetX" || l === "shadowOffsetY" || l === "shadowBlur") && (c = c / this.canvas.getPixelRatio()), e._trace({
        property: l,
        val: c
      });
    };
  }
  _applyGlobalCompositeOperation(e) {
    const r = e.attrs.globalCompositeOperation;
    var n = !r || r === "source-over";
    n || this.setAttr("globalCompositeOperation", r);
  }
}
wR.forEach(function(t) {
  Object.defineProperty(Ll.prototype, t, {
    get() {
      return this._context[t];
    },
    set(e) {
      this._context[t] = e;
    }
  });
});
class CR extends Ll {
  constructor(e) {
    super(e), this._context = e._canvas.getContext("2d");
  }
  _fillColor(e) {
    var r = e.fill();
    this.setAttr("fillStyle", r), e._fillFunc(this);
  }
  _fillPattern(e) {
    this.setAttr("fillStyle", e._getFillPattern()), e._fillFunc(this);
  }
  _fillLinearGradient(e) {
    var r = e._getLinearGradient();
    r && (this.setAttr("fillStyle", r), e._fillFunc(this));
  }
  _fillRadialGradient(e) {
    var r = e._getRadialGradient();
    r && (this.setAttr("fillStyle", r), e._fillFunc(this));
  }
  _fill(e) {
    var r = e.fill(), n = e.getFillPriority();
    if (r && n === "color") {
      this._fillColor(e);
      return;
    }
    var i = e.getFillPatternImage();
    if (i && n === "pattern") {
      this._fillPattern(e);
      return;
    }
    var s = e.getFillLinearGradientColorStops();
    if (s && n === "linear-gradient") {
      this._fillLinearGradient(e);
      return;
    }
    var o = e.getFillRadialGradientColorStops();
    if (o && n === "radial-gradient") {
      this._fillRadialGradient(e);
      return;
    }
    r ? this._fillColor(e) : i ? this._fillPattern(e) : s ? this._fillLinearGradient(e) : o && this._fillRadialGradient(e);
  }
  _strokeLinearGradient(e) {
    var r = e.getStrokeLinearGradientStartPoint(), n = e.getStrokeLinearGradientEndPoint(), i = e.getStrokeLinearGradientColorStops(), s = this.createLinearGradient(r.x, r.y, n.x, n.y);
    if (i) {
      for (var o = 0; o < i.length; o += 2)
        s.addColorStop(i[o], i[o + 1]);
      this.setAttr("strokeStyle", s);
    }
  }
  _stroke(e) {
    var r = e.dash(), n = e.getStrokeScaleEnabled();
    if (e.hasStroke()) {
      if (!n) {
        this.save();
        var i = this.getCanvas().getPixelRatio();
        this.setTransform(i, 0, 0, i, 0, 0);
      }
      this._applyLineCap(e), r && e.dashEnabled() && (this.setLineDash(r), this.setAttr("lineDashOffset", e.dashOffset())), this.setAttr("lineWidth", e.strokeWidth()), e.getShadowForStrokeEnabled() || this.setAttr("shadowColor", "rgba(0,0,0,0)");
      var s = e.getStrokeLinearGradientColorStops();
      s ? this._strokeLinearGradient(e) : this.setAttr("strokeStyle", e.stroke()), e._strokeFunc(this), n || this.restore();
    }
  }
  _applyShadow(e) {
    var r, n, i, s = (r = e.getShadowRGBA()) !== null && r !== void 0 ? r : "black", o = (n = e.getShadowBlur()) !== null && n !== void 0 ? n : 5, l = (i = e.getShadowOffset()) !== null && i !== void 0 ? i : {
      x: 0,
      y: 0
    }, c = e.getAbsoluteScale(), h = this.canvas.getPixelRatio(), d = c.x * h, g = c.y * h;
    this.setAttr("shadowColor", s), this.setAttr("shadowBlur", o * Math.min(Math.abs(d), Math.abs(g))), this.setAttr("shadowOffsetX", l.x * d), this.setAttr("shadowOffsetY", l.y * g);
  }
}
class xR extends Ll {
  constructor(e) {
    super(e), this._context = e._canvas.getContext("2d", {
      willReadFrequently: !0
    });
  }
  _fill(e) {
    this.save(), this.setAttr("fillStyle", e.colorKey), e._fillFuncHit(this), this.restore();
  }
  strokeShape(e) {
    e.hasHitStroke() && this._stroke(e);
  }
  _stroke(e) {
    if (e.hasHitStroke()) {
      var r = e.getStrokeScaleEnabled();
      if (!r) {
        this.save();
        var n = this.getCanvas().getPixelRatio();
        this.setTransform(n, 0, 0, n, 0, 0);
      }
      this._applyLineCap(e);
      var i = e.hitStrokeWidth(), s = i === "auto" ? e.strokeWidth() : i;
      this.setAttr("lineWidth", s), this.setAttr("strokeStyle", e.colorKey), e._strokeFuncHit(this), r || this.restore();
    }
  }
}
var Mu;
function OR() {
  if (Mu)
    return Mu;
  var t = W.createCanvasElement(), e = t.getContext("2d");
  return Mu = function() {
    var r = ve._global.devicePixelRatio || 1, n = e.webkitBackingStorePixelRatio || e.mozBackingStorePixelRatio || e.msBackingStorePixelRatio || e.oBackingStorePixelRatio || e.backingStorePixelRatio || 1;
    return r / n;
  }(), W.releaseCanvas(t), Mu;
}
class Dl {
  constructor(e) {
    this.pixelRatio = 1, this.width = 0, this.height = 0, this.isCache = !1;
    var r = e || {}, n = r.pixelRatio || ve.pixelRatio || OR();
    this.pixelRatio = n, this._canvas = W.createCanvasElement(), this._canvas.style.padding = "0", this._canvas.style.margin = "0", this._canvas.style.border = "0", this._canvas.style.background = "transparent", this._canvas.style.position = "absolute", this._canvas.style.top = "0", this._canvas.style.left = "0";
  }
  getContext() {
    return this.context;
  }
  getPixelRatio() {
    return this.pixelRatio;
  }
  setPixelRatio(e) {
    var r = this.pixelRatio;
    this.pixelRatio = e, this.setSize(this.getWidth() / r, this.getHeight() / r);
  }
  setWidth(e) {
    this.width = this._canvas.width = e * this.pixelRatio, this._canvas.style.width = e + "px";
    var r = this.pixelRatio, n = this.getContext()._context;
    n.scale(r, r);
  }
  setHeight(e) {
    this.height = this._canvas.height = e * this.pixelRatio, this._canvas.style.height = e + "px";
    var r = this.pixelRatio, n = this.getContext()._context;
    n.scale(r, r);
  }
  getWidth() {
    return this.width;
  }
  getHeight() {
    return this.height;
  }
  setSize(e, r) {
    this.setWidth(e || 0), this.setHeight(r || 0);
  }
  toDataURL(e, r) {
    try {
      return this._canvas.toDataURL(e, r);
    } catch {
      try {
        return this._canvas.toDataURL();
      } catch (i) {
        return W.error("Unable to get data URL. " + i.message + " For more info read https://konvajs.org/docs/posts/Tainted_Canvas.html."), "";
      }
    }
  }
}
I.addGetterSetter(Dl, "pixelRatio", void 0, le());
class Va extends Dl {
  constructor(e = { width: 0, height: 0 }) {
    super(e), this.context = new CR(this), this.setSize(e.width, e.height);
  }
}
class Nh extends Dl {
  constructor(e = { width: 0, height: 0 }) {
    super(e), this.hitCanvas = !0, this.context = new xR(this), this.setSize(e.width, e.height);
  }
}
const ze = {
  get isDragging() {
    var t = !1;
    return ze._dragElements.forEach((e) => {
      e.dragStatus === "dragging" && (t = !0);
    }), t;
  },
  justDragged: !1,
  get node() {
    var t;
    return ze._dragElements.forEach((e) => {
      t = e.node;
    }), t;
  },
  _dragElements: /* @__PURE__ */ new Map(),
  _drag(t) {
    const e = [];
    ze._dragElements.forEach((r, n) => {
      const { node: i } = r, s = i.getStage();
      s.setPointersPositions(t), r.pointerId === void 0 && (r.pointerId = W._getFirstPointerId(t));
      const o = s._changedPointerPositions.find((h) => h.id === r.pointerId);
      if (o) {
        if (r.dragStatus !== "dragging") {
          var l = i.dragDistance(), c = Math.max(Math.abs(o.x - r.startPointerPos.x), Math.abs(o.y - r.startPointerPos.y));
          if (c < l || (i.startDrag({ evt: t }), !i.isDragging()))
            return;
        }
        i._setDragPosition(t, r), e.push(i);
      }
    }), e.forEach((r) => {
      r.fire("dragmove", {
        type: "dragmove",
        target: r,
        evt: t
      }, !0);
    });
  },
  _endDragBefore(t) {
    const e = [];
    ze._dragElements.forEach((r) => {
      const { node: n } = r, i = n.getStage();
      if (t && i.setPointersPositions(t), !i._changedPointerPositions.find((l) => l.id === r.pointerId))
        return;
      (r.dragStatus === "dragging" || r.dragStatus === "stopped") && (ze.justDragged = !0, ve._mouseListenClick = !1, ve._touchListenClick = !1, ve._pointerListenClick = !1, r.dragStatus = "stopped");
      const o = r.node.getLayer() || r.node instanceof ve.Stage && r.node;
      o && e.indexOf(o) === -1 && e.push(o);
    }), e.forEach((r) => {
      r.draw();
    });
  },
  _endDragAfter(t) {
    ze._dragElements.forEach((e, r) => {
      e.dragStatus === "stopped" && e.node.fire("dragend", {
        type: "dragend",
        target: e.node,
        evt: t
      }, !0), e.dragStatus !== "dragging" && ze._dragElements.delete(r);
    });
  }
};
ve.isBrowser && (window.addEventListener("mouseup", ze._endDragBefore, !0), window.addEventListener("touchend", ze._endDragBefore, !0), window.addEventListener("mousemove", ze._drag), window.addEventListener("touchmove", ze._drag), window.addEventListener("mouseup", ze._endDragAfter, !1), window.addEventListener("touchend", ze._endDragAfter, !1));
var nl = "absoluteOpacity", Nu = "allEventListeners", Jn = "absoluteTransform", Mm = "absoluteScale", zi = "canvas", TR = "Change", PR = "children", RR = "konva", th = "listening", Nm = "mouseenter", Bm = "mouseleave", Fm = "set", Gm = "Shape", il = " ", jm = "stage", yi = "transform", AR = "Stage", rh = "visible", kR = [
  "xChange.konva",
  "yChange.konva",
  "scaleXChange.konva",
  "scaleYChange.konva",
  "skewXChange.konva",
  "skewYChange.konva",
  "rotationChange.konva",
  "offsetXChange.konva",
  "offsetYChange.konva",
  "transformsEnabledChange.konva"
].join(il);
let IR = 1;
class ue {
  constructor(e) {
    this._id = IR++, this.eventListeners = {}, this.attrs = {}, this.index = 0, this._allEventListeners = null, this.parent = null, this._cache = /* @__PURE__ */ new Map(), this._attachedDepsListeners = /* @__PURE__ */ new Map(), this._lastPos = null, this._batchingTransformChange = !1, this._needClearTransformCache = !1, this._filterUpToDate = !1, this._isUnderCache = !1, this._dragEventId = null, this._shouldFireChangeEvents = !1, this.setAttrs(e), this._shouldFireChangeEvents = !0;
  }
  hasChildren() {
    return !1;
  }
  _clearCache(e) {
    (e === yi || e === Jn) && this._cache.get(e) ? this._cache.get(e).dirty = !0 : e ? this._cache.delete(e) : this._cache.clear();
  }
  _getCache(e, r) {
    var n = this._cache.get(e), i = e === yi || e === Jn, s = n === void 0 || i && n.dirty === !0;
    return s && (n = r.call(this), this._cache.set(e, n)), n;
  }
  _calculate(e, r, n) {
    if (!this._attachedDepsListeners.get(e)) {
      const i = r.map((s) => s + "Change.konva").join(il);
      this.on(i, () => {
        this._clearCache(e);
      }), this._attachedDepsListeners.set(e, !0);
    }
    return this._getCache(e, n);
  }
  _getCanvasCache() {
    return this._cache.get(zi);
  }
  _clearSelfAndDescendantCache(e) {
    this._clearCache(e), e === Jn && this.fire("absoluteTransformChange");
  }
  clearCache() {
    if (this._cache.has(zi)) {
      const { scene: e, filter: r, hit: n } = this._cache.get(zi);
      W.releaseCanvas(e, r, n), this._cache.delete(zi);
    }
    return this._clearSelfAndDescendantCache(), this._requestDraw(), this;
  }
  cache(e) {
    var r = e || {}, n = {};
    (r.x === void 0 || r.y === void 0 || r.width === void 0 || r.height === void 0) && (n = this.getClientRect({
      skipTransform: !0,
      relativeTo: this.getParent()
    }));
    var i = Math.ceil(r.width || n.width), s = Math.ceil(r.height || n.height), o = r.pixelRatio, l = r.x === void 0 ? Math.floor(n.x) : r.x, c = r.y === void 0 ? Math.floor(n.y) : r.y, h = r.offset || 0, d = r.drawBorder || !1, g = r.hitCanvasPixelRatio || 1;
    if (!i || !s) {
      W.error("Can not cache the node. Width or height of the node equals 0. Caching is skipped.");
      return;
    }
    i += h * 2 + 1, s += h * 2 + 1, l -= h, c -= h;
    var p = new Va({
      pixelRatio: o,
      width: i,
      height: s
    }), y = new Va({
      pixelRatio: o,
      width: 0,
      height: 0
    }), v = new Nh({
      pixelRatio: g,
      width: i,
      height: s
    }), _ = p.getContext(), b = v.getContext();
    return v.isCache = !0, p.isCache = !0, this._cache.delete(zi), this._filterUpToDate = !1, r.imageSmoothingEnabled === !1 && (p.getContext()._context.imageSmoothingEnabled = !1, y.getContext()._context.imageSmoothingEnabled = !1), _.save(), b.save(), _.translate(-l, -c), b.translate(-l, -c), this._isUnderCache = !0, this._clearSelfAndDescendantCache(nl), this._clearSelfAndDescendantCache(Mm), this.drawScene(p, this), this.drawHit(v, this), this._isUnderCache = !1, _.restore(), b.restore(), d && (_.save(), _.beginPath(), _.rect(0, 0, i, s), _.closePath(), _.setAttr("strokeStyle", "red"), _.setAttr("lineWidth", 5), _.stroke(), _.restore()), this._cache.set(zi, {
      scene: p,
      filter: y,
      hit: v,
      x: l,
      y: c
    }), this._requestDraw(), this;
  }
  isCached() {
    return this._cache.has(zi);
  }
  getClientRect(e) {
    throw new Error('abstract "getClientRect" method call');
  }
  _transformedRect(e, r) {
    var n = [
      { x: e.x, y: e.y },
      { x: e.x + e.width, y: e.y },
      { x: e.x + e.width, y: e.y + e.height },
      { x: e.x, y: e.y + e.height }
    ], i, s, o, l, c = this.getAbsoluteTransform(r);
    return n.forEach(function(h) {
      var d = c.point(h);
      i === void 0 && (i = o = d.x, s = l = d.y), i = Math.min(i, d.x), s = Math.min(s, d.y), o = Math.max(o, d.x), l = Math.max(l, d.y);
    }), {
      x: i,
      y: s,
      width: o - i,
      height: l - s
    };
  }
  _drawCachedSceneCanvas(e) {
    e.save(), e._applyOpacity(this), e._applyGlobalCompositeOperation(this);
    const r = this._getCanvasCache();
    e.translate(r.x, r.y);
    var n = this._getCachedSceneCanvas(), i = n.pixelRatio;
    e.drawImage(n._canvas, 0, 0, n.width / i, n.height / i), e.restore();
  }
  _drawCachedHitCanvas(e) {
    var r = this._getCanvasCache(), n = r.hit;
    e.save(), e.translate(r.x, r.y), e.drawImage(n._canvas, 0, 0, n.width / n.pixelRatio, n.height / n.pixelRatio), e.restore();
  }
  _getCachedSceneCanvas() {
    var e = this.filters(), r = this._getCanvasCache(), n = r.scene, i = r.filter, s = i.getContext(), o, l, c, h;
    if (e) {
      if (!this._filterUpToDate) {
        var d = n.pixelRatio;
        i.setSize(n.width / n.pixelRatio, n.height / n.pixelRatio);
        try {
          for (o = e.length, s.clear(), s.drawImage(n._canvas, 0, 0, n.getWidth() / d, n.getHeight() / d), l = s.getImageData(0, 0, i.getWidth(), i.getHeight()), c = 0; c < o; c++) {
            if (h = e[c], typeof h != "function") {
              W.error("Filter should be type of function, but got " + typeof h + " instead. Please check correct filters");
              continue;
            }
            h.call(this, l), s.putImageData(l, 0, 0);
          }
        } catch (g) {
          W.error("Unable to apply filter. " + g.message + " This post my help you https://konvajs.org/docs/posts/Tainted_Canvas.html.");
        }
        this._filterUpToDate = !0;
      }
      return i;
    }
    return n;
  }
  on(e, r) {
    if (this._cache && this._cache.delete(Nu), arguments.length === 3)
      return this._delegate.apply(this, arguments);
    var n = e.split(il), i = n.length, s, o, l, c, h;
    for (s = 0; s < i; s++)
      o = n[s], l = o.split("."), c = l[0], h = l[1] || "", this.eventListeners[c] || (this.eventListeners[c] = []), this.eventListeners[c].push({
        name: h,
        handler: r
      });
    return this;
  }
  off(e, r) {
    var n = (e || "").split(il), i = n.length, s, o, l, c, h, d;
    if (this._cache && this._cache.delete(Nu), !e)
      for (o in this.eventListeners)
        this._off(o);
    for (s = 0; s < i; s++)
      if (l = n[s], c = l.split("."), h = c[0], d = c[1], h)
        this.eventListeners[h] && this._off(h, d, r);
      else
        for (o in this.eventListeners)
          this._off(o, d, r);
    return this;
  }
  dispatchEvent(e) {
    var r = {
      target: this,
      type: e.type,
      evt: e
    };
    return this.fire(e.type, r), this;
  }
  addEventListener(e, r) {
    return this.on(e, function(n) {
      r.call(this, n.evt);
    }), this;
  }
  removeEventListener(e) {
    return this.off(e), this;
  }
  _delegate(e, r, n) {
    var i = this;
    this.on(e, function(s) {
      for (var o = s.target.findAncestors(r, !0, i), l = 0; l < o.length; l++)
        s = W.cloneObject(s), s.currentTarget = o[l], n.call(o[l], s);
    });
  }
  remove() {
    return this.isDragging() && this.stopDrag(), ze._dragElements.delete(this._id), this._remove(), this;
  }
  _clearCaches() {
    this._clearSelfAndDescendantCache(Jn), this._clearSelfAndDescendantCache(nl), this._clearSelfAndDescendantCache(Mm), this._clearSelfAndDescendantCache(jm), this._clearSelfAndDescendantCache(rh), this._clearSelfAndDescendantCache(th);
  }
  _remove() {
    this._clearCaches();
    var e = this.getParent();
    e && e.children && (e.children.splice(this.index, 1), e._setChildrenIndices(), this.parent = null);
  }
  destroy() {
    return this.remove(), this.clearCache(), this;
  }
  getAttr(e) {
    var r = "get" + W._capitalize(e);
    return W._isFunction(this[r]) ? this[r]() : this.attrs[e];
  }
  getAncestors() {
    for (var e = this.getParent(), r = []; e; )
      r.push(e), e = e.getParent();
    return r;
  }
  getAttrs() {
    return this.attrs || {};
  }
  setAttrs(e) {
    return this._batchTransformChanges(() => {
      var r, n;
      if (!e)
        return this;
      for (r in e)
        r !== PR && (n = Fm + W._capitalize(r), W._isFunction(this[n]) ? this[n](e[r]) : this._setAttr(r, e[r]));
    }), this;
  }
  isListening() {
    return this._getCache(th, this._isListening);
  }
  _isListening(e) {
    if (!this.listening())
      return !1;
    const n = this.getParent();
    return n && n !== e && this !== e ? n._isListening(e) : !0;
  }
  isVisible() {
    return this._getCache(rh, this._isVisible);
  }
  _isVisible(e) {
    if (!this.visible())
      return !1;
    const n = this.getParent();
    return n && n !== e && this !== e ? n._isVisible(e) : !0;
  }
  shouldDrawHit(e, r = !1) {
    if (e)
      return this._isVisible(e) && this._isListening(e);
    var n = this.getLayer(), i = !1;
    ze._dragElements.forEach((o) => {
      o.dragStatus === "dragging" && (o.node.nodeType === "Stage" || o.node.getLayer() === n) && (i = !0);
    });
    var s = !r && !ve.hitOnDragEnabled && i;
    return this.isListening() && this.isVisible() && !s;
  }
  show() {
    return this.visible(!0), this;
  }
  hide() {
    return this.visible(!1), this;
  }
  getZIndex() {
    return this.index || 0;
  }
  getAbsoluteZIndex() {
    var e = this.getDepth(), r = this, n = 0, i, s, o, l;
    function c(h) {
      for (i = [], s = h.length, o = 0; o < s; o++)
        l = h[o], n++, l.nodeType !== Gm && (i = i.concat(l.getChildren().slice())), l._id === r._id && (o = s);
      i.length > 0 && i[0].getDepth() <= e && c(i);
    }
    return r.nodeType !== AR && c(r.getStage().getChildren()), n;
  }
  getDepth() {
    for (var e = 0, r = this.parent; r; )
      e++, r = r.parent;
    return e;
  }
  _batchTransformChanges(e) {
    this._batchingTransformChange = !0, e(), this._batchingTransformChange = !1, this._needClearTransformCache && (this._clearCache(yi), this._clearSelfAndDescendantCache(Jn)), this._needClearTransformCache = !1;
  }
  setPosition(e) {
    return this._batchTransformChanges(() => {
      this.x(e.x), this.y(e.y);
    }), this;
  }
  getPosition() {
    return {
      x: this.x(),
      y: this.y()
    };
  }
  getRelativePointerPosition() {
    if (!this.getStage())
      return null;
    var e = this.getStage().getPointerPosition();
    if (!e)
      return null;
    var r = this.getAbsoluteTransform().copy();
    return r.invert(), r.point(e);
  }
  getAbsolutePosition(e) {
    let r = !1, n = this.parent;
    for (; n; ) {
      if (n.isCached()) {
        r = !0;
        break;
      }
      n = n.parent;
    }
    r && !e && (e = !0);
    var i = this.getAbsoluteTransform(e).getMatrix(), s = new Ir(), o = this.offset();
    return s.m = i.slice(), s.translate(o.x, o.y), s.getTranslation();
  }
  setAbsolutePosition(e) {
    var r = this._clearTransform();
    this.attrs.x = r.x, this.attrs.y = r.y, delete r.x, delete r.y, this._clearCache(yi);
    var n = this._getAbsoluteTransform().copy();
    return n.invert(), n.translate(e.x, e.y), e = {
      x: this.attrs.x + n.getTranslation().x,
      y: this.attrs.y + n.getTranslation().y
    }, this._setTransform(r), this.setPosition({ x: e.x, y: e.y }), this._clearCache(yi), this._clearSelfAndDescendantCache(Jn), this;
  }
  _setTransform(e) {
    var r;
    for (r in e)
      this.attrs[r] = e[r];
  }
  _clearTransform() {
    var e = {
      x: this.x(),
      y: this.y(),
      rotation: this.rotation(),
      scaleX: this.scaleX(),
      scaleY: this.scaleY(),
      offsetX: this.offsetX(),
      offsetY: this.offsetY(),
      skewX: this.skewX(),
      skewY: this.skewY()
    };
    return this.attrs.x = 0, this.attrs.y = 0, this.attrs.rotation = 0, this.attrs.scaleX = 1, this.attrs.scaleY = 1, this.attrs.offsetX = 0, this.attrs.offsetY = 0, this.attrs.skewX = 0, this.attrs.skewY = 0, e;
  }
  move(e) {
    var r = e.x, n = e.y, i = this.x(), s = this.y();
    return r !== void 0 && (i += r), n !== void 0 && (s += n), this.setPosition({ x: i, y: s }), this;
  }
  _eachAncestorReverse(e, r) {
    var n = [], i = this.getParent(), s, o;
    if (!(r && r._id === this._id)) {
      for (n.unshift(this); i && (!r || i._id !== r._id); )
        n.unshift(i), i = i.parent;
      for (s = n.length, o = 0; o < s; o++)
        e(n[o]);
    }
  }
  rotate(e) {
    return this.rotation(this.rotation() + e), this;
  }
  moveToTop() {
    if (!this.parent)
      return W.warn("Node has no parent. moveToTop function is ignored."), !1;
    var e = this.index, r = this.parent.getChildren().length;
    return e < r - 1 ? (this.parent.children.splice(e, 1), this.parent.children.push(this), this.parent._setChildrenIndices(), !0) : !1;
  }
  moveUp() {
    if (!this.parent)
      return W.warn("Node has no parent. moveUp function is ignored."), !1;
    var e = this.index, r = this.parent.getChildren().length;
    return e < r - 1 ? (this.parent.children.splice(e, 1), this.parent.children.splice(e + 1, 0, this), this.parent._setChildrenIndices(), !0) : !1;
  }
  moveDown() {
    if (!this.parent)
      return W.warn("Node has no parent. moveDown function is ignored."), !1;
    var e = this.index;
    return e > 0 ? (this.parent.children.splice(e, 1), this.parent.children.splice(e - 1, 0, this), this.parent._setChildrenIndices(), !0) : !1;
  }
  moveToBottom() {
    if (!this.parent)
      return W.warn("Node has no parent. moveToBottom function is ignored."), !1;
    var e = this.index;
    return e > 0 ? (this.parent.children.splice(e, 1), this.parent.children.unshift(this), this.parent._setChildrenIndices(), !0) : !1;
  }
  setZIndex(e) {
    if (!this.parent)
      return W.warn("Node has no parent. zIndex parameter is ignored."), this;
    (e < 0 || e >= this.parent.children.length) && W.warn("Unexpected value " + e + " for zIndex property. zIndex is just index of a node in children of its parent. Expected value is from 0 to " + (this.parent.children.length - 1) + ".");
    var r = this.index;
    return this.parent.children.splice(r, 1), this.parent.children.splice(e, 0, this), this.parent._setChildrenIndices(), this;
  }
  getAbsoluteOpacity() {
    return this._getCache(nl, this._getAbsoluteOpacity);
  }
  _getAbsoluteOpacity() {
    var e = this.opacity(), r = this.getParent();
    return r && !r._isUnderCache && (e *= r.getAbsoluteOpacity()), e;
  }
  moveTo(e) {
    return this.getParent() !== e && (this._remove(), e.add(this)), this;
  }
  toObject() {
    var e = {}, r = this.getAttrs(), n, i, s, o, l;
    e.attrs = {};
    for (n in r)
      i = r[n], l = W.isObject(i) && !W._isPlainObject(i) && !W._isArray(i), !l && (s = typeof this[n] == "function" && this[n], delete r[n], o = s ? s.call(this) : null, r[n] = i, o !== i && (e.attrs[n] = i));
    return e.className = this.getClassName(), W._prepareToStringify(e);
  }
  toJSON() {
    return JSON.stringify(this.toObject());
  }
  getParent() {
    return this.parent;
  }
  findAncestors(e, r, n) {
    var i = [];
    r && this._isMatch(e) && i.push(this);
    for (var s = this.parent; s; ) {
      if (s === n)
        return i;
      s._isMatch(e) && i.push(s), s = s.parent;
    }
    return i;
  }
  isAncestorOf(e) {
    return !1;
  }
  findAncestor(e, r, n) {
    return this.findAncestors(e, r, n)[0];
  }
  _isMatch(e) {
    if (!e)
      return !1;
    if (typeof e == "function")
      return e(this);
    var r = e.replace(/ /g, "").split(","), n = r.length, i, s;
    for (i = 0; i < n; i++)
      if (s = r[i], W.isValidSelector(s) || (W.warn('Selector "' + s + '" is invalid. Allowed selectors examples are "#foo", ".bar" or "Group".'), W.warn('If you have a custom shape with such className, please change it to start with upper letter like "Triangle".'), W.warn("Konva is awesome, right?")), s.charAt(0) === "#") {
        if (this.id() === s.slice(1))
          return !0;
      } else if (s.charAt(0) === ".") {
        if (this.hasName(s.slice(1)))
          return !0;
      } else if (this.className === s || this.nodeType === s)
        return !0;
    return !1;
  }
  getLayer() {
    var e = this.getParent();
    return e ? e.getLayer() : null;
  }
  getStage() {
    return this._getCache(jm, this._getStage);
  }
  _getStage() {
    var e = this.getParent();
    if (e)
      return e.getStage();
  }
  fire(e, r = {}, n) {
    return r.target = r.target || this, n ? this._fireAndBubble(e, r) : this._fire(e, r), this;
  }
  getAbsoluteTransform(e) {
    return e ? this._getAbsoluteTransform(e) : this._getCache(Jn, this._getAbsoluteTransform);
  }
  _getAbsoluteTransform(e) {
    var r;
    if (e)
      return r = new Ir(), this._eachAncestorReverse(function(i) {
        var s = i.transformsEnabled();
        s === "all" ? r.multiply(i.getTransform()) : s === "position" && r.translate(i.x() - i.offsetX(), i.y() - i.offsetY());
      }, e), r;
    r = this._cache.get(Jn) || new Ir(), this.parent ? this.parent.getAbsoluteTransform().copyInto(r) : r.reset();
    var n = this.transformsEnabled();
    if (n === "all")
      r.multiply(this.getTransform());
    else if (n === "position") {
      const i = this.attrs.x || 0, s = this.attrs.y || 0, o = this.attrs.offsetX || 0, l = this.attrs.offsetY || 0;
      r.translate(i - o, s - l);
    }
    return r.dirty = !1, r;
  }
  getAbsoluteScale(e) {
    for (var r = this; r; )
      r._isUnderCache && (e = r), r = r.getParent();
    const i = this.getAbsoluteTransform(e).decompose();
    return {
      x: i.scaleX,
      y: i.scaleY
    };
  }
  getAbsoluteRotation() {
    return this.getAbsoluteTransform().decompose().rotation;
  }
  getTransform() {
    return this._getCache(yi, this._getTransform);
  }
  _getTransform() {
    var e, r, n = this._cache.get(yi) || new Ir();
    n.reset();
    var i = this.x(), s = this.y(), o = ve.getAngle(this.rotation()), l = (e = this.attrs.scaleX) !== null && e !== void 0 ? e : 1, c = (r = this.attrs.scaleY) !== null && r !== void 0 ? r : 1, h = this.attrs.skewX || 0, d = this.attrs.skewY || 0, g = this.attrs.offsetX || 0, p = this.attrs.offsetY || 0;
    return (i !== 0 || s !== 0) && n.translate(i, s), o !== 0 && n.rotate(o), (h !== 0 || d !== 0) && n.skew(h, d), (l !== 1 || c !== 1) && n.scale(l, c), (g !== 0 || p !== 0) && n.translate(-1 * g, -1 * p), n.dirty = !1, n;
  }
  clone(e) {
    var r = W.cloneObject(this.attrs), n, i, s, o, l;
    for (n in e)
      r[n] = e[n];
    var c = new this.constructor(r);
    for (n in this.eventListeners)
      for (i = this.eventListeners[n], s = i.length, o = 0; o < s; o++)
        l = i[o], l.name.indexOf(RR) < 0 && (c.eventListeners[n] || (c.eventListeners[n] = []), c.eventListeners[n].push(l));
    return c;
  }
  _toKonvaCanvas(e) {
    e = e || {};
    var r = this.getClientRect(), n = this.getStage(), i = e.x !== void 0 ? e.x : Math.floor(r.x), s = e.y !== void 0 ? e.y : Math.floor(r.y), o = e.pixelRatio || 1, l = new Va({
      width: e.width || Math.ceil(r.width) || (n ? n.width() : 0),
      height: e.height || Math.ceil(r.height) || (n ? n.height() : 0),
      pixelRatio: o
    }), c = l.getContext();
    return e.imageSmoothingEnabled === !1 && (c._context.imageSmoothingEnabled = !1), c.save(), (i || s) && c.translate(-1 * i, -1 * s), this.drawScene(l), c.restore(), l;
  }
  toCanvas(e) {
    return this._toKonvaCanvas(e)._canvas;
  }
  toDataURL(e) {
    e = e || {};
    var r = e.mimeType || null, n = e.quality || null, i = this._toKonvaCanvas(e).toDataURL(r, n);
    return e.callback && e.callback(i), i;
  }
  toImage(e) {
    return new Promise((r, n) => {
      try {
        const i = e == null ? void 0 : e.callback;
        i && delete e.callback, W._urlToImage(this.toDataURL(e), function(s) {
          r(s), i == null || i(s);
        });
      } catch (i) {
        n(i);
      }
    });
  }
  toBlob(e) {
    return new Promise((r, n) => {
      try {
        const i = e == null ? void 0 : e.callback;
        i && delete e.callback, this.toCanvas(e).toBlob((s) => {
          r(s), i == null || i(s);
        });
      } catch (i) {
        n(i);
      }
    });
  }
  setSize(e) {
    return this.width(e.width), this.height(e.height), this;
  }
  getSize() {
    return {
      width: this.width(),
      height: this.height()
    };
  }
  getClassName() {
    return this.className || this.nodeType;
  }
  getType() {
    return this.nodeType;
  }
  getDragDistance() {
    return this.attrs.dragDistance !== void 0 ? this.attrs.dragDistance : this.parent ? this.parent.getDragDistance() : ve.dragDistance;
  }
  _off(e, r, n) {
    var i = this.eventListeners[e], s, o, l;
    for (s = 0; s < i.length; s++)
      if (o = i[s].name, l = i[s].handler, (o !== "konva" || r === "konva") && (!r || o === r) && (!n || n === l)) {
        if (i.splice(s, 1), i.length === 0) {
          delete this.eventListeners[e];
          break;
        }
        s--;
      }
  }
  _fireChangeEvent(e, r, n) {
    this._fire(e + TR, {
      oldVal: r,
      newVal: n
    });
  }
  addName(e) {
    if (!this.hasName(e)) {
      var r = this.name(), n = r ? r + " " + e : e;
      this.name(n);
    }
    return this;
  }
  hasName(e) {
    if (!e)
      return !1;
    const r = this.name();
    if (!r)
      return !1;
    var n = (r || "").split(/\s/g);
    return n.indexOf(e) !== -1;
  }
  removeName(e) {
    var r = (this.name() || "").split(/\s/g), n = r.indexOf(e);
    return n !== -1 && (r.splice(n, 1), this.name(r.join(" "))), this;
  }
  setAttr(e, r) {
    var n = this[Fm + W._capitalize(e)];
    return W._isFunction(n) ? n.call(this, r) : this._setAttr(e, r), this;
  }
  _requestDraw() {
    if (ve.autoDrawEnabled) {
      const e = this.getLayer() || this.getStage();
      e == null || e.batchDraw();
    }
  }
  _setAttr(e, r) {
    var n = this.attrs[e];
    n === r && !W.isObject(r) || (r == null ? delete this.attrs[e] : this.attrs[e] = r, this._shouldFireChangeEvents && this._fireChangeEvent(e, n, r), this._requestDraw());
  }
  _setComponentAttr(e, r, n) {
    var i;
    n !== void 0 && (i = this.attrs[e], i || (this.attrs[e] = this.getAttr(e)), this.attrs[e][r] = n, this._fireChangeEvent(e, i, n));
  }
  _fireAndBubble(e, r, n) {
    r && this.nodeType === Gm && (r.target = this);
    var i = (e === Nm || e === Bm) && (n && (this === n || this.isAncestorOf && this.isAncestorOf(n)) || this.nodeType === "Stage" && !n);
    if (!i) {
      this._fire(e, r);
      var s = (e === Nm || e === Bm) && n && n.isAncestorOf && n.isAncestorOf(this) && !n.isAncestorOf(this.parent);
      (r && !r.cancelBubble || !r) && this.parent && this.parent.isListening() && !s && (n && n.parent ? this._fireAndBubble.call(this.parent, e, r, n) : this._fireAndBubble.call(this.parent, e, r));
    }
  }
  _getProtoListeners(e) {
    let r = this._cache.get(Nu);
    if (!r) {
      r = {};
      let i = Object.getPrototypeOf(this);
      for (; i; ) {
        if (!i.eventListeners) {
          i = Object.getPrototypeOf(i);
          continue;
        }
        for (var n in i.eventListeners) {
          const s = i.eventListeners[n], o = r[n] || [];
          r[n] = s.concat(o);
        }
        i = Object.getPrototypeOf(i);
      }
      this._cache.set(Nu, r);
    }
    return r[e];
  }
  _fire(e, r) {
    r = r || {}, r.currentTarget = this, r.type = e;
    const n = this._getProtoListeners(e);
    if (n)
      for (var i = 0; i < n.length; i++)
        n[i].handler.call(this, r);
    const s = this.eventListeners[e];
    if (s)
      for (var i = 0; i < s.length; i++)
        s[i].handler.call(this, r);
  }
  draw() {
    return this.drawScene(), this.drawHit(), this;
  }
  _createDragElement(e) {
    var r = e ? e.pointerId : void 0, n = this.getStage(), i = this.getAbsolutePosition(), s = n._getPointerById(r) || n._changedPointerPositions[0] || i;
    ze._dragElements.set(this._id, {
      node: this,
      startPointerPos: s,
      offset: {
        x: s.x - i.x,
        y: s.y - i.y
      },
      dragStatus: "ready",
      pointerId: r
    });
  }
  startDrag(e, r = !0) {
    ze._dragElements.has(this._id) || this._createDragElement(e);
    const n = ze._dragElements.get(this._id);
    n.dragStatus = "dragging", this.fire("dragstart", {
      type: "dragstart",
      target: this,
      evt: e && e.evt
    }, r);
  }
  _setDragPosition(e, r) {
    const n = this.getStage()._getPointerById(r.pointerId);
    if (n) {
      var i = {
        x: n.x - r.offset.x,
        y: n.y - r.offset.y
      }, s = this.dragBoundFunc();
      if (s !== void 0) {
        const o = s.call(this, i, e);
        o ? i = o : W.warn("dragBoundFunc did not return any value. That is unexpected behavior. You must return new absolute position from dragBoundFunc.");
      }
      (!this._lastPos || this._lastPos.x !== i.x || this._lastPos.y !== i.y) && (this.setAbsolutePosition(i), this._requestDraw()), this._lastPos = i;
    }
  }
  stopDrag(e) {
    const r = ze._dragElements.get(this._id);
    r && (r.dragStatus = "stopped"), ze._endDragBefore(e), ze._endDragAfter(e);
  }
  setDraggable(e) {
    this._setAttr("draggable", e), this._dragChange();
  }
  isDragging() {
    const e = ze._dragElements.get(this._id);
    return e ? e.dragStatus === "dragging" : !1;
  }
  _listenDrag() {
    this._dragCleanup(), this.on("mousedown.konva touchstart.konva", function(e) {
      var r = e.evt.button !== void 0, n = !r || ve.dragButtons.indexOf(e.evt.button) >= 0;
      if (n && !this.isDragging()) {
        var i = !1;
        ze._dragElements.forEach((s) => {
          this.isAncestorOf(s.node) && (i = !0);
        }), i || this._createDragElement(e);
      }
    });
  }
  _dragChange() {
    if (this.attrs.draggable)
      this._listenDrag();
    else {
      this._dragCleanup();
      var e = this.getStage();
      if (!e)
        return;
      const r = ze._dragElements.get(this._id), n = r && r.dragStatus === "dragging", i = r && r.dragStatus === "ready";
      n ? this.stopDrag() : i && ze._dragElements.delete(this._id);
    }
  }
  _dragCleanup() {
    this.off("mousedown.konva"), this.off("touchstart.konva");
  }
  isClientRectOnScreen(e = { x: 0, y: 0 }) {
    const r = this.getStage();
    if (!r)
      return !1;
    const n = {
      x: -e.x,
      y: -e.y,
      width: r.width() + 2 * e.x,
      height: r.height() + 2 * e.y
    };
    return W.haveIntersection(n, this.getClientRect());
  }
  static create(e, r) {
    return W._isString(e) && (e = JSON.parse(e)), this._createNode(e, r);
  }
  static _createNode(e, r) {
    var n = ue.prototype.getClassName.call(e), i = e.children, s, o, l;
    r && (e.attrs.container = r), ve[n] || (W.warn('Can not find a node with class name "' + n + '". Fallback to "Shape".'), n = "Shape");
    const c = ve[n];
    if (s = new c(e.attrs), i)
      for (o = i.length, l = 0; l < o; l++)
        s.add(ue._createNode(i[l]));
    return s;
  }
}
ue.prototype.nodeType = "Node";
ue.prototype._attrsAffectingSize = [];
ue.prototype.eventListeners = {};
ue.prototype.on.call(ue.prototype, kR, function() {
  if (this._batchingTransformChange) {
    this._needClearTransformCache = !0;
    return;
  }
  this._clearCache(yi), this._clearSelfAndDescendantCache(Jn);
});
ue.prototype.on.call(ue.prototype, "visibleChange.konva", function() {
  this._clearSelfAndDescendantCache(rh);
});
ue.prototype.on.call(ue.prototype, "listeningChange.konva", function() {
  this._clearSelfAndDescendantCache(th);
});
ue.prototype.on.call(ue.prototype, "opacityChange.konva", function() {
  this._clearSelfAndDescendantCache(nl);
});
const ct = I.addGetterSetter;
ct(ue, "zIndex");
ct(ue, "absolutePosition");
ct(ue, "position");
ct(ue, "x", 0, le());
ct(ue, "y", 0, le());
ct(ue, "globalCompositeOperation", "source-over", es());
ct(ue, "opacity", 1, le());
ct(ue, "name", "", es());
ct(ue, "id", "", es());
ct(ue, "rotation", 0, le());
I.addComponentsGetterSetter(ue, "scale", ["x", "y"]);
ct(ue, "scaleX", 1, le());
ct(ue, "scaleY", 1, le());
I.addComponentsGetterSetter(ue, "skew", ["x", "y"]);
ct(ue, "skewX", 0, le());
ct(ue, "skewY", 0, le());
I.addComponentsGetterSetter(ue, "offset", ["x", "y"]);
ct(ue, "offsetX", 0, le());
ct(ue, "offsetY", 0, le());
ct(ue, "dragDistance", null, le());
ct(ue, "width", 0, le());
ct(ue, "height", 0, le());
ct(ue, "listening", !0, hn());
ct(ue, "preventDefault", !0, hn());
ct(ue, "filters", null, function(t) {
  return this._filterUpToDate = !1, t;
});
ct(ue, "visible", !0, hn());
ct(ue, "transformsEnabled", "all", es());
ct(ue, "size");
ct(ue, "dragBoundFunc");
ct(ue, "draggable", !1, hn());
I.backCompat(ue, {
  rotateDeg: "rotate",
  setRotationDeg: "setRotation",
  getRotationDeg: "getRotation"
});
class Lr extends ue {
  constructor() {
    super(...arguments), this.children = [];
  }
  getChildren(e) {
    if (!e)
      return this.children || [];
    const r = this.children || [];
    var n = [];
    return r.forEach(function(i) {
      e(i) && n.push(i);
    }), n;
  }
  hasChildren() {
    return this.getChildren().length > 0;
  }
  removeChildren() {
    return this.getChildren().forEach((e) => {
      e.parent = null, e.index = 0, e.remove();
    }), this.children = [], this._requestDraw(), this;
  }
  destroyChildren() {
    return this.getChildren().forEach((e) => {
      e.parent = null, e.index = 0, e.destroy();
    }), this.children = [], this._requestDraw(), this;
  }
  add(...e) {
    if (e.length === 0)
      return this;
    if (e.length > 1) {
      for (var r = 0; r < e.length; r++)
        this.add(e[r]);
      return this;
    }
    const n = e[0];
    return n.getParent() ? (n.moveTo(this), this) : (this._validateAdd(n), n.index = this.getChildren().length, n.parent = this, n._clearCaches(), this.getChildren().push(n), this._fire("add", {
      child: n
    }), this._requestDraw(), this);
  }
  destroy() {
    return this.hasChildren() && this.destroyChildren(), super.destroy(), this;
  }
  find(e) {
    return this._generalFind(e, !1);
  }
  findOne(e) {
    var r = this._generalFind(e, !0);
    return r.length > 0 ? r[0] : void 0;
  }
  _generalFind(e, r) {
    var n = [];
    return this._descendants((i) => {
      const s = i._isMatch(e);
      return s && n.push(i), !!(s && r);
    }), n;
  }
  _descendants(e) {
    let r = !1;
    const n = this.getChildren();
    for (const i of n) {
      if (r = e(i), r)
        return !0;
      if (i.hasChildren() && (r = i._descendants(e), r))
        return !0;
    }
    return !1;
  }
  toObject() {
    var e = ue.prototype.toObject.call(this);
    return e.children = [], this.getChildren().forEach((r) => {
      e.children.push(r.toObject());
    }), e;
  }
  isAncestorOf(e) {
    for (var r = e.getParent(); r; ) {
      if (r._id === this._id)
        return !0;
      r = r.getParent();
    }
    return !1;
  }
  clone(e) {
    var r = ue.prototype.clone.call(this, e);
    return this.getChildren().forEach(function(n) {
      r.add(n.clone());
    }), r;
  }
  getAllIntersections(e) {
    var r = [];
    return this.find("Shape").forEach(function(n) {
      n.isVisible() && n.intersects(e) && r.push(n);
    }), r;
  }
  _clearSelfAndDescendantCache(e) {
    var r;
    super._clearSelfAndDescendantCache(e), !this.isCached() && ((r = this.children) === null || r === void 0 || r.forEach(function(n) {
      n._clearSelfAndDescendantCache(e);
    }));
  }
  _setChildrenIndices() {
    var e;
    (e = this.children) === null || e === void 0 || e.forEach(function(r, n) {
      r.index = n;
    }), this._requestDraw();
  }
  drawScene(e, r) {
    var n = this.getLayer(), i = e || n && n.getCanvas(), s = i && i.getContext(), o = this._getCanvasCache(), l = o && o.scene, c = i && i.isCache;
    if (!this.isVisible() && !c)
      return this;
    if (l) {
      s.save();
      var h = this.getAbsoluteTransform(r).getMatrix();
      s.transform(h[0], h[1], h[2], h[3], h[4], h[5]), this._drawCachedSceneCanvas(s), s.restore();
    } else
      this._drawChildren("drawScene", i, r);
    return this;
  }
  drawHit(e, r) {
    if (!this.shouldDrawHit(r))
      return this;
    var n = this.getLayer(), i = e || n && n.hitCanvas, s = i && i.getContext(), o = this._getCanvasCache(), l = o && o.hit;
    if (l) {
      s.save();
      var c = this.getAbsoluteTransform(r).getMatrix();
      s.transform(c[0], c[1], c[2], c[3], c[4], c[5]), this._drawCachedHitCanvas(s), s.restore();
    } else
      this._drawChildren("drawHit", i, r);
    return this;
  }
  _drawChildren(e, r, n) {
    var i, s = r && r.getContext(), o = this.clipWidth(), l = this.clipHeight(), c = this.clipFunc(), h = o && l || c;
    const d = n === this;
    if (h) {
      s.save();
      var g = this.getAbsoluteTransform(n), p = g.getMatrix();
      if (s.transform(p[0], p[1], p[2], p[3], p[4], p[5]), s.beginPath(), c)
        c.call(this, s, this);
      else {
        var y = this.clipX(), v = this.clipY();
        s.rect(y, v, o, l);
      }
      s.clip(), p = g.copy().invert().getMatrix(), s.transform(p[0], p[1], p[2], p[3], p[4], p[5]);
    }
    var _ = !d && this.globalCompositeOperation() !== "source-over" && e === "drawScene";
    _ && (s.save(), s._applyGlobalCompositeOperation(this)), (i = this.children) === null || i === void 0 || i.forEach(function(b) {
      b[e](r, n);
    }), _ && s.restore(), h && s.restore();
  }
  getClientRect(e) {
    var r;
    e = e || {};
    var n = e.skipTransform, i = e.relativeTo, s, o, l, c, h = {
      x: 1 / 0,
      y: 1 / 0,
      width: 0,
      height: 0
    }, d = this;
    (r = this.children) === null || r === void 0 || r.forEach(function(_) {
      if (_.visible()) {
        var b = _.getClientRect({
          relativeTo: d,
          skipShadow: e.skipShadow,
          skipStroke: e.skipStroke
        });
        b.width === 0 && b.height === 0 || (s === void 0 ? (s = b.x, o = b.y, l = b.x + b.width, c = b.y + b.height) : (s = Math.min(s, b.x), o = Math.min(o, b.y), l = Math.max(l, b.x + b.width), c = Math.max(c, b.y + b.height)));
      }
    });
    for (var g = this.find("Shape"), p = !1, y = 0; y < g.length; y++) {
      var v = g[y];
      if (v._isVisible(this)) {
        p = !0;
        break;
      }
    }
    return p && s !== void 0 ? h = {
      x: s,
      y: o,
      width: l - s,
      height: c - o
    } : h = {
      x: 0,
      y: 0,
      width: 0,
      height: 0
    }, n ? h : this._transformedRect(h, i);
  }
}
I.addComponentsGetterSetter(Lr, "clip", [
  "x",
  "y",
  "width",
  "height"
]);
I.addGetterSetter(Lr, "clipX", void 0, le());
I.addGetterSetter(Lr, "clipY", void 0, le());
I.addGetterSetter(Lr, "clipWidth", void 0, le());
I.addGetterSetter(Lr, "clipHeight", void 0, le());
I.addGetterSetter(Lr, "clipFunc");
const ho = /* @__PURE__ */ new Map(), a0 = ve._global.PointerEvent !== void 0;
function Ef(t) {
  return ho.get(t);
}
function Bh(t) {
  return {
    evt: t,
    pointerId: t.pointerId
  };
}
function s0(t, e) {
  return ho.get(t) === e;
}
function o0(t, e) {
  ro(t), e.getStage() && (ho.set(t, e), a0 && e._fire("gotpointercapture", Bh(new PointerEvent("gotpointercapture"))));
}
function ro(t, e) {
  const r = ho.get(t);
  if (!r)
    return;
  const n = r.getStage();
  n && n.content, ho.delete(t), a0 && r._fire("lostpointercapture", Bh(new PointerEvent("lostpointercapture")));
}
var LR = "Stage", DR = "string", $m = "px", MR = "mouseout", u0 = "mouseleave", l0 = "mouseover", c0 = "mouseenter", f0 = "mousemove", h0 = "mousedown", d0 = "mouseup", Ks = "pointermove", Xs = "pointerdown", Wa = "pointerup", Js = "pointercancel", NR = "lostpointercapture", Bu = "pointerout", Fu = "pointerleave", Gu = "pointerover", ju = "pointerenter", nh = "contextmenu", g0 = "touchstart", p0 = "touchend", m0 = "touchmove", v0 = "touchcancel", ih = "wheel", BR = 5, FR = [
  [c0, "_pointerenter"],
  [h0, "_pointerdown"],
  [f0, "_pointermove"],
  [d0, "_pointerup"],
  [u0, "_pointerleave"],
  [g0, "_pointerdown"],
  [m0, "_pointermove"],
  [p0, "_pointerup"],
  [v0, "_pointercancel"],
  [l0, "_pointerover"],
  [ih, "_wheel"],
  [nh, "_contextmenu"],
  [Xs, "_pointerdown"],
  [Ks, "_pointermove"],
  [Wa, "_pointerup"],
  [Js, "_pointercancel"],
  [NR, "_lostpointercapture"]
];
const Cf = {
  mouse: {
    [Bu]: MR,
    [Fu]: u0,
    [Gu]: l0,
    [ju]: c0,
    [Ks]: f0,
    [Xs]: h0,
    [Wa]: d0,
    [Js]: "mousecancel",
    pointerclick: "click",
    pointerdblclick: "dblclick"
  },
  touch: {
    [Bu]: "touchout",
    [Fu]: "touchleave",
    [Gu]: "touchover",
    [ju]: "touchenter",
    [Ks]: m0,
    [Xs]: g0,
    [Wa]: p0,
    [Js]: v0,
    pointerclick: "tap",
    pointerdblclick: "dbltap"
  },
  pointer: {
    [Bu]: Bu,
    [Fu]: Fu,
    [Gu]: Gu,
    [ju]: ju,
    [Ks]: Ks,
    [Xs]: Xs,
    [Wa]: Wa,
    [Js]: Js,
    pointerclick: "pointerclick",
    pointerdblclick: "pointerdblclick"
  }
}, Qs = (t) => t.indexOf("pointer") >= 0 ? "pointer" : t.indexOf("touch") >= 0 ? "touch" : "mouse", Ga = (t) => {
  const e = Qs(t);
  if (e === "pointer")
    return ve.pointerEventsEnabled && Cf.pointer;
  if (e === "touch")
    return Cf.touch;
  if (e === "mouse")
    return Cf.mouse;
};
function Wm(t = {}) {
  return (t.clipFunc || t.clipWidth || t.clipHeight) && W.warn("Stage does not support clipping. Please use clip for Layers or Groups."), t;
}
const GR = "Pointer position is missing and not registered by the stage. Looks like it is outside of the stage container. You can set it manually from event: stage.setPointersPositions(event);", al = [];
class Ml extends Lr {
  constructor(e) {
    super(Wm(e)), this._pointerPositions = [], this._changedPointerPositions = [], this._buildDOM(), this._bindContentEvents(), al.push(this), this.on("widthChange.konva heightChange.konva", this._resizeDOM), this.on("visibleChange.konva", this._checkVisibility), this.on("clipWidthChange.konva clipHeightChange.konva clipFuncChange.konva", () => {
      Wm(this.attrs);
    }), this._checkVisibility();
  }
  _validateAdd(e) {
    const r = e.getType() === "Layer", n = e.getType() === "FastLayer";
    r || n || W.throw("You may only add layers to the stage.");
  }
  _checkVisibility() {
    if (!this.content)
      return;
    const e = this.visible() ? "" : "none";
    this.content.style.display = e;
  }
  setContainer(e) {
    if (typeof e === DR) {
      if (e.charAt(0) === ".") {
        var r = e.slice(1);
        e = document.getElementsByClassName(r)[0];
      } else {
        var n;
        e.charAt(0) !== "#" ? n = e : n = e.slice(1), e = document.getElementById(n);
      }
      if (!e)
        throw "Can not find container in document with id " + n;
    }
    return this._setAttr("container", e), this.content && (this.content.parentElement && this.content.parentElement.removeChild(this.content), e.appendChild(this.content)), this;
  }
  shouldDrawHit() {
    return !0;
  }
  clear() {
    var e = this.children, r = e.length, n;
    for (n = 0; n < r; n++)
      e[n].clear();
    return this;
  }
  clone(e) {
    return e || (e = {}), e.container = typeof document < "u" && document.createElement("div"), Lr.prototype.clone.call(this, e);
  }
  destroy() {
    super.destroy();
    var e = this.content;
    e && W._isInDocument(e) && this.container().removeChild(e);
    var r = al.indexOf(this);
    return r > -1 && al.splice(r, 1), W.releaseCanvas(this.bufferCanvas._canvas, this.bufferHitCanvas._canvas), this;
  }
  getPointerPosition() {
    const e = this._pointerPositions[0] || this._changedPointerPositions[0];
    return e ? {
      x: e.x,
      y: e.y
    } : (W.warn(GR), null);
  }
  _getPointerById(e) {
    return this._pointerPositions.find((r) => r.id === e);
  }
  getPointersPositions() {
    return this._pointerPositions;
  }
  getStage() {
    return this;
  }
  getContent() {
    return this.content;
  }
  _toKonvaCanvas(e) {
    e = e || {}, e.x = e.x || 0, e.y = e.y || 0, e.width = e.width || this.width(), e.height = e.height || this.height();
    var r = new Va({
      width: e.width,
      height: e.height,
      pixelRatio: e.pixelRatio || 1
    }), n = r.getContext()._context, i = this.children;
    return (e.x || e.y) && n.translate(-1 * e.x, -1 * e.y), i.forEach(function(s) {
      if (s.isVisible()) {
        var o = s._toKonvaCanvas(e);
        n.drawImage(o._canvas, e.x, e.y, o.getWidth() / o.getPixelRatio(), o.getHeight() / o.getPixelRatio());
      }
    }), r;
  }
  getIntersection(e) {
    if (!e)
      return null;
    var r = this.children, n = r.length, i = n - 1, s;
    for (s = i; s >= 0; s--) {
      const o = r[s].getIntersection(e);
      if (o)
        return o;
    }
    return null;
  }
  _resizeDOM() {
    var e = this.width(), r = this.height();
    this.content && (this.content.style.width = e + $m, this.content.style.height = r + $m), this.bufferCanvas.setSize(e, r), this.bufferHitCanvas.setSize(e, r), this.children.forEach((n) => {
      n.setSize({ width: e, height: r }), n.draw();
    });
  }
  add(e, ...r) {
    if (arguments.length > 1) {
      for (var n = 0; n < arguments.length; n++)
        this.add(arguments[n]);
      return this;
    }
    super.add(e);
    var i = this.children.length;
    return i > BR && W.warn("The stage has " + i + " layers. Recommended maximum number of layers is 3-5. Adding more layers into the stage may drop the performance. Rethink your tree structure, you can use Konva.Group."), e.setSize({ width: this.width(), height: this.height() }), e.draw(), ve.isBrowser && this.content.appendChild(e.canvas._canvas), this;
  }
  getParent() {
    return null;
  }
  getLayer() {
    return null;
  }
  hasPointerCapture(e) {
    return s0(e, this);
  }
  setPointerCapture(e) {
    o0(e, this);
  }
  releaseCapture(e) {
    ro(e);
  }
  getLayers() {
    return this.children;
  }
  _bindContentEvents() {
    ve.isBrowser && FR.forEach(([e, r]) => {
      this.content.addEventListener(e, (n) => {
        this[r](n);
      }, { passive: !1 });
    });
  }
  _pointerenter(e) {
    this.setPointersPositions(e);
    const r = Ga(e.type);
    this._fire(r.pointerenter, {
      evt: e,
      target: this,
      currentTarget: this
    });
  }
  _pointerover(e) {
    this.setPointersPositions(e);
    const r = Ga(e.type);
    this._fire(r.pointerover, {
      evt: e,
      target: this,
      currentTarget: this
    });
  }
  _getTargetShape(e) {
    let r = this[e + "targetShape"];
    return r && !r.getStage() && (r = null), r;
  }
  _pointerleave(e) {
    const r = Ga(e.type), n = Qs(e.type);
    if (r) {
      this.setPointersPositions(e);
      var i = this._getTargetShape(n), s = !ze.isDragging || ve.hitOnDragEnabled;
      i && s ? (i._fireAndBubble(r.pointerout, { evt: e }), i._fireAndBubble(r.pointerleave, { evt: e }), this._fire(r.pointerleave, {
        evt: e,
        target: this,
        currentTarget: this
      }), this[n + "targetShape"] = null) : s && (this._fire(r.pointerleave, {
        evt: e,
        target: this,
        currentTarget: this
      }), this._fire(r.pointerout, {
        evt: e,
        target: this,
        currentTarget: this
      })), this.pointerPos = void 0, this._pointerPositions = [];
    }
  }
  _pointerdown(e) {
    const r = Ga(e.type), n = Qs(e.type);
    if (r) {
      this.setPointersPositions(e);
      var i = !1;
      this._changedPointerPositions.forEach((s) => {
        var o = this.getIntersection(s);
        if (ze.justDragged = !1, ve["_" + n + "ListenClick"] = !0, !(o && o.isListening()))
          return;
        ve.capturePointerEventsEnabled && o.setPointerCapture(s.id), this[n + "ClickStartShape"] = o, o._fireAndBubble(r.pointerdown, {
          evt: e,
          pointerId: s.id
        }), i = !0;
        const c = e.type.indexOf("touch") >= 0;
        o.preventDefault() && e.cancelable && c && e.preventDefault();
      }), i || this._fire(r.pointerdown, {
        evt: e,
        target: this,
        currentTarget: this,
        pointerId: this._pointerPositions[0].id
      });
    }
  }
  _pointermove(e) {
    const r = Ga(e.type), n = Qs(e.type);
    if (!r)
      return;
    ze.isDragging && ze.node.preventDefault() && e.cancelable && e.preventDefault(), this.setPointersPositions(e);
    var i = !ze.isDragging || ve.hitOnDragEnabled;
    if (!i)
      return;
    var s = {};
    let o = !1;
    var l = this._getTargetShape(n);
    this._changedPointerPositions.forEach((c) => {
      const h = Ef(c.id) || this.getIntersection(c), d = c.id, g = { evt: e, pointerId: d };
      var p = l !== h;
      if (p && l && (l._fireAndBubble(r.pointerout, Object.assign({}, g), h), l._fireAndBubble(r.pointerleave, Object.assign({}, g), h)), h) {
        if (s[h._id])
          return;
        s[h._id] = !0;
      }
      h && h.isListening() ? (o = !0, p && (h._fireAndBubble(r.pointerover, Object.assign({}, g), l), h._fireAndBubble(r.pointerenter, Object.assign({}, g), l), this[n + "targetShape"] = h), h._fireAndBubble(r.pointermove, Object.assign({}, g))) : l && (this._fire(r.pointerover, {
        evt: e,
        target: this,
        currentTarget: this,
        pointerId: d
      }), this[n + "targetShape"] = null);
    }), o || this._fire(r.pointermove, {
      evt: e,
      target: this,
      currentTarget: this,
      pointerId: this._changedPointerPositions[0].id
    });
  }
  _pointerup(e) {
    const r = Ga(e.type), n = Qs(e.type);
    if (!r)
      return;
    this.setPointersPositions(e);
    const i = this[n + "ClickStartShape"], s = this[n + "ClickEndShape"];
    var o = {};
    let l = !1;
    this._changedPointerPositions.forEach((c) => {
      const h = Ef(c.id) || this.getIntersection(c);
      if (h) {
        if (h.releaseCapture(c.id), o[h._id])
          return;
        o[h._id] = !0;
      }
      const d = c.id, g = { evt: e, pointerId: d };
      let p = !1;
      ve["_" + n + "InDblClickWindow"] ? (p = !0, clearTimeout(this[n + "DblTimeout"])) : ze.justDragged || (ve["_" + n + "InDblClickWindow"] = !0, clearTimeout(this[n + "DblTimeout"])), this[n + "DblTimeout"] = setTimeout(function() {
        ve["_" + n + "InDblClickWindow"] = !1;
      }, ve.dblClickWindow), h && h.isListening() ? (l = !0, this[n + "ClickEndShape"] = h, h._fireAndBubble(r.pointerup, Object.assign({}, g)), ve["_" + n + "ListenClick"] && i && i === h && (h._fireAndBubble(r.pointerclick, Object.assign({}, g)), p && s && s === h && h._fireAndBubble(r.pointerdblclick, Object.assign({}, g)))) : (this[n + "ClickEndShape"] = null, ve["_" + n + "ListenClick"] && this._fire(r.pointerclick, {
        evt: e,
        target: this,
        currentTarget: this,
        pointerId: d
      }), p && this._fire(r.pointerdblclick, {
        evt: e,
        target: this,
        currentTarget: this,
        pointerId: d
      }));
    }), l || this._fire(r.pointerup, {
      evt: e,
      target: this,
      currentTarget: this,
      pointerId: this._changedPointerPositions[0].id
    }), ve["_" + n + "ListenClick"] = !1, e.cancelable && n !== "touch" && e.preventDefault();
  }
  _contextmenu(e) {
    this.setPointersPositions(e);
    var r = this.getIntersection(this.getPointerPosition());
    r && r.isListening() ? r._fireAndBubble(nh, { evt: e }) : this._fire(nh, {
      evt: e,
      target: this,
      currentTarget: this
    });
  }
  _wheel(e) {
    this.setPointersPositions(e);
    var r = this.getIntersection(this.getPointerPosition());
    r && r.isListening() ? r._fireAndBubble(ih, { evt: e }) : this._fire(ih, {
      evt: e,
      target: this,
      currentTarget: this
    });
  }
  _pointercancel(e) {
    this.setPointersPositions(e);
    const r = Ef(e.pointerId) || this.getIntersection(this.getPointerPosition());
    r && r._fireAndBubble(Wa, Bh(e)), ro(e.pointerId);
  }
  _lostpointercapture(e) {
    ro(e.pointerId);
  }
  setPointersPositions(e) {
    var r = this._getContentPosition(), n = null, i = null;
    e = e || window.event, e.touches !== void 0 ? (this._pointerPositions = [], this._changedPointerPositions = [], Array.prototype.forEach.call(e.touches, (s) => {
      this._pointerPositions.push({
        id: s.identifier,
        x: (s.clientX - r.left) / r.scaleX,
        y: (s.clientY - r.top) / r.scaleY
      });
    }), Array.prototype.forEach.call(e.changedTouches || e.touches, (s) => {
      this._changedPointerPositions.push({
        id: s.identifier,
        x: (s.clientX - r.left) / r.scaleX,
        y: (s.clientY - r.top) / r.scaleY
      });
    })) : (n = (e.clientX - r.left) / r.scaleX, i = (e.clientY - r.top) / r.scaleY, this.pointerPos = {
      x: n,
      y: i
    }, this._pointerPositions = [{ x: n, y: i, id: W._getFirstPointerId(e) }], this._changedPointerPositions = [
      { x: n, y: i, id: W._getFirstPointerId(e) }
    ]);
  }
  _setPointerPosition(e) {
    W.warn('Method _setPointerPosition is deprecated. Use "stage.setPointersPositions(event)" instead.'), this.setPointersPositions(e);
  }
  _getContentPosition() {
    if (!this.content || !this.content.getBoundingClientRect)
      return {
        top: 0,
        left: 0,
        scaleX: 1,
        scaleY: 1
      };
    var e = this.content.getBoundingClientRect();
    return {
      top: e.top,
      left: e.left,
      scaleX: e.width / this.content.clientWidth || 1,
      scaleY: e.height / this.content.clientHeight || 1
    };
  }
  _buildDOM() {
    if (this.bufferCanvas = new Va({
      width: this.width(),
      height: this.height()
    }), this.bufferHitCanvas = new Nh({
      pixelRatio: 1,
      width: this.width(),
      height: this.height()
    }), !!ve.isBrowser) {
      var e = this.container();
      if (!e)
        throw "Stage has no container. A container is required.";
      e.innerHTML = "", this.content = document.createElement("div"), this.content.style.position = "relative", this.content.style.userSelect = "none", this.content.className = "konvajs-content", this.content.setAttribute("role", "presentation"), e.appendChild(this.content), this._resizeDOM();
    }
  }
  cache() {
    return W.warn("Cache function is not allowed for stage. You may use cache only for layers, groups and shapes."), this;
  }
  clearCache() {
    return this;
  }
  batchDraw() {
    return this.getChildren().forEach(function(e) {
      e.batchDraw();
    }), this;
  }
}
Ml.prototype.nodeType = LR;
St(Ml);
I.addGetterSetter(Ml, "container");
var y0 = "hasShadow", b0 = "shadowRGBA", _0 = "patternImage", S0 = "linearGradient", w0 = "radialGradient";
let $u;
function xf() {
  return $u || ($u = W.createCanvasElement().getContext("2d"), $u);
}
const no = {};
function jR(t) {
  t.fill();
}
function $R(t) {
  t.stroke();
}
function WR(t) {
  t.fill();
}
function UR(t) {
  t.stroke();
}
function HR() {
  this._clearCache(y0);
}
function zR() {
  this._clearCache(b0);
}
function VR() {
  this._clearCache(_0);
}
function qR() {
  this._clearCache(S0);
}
function YR() {
  this._clearCache(w0);
}
class se extends ue {
  constructor(e) {
    super(e);
    let r;
    for (; r = W.getRandomColor(), !(r && !(r in no)); )
      ;
    this.colorKey = r, no[r] = this;
  }
  getContext() {
    return W.warn("shape.getContext() method is deprecated. Please do not use it."), this.getLayer().getContext();
  }
  getCanvas() {
    return W.warn("shape.getCanvas() method is deprecated. Please do not use it."), this.getLayer().getCanvas();
  }
  getSceneFunc() {
    return this.attrs.sceneFunc || this._sceneFunc;
  }
  getHitFunc() {
    return this.attrs.hitFunc || this._hitFunc;
  }
  hasShadow() {
    return this._getCache(y0, this._hasShadow);
  }
  _hasShadow() {
    return this.shadowEnabled() && this.shadowOpacity() !== 0 && !!(this.shadowColor() || this.shadowBlur() || this.shadowOffsetX() || this.shadowOffsetY());
  }
  _getFillPattern() {
    return this._getCache(_0, this.__getFillPattern);
  }
  __getFillPattern() {
    if (this.fillPatternImage()) {
      var e = xf();
      const r = e.createPattern(this.fillPatternImage(), this.fillPatternRepeat() || "repeat");
      if (r && r.setTransform) {
        const n = new Ir();
        n.translate(this.fillPatternX(), this.fillPatternY()), n.rotate(ve.getAngle(this.fillPatternRotation())), n.scale(this.fillPatternScaleX(), this.fillPatternScaleY()), n.translate(-1 * this.fillPatternOffsetX(), -1 * this.fillPatternOffsetY());
        const i = n.getMatrix(), s = typeof DOMMatrix > "u" ? {
          a: i[0],
          b: i[1],
          c: i[2],
          d: i[3],
          e: i[4],
          f: i[5]
        } : new DOMMatrix(i);
        r.setTransform(s);
      }
      return r;
    }
  }
  _getLinearGradient() {
    return this._getCache(S0, this.__getLinearGradient);
  }
  __getLinearGradient() {
    var e = this.fillLinearGradientColorStops();
    if (e) {
      for (var r = xf(), n = this.fillLinearGradientStartPoint(), i = this.fillLinearGradientEndPoint(), s = r.createLinearGradient(n.x, n.y, i.x, i.y), o = 0; o < e.length; o += 2)
        s.addColorStop(e[o], e[o + 1]);
      return s;
    }
  }
  _getRadialGradient() {
    return this._getCache(w0, this.__getRadialGradient);
  }
  __getRadialGradient() {
    var e = this.fillRadialGradientColorStops();
    if (e) {
      for (var r = xf(), n = this.fillRadialGradientStartPoint(), i = this.fillRadialGradientEndPoint(), s = r.createRadialGradient(n.x, n.y, this.fillRadialGradientStartRadius(), i.x, i.y, this.fillRadialGradientEndRadius()), o = 0; o < e.length; o += 2)
        s.addColorStop(e[o], e[o + 1]);
      return s;
    }
  }
  getShadowRGBA() {
    return this._getCache(b0, this._getShadowRGBA);
  }
  _getShadowRGBA() {
    if (this.hasShadow()) {
      var e = W.colorToRGBA(this.shadowColor());
      if (e)
        return "rgba(" + e.r + "," + e.g + "," + e.b + "," + e.a * (this.shadowOpacity() || 1) + ")";
    }
  }
  hasFill() {
    return this._calculate("hasFill", [
      "fillEnabled",
      "fill",
      "fillPatternImage",
      "fillLinearGradientColorStops",
      "fillRadialGradientColorStops"
    ], () => this.fillEnabled() && !!(this.fill() || this.fillPatternImage() || this.fillLinearGradientColorStops() || this.fillRadialGradientColorStops()));
  }
  hasStroke() {
    return this._calculate("hasStroke", [
      "strokeEnabled",
      "strokeWidth",
      "stroke",
      "strokeLinearGradientColorStops"
    ], () => this.strokeEnabled() && this.strokeWidth() && !!(this.stroke() || this.strokeLinearGradientColorStops()));
  }
  hasHitStroke() {
    const e = this.hitStrokeWidth();
    return e === "auto" ? this.hasStroke() : this.strokeEnabled() && !!e;
  }
  intersects(e) {
    var r = this.getStage(), n = r.bufferHitCanvas, i;
    return n.getContext().clear(), this.drawHit(n, null, !0), i = n.context.getImageData(Math.round(e.x), Math.round(e.y), 1, 1).data, i[3] > 0;
  }
  destroy() {
    return ue.prototype.destroy.call(this), delete no[this.colorKey], delete this.colorKey, this;
  }
  _useBufferCanvas(e) {
    var r;
    if (!this.getStage() || !((r = this.attrs.perfectDrawEnabled) !== null && r !== void 0 ? r : !0))
      return !1;
    const i = e || this.hasFill(), s = this.hasStroke(), o = this.getAbsoluteOpacity() !== 1;
    if (i && s && o)
      return !0;
    const l = this.hasShadow(), c = this.shadowForStrokeEnabled();
    return !!(i && s && l && c);
  }
  setStrokeHitEnabled(e) {
    W.warn("strokeHitEnabled property is deprecated. Please use hitStrokeWidth instead."), e ? this.hitStrokeWidth("auto") : this.hitStrokeWidth(0);
  }
  getStrokeHitEnabled() {
    return this.hitStrokeWidth() !== 0;
  }
  getSelfRect() {
    var e = this.size();
    return {
      x: this._centroid ? -e.width / 2 : 0,
      y: this._centroid ? -e.height / 2 : 0,
      width: e.width,
      height: e.height
    };
  }
  getClientRect(e = {}) {
    const r = e.skipTransform, n = e.relativeTo, i = this.getSelfRect(), o = !e.skipStroke && this.hasStroke() && this.strokeWidth() || 0, l = i.width + o, c = i.height + o, h = !e.skipShadow && this.hasShadow(), d = h ? this.shadowOffsetX() : 0, g = h ? this.shadowOffsetY() : 0, p = l + Math.abs(d), y = c + Math.abs(g), v = h && this.shadowBlur() || 0, _ = p + v * 2, b = y + v * 2, C = {
      width: _,
      height: b,
      x: -(o / 2 + v) + Math.min(d, 0) + i.x,
      y: -(o / 2 + v) + Math.min(g, 0) + i.y
    };
    return r ? C : this._transformedRect(C, n);
  }
  drawScene(e, r) {
    var n = this.getLayer(), i = e || n.getCanvas(), s = i.getContext(), o = this._getCanvasCache(), l = this.getSceneFunc(), c = this.hasShadow(), h, d, g, p = i.isCache, y = r === this;
    if (!this.isVisible() && !y)
      return this;
    if (o) {
      s.save();
      var v = this.getAbsoluteTransform(r).getMatrix();
      return s.transform(v[0], v[1], v[2], v[3], v[4], v[5]), this._drawCachedSceneCanvas(s), s.restore(), this;
    }
    if (!l)
      return this;
    if (s.save(), this._useBufferCanvas() && !p) {
      h = this.getStage(), d = h.bufferCanvas, g = d.getContext(), g.clear(), g.save(), g._applyLineJoin(this);
      var _ = this.getAbsoluteTransform(r).getMatrix();
      g.transform(_[0], _[1], _[2], _[3], _[4], _[5]), l.call(this, g, this), g.restore();
      var b = d.pixelRatio;
      c && s._applyShadow(this), s._applyOpacity(this), s._applyGlobalCompositeOperation(this), s.drawImage(d._canvas, 0, 0, d.width / b, d.height / b);
    } else {
      if (s._applyLineJoin(this), !y) {
        var _ = this.getAbsoluteTransform(r).getMatrix();
        s.transform(_[0], _[1], _[2], _[3], _[4], _[5]), s._applyOpacity(this), s._applyGlobalCompositeOperation(this);
      }
      c && s._applyShadow(this), l.call(this, s, this);
    }
    return s.restore(), this;
  }
  drawHit(e, r, n = !1) {
    if (!this.shouldDrawHit(r, n))
      return this;
    var i = this.getLayer(), s = e || i.hitCanvas, o = s && s.getContext(), l = this.hitFunc() || this.sceneFunc(), c = this._getCanvasCache(), h = c && c.hit;
    if (this.colorKey || W.warn("Looks like your canvas has a destroyed shape in it. Do not reuse shape after you destroyed it. If you want to reuse shape you should call remove() instead of destroy()"), h) {
      o.save();
      var d = this.getAbsoluteTransform(r).getMatrix();
      return o.transform(d[0], d[1], d[2], d[3], d[4], d[5]), this._drawCachedHitCanvas(o), o.restore(), this;
    }
    if (!l)
      return this;
    if (o.save(), o._applyLineJoin(this), !(this === r)) {
      var p = this.getAbsoluteTransform(r).getMatrix();
      o.transform(p[0], p[1], p[2], p[3], p[4], p[5]);
    }
    return l.call(this, o, this), o.restore(), this;
  }
  drawHitFromCache(e = 0) {
    var r = this._getCanvasCache(), n = this._getCachedSceneCanvas(), i = r.hit, s = i.getContext(), o = i.getWidth(), l = i.getHeight(), c, h, d, g, p, y;
    s.clear(), s.drawImage(n._canvas, 0, 0, o, l);
    try {
      for (c = s.getImageData(0, 0, o, l), h = c.data, d = h.length, g = W._hexToRgb(this.colorKey), p = 0; p < d; p += 4)
        y = h[p + 3], y > e ? (h[p] = g.r, h[p + 1] = g.g, h[p + 2] = g.b, h[p + 3] = 255) : h[p + 3] = 0;
      s.putImageData(c, 0, 0);
    } catch (v) {
      W.error("Unable to draw hit graph from cached scene canvas. " + v.message);
    }
    return this;
  }
  hasPointerCapture(e) {
    return s0(e, this);
  }
  setPointerCapture(e) {
    o0(e, this);
  }
  releaseCapture(e) {
    ro(e);
  }
}
se.prototype._fillFunc = jR;
se.prototype._strokeFunc = $R;
se.prototype._fillFuncHit = WR;
se.prototype._strokeFuncHit = UR;
se.prototype._centroid = !1;
se.prototype.nodeType = "Shape";
St(se);
se.prototype.eventListeners = {};
se.prototype.on.call(se.prototype, "shadowColorChange.konva shadowBlurChange.konva shadowOffsetChange.konva shadowOpacityChange.konva shadowEnabledChange.konva", HR);
se.prototype.on.call(se.prototype, "shadowColorChange.konva shadowOpacityChange.konva shadowEnabledChange.konva", zR);
se.prototype.on.call(se.prototype, "fillPriorityChange.konva fillPatternImageChange.konva fillPatternRepeatChange.konva fillPatternScaleXChange.konva fillPatternScaleYChange.konva fillPatternOffsetXChange.konva fillPatternOffsetYChange.konva fillPatternXChange.konva fillPatternYChange.konva fillPatternRotationChange.konva", VR);
se.prototype.on.call(se.prototype, "fillPriorityChange.konva fillLinearGradientColorStopsChange.konva fillLinearGradientStartPointXChange.konva fillLinearGradientStartPointYChange.konva fillLinearGradientEndPointXChange.konva fillLinearGradientEndPointYChange.konva", qR);
se.prototype.on.call(se.prototype, "fillPriorityChange.konva fillRadialGradientColorStopsChange.konva fillRadialGradientStartPointXChange.konva fillRadialGradientStartPointYChange.konva fillRadialGradientEndPointXChange.konva fillRadialGradientEndPointYChange.konva fillRadialGradientStartRadiusChange.konva fillRadialGradientEndRadiusChange.konva", YR);
I.addGetterSetter(se, "stroke", void 0, i0());
I.addGetterSetter(se, "strokeWidth", 2, le());
I.addGetterSetter(se, "fillAfterStrokeEnabled", !1);
I.addGetterSetter(se, "hitStrokeWidth", "auto", Mh());
I.addGetterSetter(se, "strokeHitEnabled", !0, hn());
I.addGetterSetter(se, "perfectDrawEnabled", !0, hn());
I.addGetterSetter(se, "shadowForStrokeEnabled", !0, hn());
I.addGetterSetter(se, "lineJoin");
I.addGetterSetter(se, "lineCap");
I.addGetterSetter(se, "sceneFunc");
I.addGetterSetter(se, "hitFunc");
I.addGetterSetter(se, "dash");
I.addGetterSetter(se, "dashOffset", 0, le());
I.addGetterSetter(se, "shadowColor", void 0, es());
I.addGetterSetter(se, "shadowBlur", 0, le());
I.addGetterSetter(se, "shadowOpacity", 1, le());
I.addComponentsGetterSetter(se, "shadowOffset", ["x", "y"]);
I.addGetterSetter(se, "shadowOffsetX", 0, le());
I.addGetterSetter(se, "shadowOffsetY", 0, le());
I.addGetterSetter(se, "fillPatternImage");
I.addGetterSetter(se, "fill", void 0, i0());
I.addGetterSetter(se, "fillPatternX", 0, le());
I.addGetterSetter(se, "fillPatternY", 0, le());
I.addGetterSetter(se, "fillLinearGradientColorStops");
I.addGetterSetter(se, "strokeLinearGradientColorStops");
I.addGetterSetter(se, "fillRadialGradientStartRadius", 0);
I.addGetterSetter(se, "fillRadialGradientEndRadius", 0);
I.addGetterSetter(se, "fillRadialGradientColorStops");
I.addGetterSetter(se, "fillPatternRepeat", "repeat");
I.addGetterSetter(se, "fillEnabled", !0);
I.addGetterSetter(se, "strokeEnabled", !0);
I.addGetterSetter(se, "shadowEnabled", !0);
I.addGetterSetter(se, "dashEnabled", !0);
I.addGetterSetter(se, "strokeScaleEnabled", !0);
I.addGetterSetter(se, "fillPriority", "color");
I.addComponentsGetterSetter(se, "fillPatternOffset", ["x", "y"]);
I.addGetterSetter(se, "fillPatternOffsetX", 0, le());
I.addGetterSetter(se, "fillPatternOffsetY", 0, le());
I.addComponentsGetterSetter(se, "fillPatternScale", ["x", "y"]);
I.addGetterSetter(se, "fillPatternScaleX", 1, le());
I.addGetterSetter(se, "fillPatternScaleY", 1, le());
I.addComponentsGetterSetter(se, "fillLinearGradientStartPoint", [
  "x",
  "y"
]);
I.addComponentsGetterSetter(se, "strokeLinearGradientStartPoint", [
  "x",
  "y"
]);
I.addGetterSetter(se, "fillLinearGradientStartPointX", 0);
I.addGetterSetter(se, "strokeLinearGradientStartPointX", 0);
I.addGetterSetter(se, "fillLinearGradientStartPointY", 0);
I.addGetterSetter(se, "strokeLinearGradientStartPointY", 0);
I.addComponentsGetterSetter(se, "fillLinearGradientEndPoint", [
  "x",
  "y"
]);
I.addComponentsGetterSetter(se, "strokeLinearGradientEndPoint", [
  "x",
  "y"
]);
I.addGetterSetter(se, "fillLinearGradientEndPointX", 0);
I.addGetterSetter(se, "strokeLinearGradientEndPointX", 0);
I.addGetterSetter(se, "fillLinearGradientEndPointY", 0);
I.addGetterSetter(se, "strokeLinearGradientEndPointY", 0);
I.addComponentsGetterSetter(se, "fillRadialGradientStartPoint", [
  "x",
  "y"
]);
I.addGetterSetter(se, "fillRadialGradientStartPointX", 0);
I.addGetterSetter(se, "fillRadialGradientStartPointY", 0);
I.addComponentsGetterSetter(se, "fillRadialGradientEndPoint", [
  "x",
  "y"
]);
I.addGetterSetter(se, "fillRadialGradientEndPointX", 0);
I.addGetterSetter(se, "fillRadialGradientEndPointY", 0);
I.addGetterSetter(se, "fillPatternRotation", 0);
I.backCompat(se, {
  dashArray: "dash",
  getDashArray: "getDash",
  setDashArray: "getDash",
  drawFunc: "sceneFunc",
  getDrawFunc: "getSceneFunc",
  setDrawFunc: "setSceneFunc",
  drawHitFunc: "hitFunc",
  getDrawHitFunc: "getHitFunc",
  setDrawHitFunc: "setHitFunc"
});
var KR = "#", XR = "beforeDraw", JR = "draw", E0 = [
  { x: 0, y: 0 },
  { x: -1, y: -1 },
  { x: 1, y: -1 },
  { x: 1, y: 1 },
  { x: -1, y: 1 }
], QR = E0.length;
class Zi extends Lr {
  constructor(e) {
    super(e), this.canvas = new Va(), this.hitCanvas = new Nh({
      pixelRatio: 1
    }), this._waitingForDraw = !1, this.on("visibleChange.konva", this._checkVisibility), this._checkVisibility(), this.on("imageSmoothingEnabledChange.konva", this._setSmoothEnabled), this._setSmoothEnabled();
  }
  createPNGStream() {
    return this.canvas._canvas.createPNGStream();
  }
  getCanvas() {
    return this.canvas;
  }
  getNativeCanvasElement() {
    return this.canvas._canvas;
  }
  getHitCanvas() {
    return this.hitCanvas;
  }
  getContext() {
    return this.getCanvas().getContext();
  }
  clear(e) {
    return this.getContext().clear(e), this.getHitCanvas().getContext().clear(e), this;
  }
  setZIndex(e) {
    super.setZIndex(e);
    var r = this.getStage();
    return r && r.content && (r.content.removeChild(this.getNativeCanvasElement()), e < r.children.length - 1 ? r.content.insertBefore(this.getNativeCanvasElement(), r.children[e + 1].getCanvas()._canvas) : r.content.appendChild(this.getNativeCanvasElement())), this;
  }
  moveToTop() {
    ue.prototype.moveToTop.call(this);
    var e = this.getStage();
    return e && e.content && (e.content.removeChild(this.getNativeCanvasElement()), e.content.appendChild(this.getNativeCanvasElement())), !0;
  }
  moveUp() {
    var e = ue.prototype.moveUp.call(this);
    if (!e)
      return !1;
    var r = this.getStage();
    return !r || !r.content ? !1 : (r.content.removeChild(this.getNativeCanvasElement()), this.index < r.children.length - 1 ? r.content.insertBefore(this.getNativeCanvasElement(), r.children[this.index + 1].getCanvas()._canvas) : r.content.appendChild(this.getNativeCanvasElement()), !0);
  }
  moveDown() {
    if (ue.prototype.moveDown.call(this)) {
      var e = this.getStage();
      if (e) {
        var r = e.children;
        e.content && (e.content.removeChild(this.getNativeCanvasElement()), e.content.insertBefore(this.getNativeCanvasElement(), r[this.index + 1].getCanvas()._canvas));
      }
      return !0;
    }
    return !1;
  }
  moveToBottom() {
    if (ue.prototype.moveToBottom.call(this)) {
      var e = this.getStage();
      if (e) {
        var r = e.children;
        e.content && (e.content.removeChild(this.getNativeCanvasElement()), e.content.insertBefore(this.getNativeCanvasElement(), r[1].getCanvas()._canvas));
      }
      return !0;
    }
    return !1;
  }
  getLayer() {
    return this;
  }
  remove() {
    var e = this.getNativeCanvasElement();
    return ue.prototype.remove.call(this), e && e.parentNode && W._isInDocument(e) && e.parentNode.removeChild(e), this;
  }
  getStage() {
    return this.parent;
  }
  setSize({ width: e, height: r }) {
    return this.canvas.setSize(e, r), this.hitCanvas.setSize(e, r), this._setSmoothEnabled(), this;
  }
  _validateAdd(e) {
    var r = e.getType();
    r !== "Group" && r !== "Shape" && W.throw("You may only add groups and shapes to a layer.");
  }
  _toKonvaCanvas(e) {
    return e = e || {}, e.width = e.width || this.getWidth(), e.height = e.height || this.getHeight(), e.x = e.x !== void 0 ? e.x : this.x(), e.y = e.y !== void 0 ? e.y : this.y(), ue.prototype._toKonvaCanvas.call(this, e);
  }
  _checkVisibility() {
    this.visible() ? this.canvas._canvas.style.display = "block" : this.canvas._canvas.style.display = "none";
  }
  _setSmoothEnabled() {
    this.getContext()._context.imageSmoothingEnabled = this.imageSmoothingEnabled();
  }
  getWidth() {
    if (this.parent)
      return this.parent.width();
  }
  setWidth() {
    W.warn('Can not change width of layer. Use "stage.width(value)" function instead.');
  }
  getHeight() {
    if (this.parent)
      return this.parent.height();
  }
  setHeight() {
    W.warn('Can not change height of layer. Use "stage.height(value)" function instead.');
  }
  batchDraw() {
    return this._waitingForDraw || (this._waitingForDraw = !0, W.requestAnimFrame(() => {
      this.draw(), this._waitingForDraw = !1;
    })), this;
  }
  getIntersection(e) {
    if (!this.isListening() || !this.isVisible())
      return null;
    for (var r = 1, n = !1; ; ) {
      for (let i = 0; i < QR; i++) {
        const s = E0[i], o = this._getIntersection({
          x: e.x + s.x * r,
          y: e.y + s.y * r
        }), l = o.shape;
        if (l)
          return l;
        if (n = !!o.antialiased, !o.antialiased)
          break;
      }
      if (n)
        r += 1;
      else
        return null;
    }
  }
  _getIntersection(e) {
    const r = this.hitCanvas.pixelRatio, n = this.hitCanvas.context.getImageData(Math.round(e.x * r), Math.round(e.y * r), 1, 1).data, i = n[3];
    if (i === 255) {
      const s = W._rgbToHex(n[0], n[1], n[2]), o = no[KR + s];
      return o ? {
        shape: o
      } : {
        antialiased: !0
      };
    } else if (i > 0)
      return {
        antialiased: !0
      };
    return {};
  }
  drawScene(e, r) {
    var n = this.getLayer(), i = e || n && n.getCanvas();
    return this._fire(XR, {
      node: this
    }), this.clearBeforeDraw() && i.getContext().clear(), Lr.prototype.drawScene.call(this, i, r), this._fire(JR, {
      node: this
    }), this;
  }
  drawHit(e, r) {
    var n = this.getLayer(), i = e || n && n.hitCanvas;
    return n && n.clearBeforeDraw() && n.getHitCanvas().getContext().clear(), Lr.prototype.drawHit.call(this, i, r), this;
  }
  enableHitGraph() {
    return this.hitGraphEnabled(!0), this;
  }
  disableHitGraph() {
    return this.hitGraphEnabled(!1), this;
  }
  setHitGraphEnabled(e) {
    W.warn("hitGraphEnabled method is deprecated. Please use layer.listening() instead."), this.listening(e);
  }
  getHitGraphEnabled(e) {
    return W.warn("hitGraphEnabled method is deprecated. Please use layer.listening() instead."), this.listening();
  }
  toggleHitCanvas() {
    if (!(!this.parent || !this.parent.content)) {
      var e = this.parent, r = !!this.hitCanvas._canvas.parentNode;
      r ? e.content.removeChild(this.hitCanvas._canvas) : e.content.appendChild(this.hitCanvas._canvas);
    }
  }
  destroy() {
    return W.releaseCanvas(this.getNativeCanvasElement(), this.getHitCanvas()._canvas), super.destroy();
  }
}
Zi.prototype.nodeType = "Layer";
St(Zi);
I.addGetterSetter(Zi, "imageSmoothingEnabled", !0);
I.addGetterSetter(Zi, "clearBeforeDraw", !0);
I.addGetterSetter(Zi, "hitGraphEnabled", !0, hn());
class Fh extends Zi {
  constructor(e) {
    super(e), this.listening(!1), W.warn('Konva.Fast layer is deprecated. Please use "new Konva.Layer({ listening: false })" instead.');
  }
}
Fh.prototype.nodeType = "FastLayer";
St(Fh);
class Ka extends Lr {
  _validateAdd(e) {
    var r = e.getType();
    r !== "Group" && r !== "Shape" && W.throw("You may only add groups and shapes to groups.");
  }
}
Ka.prototype.nodeType = "Group";
St(Ka);
var Of = function() {
  return za.performance && za.performance.now ? function() {
    return za.performance.now();
  } : function() {
    return new Date().getTime();
  };
}();
class Yr {
  constructor(e, r) {
    this.id = Yr.animIdCounter++, this.frame = {
      time: 0,
      timeDiff: 0,
      lastTime: Of(),
      frameRate: 0
    }, this.func = e, this.setLayers(r);
  }
  setLayers(e) {
    var r = [];
    return e ? e.length > 0 ? r = e : r = [e] : r = [], this.layers = r, this;
  }
  getLayers() {
    return this.layers;
  }
  addLayer(e) {
    var r = this.layers, n = r.length, i;
    for (i = 0; i < n; i++)
      if (r[i]._id === e._id)
        return !1;
    return this.layers.push(e), !0;
  }
  isRunning() {
    var e = Yr, r = e.animations, n = r.length, i;
    for (i = 0; i < n; i++)
      if (r[i].id === this.id)
        return !0;
    return !1;
  }
  start() {
    return this.stop(), this.frame.timeDiff = 0, this.frame.lastTime = Of(), Yr._addAnimation(this), this;
  }
  stop() {
    return Yr._removeAnimation(this), this;
  }
  _updateFrameObject(e) {
    this.frame.timeDiff = e - this.frame.lastTime, this.frame.lastTime = e, this.frame.time += this.frame.timeDiff, this.frame.frameRate = 1e3 / this.frame.timeDiff;
  }
  static _addAnimation(e) {
    this.animations.push(e), this._handleAnimation();
  }
  static _removeAnimation(e) {
    var r = e.id, n = this.animations, i = n.length, s;
    for (s = 0; s < i; s++)
      if (n[s].id === r) {
        this.animations.splice(s, 1);
        break;
      }
  }
  static _runFrames() {
    var e = {}, r = this.animations, n, i, s, o, l, c, h, d, g;
    for (o = 0; o < r.length; o++)
      if (n = r[o], i = n.layers, s = n.func, n._updateFrameObject(Of()), c = i.length, s ? g = s.call(n, n.frame) !== !1 : g = !0, !!g)
        for (l = 0; l < c; l++)
          h = i[l], h._id !== void 0 && (e[h._id] = h);
    for (d in e)
      e.hasOwnProperty(d) && e[d].batchDraw();
  }
  static _animationLoop() {
    var e = Yr;
    e.animations.length ? (e._runFrames(), W.requestAnimFrame(e._animationLoop)) : e.animRunning = !1;
  }
  static _handleAnimation() {
    this.animRunning || (this.animRunning = !0, W.requestAnimFrame(this._animationLoop));
  }
}
Yr.animations = [];
Yr.animIdCounter = 0;
Yr.animRunning = !1;
var ZR = {
  node: 1,
  duration: 1,
  easing: 1,
  onFinish: 1,
  yoyo: 1
}, eA = 1, Um = 2, Hm = 3, tA = 0, zm = ["fill", "stroke", "shadowColor"];
class rA {
  constructor(e, r, n, i, s, o, l) {
    this.prop = e, this.propFunc = r, this.begin = i, this._pos = i, this.duration = o, this._change = 0, this.prevPos = 0, this.yoyo = l, this._time = 0, this._position = 0, this._startTime = 0, this._finish = 0, this.func = n, this._change = s - this.begin, this.pause();
  }
  fire(e) {
    var r = this[e];
    r && r();
  }
  setTime(e) {
    e > this.duration ? this.yoyo ? (this._time = this.duration, this.reverse()) : this.finish() : e < 0 ? this.yoyo ? (this._time = 0, this.play()) : this.reset() : (this._time = e, this.update());
  }
  getTime() {
    return this._time;
  }
  setPosition(e) {
    this.prevPos = this._pos, this.propFunc(e), this._pos = e;
  }
  getPosition(e) {
    return e === void 0 && (e = this._time), this.func(e, this.begin, this._change, this.duration);
  }
  play() {
    this.state = Um, this._startTime = this.getTimer() - this._time, this.onEnterFrame(), this.fire("onPlay");
  }
  reverse() {
    this.state = Hm, this._time = this.duration - this._time, this._startTime = this.getTimer() - this._time, this.onEnterFrame(), this.fire("onReverse");
  }
  seek(e) {
    this.pause(), this._time = e, this.update(), this.fire("onSeek");
  }
  reset() {
    this.pause(), this._time = 0, this.update(), this.fire("onReset");
  }
  finish() {
    this.pause(), this._time = this.duration, this.update(), this.fire("onFinish");
  }
  update() {
    this.setPosition(this.getPosition(this._time)), this.fire("onUpdate");
  }
  onEnterFrame() {
    var e = this.getTimer() - this._startTime;
    this.state === Um ? this.setTime(e) : this.state === Hm && this.setTime(this.duration - e);
  }
  pause() {
    this.state = eA, this.fire("onPause");
  }
  getTimer() {
    return new Date().getTime();
  }
}
class Rt {
  constructor(e) {
    var r = this, n = e.node, i = n._id, s, o = e.easing || io.Linear, l = !!e.yoyo, c;
    typeof e.duration > "u" ? s = 0.3 : e.duration === 0 ? s = 1e-3 : s = e.duration, this.node = n, this._id = tA++;
    var h = n.getLayer() || (n instanceof ve.Stage ? n.getLayers() : null);
    h || W.error("Tween constructor have `node` that is not in a layer. Please add node into layer first."), this.anim = new Yr(function() {
      r.tween.onEnterFrame();
    }, h), this.tween = new rA(c, function(d) {
      r._tweenFunc(d);
    }, o, 0, 1, s * 1e3, l), this._addListeners(), Rt.attrs[i] || (Rt.attrs[i] = {}), Rt.attrs[i][this._id] || (Rt.attrs[i][this._id] = {}), Rt.tweens[i] || (Rt.tweens[i] = {});
    for (c in e)
      ZR[c] === void 0 && this._addAttr(c, e[c]);
    this.reset(), this.onFinish = e.onFinish, this.onReset = e.onReset, this.onUpdate = e.onUpdate;
  }
  _addAttr(e, r) {
    var n = this.node, i = n._id, s, o, l, c, h, d, g, p;
    if (l = Rt.tweens[i][e], l && delete Rt.attrs[i][l][e], s = n.getAttr(e), W._isArray(r))
      if (o = [], h = Math.max(r.length, s.length), e === "points" && r.length !== s.length && (r.length > s.length ? (g = s, s = W._prepareArrayForTween(s, r, n.closed())) : (d = r, r = W._prepareArrayForTween(r, s, n.closed()))), e.indexOf("fill") === 0)
        for (c = 0; c < h; c++)
          if (c % 2 === 0)
            o.push(r[c] - s[c]);
          else {
            var y = W.colorToRGBA(s[c]);
            p = W.colorToRGBA(r[c]), s[c] = y, o.push({
              r: p.r - y.r,
              g: p.g - y.g,
              b: p.b - y.b,
              a: p.a - y.a
            });
          }
      else
        for (c = 0; c < h; c++)
          o.push(r[c] - s[c]);
    else
      zm.indexOf(e) !== -1 ? (s = W.colorToRGBA(s), p = W.colorToRGBA(r), o = {
        r: p.r - s.r,
        g: p.g - s.g,
        b: p.b - s.b,
        a: p.a - s.a
      }) : o = r - s;
    Rt.attrs[i][this._id][e] = {
      start: s,
      diff: o,
      end: r,
      trueEnd: d,
      trueStart: g
    }, Rt.tweens[i][e] = this._id;
  }
  _tweenFunc(e) {
    var r = this.node, n = Rt.attrs[r._id][this._id], i, s, o, l, c, h, d, g;
    for (i in n) {
      if (s = n[i], o = s.start, l = s.diff, g = s.end, W._isArray(o))
        if (c = [], d = Math.max(o.length, g.length), i.indexOf("fill") === 0)
          for (h = 0; h < d; h++)
            h % 2 === 0 ? c.push((o[h] || 0) + l[h] * e) : c.push("rgba(" + Math.round(o[h].r + l[h].r * e) + "," + Math.round(o[h].g + l[h].g * e) + "," + Math.round(o[h].b + l[h].b * e) + "," + (o[h].a + l[h].a * e) + ")");
        else
          for (h = 0; h < d; h++)
            c.push((o[h] || 0) + l[h] * e);
      else
        zm.indexOf(i) !== -1 ? c = "rgba(" + Math.round(o.r + l.r * e) + "," + Math.round(o.g + l.g * e) + "," + Math.round(o.b + l.b * e) + "," + (o.a + l.a * e) + ")" : c = o + l * e;
      r.setAttr(i, c);
    }
  }
  _addListeners() {
    this.tween.onPlay = () => {
      this.anim.start();
    }, this.tween.onReverse = () => {
      this.anim.start();
    }, this.tween.onPause = () => {
      this.anim.stop();
    }, this.tween.onFinish = () => {
      var e = this.node, r = Rt.attrs[e._id][this._id];
      r.points && r.points.trueEnd && e.setAttr("points", r.points.trueEnd), this.onFinish && this.onFinish.call(this);
    }, this.tween.onReset = () => {
      var e = this.node, r = Rt.attrs[e._id][this._id];
      r.points && r.points.trueStart && e.points(r.points.trueStart), this.onReset && this.onReset();
    }, this.tween.onUpdate = () => {
      this.onUpdate && this.onUpdate.call(this);
    };
  }
  play() {
    return this.tween.play(), this;
  }
  reverse() {
    return this.tween.reverse(), this;
  }
  reset() {
    return this.tween.reset(), this;
  }
  seek(e) {
    return this.tween.seek(e * 1e3), this;
  }
  pause() {
    return this.tween.pause(), this;
  }
  finish() {
    return this.tween.finish(), this;
  }
  destroy() {
    var e = this.node._id, r = this._id, n = Rt.tweens[e], i;
    this.pause();
    for (i in n)
      delete Rt.tweens[e][i];
    delete Rt.attrs[e][r];
  }
}
Rt.attrs = {};
Rt.tweens = {};
ue.prototype.to = function(t) {
  var e = t.onFinish;
  t.node = this, t.onFinish = function() {
    this.destroy(), e && e();
  };
  var r = new Rt(t);
  r.play();
};
const io = {
  BackEaseIn(t, e, r, n) {
    var i = 1.70158;
    return r * (t /= n) * t * ((i + 1) * t - i) + e;
  },
  BackEaseOut(t, e, r, n) {
    var i = 1.70158;
    return r * ((t = t / n - 1) * t * ((i + 1) * t + i) + 1) + e;
  },
  BackEaseInOut(t, e, r, n) {
    var i = 1.70158;
    return (t /= n / 2) < 1 ? r / 2 * (t * t * (((i *= 1.525) + 1) * t - i)) + e : r / 2 * ((t -= 2) * t * (((i *= 1.525) + 1) * t + i) + 2) + e;
  },
  ElasticEaseIn(t, e, r, n, i, s) {
    var o = 0;
    return t === 0 ? e : (t /= n) === 1 ? e + r : (s || (s = n * 0.3), !i || i < Math.abs(r) ? (i = r, o = s / 4) : o = s / (2 * Math.PI) * Math.asin(r / i), -(i * Math.pow(2, 10 * (t -= 1)) * Math.sin((t * n - o) * (2 * Math.PI) / s)) + e);
  },
  ElasticEaseOut(t, e, r, n, i, s) {
    var o = 0;
    return t === 0 ? e : (t /= n) === 1 ? e + r : (s || (s = n * 0.3), !i || i < Math.abs(r) ? (i = r, o = s / 4) : o = s / (2 * Math.PI) * Math.asin(r / i), i * Math.pow(2, -10 * t) * Math.sin((t * n - o) * (2 * Math.PI) / s) + r + e);
  },
  ElasticEaseInOut(t, e, r, n, i, s) {
    var o = 0;
    return t === 0 ? e : (t /= n / 2) === 2 ? e + r : (s || (s = n * (0.3 * 1.5)), !i || i < Math.abs(r) ? (i = r, o = s / 4) : o = s / (2 * Math.PI) * Math.asin(r / i), t < 1 ? -0.5 * (i * Math.pow(2, 10 * (t -= 1)) * Math.sin((t * n - o) * (2 * Math.PI) / s)) + e : i * Math.pow(2, -10 * (t -= 1)) * Math.sin((t * n - o) * (2 * Math.PI) / s) * 0.5 + r + e);
  },
  BounceEaseOut(t, e, r, n) {
    return (t /= n) < 1 / 2.75 ? r * (7.5625 * t * t) + e : t < 2 / 2.75 ? r * (7.5625 * (t -= 1.5 / 2.75) * t + 0.75) + e : t < 2.5 / 2.75 ? r * (7.5625 * (t -= 2.25 / 2.75) * t + 0.9375) + e : r * (7.5625 * (t -= 2.625 / 2.75) * t + 0.984375) + e;
  },
  BounceEaseIn(t, e, r, n) {
    return r - io.BounceEaseOut(n - t, 0, r, n) + e;
  },
  BounceEaseInOut(t, e, r, n) {
    return t < n / 2 ? io.BounceEaseIn(t * 2, 0, r, n) * 0.5 + e : io.BounceEaseOut(t * 2 - n, 0, r, n) * 0.5 + r * 0.5 + e;
  },
  EaseIn(t, e, r, n) {
    return r * (t /= n) * t + e;
  },
  EaseOut(t, e, r, n) {
    return -r * (t /= n) * (t - 2) + e;
  },
  EaseInOut(t, e, r, n) {
    return (t /= n / 2) < 1 ? r / 2 * t * t + e : -r / 2 * (--t * (t - 2) - 1) + e;
  },
  StrongEaseIn(t, e, r, n) {
    return r * (t /= n) * t * t * t * t + e;
  },
  StrongEaseOut(t, e, r, n) {
    return r * ((t = t / n - 1) * t * t * t * t + 1) + e;
  },
  StrongEaseInOut(t, e, r, n) {
    return (t /= n / 2) < 1 ? r / 2 * t * t * t * t * t + e : r / 2 * ((t -= 2) * t * t * t * t + 2) + e;
  },
  Linear(t, e, r, n) {
    return r * t / n + e;
  }
}, Vm = W._assign(ve, {
  Util: W,
  Transform: Ir,
  Node: ue,
  Container: Lr,
  Stage: Ml,
  stages: al,
  Layer: Zi,
  FastLayer: Fh,
  Group: Ka,
  DD: ze,
  Shape: se,
  shapes: no,
  Animation: Yr,
  Tween: Rt,
  Easings: io,
  Context: Ll,
  Canvas: Dl
});
class ti extends se {
  _sceneFunc(e) {
    var r = ve.getAngle(this.angle()), n = this.clockwise();
    e.beginPath(), e.arc(0, 0, this.outerRadius(), 0, r, n), e.arc(0, 0, this.innerRadius(), r, 0, !n), e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    return this.outerRadius() * 2;
  }
  getHeight() {
    return this.outerRadius() * 2;
  }
  setWidth(e) {
    this.outerRadius(e / 2);
  }
  setHeight(e) {
    this.outerRadius(e / 2);
  }
  getSelfRect() {
    const e = this.innerRadius(), r = this.outerRadius(), n = this.clockwise(), i = ve.getAngle(n ? 360 - this.angle() : this.angle()), s = Math.cos(Math.min(i, Math.PI)), o = 1, l = Math.sin(Math.min(Math.max(Math.PI, i), 3 * Math.PI / 2)), c = Math.sin(Math.min(i, Math.PI / 2)), h = s * (s > 0 ? e : r), d = o * r, g = l * (l > 0 ? e : r), p = c * (c > 0 ? r : e);
    return {
      x: h,
      y: n ? -1 * p : g,
      width: d - h,
      height: p - g
    };
  }
}
ti.prototype._centroid = !0;
ti.prototype.className = "Arc";
ti.prototype._attrsAffectingSize = ["innerRadius", "outerRadius"];
St(ti);
I.addGetterSetter(ti, "innerRadius", 0, le());
I.addGetterSetter(ti, "outerRadius", 0, le());
I.addGetterSetter(ti, "angle", 0, le());
I.addGetterSetter(ti, "clockwise", !1, hn());
function ah(t, e, r, n, i, s, o) {
  var l = Math.sqrt(Math.pow(r - t, 2) + Math.pow(n - e, 2)), c = Math.sqrt(Math.pow(i - r, 2) + Math.pow(s - n, 2)), h = o * l / (l + c), d = o * c / (l + c), g = r - h * (i - t), p = n - h * (s - e), y = r + d * (i - t), v = n + d * (s - e);
  return [g, p, y, v];
}
function qm(t, e) {
  var r = t.length, n = [], i, s;
  for (i = 2; i < r - 2; i += 2)
    s = ah(t[i - 2], t[i - 1], t[i], t[i + 1], t[i + 2], t[i + 3], e), !isNaN(s[0]) && (n.push(s[0]), n.push(s[1]), n.push(t[i]), n.push(t[i + 1]), n.push(s[2]), n.push(s[3]));
  return n;
}
class ri extends se {
  constructor(e) {
    super(e), this.on("pointsChange.konva tensionChange.konva closedChange.konva bezierChange.konva", function() {
      this._clearCache("tensionPoints");
    });
  }
  _sceneFunc(e) {
    var r = this.points(), n = r.length, i = this.tension(), s = this.closed(), o = this.bezier(), l, c, h;
    if (n) {
      if (e.beginPath(), e.moveTo(r[0], r[1]), i !== 0 && n > 4) {
        for (l = this.getTensionPoints(), c = l.length, h = s ? 0 : 4, s || e.quadraticCurveTo(l[0], l[1], l[2], l[3]); h < c - 2; )
          e.bezierCurveTo(l[h++], l[h++], l[h++], l[h++], l[h++], l[h++]);
        s || e.quadraticCurveTo(l[c - 2], l[c - 1], r[n - 2], r[n - 1]);
      } else if (o)
        for (h = 2; h < n; )
          e.bezierCurveTo(r[h++], r[h++], r[h++], r[h++], r[h++], r[h++]);
      else
        for (h = 2; h < n; h += 2)
          e.lineTo(r[h], r[h + 1]);
      s ? (e.closePath(), e.fillStrokeShape(this)) : e.strokeShape(this);
    }
  }
  getTensionPoints() {
    return this._getCache("tensionPoints", this._getTensionPoints);
  }
  _getTensionPoints() {
    return this.closed() ? this._getTensionPointsClosed() : qm(this.points(), this.tension());
  }
  _getTensionPointsClosed() {
    var e = this.points(), r = e.length, n = this.tension(), i = ah(e[r - 2], e[r - 1], e[0], e[1], e[2], e[3], n), s = ah(e[r - 4], e[r - 3], e[r - 2], e[r - 1], e[0], e[1], n), o = qm(e, n), l = [i[2], i[3]].concat(o).concat([
      s[0],
      s[1],
      e[r - 2],
      e[r - 1],
      s[2],
      s[3],
      i[0],
      i[1],
      e[0],
      e[1]
    ]);
    return l;
  }
  getWidth() {
    return this.getSelfRect().width;
  }
  getHeight() {
    return this.getSelfRect().height;
  }
  getSelfRect() {
    var e = this.points();
    if (e.length < 4)
      return {
        x: e[0] || 0,
        y: e[1] || 0,
        width: 0,
        height: 0
      };
    this.tension() !== 0 ? e = [
      e[0],
      e[1],
      ...this._getTensionPoints(),
      e[e.length - 2],
      e[e.length - 1]
    ] : e = this.points();
    for (var r = e[0], n = e[0], i = e[1], s = e[1], o, l, c = 0; c < e.length / 2; c++)
      o = e[c * 2], l = e[c * 2 + 1], r = Math.min(r, o), n = Math.max(n, o), i = Math.min(i, l), s = Math.max(s, l);
    return {
      x: r,
      y: i,
      width: n - r,
      height: s - i
    };
  }
}
ri.prototype.className = "Line";
ri.prototype._attrsAffectingSize = ["points", "bezier", "tension"];
St(ri);
I.addGetterSetter(ri, "closed", !1);
I.addGetterSetter(ri, "bezier", !1);
I.addGetterSetter(ri, "tension", 0, le());
I.addGetterSetter(ri, "points", [], hR());
class st extends se {
  constructor(e) {
    super(e), this.dataArray = [], this.pathLength = 0, this.dataArray = st.parsePathData(this.data()), this.pathLength = 0;
    for (var r = 0; r < this.dataArray.length; ++r)
      this.pathLength += this.dataArray[r].pathLength;
    this.on("dataChange.konva", function() {
      this.dataArray = st.parsePathData(this.data()), this.pathLength = 0;
      for (var n = 0; n < this.dataArray.length; ++n)
        this.pathLength += this.dataArray[n].pathLength;
    });
  }
  _sceneFunc(e) {
    var r = this.dataArray;
    e.beginPath();
    for (var n = !1, i = 0; i < r.length; i++) {
      var s = r[i].command, o = r[i].points;
      switch (s) {
        case "L":
          e.lineTo(o[0], o[1]);
          break;
        case "M":
          e.moveTo(o[0], o[1]);
          break;
        case "C":
          e.bezierCurveTo(o[0], o[1], o[2], o[3], o[4], o[5]);
          break;
        case "Q":
          e.quadraticCurveTo(o[0], o[1], o[2], o[3]);
          break;
        case "A":
          var l = o[0], c = o[1], h = o[2], d = o[3], g = o[4], p = o[5], y = o[6], v = o[7], _ = h > d ? h : d, b = h > d ? 1 : h / d, C = h > d ? d / h : 1;
          e.translate(l, c), e.rotate(y), e.scale(b, C), e.arc(0, 0, _, g, g + p, 1 - v), e.scale(1 / b, 1 / C), e.rotate(-y), e.translate(-l, -c);
          break;
        case "z":
          n = !0, e.closePath();
          break;
      }
    }
    !n && !this.hasFill() ? e.strokeShape(this) : e.fillStrokeShape(this);
  }
  getSelfRect() {
    var e = [];
    this.dataArray.forEach(function(h) {
      if (h.command === "A") {
        var d = h.points[4], g = h.points[5], p = h.points[4] + g, y = Math.PI / 180;
        if (Math.abs(d - p) < y && (y = Math.abs(d - p)), g < 0)
          for (let v = d - y; v > p; v -= y) {
            const _ = st.getPointOnEllipticalArc(h.points[0], h.points[1], h.points[2], h.points[3], v, 0);
            e.push(_.x, _.y);
          }
        else
          for (let v = d + y; v < p; v += y) {
            const _ = st.getPointOnEllipticalArc(h.points[0], h.points[1], h.points[2], h.points[3], v, 0);
            e.push(_.x, _.y);
          }
      } else if (h.command === "C")
        for (let v = 0; v <= 1; v += 0.01) {
          const _ = st.getPointOnCubicBezier(v, h.start.x, h.start.y, h.points[0], h.points[1], h.points[2], h.points[3], h.points[4], h.points[5]);
          e.push(_.x, _.y);
        }
      else
        e = e.concat(h.points);
    });
    for (var r = e[0], n = e[0], i = e[1], s = e[1], o, l, c = 0; c < e.length / 2; c++)
      o = e[c * 2], l = e[c * 2 + 1], isNaN(o) || (r = Math.min(r, o), n = Math.max(n, o)), isNaN(l) || (i = Math.min(i, l), s = Math.max(s, l));
    return {
      x: r,
      y: i,
      width: n - r,
      height: s - i
    };
  }
  getLength() {
    return this.pathLength;
  }
  getPointAtLength(e) {
    var r, n = 0, i = this.dataArray.length;
    if (!i)
      return null;
    for (; n < i && e > this.dataArray[n].pathLength; )
      e -= this.dataArray[n].pathLength, ++n;
    if (n === i)
      return r = this.dataArray[n - 1].points.slice(-2), {
        x: r[0],
        y: r[1]
      };
    if (e < 0.01)
      return r = this.dataArray[n].points.slice(0, 2), {
        x: r[0],
        y: r[1]
      };
    var s = this.dataArray[n], o = s.points;
    switch (s.command) {
      case "L":
        return st.getPointOnLine(e, s.start.x, s.start.y, o[0], o[1]);
      case "C":
        return st.getPointOnCubicBezier(e / s.pathLength, s.start.x, s.start.y, o[0], o[1], o[2], o[3], o[4], o[5]);
      case "Q":
        return st.getPointOnQuadraticBezier(e / s.pathLength, s.start.x, s.start.y, o[0], o[1], o[2], o[3]);
      case "A":
        var l = o[0], c = o[1], h = o[2], d = o[3], g = o[4], p = o[5], y = o[6];
        return g += p * e / s.pathLength, st.getPointOnEllipticalArc(l, c, h, d, g, y);
    }
    return null;
  }
  static getLineLength(e, r, n, i) {
    return Math.sqrt((n - e) * (n - e) + (i - r) * (i - r));
  }
  static getPointOnLine(e, r, n, i, s, o, l) {
    o === void 0 && (o = r), l === void 0 && (l = n);
    var c = (s - n) / (i - r + 1e-8), h = Math.sqrt(e * e / (1 + c * c));
    i < r && (h *= -1);
    var d = c * h, g;
    if (i === r)
      g = {
        x: o,
        y: l + d
      };
    else if ((l - n) / (o - r + 1e-8) === c)
      g = {
        x: o + h,
        y: l + d
      };
    else {
      var p, y, v = this.getLineLength(r, n, i, s), _ = (o - r) * (i - r) + (l - n) * (s - n);
      _ = _ / (v * v), p = r + _ * (i - r), y = n + _ * (s - n);
      var b = this.getLineLength(o, l, p, y), C = Math.sqrt(e * e - b * b);
      h = Math.sqrt(C * C / (1 + c * c)), i < r && (h *= -1), d = c * h, g = {
        x: p + h,
        y: y + d
      };
    }
    return g;
  }
  static getPointOnCubicBezier(e, r, n, i, s, o, l, c, h) {
    function d(b) {
      return b * b * b;
    }
    function g(b) {
      return 3 * b * b * (1 - b);
    }
    function p(b) {
      return 3 * b * (1 - b) * (1 - b);
    }
    function y(b) {
      return (1 - b) * (1 - b) * (1 - b);
    }
    var v = c * d(e) + o * g(e) + i * p(e) + r * y(e), _ = h * d(e) + l * g(e) + s * p(e) + n * y(e);
    return {
      x: v,
      y: _
    };
  }
  static getPointOnQuadraticBezier(e, r, n, i, s, o, l) {
    function c(y) {
      return y * y;
    }
    function h(y) {
      return 2 * y * (1 - y);
    }
    function d(y) {
      return (1 - y) * (1 - y);
    }
    var g = o * c(e) + i * h(e) + r * d(e), p = l * c(e) + s * h(e) + n * d(e);
    return {
      x: g,
      y: p
    };
  }
  static getPointOnEllipticalArc(e, r, n, i, s, o) {
    var l = Math.cos(o), c = Math.sin(o), h = {
      x: n * Math.cos(s),
      y: i * Math.sin(s)
    };
    return {
      x: e + (h.x * l - h.y * c),
      y: r + (h.x * c + h.y * l)
    };
  }
  static parsePathData(e) {
    if (!e)
      return [];
    var r = e, n = [
      "m",
      "M",
      "l",
      "L",
      "v",
      "V",
      "h",
      "H",
      "z",
      "Z",
      "c",
      "C",
      "q",
      "Q",
      "t",
      "T",
      "s",
      "S",
      "a",
      "A"
    ];
    r = r.replace(new RegExp(" ", "g"), ",");
    for (var i = 0; i < n.length; i++)
      r = r.replace(new RegExp(n[i], "g"), "|" + n[i]);
    var s = r.split("|"), o = [], l = [], c = 0, h = 0, d = /([-+]?((\d+\.\d+)|((\d+)|(\.\d+)))(?:e[-+]?\d+)?)/gi, g;
    for (i = 1; i < s.length; i++) {
      var p = s[i], y = p.charAt(0);
      for (p = p.slice(1), l.length = 0; g = d.exec(p); )
        l.push(g[0]);
      for (var v = [], _ = 0, b = l.length; _ < b; _++) {
        if (l[_] === "00") {
          v.push(0, 0);
          continue;
        }
        var C = parseFloat(l[_]);
        isNaN(C) ? v.push(0) : v.push(C);
      }
      for (; v.length > 0 && !isNaN(v[0]); ) {
        var w = null, O = [], k = c, P = h, M, U, V, Y, X, Z, ce, $, R, F;
        switch (y) {
          case "l":
            c += v.shift(), h += v.shift(), w = "L", O.push(c, h);
            break;
          case "L":
            c = v.shift(), h = v.shift(), O.push(c, h);
            break;
          case "m":
            var H = v.shift(), re = v.shift();
            if (c += H, h += re, w = "M", o.length > 2 && o[o.length - 1].command === "z") {
              for (var ae = o.length - 2; ae >= 0; ae--)
                if (o[ae].command === "M") {
                  c = o[ae].points[0] + H, h = o[ae].points[1] + re;
                  break;
                }
            }
            O.push(c, h), y = "l";
            break;
          case "M":
            c = v.shift(), h = v.shift(), w = "M", O.push(c, h), y = "L";
            break;
          case "h":
            c += v.shift(), w = "L", O.push(c, h);
            break;
          case "H":
            c = v.shift(), w = "L", O.push(c, h);
            break;
          case "v":
            h += v.shift(), w = "L", O.push(c, h);
            break;
          case "V":
            h = v.shift(), w = "L", O.push(c, h);
            break;
          case "C":
            O.push(v.shift(), v.shift(), v.shift(), v.shift()), c = v.shift(), h = v.shift(), O.push(c, h);
            break;
          case "c":
            O.push(c + v.shift(), h + v.shift(), c + v.shift(), h + v.shift()), c += v.shift(), h += v.shift(), w = "C", O.push(c, h);
            break;
          case "S":
            U = c, V = h, M = o[o.length - 1], M.command === "C" && (U = c + (c - M.points[2]), V = h + (h - M.points[3])), O.push(U, V, v.shift(), v.shift()), c = v.shift(), h = v.shift(), w = "C", O.push(c, h);
            break;
          case "s":
            U = c, V = h, M = o[o.length - 1], M.command === "C" && (U = c + (c - M.points[2]), V = h + (h - M.points[3])), O.push(U, V, c + v.shift(), h + v.shift()), c += v.shift(), h += v.shift(), w = "C", O.push(c, h);
            break;
          case "Q":
            O.push(v.shift(), v.shift()), c = v.shift(), h = v.shift(), O.push(c, h);
            break;
          case "q":
            O.push(c + v.shift(), h + v.shift()), c += v.shift(), h += v.shift(), w = "Q", O.push(c, h);
            break;
          case "T":
            U = c, V = h, M = o[o.length - 1], M.command === "Q" && (U = c + (c - M.points[0]), V = h + (h - M.points[1])), c = v.shift(), h = v.shift(), w = "Q", O.push(U, V, c, h);
            break;
          case "t":
            U = c, V = h, M = o[o.length - 1], M.command === "Q" && (U = c + (c - M.points[0]), V = h + (h - M.points[1])), c += v.shift(), h += v.shift(), w = "Q", O.push(U, V, c, h);
            break;
          case "A":
            Y = v.shift(), X = v.shift(), Z = v.shift(), ce = v.shift(), $ = v.shift(), R = c, F = h, c = v.shift(), h = v.shift(), w = "A", O = this.convertEndpointToCenterParameterization(R, F, c, h, ce, $, Y, X, Z);
            break;
          case "a":
            Y = v.shift(), X = v.shift(), Z = v.shift(), ce = v.shift(), $ = v.shift(), R = c, F = h, c += v.shift(), h += v.shift(), w = "A", O = this.convertEndpointToCenterParameterization(R, F, c, h, ce, $, Y, X, Z);
            break;
        }
        o.push({
          command: w || y,
          points: O,
          start: {
            x: k,
            y: P
          },
          pathLength: this.calcLength(k, P, w || y, O)
        });
      }
      (y === "z" || y === "Z") && o.push({
        command: "z",
        points: [],
        start: void 0,
        pathLength: 0
      });
    }
    return o;
  }
  static calcLength(e, r, n, i) {
    var s, o, l, c, h = st;
    switch (n) {
      case "L":
        return h.getLineLength(e, r, i[0], i[1]);
      case "C":
        for (s = 0, o = h.getPointOnCubicBezier(0, e, r, i[0], i[1], i[2], i[3], i[4], i[5]), c = 0.01; c <= 1; c += 0.01)
          l = h.getPointOnCubicBezier(c, e, r, i[0], i[1], i[2], i[3], i[4], i[5]), s += h.getLineLength(o.x, o.y, l.x, l.y), o = l;
        return s;
      case "Q":
        for (s = 0, o = h.getPointOnQuadraticBezier(0, e, r, i[0], i[1], i[2], i[3]), c = 0.01; c <= 1; c += 0.01)
          l = h.getPointOnQuadraticBezier(c, e, r, i[0], i[1], i[2], i[3]), s += h.getLineLength(o.x, o.y, l.x, l.y), o = l;
        return s;
      case "A":
        s = 0;
        var d = i[4], g = i[5], p = i[4] + g, y = Math.PI / 180;
        if (Math.abs(d - p) < y && (y = Math.abs(d - p)), o = h.getPointOnEllipticalArc(i[0], i[1], i[2], i[3], d, 0), g < 0)
          for (c = d - y; c > p; c -= y)
            l = h.getPointOnEllipticalArc(i[0], i[1], i[2], i[3], c, 0), s += h.getLineLength(o.x, o.y, l.x, l.y), o = l;
        else
          for (c = d + y; c < p; c += y)
            l = h.getPointOnEllipticalArc(i[0], i[1], i[2], i[3], c, 0), s += h.getLineLength(o.x, o.y, l.x, l.y), o = l;
        return l = h.getPointOnEllipticalArc(i[0], i[1], i[2], i[3], p, 0), s += h.getLineLength(o.x, o.y, l.x, l.y), s;
    }
    return 0;
  }
  static convertEndpointToCenterParameterization(e, r, n, i, s, o, l, c, h) {
    var d = h * (Math.PI / 180), g = Math.cos(d) * (e - n) / 2 + Math.sin(d) * (r - i) / 2, p = -1 * Math.sin(d) * (e - n) / 2 + Math.cos(d) * (r - i) / 2, y = g * g / (l * l) + p * p / (c * c);
    y > 1 && (l *= Math.sqrt(y), c *= Math.sqrt(y));
    var v = Math.sqrt((l * l * (c * c) - l * l * (p * p) - c * c * (g * g)) / (l * l * (p * p) + c * c * (g * g)));
    s === o && (v *= -1), isNaN(v) && (v = 0);
    var _ = v * l * p / c, b = v * -c * g / l, C = (e + n) / 2 + Math.cos(d) * _ - Math.sin(d) * b, w = (r + i) / 2 + Math.sin(d) * _ + Math.cos(d) * b, O = function(X) {
      return Math.sqrt(X[0] * X[0] + X[1] * X[1]);
    }, k = function(X, Z) {
      return (X[0] * Z[0] + X[1] * Z[1]) / (O(X) * O(Z));
    }, P = function(X, Z) {
      return (X[0] * Z[1] < X[1] * Z[0] ? -1 : 1) * Math.acos(k(X, Z));
    }, M = P([1, 0], [(g - _) / l, (p - b) / c]), U = [(g - _) / l, (p - b) / c], V = [(-1 * g - _) / l, (-1 * p - b) / c], Y = P(U, V);
    return k(U, V) <= -1 && (Y = Math.PI), k(U, V) >= 1 && (Y = 0), o === 0 && Y > 0 && (Y = Y - 2 * Math.PI), o === 1 && Y < 0 && (Y = Y + 2 * Math.PI), [C, w, l, c, M, Y, d, o];
  }
}
st.prototype.className = "Path";
st.prototype._attrsAffectingSize = ["data"];
St(st);
I.addGetterSetter(st, "data");
class ea extends ri {
  _sceneFunc(e) {
    super._sceneFunc(e);
    var r = Math.PI * 2, n = this.points(), i = n, s = this.tension() !== 0 && n.length > 4;
    s && (i = this.getTensionPoints());
    var o = this.pointerLength(), l = n.length, c, h;
    if (s) {
      const p = [
        i[i.length - 4],
        i[i.length - 3],
        i[i.length - 2],
        i[i.length - 1],
        n[l - 2],
        n[l - 1]
      ], y = st.calcLength(i[i.length - 4], i[i.length - 3], "C", p), v = st.getPointOnQuadraticBezier(Math.min(1, 1 - o / y), p[0], p[1], p[2], p[3], p[4], p[5]);
      c = n[l - 2] - v.x, h = n[l - 1] - v.y;
    } else
      c = n[l - 2] - n[l - 4], h = n[l - 1] - n[l - 3];
    var d = (Math.atan2(h, c) + r) % r, g = this.pointerWidth();
    this.pointerAtEnding() && (e.save(), e.beginPath(), e.translate(n[l - 2], n[l - 1]), e.rotate(d), e.moveTo(0, 0), e.lineTo(-o, g / 2), e.lineTo(-o, -g / 2), e.closePath(), e.restore(), this.__fillStroke(e)), this.pointerAtBeginning() && (e.save(), e.beginPath(), e.translate(n[0], n[1]), s ? (c = (i[0] + i[2]) / 2 - n[0], h = (i[1] + i[3]) / 2 - n[1]) : (c = n[2] - n[0], h = n[3] - n[1]), e.rotate((Math.atan2(-h, -c) + r) % r), e.moveTo(0, 0), e.lineTo(-o, g / 2), e.lineTo(-o, -g / 2), e.closePath(), e.restore(), this.__fillStroke(e));
  }
  __fillStroke(e) {
    var r = this.dashEnabled();
    r && (this.attrs.dashEnabled = !1, e.setLineDash([])), e.fillStrokeShape(this), r && (this.attrs.dashEnabled = !0);
  }
  getSelfRect() {
    const e = super.getSelfRect(), r = this.pointerWidth() / 2;
    return {
      x: e.x - r,
      y: e.y - r,
      width: e.width + r * 2,
      height: e.height + r * 2
    };
  }
}
ea.prototype.className = "Arrow";
St(ea);
I.addGetterSetter(ea, "pointerLength", 10, le());
I.addGetterSetter(ea, "pointerWidth", 10, le());
I.addGetterSetter(ea, "pointerAtBeginning", !1);
I.addGetterSetter(ea, "pointerAtEnding", !0);
class ts extends se {
  _sceneFunc(e) {
    e.beginPath(), e.arc(0, 0, this.attrs.radius || 0, 0, Math.PI * 2, !1), e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    return this.radius() * 2;
  }
  getHeight() {
    return this.radius() * 2;
  }
  setWidth(e) {
    this.radius() !== e / 2 && this.radius(e / 2);
  }
  setHeight(e) {
    this.radius() !== e / 2 && this.radius(e / 2);
  }
}
ts.prototype._centroid = !0;
ts.prototype.className = "Circle";
ts.prototype._attrsAffectingSize = ["radius"];
St(ts);
I.addGetterSetter(ts, "radius", 0, le());
class Pi extends se {
  _sceneFunc(e) {
    var r = this.radiusX(), n = this.radiusY();
    e.beginPath(), e.save(), r !== n && e.scale(1, n / r), e.arc(0, 0, r, 0, Math.PI * 2, !1), e.restore(), e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    return this.radiusX() * 2;
  }
  getHeight() {
    return this.radiusY() * 2;
  }
  setWidth(e) {
    this.radiusX(e / 2);
  }
  setHeight(e) {
    this.radiusY(e / 2);
  }
}
Pi.prototype.className = "Ellipse";
Pi.prototype._centroid = !0;
Pi.prototype._attrsAffectingSize = ["radiusX", "radiusY"];
St(Pi);
I.addComponentsGetterSetter(Pi, "radius", ["x", "y"]);
I.addGetterSetter(Pi, "radiusX", 0, le());
I.addGetterSetter(Pi, "radiusY", 0, le());
let In = class C0 extends se {
  constructor(e) {
    super(e), this.on("imageChange.konva", () => {
      this._setImageLoad();
    }), this._setImageLoad();
  }
  _setImageLoad() {
    const e = this.image();
    e && e.complete || e && e.readyState === 4 || e && e.addEventListener && e.addEventListener("load", () => {
      this._requestDraw();
    });
  }
  _useBufferCanvas() {
    return super._useBufferCanvas(!0);
  }
  _sceneFunc(e) {
    const r = this.getWidth(), n = this.getHeight(), i = this.cornerRadius(), s = this.attrs.image;
    let o;
    if (s) {
      const l = this.attrs.cropWidth, c = this.attrs.cropHeight;
      l && c ? o = [
        s,
        this.cropX(),
        this.cropY(),
        l,
        c,
        0,
        0,
        r,
        n
      ] : o = [s, 0, 0, r, n];
    }
    (this.hasFill() || this.hasStroke() || i) && (e.beginPath(), i ? W.drawRoundedRectPath(e, r, n, i) : e.rect(0, 0, r, n), e.closePath(), e.fillStrokeShape(this)), s && (i && e.clip(), e.drawImage.apply(e, o));
  }
  _hitFunc(e) {
    var r = this.width(), n = this.height(), i = this.cornerRadius();
    e.beginPath(), i ? W.drawRoundedRectPath(e, r, n, i) : e.rect(0, 0, r, n), e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    var e, r;
    return (e = this.attrs.width) !== null && e !== void 0 ? e : (r = this.image()) === null || r === void 0 ? void 0 : r.width;
  }
  getHeight() {
    var e, r;
    return (e = this.attrs.height) !== null && e !== void 0 ? e : (r = this.image()) === null || r === void 0 ? void 0 : r.height;
  }
  static fromURL(e, r, n = null) {
    var i = W.createImageElement();
    i.onload = function() {
      var s = new C0({
        image: i
      });
      r(s);
    }, i.onerror = n, i.crossOrigin = "Anonymous", i.src = e;
  }
};
In.prototype.className = "Image";
St(In);
I.addGetterSetter(In, "cornerRadius", 0, Dh(4));
I.addGetterSetter(In, "image");
I.addComponentsGetterSetter(In, "crop", ["x", "y", "width", "height"]);
I.addGetterSetter(In, "cropX", 0, le());
I.addGetterSetter(In, "cropY", 0, le());
I.addGetterSetter(In, "cropWidth", 0, le());
I.addGetterSetter(In, "cropHeight", 0, le());
var x0 = [
  "fontFamily",
  "fontSize",
  "fontStyle",
  "padding",
  "lineHeight",
  "text",
  "width",
  "height",
  "pointerDirection",
  "pointerWidth",
  "pointerHeight"
], nA = "Change.konva", iA = "none", sh = "up", oh = "right", uh = "down", lh = "left", aA = x0.length;
class Gh extends Ka {
  constructor(e) {
    super(e), this.on("add.konva", function(r) {
      this._addListeners(r.child), this._sync();
    });
  }
  getText() {
    return this.find("Text")[0];
  }
  getTag() {
    return this.find("Tag")[0];
  }
  _addListeners(e) {
    var r = this, n, i = function() {
      r._sync();
    };
    for (n = 0; n < aA; n++)
      e.on(x0[n] + nA, i);
  }
  getWidth() {
    return this.getText().width();
  }
  getHeight() {
    return this.getText().height();
  }
  _sync() {
    var e = this.getText(), r = this.getTag(), n, i, s, o, l, c, h;
    if (e && r) {
      switch (n = e.width(), i = e.height(), s = r.pointerDirection(), o = r.pointerWidth(), h = r.pointerHeight(), l = 0, c = 0, s) {
        case sh:
          l = n / 2, c = -1 * h;
          break;
        case oh:
          l = n + o, c = i / 2;
          break;
        case uh:
          l = n / 2, c = i + h;
          break;
        case lh:
          l = -1 * o, c = i / 2;
          break;
      }
      r.setAttrs({
        x: -1 * l,
        y: -1 * c,
        width: n,
        height: i
      }), e.setAttrs({
        x: -1 * l,
        y: -1 * c
      });
    }
  }
}
Gh.prototype.className = "Label";
St(Gh);
class ta extends se {
  _sceneFunc(e) {
    var r = this.width(), n = this.height(), i = this.pointerDirection(), s = this.pointerWidth(), o = this.pointerHeight(), l = this.cornerRadius();
    let c = 0, h = 0, d = 0, g = 0;
    typeof l == "number" ? c = h = d = g = Math.min(l, r / 2, n / 2) : (c = Math.min(l[0] || 0, r / 2, n / 2), h = Math.min(l[1] || 0, r / 2, n / 2), g = Math.min(l[2] || 0, r / 2, n / 2), d = Math.min(l[3] || 0, r / 2, n / 2)), e.beginPath(), e.moveTo(c, 0), i === sh && (e.lineTo((r - s) / 2, 0), e.lineTo(r / 2, -1 * o), e.lineTo((r + s) / 2, 0)), e.lineTo(r - h, 0), e.arc(r - h, h, h, Math.PI * 3 / 2, 0, !1), i === oh && (e.lineTo(r, (n - o) / 2), e.lineTo(r + s, n / 2), e.lineTo(r, (n + o) / 2)), e.lineTo(r, n - g), e.arc(r - g, n - g, g, 0, Math.PI / 2, !1), i === uh && (e.lineTo((r + s) / 2, n), e.lineTo(r / 2, n + o), e.lineTo((r - s) / 2, n)), e.lineTo(d, n), e.arc(d, n - d, d, Math.PI / 2, Math.PI, !1), i === lh && (e.lineTo(0, (n + o) / 2), e.lineTo(-1 * s, n / 2), e.lineTo(0, (n - o) / 2)), e.lineTo(0, c), e.arc(c, c, c, Math.PI, Math.PI * 3 / 2, !1), e.closePath(), e.fillStrokeShape(this);
  }
  getSelfRect() {
    var e = 0, r = 0, n = this.pointerWidth(), i = this.pointerHeight(), s = this.pointerDirection(), o = this.width(), l = this.height();
    return s === sh ? (r -= i, l += i) : s === uh ? l += i : s === lh ? (e -= n * 1.5, o += n) : s === oh && (o += n * 1.5), {
      x: e,
      y: r,
      width: o,
      height: l
    };
  }
}
ta.prototype.className = "Tag";
St(ta);
I.addGetterSetter(ta, "pointerDirection", iA);
I.addGetterSetter(ta, "pointerWidth", 0, le());
I.addGetterSetter(ta, "pointerHeight", 0, le());
I.addGetterSetter(ta, "cornerRadius", 0, Dh(4));
class bo extends se {
  _sceneFunc(e) {
    var r = this.cornerRadius(), n = this.width(), i = this.height();
    e.beginPath(), r ? W.drawRoundedRectPath(e, n, i, r) : e.rect(0, 0, n, i), e.closePath(), e.fillStrokeShape(this);
  }
}
bo.prototype.className = "Rect";
St(bo);
I.addGetterSetter(bo, "cornerRadius", 0, Dh(4));
class ra extends se {
  _sceneFunc(e) {
    const r = this._getPoints();
    e.beginPath(), e.moveTo(r[0].x, r[0].y);
    for (var n = 1; n < r.length; n++)
      e.lineTo(r[n].x, r[n].y);
    e.closePath(), e.fillStrokeShape(this);
  }
  _getPoints() {
    const e = this.attrs.sides, r = this.attrs.radius || 0, n = [];
    for (var i = 0; i < e; i++)
      n.push({
        x: r * Math.sin(i * 2 * Math.PI / e),
        y: -1 * r * Math.cos(i * 2 * Math.PI / e)
      });
    return n;
  }
  getSelfRect() {
    const e = this._getPoints();
    var r = e[0].x, n = e[0].y, i = e[0].x, s = e[0].y;
    return e.forEach((o) => {
      r = Math.min(r, o.x), n = Math.max(n, o.x), i = Math.min(i, o.y), s = Math.max(s, o.y);
    }), {
      x: r,
      y: i,
      width: n - r,
      height: s - i
    };
  }
  getWidth() {
    return this.radius() * 2;
  }
  getHeight() {
    return this.radius() * 2;
  }
  setWidth(e) {
    this.radius(e / 2);
  }
  setHeight(e) {
    this.radius(e / 2);
  }
}
ra.prototype.className = "RegularPolygon";
ra.prototype._centroid = !0;
ra.prototype._attrsAffectingSize = ["radius"];
St(ra);
I.addGetterSetter(ra, "radius", 0, le());
I.addGetterSetter(ra, "sides", 0, le());
var Ym = Math.PI * 2;
class na extends se {
  _sceneFunc(e) {
    e.beginPath(), e.arc(0, 0, this.innerRadius(), 0, Ym, !1), e.moveTo(this.outerRadius(), 0), e.arc(0, 0, this.outerRadius(), Ym, 0, !0), e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    return this.outerRadius() * 2;
  }
  getHeight() {
    return this.outerRadius() * 2;
  }
  setWidth(e) {
    this.outerRadius(e / 2);
  }
  setHeight(e) {
    this.outerRadius(e / 2);
  }
}
na.prototype.className = "Ring";
na.prototype._centroid = !0;
na.prototype._attrsAffectingSize = ["innerRadius", "outerRadius"];
St(na);
I.addGetterSetter(na, "innerRadius", 0, le());
I.addGetterSetter(na, "outerRadius", 0, le());
class Ln extends se {
  constructor(e) {
    super(e), this._updated = !0, this.anim = new Yr(() => {
      var r = this._updated;
      return this._updated = !1, r;
    }), this.on("animationChange.konva", function() {
      this.frameIndex(0);
    }), this.on("frameIndexChange.konva", function() {
      this._updated = !0;
    }), this.on("frameRateChange.konva", function() {
      this.anim.isRunning() && (clearInterval(this.interval), this._setInterval());
    });
  }
  _sceneFunc(e) {
    var r = this.animation(), n = this.frameIndex(), i = n * 4, s = this.animations()[r], o = this.frameOffsets(), l = s[i + 0], c = s[i + 1], h = s[i + 2], d = s[i + 3], g = this.image();
    if ((this.hasFill() || this.hasStroke()) && (e.beginPath(), e.rect(0, 0, h, d), e.closePath(), e.fillStrokeShape(this)), g)
      if (o) {
        var p = o[r], y = n * 2;
        e.drawImage(g, l, c, h, d, p[y + 0], p[y + 1], h, d);
      } else
        e.drawImage(g, l, c, h, d, 0, 0, h, d);
  }
  _hitFunc(e) {
    var r = this.animation(), n = this.frameIndex(), i = n * 4, s = this.animations()[r], o = this.frameOffsets(), l = s[i + 2], c = s[i + 3];
    if (e.beginPath(), o) {
      var h = o[r], d = n * 2;
      e.rect(h[d + 0], h[d + 1], l, c);
    } else
      e.rect(0, 0, l, c);
    e.closePath(), e.fillShape(this);
  }
  _useBufferCanvas() {
    return super._useBufferCanvas(!0);
  }
  _setInterval() {
    var e = this;
    this.interval = setInterval(function() {
      e._updateIndex();
    }, 1e3 / this.frameRate());
  }
  start() {
    if (!this.isRunning()) {
      var e = this.getLayer();
      this.anim.setLayers(e), this._setInterval(), this.anim.start();
    }
  }
  stop() {
    this.anim.stop(), clearInterval(this.interval);
  }
  isRunning() {
    return this.anim.isRunning();
  }
  _updateIndex() {
    var e = this.frameIndex(), r = this.animation(), n = this.animations(), i = n[r], s = i.length / 4;
    e < s - 1 ? this.frameIndex(e + 1) : this.frameIndex(0);
  }
}
Ln.prototype.className = "Sprite";
St(Ln);
I.addGetterSetter(Ln, "animation");
I.addGetterSetter(Ln, "animations");
I.addGetterSetter(Ln, "frameOffsets");
I.addGetterSetter(Ln, "image");
I.addGetterSetter(Ln, "frameIndex", 0, le());
I.addGetterSetter(Ln, "frameRate", 17, le());
I.backCompat(Ln, {
  index: "frameIndex",
  getIndex: "getFrameIndex",
  setIndex: "setFrameIndex"
});
class Ri extends se {
  _sceneFunc(e) {
    var r = this.innerRadius(), n = this.outerRadius(), i = this.numPoints();
    e.beginPath(), e.moveTo(0, 0 - n);
    for (var s = 1; s < i * 2; s++) {
      var o = s % 2 === 0 ? n : r, l = o * Math.sin(s * Math.PI / i), c = -1 * o * Math.cos(s * Math.PI / i);
      e.lineTo(l, c);
    }
    e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    return this.outerRadius() * 2;
  }
  getHeight() {
    return this.outerRadius() * 2;
  }
  setWidth(e) {
    this.outerRadius(e / 2);
  }
  setHeight(e) {
    this.outerRadius(e / 2);
  }
}
Ri.prototype.className = "Star";
Ri.prototype._centroid = !0;
Ri.prototype._attrsAffectingSize = ["innerRadius", "outerRadius"];
St(Ri);
I.addGetterSetter(Ri, "numPoints", 5, le());
I.addGetterSetter(Ri, "innerRadius", 0, le());
I.addGetterSetter(Ri, "outerRadius", 0, le());
function O0(t) {
  return Array.from(t);
}
var ja = "auto", sA = "center", Ws = "justify", oA = "Change.konva", uA = "2d", Km = "-", T0 = "left", lA = "text", cA = "Text", fA = "top", hA = "bottom", Xm = "middle", P0 = "normal", dA = "px ", Wu = " ", gA = "right", pA = "word", mA = "char", Jm = "none", Tf = "…", R0 = [
  "fontFamily",
  "fontSize",
  "fontStyle",
  "fontVariant",
  "padding",
  "align",
  "verticalAlign",
  "lineHeight",
  "text",
  "width",
  "height",
  "wrap",
  "ellipsis",
  "letterSpacing"
], vA = R0.length;
function yA(t) {
  return t.split(",").map((e) => {
    e = e.trim();
    const r = e.indexOf(" ") >= 0, n = e.indexOf('"') >= 0 || e.indexOf("'") >= 0;
    return r && !n && (e = `"${e}"`), e;
  }).join(", ");
}
var Uu;
function Pf() {
  return Uu || (Uu = W.createCanvasElement().getContext(uA), Uu);
}
function bA(t) {
  t.fillText(this._partialText, this._partialTextX, this._partialTextY);
}
function _A(t) {
  t.strokeText(this._partialText, this._partialTextX, this._partialTextY);
}
function SA(t) {
  return t = t || {}, !t.fillLinearGradientColorStops && !t.fillRadialGradientColorStops && !t.fillPatternImage && (t.fill = t.fill || "black"), t;
}
let _t = class extends se {
  constructor(e) {
    super(SA(e)), this._partialTextX = 0, this._partialTextY = 0;
    for (var r = 0; r < vA; r++)
      this.on(R0[r] + oA, this._setTextData);
    this._setTextData();
  }
  _sceneFunc(e) {
    var r = this.textArr, n = r.length;
    if (this.text()) {
      var i = this.padding(), s = this.fontSize(), o = this.lineHeight() * s, l = this.verticalAlign(), c = 0, h = this.align(), d = this.getWidth(), g = this.letterSpacing(), p = this.fill(), y = this.textDecoration(), v = y.indexOf("underline") !== -1, _ = y.indexOf("line-through") !== -1, b, C = 0, C = o / 2, w = 0, O = 0;
      for (e.setAttr("font", this._getContextFont()), e.setAttr("textBaseline", Xm), e.setAttr("textAlign", T0), l === Xm ? c = (this.getHeight() - n * o - i * 2) / 2 : l === hA && (c = this.getHeight() - n * o - i * 2), e.translate(i, c + i), b = 0; b < n; b++) {
        var w = 0, O = 0, k = r[b], P = k.text, M = k.width, U = k.lastInParagraph, V, Y, X;
        if (e.save(), h === gA ? w += d - M - i * 2 : h === sA && (w += (d - M - i * 2) / 2), v) {
          e.save(), e.beginPath(), e.moveTo(w, C + O + Math.round(s / 2)), V = P.split(" ").length - 1, Y = V === 0, X = h === Ws && !U ? d - i * 2 : M, e.lineTo(w + Math.round(X), C + O + Math.round(s / 2)), e.lineWidth = s / 15;
          const H = this._getLinearGradient();
          e.strokeStyle = H || p, e.stroke(), e.restore();
        }
        if (_) {
          e.save(), e.beginPath(), e.moveTo(w, C + O), V = P.split(" ").length - 1, Y = V === 0, X = h === Ws && U && !Y ? d - i * 2 : M, e.lineTo(w + Math.round(X), C + O), e.lineWidth = s / 15;
          const H = this._getLinearGradient();
          e.strokeStyle = H || p, e.stroke(), e.restore();
        }
        if (g !== 0 || h === Ws) {
          V = P.split(" ").length - 1;
          for (var Z = O0(P), ce = 0; ce < Z.length; ce++) {
            var $ = Z[ce];
            $ === " " && !U && h === Ws && (w += (d - i * 2 - M) / V), this._partialTextX = w, this._partialTextY = C + O, this._partialText = $, e.fillStrokeShape(this), w += this.measureSize($).width + g;
          }
        } else
          this._partialTextX = w, this._partialTextY = C + O, this._partialText = P, e.fillStrokeShape(this);
        e.restore(), n > 1 && (C += o);
      }
    }
  }
  _hitFunc(e) {
    var r = this.getWidth(), n = this.getHeight();
    e.beginPath(), e.rect(0, 0, r, n), e.closePath(), e.fillStrokeShape(this);
  }
  setText(e) {
    var r = W._isString(e) ? e : e == null ? "" : e + "";
    return this._setAttr(lA, r), this;
  }
  getWidth() {
    var e = this.attrs.width === ja || this.attrs.width === void 0;
    return e ? this.getTextWidth() + this.padding() * 2 : this.attrs.width;
  }
  getHeight() {
    var e = this.attrs.height === ja || this.attrs.height === void 0;
    return e ? this.fontSize() * this.textArr.length * this.lineHeight() + this.padding() * 2 : this.attrs.height;
  }
  getTextWidth() {
    return this.textWidth;
  }
  getTextHeight() {
    return W.warn("text.getTextHeight() method is deprecated. Use text.height() - for full height and text.fontSize() - for one line height."), this.textHeight;
  }
  measureSize(e) {
    var r = Pf(), n = this.fontSize(), i;
    return r.save(), r.font = this._getContextFont(), i = r.measureText(e), r.restore(), {
      width: i.width,
      height: n
    };
  }
  _getContextFont() {
    return this.fontStyle() + Wu + this.fontVariant() + Wu + (this.fontSize() + dA) + yA(this.fontFamily());
  }
  _addTextLine(e) {
    this.align() === Ws && (e = e.trim());
    var n = this._getTextWidth(e);
    return this.textArr.push({
      text: e,
      width: n,
      lastInParagraph: !1
    });
  }
  _getTextWidth(e) {
    var r = this.letterSpacing(), n = e.length;
    return Pf().measureText(e).width + (n ? r * (n - 1) : 0);
  }
  _setTextData() {
    var e = this.text().split(`
`), r = +this.fontSize(), n = 0, i = this.lineHeight() * r, s = this.attrs.width, o = this.attrs.height, l = s !== ja && s !== void 0, c = o !== ja && o !== void 0, h = this.padding(), d = s - h * 2, g = o - h * 2, p = 0, y = this.wrap(), v = y !== Jm, _ = y !== mA && v, b = this.ellipsis();
    this.textArr = [], Pf().font = this._getContextFont();
    for (var C = b ? this._getTextWidth(Tf) : 0, w = 0, O = e.length; w < O; ++w) {
      var k = e[w], P = this._getTextWidth(k);
      if (l && P > d)
        for (; k.length > 0; ) {
          for (var M = 0, U = k.length, V = "", Y = 0; M < U; ) {
            var X = M + U >>> 1, Z = k.slice(0, X + 1), ce = this._getTextWidth(Z) + C;
            ce <= d ? (M = X + 1, V = Z, Y = ce) : U = X;
          }
          if (V) {
            if (_) {
              var $, R = k[V.length], F = R === Wu || R === Km;
              F && Y <= d ? $ = V.length : $ = Math.max(V.lastIndexOf(Wu), V.lastIndexOf(Km)) + 1, $ > 0 && (M = $, V = V.slice(0, M), Y = this._getTextWidth(V));
            }
            V = V.trimRight(), this._addTextLine(V), n = Math.max(n, Y), p += i;
            var H = this._shouldHandleEllipsis(p);
            if (H) {
              this._tryToAddEllipsisToLastLine();
              break;
            }
            if (k = k.slice(M), k = k.trimLeft(), k.length > 0 && (P = this._getTextWidth(k), P <= d)) {
              this._addTextLine(k), p += i, n = Math.max(n, P);
              break;
            }
          } else
            break;
        }
      else
        this._addTextLine(k), p += i, n = Math.max(n, P), this._shouldHandleEllipsis(p) && w < O - 1 && this._tryToAddEllipsisToLastLine();
      if (this.textArr[this.textArr.length - 1] && (this.textArr[this.textArr.length - 1].lastInParagraph = !0), c && p + i > g)
        break;
    }
    this.textHeight = r, this.textWidth = n;
  }
  _shouldHandleEllipsis(e) {
    var r = +this.fontSize(), n = this.lineHeight() * r, i = this.attrs.height, s = i !== ja && i !== void 0, o = this.padding(), l = i - o * 2, c = this.wrap(), h = c !== Jm;
    return !h || s && e + n > l;
  }
  _tryToAddEllipsisToLastLine() {
    var e = this.attrs.width, r = e !== ja && e !== void 0, n = this.padding(), i = e - n * 2, s = this.ellipsis(), o = this.textArr[this.textArr.length - 1];
    if (!(!o || !s)) {
      if (r) {
        var l = this._getTextWidth(o.text + Tf) < i;
        l || (o.text = o.text.slice(0, o.text.length - 3));
      }
      this.textArr.splice(this.textArr.length - 1, 1), this._addTextLine(o.text + Tf);
    }
  }
  getStrokeScaleEnabled() {
    return !0;
  }
};
_t.prototype._fillFunc = bA;
_t.prototype._strokeFunc = _A;
_t.prototype.className = cA;
_t.prototype._attrsAffectingSize = [
  "text",
  "fontSize",
  "padding",
  "wrap",
  "lineHeight",
  "letterSpacing"
];
St(_t);
I.overWriteSetter(_t, "width", Mh());
I.overWriteSetter(_t, "height", Mh());
I.addGetterSetter(_t, "fontFamily", "Arial");
I.addGetterSetter(_t, "fontSize", 12, le());
I.addGetterSetter(_t, "fontStyle", P0);
I.addGetterSetter(_t, "fontVariant", P0);
I.addGetterSetter(_t, "padding", 0, le());
I.addGetterSetter(_t, "align", T0);
I.addGetterSetter(_t, "verticalAlign", fA);
I.addGetterSetter(_t, "lineHeight", 1, le());
I.addGetterSetter(_t, "wrap", pA);
I.addGetterSetter(_t, "ellipsis", !1, hn());
I.addGetterSetter(_t, "letterSpacing", 0, le());
I.addGetterSetter(_t, "text", "", es());
I.addGetterSetter(_t, "textDecoration", "");
var wA = "", A0 = "normal";
function k0(t) {
  t.fillText(this.partialText, 0, 0);
}
function I0(t) {
  t.strokeText(this.partialText, 0, 0);
}
class jt extends se {
  constructor(e) {
    super(e), this.dummyCanvas = W.createCanvasElement(), this.dataArray = [], this.dataArray = st.parsePathData(this.attrs.data), this.on("dataChange.konva", function() {
      this.dataArray = st.parsePathData(this.attrs.data), this._setTextData();
    }), this.on("textChange.konva alignChange.konva letterSpacingChange.konva kerningFuncChange.konva fontSizeChange.konva fontFamilyChange.konva", this._setTextData), this._setTextData();
  }
  _sceneFunc(e) {
    e.setAttr("font", this._getContextFont()), e.setAttr("textBaseline", this.textBaseline()), e.setAttr("textAlign", "left"), e.save();
    var r = this.textDecoration(), n = this.fill(), i = this.fontSize(), s = this.glyphInfo;
    r === "underline" && e.beginPath();
    for (var o = 0; o < s.length; o++) {
      e.save();
      var l = s[o].p0;
      e.translate(l.x, l.y), e.rotate(s[o].rotation), this.partialText = s[o].text, e.fillStrokeShape(this), r === "underline" && (o === 0 && e.moveTo(0, i / 2 + 1), e.lineTo(i, i / 2 + 1)), e.restore();
    }
    r === "underline" && (e.strokeStyle = n, e.lineWidth = i / 20, e.stroke()), e.restore();
  }
  _hitFunc(e) {
    e.beginPath();
    var r = this.glyphInfo;
    if (r.length >= 1) {
      var n = r[0].p0;
      e.moveTo(n.x, n.y);
    }
    for (var i = 0; i < r.length; i++) {
      var s = r[i].p1;
      e.lineTo(s.x, s.y);
    }
    e.setAttr("lineWidth", this.fontSize()), e.setAttr("strokeStyle", this.colorKey), e.stroke();
  }
  getTextWidth() {
    return this.textWidth;
  }
  getTextHeight() {
    return W.warn("text.getTextHeight() method is deprecated. Use text.height() - for full height and text.fontSize() - for one line height."), this.textHeight;
  }
  setText(e) {
    return _t.prototype.setText.call(this, e);
  }
  _getContextFont() {
    return _t.prototype._getContextFont.call(this);
  }
  _getTextSize(e) {
    var r = this.dummyCanvas, n = r.getContext("2d");
    n.save(), n.font = this._getContextFont();
    var i = n.measureText(e);
    return n.restore(), {
      width: i.width,
      height: parseInt(this.attrs.fontSize, 10)
    };
  }
  _setTextData() {
    var e = this, r = this._getTextSize(this.attrs.text), n = this.letterSpacing(), i = this.align(), s = this.kerningFunc();
    this.textWidth = r.width, this.textHeight = r.height;
    var o = Math.max(this.textWidth + ((this.attrs.text || "").length - 1) * n, 0);
    this.glyphInfo = [];
    for (var l = 0, c = 0; c < e.dataArray.length; c++)
      e.dataArray[c].pathLength > 0 && (l += e.dataArray[c].pathLength);
    var h = 0;
    i === "center" && (h = Math.max(0, l / 2 - o / 2)), i === "right" && (h = Math.max(0, l - o));
    for (var d = O0(this.text()), g = this.text().split(" ").length - 1, p, y, v, _ = -1, b = 0, C = function() {
      b = 0;
      for (var ce = e.dataArray, $ = _ + 1; $ < ce.length; $++) {
        if (ce[$].pathLength > 0)
          return _ = $, ce[$];
        ce[$].command === "M" && (p = {
          x: ce[$].points[0],
          y: ce[$].points[1]
        });
      }
      return {};
    }, w = function(ce) {
      var $ = e._getTextSize(ce).width + n;
      ce === " " && i === "justify" && ($ += (l - o) / g);
      var R = 0, F = 0;
      for (y = void 0; Math.abs($ - R) / $ > 0.01 && F < 20; ) {
        F++;
        for (var H = R; v === void 0; )
          v = C(), v && H + v.pathLength < $ && (H += v.pathLength, v = void 0);
        if (Object.keys(v).length === 0 || p === void 0)
          return;
        var re = !1;
        switch (v.command) {
          case "L":
            st.getLineLength(p.x, p.y, v.points[0], v.points[1]) > $ ? y = st.getPointOnLine($, p.x, p.y, v.points[0], v.points[1], p.x, p.y) : v = void 0;
            break;
          case "A":
            var ae = v.points[4], ne = v.points[5], oe = v.points[4] + ne;
            b === 0 ? b = ae + 1e-8 : $ > R ? b += Math.PI / 180 * ne / Math.abs(ne) : b -= Math.PI / 360 * ne / Math.abs(ne), (ne < 0 && b < oe || ne >= 0 && b > oe) && (b = oe, re = !0), y = st.getPointOnEllipticalArc(v.points[0], v.points[1], v.points[2], v.points[3], b, v.points[6]);
            break;
          case "C":
            b === 0 ? $ > v.pathLength ? b = 1e-8 : b = $ / v.pathLength : $ > R ? b += ($ - R) / v.pathLength / 2 : b = Math.max(b - (R - $) / v.pathLength / 2, 0), b > 1 && (b = 1, re = !0), y = st.getPointOnCubicBezier(b, v.start.x, v.start.y, v.points[0], v.points[1], v.points[2], v.points[3], v.points[4], v.points[5]);
            break;
          case "Q":
            b === 0 ? b = $ / v.pathLength : $ > R ? b += ($ - R) / v.pathLength : b -= (R - $) / v.pathLength, b > 1 && (b = 1, re = !0), y = st.getPointOnQuadraticBezier(b, v.start.x, v.start.y, v.points[0], v.points[1], v.points[2], v.points[3]);
            break;
        }
        y !== void 0 && (R = st.getLineLength(p.x, p.y, y.x, y.y)), re && (re = !1, v = void 0);
      }
    }, O = "C", k = e._getTextSize(O).width + n, P = h / k - 1, M = 0; M < P && (w(O), !(p === void 0 || y === void 0)); M++)
      p = y;
    for (var U = 0; U < d.length && (w(d[U]), !(p === void 0 || y === void 0)); U++) {
      var V = st.getLineLength(p.x, p.y, y.x, y.y), Y = 0;
      if (s)
        try {
          Y = s(d[U - 1], d[U]) * this.fontSize();
        } catch {
          Y = 0;
        }
      p.x += Y, y.x += Y, this.textWidth += Y;
      var X = st.getPointOnLine(Y + V / 2, p.x, p.y, y.x, y.y), Z = Math.atan2(y.y - p.y, y.x - p.x);
      this.glyphInfo.push({
        transposeX: X.x,
        transposeY: X.y,
        text: d[U],
        rotation: Z,
        p0: p,
        p1: y
      }), p = y;
    }
  }
  getSelfRect() {
    if (!this.glyphInfo.length)
      return {
        x: 0,
        y: 0,
        width: 0,
        height: 0
      };
    var e = [];
    this.glyphInfo.forEach(function(d) {
      e.push(d.p0.x), e.push(d.p0.y), e.push(d.p1.x), e.push(d.p1.y);
    });
    for (var r = e[0] || 0, n = e[0] || 0, i = e[1] || 0, s = e[1] || 0, o, l, c = 0; c < e.length / 2; c++)
      o = e[c * 2], l = e[c * 2 + 1], r = Math.min(r, o), n = Math.max(n, o), i = Math.min(i, l), s = Math.max(s, l);
    var h = this.fontSize();
    return {
      x: r - h / 2,
      y: i - h / 2,
      width: n - r + h,
      height: s - i + h
    };
  }
  destroy() {
    return W.releaseCanvas(this.dummyCanvas), super.destroy();
  }
}
jt.prototype._fillFunc = k0;
jt.prototype._strokeFunc = I0;
jt.prototype._fillFuncHit = k0;
jt.prototype._strokeFuncHit = I0;
jt.prototype.className = "TextPath";
jt.prototype._attrsAffectingSize = ["text", "fontSize", "data"];
St(jt);
I.addGetterSetter(jt, "data");
I.addGetterSetter(jt, "fontFamily", "Arial");
I.addGetterSetter(jt, "fontSize", 12, le());
I.addGetterSetter(jt, "fontStyle", A0);
I.addGetterSetter(jt, "align", "left");
I.addGetterSetter(jt, "letterSpacing", 0, le());
I.addGetterSetter(jt, "textBaseline", "middle");
I.addGetterSetter(jt, "fontVariant", A0);
I.addGetterSetter(jt, "text", wA);
I.addGetterSetter(jt, "textDecoration", null);
I.addGetterSetter(jt, "kerningFunc", null);
var L0 = "tr-konva", EA = [
  "resizeEnabledChange",
  "rotateAnchorOffsetChange",
  "rotateEnabledChange",
  "enabledAnchorsChange",
  "anchorSizeChange",
  "borderEnabledChange",
  "borderStrokeChange",
  "borderStrokeWidthChange",
  "borderDashChange",
  "anchorStrokeChange",
  "anchorStrokeWidthChange",
  "anchorFillChange",
  "anchorCornerRadiusChange",
  "ignoreStrokeChange"
].map((t) => t + `.${L0}`).join(" "), Qm = "nodesRect", CA = [
  "widthChange",
  "heightChange",
  "scaleXChange",
  "scaleYChange",
  "skewXChange",
  "skewYChange",
  "rotationChange",
  "offsetXChange",
  "offsetYChange",
  "transformsEnabledChange",
  "strokeWidthChange"
], xA = {
  "top-left": -45,
  "top-center": 0,
  "top-right": 45,
  "middle-right": -90,
  "middle-left": 90,
  "bottom-left": -135,
  "bottom-center": 180,
  "bottom-right": 135
};
const OA = "ontouchstart" in ve._global;
function TA(t, e) {
  if (t === "rotater")
    return "crosshair";
  e += W.degToRad(xA[t] || 0);
  var r = (W.radToDeg(e) % 360 + 360) % 360;
  return W._inRange(r, 315 + 22.5, 360) || W._inRange(r, 0, 22.5) ? "ns-resize" : W._inRange(r, 45 - 22.5, 45 + 22.5) ? "nesw-resize" : W._inRange(r, 90 - 22.5, 90 + 22.5) ? "ew-resize" : W._inRange(r, 135 - 22.5, 135 + 22.5) ? "nwse-resize" : W._inRange(r, 180 - 22.5, 180 + 22.5) ? "ns-resize" : W._inRange(r, 225 - 22.5, 225 + 22.5) ? "nesw-resize" : W._inRange(r, 270 - 22.5, 270 + 22.5) ? "ew-resize" : W._inRange(r, 315 - 22.5, 315 + 22.5) ? "nwse-resize" : (W.error("Transformer has unknown angle for cursor detection: " + r), "pointer");
}
var El = [
  "top-left",
  "top-center",
  "top-right",
  "middle-right",
  "middle-left",
  "bottom-left",
  "bottom-center",
  "bottom-right"
], Zm = 1e8;
function PA(t) {
  return {
    x: t.x + t.width / 2 * Math.cos(t.rotation) + t.height / 2 * Math.sin(-t.rotation),
    y: t.y + t.height / 2 * Math.cos(t.rotation) + t.width / 2 * Math.sin(t.rotation)
  };
}
function D0(t, e, r) {
  const n = r.x + (t.x - r.x) * Math.cos(e) - (t.y - r.y) * Math.sin(e), i = r.y + (t.x - r.x) * Math.sin(e) + (t.y - r.y) * Math.cos(e);
  return Object.assign(Object.assign({}, t), {
    rotation: t.rotation + e,
    x: n,
    y: i
  });
}
function RA(t, e) {
  const r = PA(t);
  return D0(t, e, r);
}
function AA(t, e, r) {
  let n = e;
  for (let i = 0; i < t.length; i++) {
    const s = ve.getAngle(t[i]), o = Math.abs(s - e) % (Math.PI * 2);
    Math.min(o, Math.PI * 2 - o) < r && (n = s);
  }
  return n;
}
class tt extends Ka {
  constructor(e) {
    super(e), this._transforming = !1, this._createElements(), this._handleMouseMove = this._handleMouseMove.bind(this), this._handleMouseUp = this._handleMouseUp.bind(this), this.update = this.update.bind(this), this.on(EA, this.update), this.getNode() && this.update();
  }
  attachTo(e) {
    return this.setNode(e), this;
  }
  setNode(e) {
    return W.warn("tr.setNode(shape), tr.node(shape) and tr.attachTo(shape) methods are deprecated. Please use tr.nodes(nodesArray) instead."), this.setNodes([e]);
  }
  getNode() {
    return this._nodes && this._nodes[0];
  }
  _getEventNamespace() {
    return L0 + this._id;
  }
  setNodes(e = []) {
    this._nodes && this._nodes.length && this.detach(), this._nodes = e, e.length === 1 && this.useSingleNodeRotation() ? this.rotation(e[0].getAbsoluteRotation()) : this.rotation(0), this._nodes.forEach((n) => {
      const i = () => {
        this.nodes().length === 1 && this.useSingleNodeRotation() && this.rotation(this.nodes()[0].getAbsoluteRotation()), this._resetTransformCache(), !this._transforming && !this.isDragging() && this.update();
      }, s = n._attrsAffectingSize.map((o) => o + "Change." + this._getEventNamespace()).join(" ");
      n.on(s, i), n.on(CA.map((o) => o + `.${this._getEventNamespace()}`).join(" "), i), n.on(`absoluteTransformChange.${this._getEventNamespace()}`, i), this._proxyDrag(n);
    }), this._resetTransformCache();
    var r = !!this.findOne(".top-left");
    return r && this.update(), this;
  }
  _proxyDrag(e) {
    let r;
    e.on(`dragstart.${this._getEventNamespace()}`, (n) => {
      r = e.getAbsolutePosition(), !this.isDragging() && e !== this.findOne(".back") && this.startDrag(n, !1);
    }), e.on(`dragmove.${this._getEventNamespace()}`, (n) => {
      if (!r)
        return;
      const i = e.getAbsolutePosition(), s = i.x - r.x, o = i.y - r.y;
      this.nodes().forEach((l) => {
        if (l === e || l.isDragging())
          return;
        const c = l.getAbsolutePosition();
        l.setAbsolutePosition({
          x: c.x + s,
          y: c.y + o
        }), l.startDrag(n);
      }), r = null;
    });
  }
  getNodes() {
    return this._nodes || [];
  }
  getActiveAnchor() {
    return this._movingAnchorName;
  }
  detach() {
    this._nodes && this._nodes.forEach((e) => {
      e.off("." + this._getEventNamespace());
    }), this._nodes = [], this._resetTransformCache();
  }
  _resetTransformCache() {
    this._clearCache(Qm), this._clearCache("transform"), this._clearSelfAndDescendantCache("absoluteTransform");
  }
  _getNodeRect() {
    return this._getCache(Qm, this.__getNodeRect);
  }
  __getNodeShape(e, r = this.rotation(), n) {
    var i = e.getClientRect({
      skipTransform: !0,
      skipShadow: !0,
      skipStroke: this.ignoreStroke()
    }), s = e.getAbsoluteScale(n), o = e.getAbsolutePosition(n), l = i.x * s.x - e.offsetX() * s.x, c = i.y * s.y - e.offsetY() * s.y;
    const h = (ve.getAngle(e.getAbsoluteRotation()) + Math.PI * 2) % (Math.PI * 2), d = {
      x: o.x + l * Math.cos(h) + c * Math.sin(-h),
      y: o.y + c * Math.cos(h) + l * Math.sin(h),
      width: i.width * s.x,
      height: i.height * s.y,
      rotation: h
    };
    return D0(d, -ve.getAngle(r), {
      x: 0,
      y: 0
    });
  }
  __getNodeRect() {
    var e = this.getNode();
    if (!e)
      return {
        x: -Zm,
        y: -Zm,
        width: 0,
        height: 0,
        rotation: 0
      };
    const r = [];
    this.nodes().map((h) => {
      const d = h.getClientRect({
        skipTransform: !0,
        skipShadow: !0,
        skipStroke: this.ignoreStroke()
      });
      var g = [
        { x: d.x, y: d.y },
        { x: d.x + d.width, y: d.y },
        { x: d.x + d.width, y: d.y + d.height },
        { x: d.x, y: d.y + d.height }
      ], p = h.getAbsoluteTransform();
      g.forEach(function(y) {
        var v = p.point(y);
        r.push(v);
      });
    });
    const n = new Ir();
    n.rotate(-ve.getAngle(this.rotation()));
    var i, s, o, l;
    r.forEach(function(h) {
      var d = n.point(h);
      i === void 0 && (i = o = d.x, s = l = d.y), i = Math.min(i, d.x), s = Math.min(s, d.y), o = Math.max(o, d.x), l = Math.max(l, d.y);
    }), n.invert();
    const c = n.point({ x: i, y: s });
    return {
      x: c.x,
      y: c.y,
      width: o - i,
      height: l - s,
      rotation: ve.getAngle(this.rotation())
    };
  }
  getX() {
    return this._getNodeRect().x;
  }
  getY() {
    return this._getNodeRect().y;
  }
  getWidth() {
    return this._getNodeRect().width;
  }
  getHeight() {
    return this._getNodeRect().height;
  }
  _createElements() {
    this._createBack(), El.forEach(function(e) {
      this._createAnchor(e);
    }.bind(this)), this._createAnchor("rotater");
  }
  _createAnchor(e) {
    var r = new bo({
      stroke: "rgb(0, 161, 255)",
      fill: "white",
      strokeWidth: 1,
      name: e + " _anchor",
      dragDistance: 0,
      draggable: !0,
      hitStrokeWidth: OA ? 10 : "auto"
    }), n = this;
    r.on("mousedown touchstart", function(i) {
      n._handleMouseDown(i);
    }), r.on("dragstart", (i) => {
      r.stopDrag(), i.cancelBubble = !0;
    }), r.on("dragend", (i) => {
      i.cancelBubble = !0;
    }), r.on("mouseenter", () => {
      var i = ve.getAngle(this.rotation()), s = TA(e, i);
      r.getStage().content && (r.getStage().content.style.cursor = s), this._cursorChange = !0;
    }), r.on("mouseout", () => {
      r.getStage().content && (r.getStage().content.style.cursor = ""), this._cursorChange = !1;
    }), this.add(r);
  }
  _createBack() {
    var e = new se({
      name: "back",
      width: 0,
      height: 0,
      draggable: !0,
      sceneFunc(r) {
        var n = this.getParent(), i = n.padding();
        r.beginPath(), r.rect(-i, -i, this.width() + i * 2, this.height() + i * 2), r.moveTo(this.width() / 2, -i), n.rotateEnabled() && r.lineTo(this.width() / 2, -n.rotateAnchorOffset() * W._sign(this.height()) - i), r.fillStrokeShape(this);
      },
      hitFunc: (r, n) => {
        if (this.shouldOverdrawWholeArea()) {
          var i = this.padding();
          r.beginPath(), r.rect(-i, -i, n.width() + i * 2, n.height() + i * 2), r.fillStrokeShape(n);
        }
      }
    });
    this.add(e), this._proxyDrag(e), e.on("dragstart", (r) => {
      r.cancelBubble = !0;
    }), e.on("dragmove", (r) => {
      r.cancelBubble = !0;
    }), e.on("dragend", (r) => {
      r.cancelBubble = !0;
    }), this.on("dragmove", (r) => {
      this.update();
    });
  }
  _handleMouseDown(e) {
    this._movingAnchorName = e.target.name().split(" ")[0];
    var r = this._getNodeRect(), n = r.width, i = r.height, s = Math.sqrt(Math.pow(n, 2) + Math.pow(i, 2));
    this.sin = Math.abs(i / s), this.cos = Math.abs(n / s), typeof window < "u" && (window.addEventListener("mousemove", this._handleMouseMove), window.addEventListener("touchmove", this._handleMouseMove), window.addEventListener("mouseup", this._handleMouseUp, !0), window.addEventListener("touchend", this._handleMouseUp, !0)), this._transforming = !0;
    var o = e.target.getAbsolutePosition(), l = e.target.getStage().getPointerPosition();
    this._anchorDragOffset = {
      x: l.x - o.x,
      y: l.y - o.y
    }, this._fire("transformstart", { evt: e.evt, target: this.getNode() }), this._nodes.forEach((c) => {
      c._fire("transformstart", { evt: e.evt, target: c });
    });
  }
  _handleMouseMove(e) {
    var r, n, i, s = this.findOne("." + this._movingAnchorName), o = s.getStage();
    o.setPointersPositions(e);
    const l = o.getPointerPosition();
    let c = {
      x: l.x - this._anchorDragOffset.x,
      y: l.y - this._anchorDragOffset.y
    };
    const h = s.getAbsolutePosition();
    this.anchorDragBoundFunc() && (c = this.anchorDragBoundFunc()(h, c, e)), s.setAbsolutePosition(c);
    const d = s.getAbsolutePosition();
    if (!(h.x === d.x && h.y === d.y)) {
      if (this._movingAnchorName === "rotater") {
        var g = this._getNodeRect();
        r = s.x() - g.width / 2, n = -s.y() + g.height / 2;
        let ce = Math.atan2(-n, r) + Math.PI / 2;
        g.height < 0 && (ce -= Math.PI);
        var p = ve.getAngle(this.rotation());
        const $ = p + ce, R = ve.getAngle(this.rotationSnapTolerance()), H = AA(this.rotationSnaps(), $, R) - g.rotation, re = RA(g, H);
        this._fitNodesInto(re, e);
        return;
      }
      var y = this.keepRatio() || e.shiftKey, w = this.centeredScaling() || e.altKey;
      if (this._movingAnchorName === "top-left") {
        if (y) {
          var v = w ? {
            x: this.width() / 2,
            y: this.height() / 2
          } : {
            x: this.findOne(".bottom-right").x(),
            y: this.findOne(".bottom-right").y()
          };
          i = Math.sqrt(Math.pow(v.x - s.x(), 2) + Math.pow(v.y - s.y(), 2));
          var _ = this.findOne(".top-left").x() > v.x ? -1 : 1, b = this.findOne(".top-left").y() > v.y ? -1 : 1;
          r = i * this.cos * _, n = i * this.sin * b, this.findOne(".top-left").x(v.x - r), this.findOne(".top-left").y(v.y - n);
        }
      } else if (this._movingAnchorName === "top-center")
        this.findOne(".top-left").y(s.y());
      else if (this._movingAnchorName === "top-right") {
        if (y) {
          var v = w ? {
            x: this.width() / 2,
            y: this.height() / 2
          } : {
            x: this.findOne(".bottom-left").x(),
            y: this.findOne(".bottom-left").y()
          };
          i = Math.sqrt(Math.pow(s.x() - v.x, 2) + Math.pow(v.y - s.y(), 2));
          var _ = this.findOne(".top-right").x() < v.x ? -1 : 1, b = this.findOne(".top-right").y() > v.y ? -1 : 1;
          r = i * this.cos * _, n = i * this.sin * b, this.findOne(".top-right").x(v.x + r), this.findOne(".top-right").y(v.y - n);
        }
        var C = s.position();
        this.findOne(".top-left").y(C.y), this.findOne(".bottom-right").x(C.x);
      } else if (this._movingAnchorName === "middle-left")
        this.findOne(".top-left").x(s.x());
      else if (this._movingAnchorName === "middle-right")
        this.findOne(".bottom-right").x(s.x());
      else if (this._movingAnchorName === "bottom-left") {
        if (y) {
          var v = w ? {
            x: this.width() / 2,
            y: this.height() / 2
          } : {
            x: this.findOne(".top-right").x(),
            y: this.findOne(".top-right").y()
          };
          i = Math.sqrt(Math.pow(v.x - s.x(), 2) + Math.pow(s.y() - v.y, 2));
          var _ = v.x < s.x() ? -1 : 1, b = s.y() < v.y ? -1 : 1;
          r = i * this.cos * _, n = i * this.sin * b, s.x(v.x - r), s.y(v.y + n);
        }
        C = s.position(), this.findOne(".top-left").x(C.x), this.findOne(".bottom-right").y(C.y);
      } else if (this._movingAnchorName === "bottom-center")
        this.findOne(".bottom-right").y(s.y());
      else if (this._movingAnchorName === "bottom-right") {
        if (y) {
          var v = w ? {
            x: this.width() / 2,
            y: this.height() / 2
          } : {
            x: this.findOne(".top-left").x(),
            y: this.findOne(".top-left").y()
          };
          i = Math.sqrt(Math.pow(s.x() - v.x, 2) + Math.pow(s.y() - v.y, 2));
          var _ = this.findOne(".bottom-right").x() < v.x ? -1 : 1, b = this.findOne(".bottom-right").y() < v.y ? -1 : 1;
          r = i * this.cos * _, n = i * this.sin * b, this.findOne(".bottom-right").x(v.x + r), this.findOne(".bottom-right").y(v.y + n);
        }
      } else
        console.error(new Error("Wrong position argument of selection resizer: " + this._movingAnchorName));
      var w = this.centeredScaling() || e.altKey;
      if (w) {
        var O = this.findOne(".top-left"), k = this.findOne(".bottom-right"), P = O.x(), M = O.y(), U = this.getWidth() - k.x(), V = this.getHeight() - k.y();
        k.move({
          x: -P,
          y: -M
        }), O.move({
          x: U,
          y: V
        });
      }
      var Y = this.findOne(".top-left").getAbsolutePosition();
      r = Y.x, n = Y.y;
      var X = this.findOne(".bottom-right").x() - this.findOne(".top-left").x(), Z = this.findOne(".bottom-right").y() - this.findOne(".top-left").y();
      this._fitNodesInto({
        x: r,
        y: n,
        width: X,
        height: Z,
        rotation: ve.getAngle(this.rotation())
      }, e);
    }
  }
  _handleMouseUp(e) {
    this._removeEvents(e);
  }
  getAbsoluteTransform() {
    return this.getTransform();
  }
  _removeEvents(e) {
    if (this._transforming) {
      this._transforming = !1, typeof window < "u" && (window.removeEventListener("mousemove", this._handleMouseMove), window.removeEventListener("touchmove", this._handleMouseMove), window.removeEventListener("mouseup", this._handleMouseUp, !0), window.removeEventListener("touchend", this._handleMouseUp, !0));
      var r = this.getNode();
      this._fire("transformend", { evt: e, target: r }), r && this._nodes.forEach((n) => {
        n._fire("transformend", { evt: e, target: n });
      }), this._movingAnchorName = null;
    }
  }
  _fitNodesInto(e, r) {
    var n = this._getNodeRect();
    const i = 1;
    if (W._inRange(e.width, -this.padding() * 2 - i, i)) {
      this.update();
      return;
    }
    if (W._inRange(e.height, -this.padding() * 2 - i, i)) {
      this.update();
      return;
    }
    const s = this.flipEnabled();
    var o = new Ir();
    if (o.rotate(ve.getAngle(this.rotation())), this._movingAnchorName && e.width < 0 && this._movingAnchorName.indexOf("left") >= 0) {
      const g = o.point({
        x: -this.padding() * 2,
        y: 0
      });
      if (e.x += g.x, e.y += g.y, e.width += this.padding() * 2, this._movingAnchorName = this._movingAnchorName.replace("left", "right"), this._anchorDragOffset.x -= g.x, this._anchorDragOffset.y -= g.y, !s) {
        this.update();
        return;
      }
    } else if (this._movingAnchorName && e.width < 0 && this._movingAnchorName.indexOf("right") >= 0) {
      const g = o.point({
        x: this.padding() * 2,
        y: 0
      });
      if (this._movingAnchorName = this._movingAnchorName.replace("right", "left"), this._anchorDragOffset.x -= g.x, this._anchorDragOffset.y -= g.y, e.width += this.padding() * 2, !s) {
        this.update();
        return;
      }
    }
    if (this._movingAnchorName && e.height < 0 && this._movingAnchorName.indexOf("top") >= 0) {
      const g = o.point({
        x: 0,
        y: -this.padding() * 2
      });
      if (e.x += g.x, e.y += g.y, this._movingAnchorName = this._movingAnchorName.replace("top", "bottom"), this._anchorDragOffset.x -= g.x, this._anchorDragOffset.y -= g.y, e.height += this.padding() * 2, !s) {
        this.update();
        return;
      }
    } else if (this._movingAnchorName && e.height < 0 && this._movingAnchorName.indexOf("bottom") >= 0) {
      const g = o.point({
        x: 0,
        y: this.padding() * 2
      });
      if (this._movingAnchorName = this._movingAnchorName.replace("bottom", "top"), this._anchorDragOffset.x -= g.x, this._anchorDragOffset.y -= g.y, e.height += this.padding() * 2, !s) {
        this.update();
        return;
      }
    }
    if (this.boundBoxFunc()) {
      const g = this.boundBoxFunc()(n, e);
      g ? e = g : W.warn("boundBoxFunc returned falsy. You should return new bound rect from it!");
    }
    const l = 1e7, c = new Ir();
    c.translate(n.x, n.y), c.rotate(n.rotation), c.scale(n.width / l, n.height / l);
    const h = new Ir();
    h.translate(e.x, e.y), h.rotate(e.rotation), h.scale(e.width / l, e.height / l);
    const d = h.multiply(c.invert());
    this._nodes.forEach((g) => {
      var p;
      const y = g.getParent().getAbsoluteTransform(), v = g.getTransform().copy();
      v.translate(g.offsetX(), g.offsetY());
      const _ = new Ir();
      _.multiply(y.copy().invert()).multiply(d).multiply(y).multiply(v);
      const b = _.decompose();
      g.setAttrs(b), this._fire("transform", { evt: r, target: g }), g._fire("transform", { evt: r, target: g }), (p = g.getLayer()) === null || p === void 0 || p.batchDraw();
    }), this.rotation(W._getRotation(e.rotation)), this._resetTransformCache(), this.update(), this.getLayer().batchDraw();
  }
  forceUpdate() {
    this._resetTransformCache(), this.update();
  }
  _batchChangeChild(e, r) {
    this.findOne(e).setAttrs(r);
  }
  update() {
    var e, r = this._getNodeRect();
    this.rotation(W._getRotation(r.rotation));
    var n = r.width, i = r.height, s = this.enabledAnchors(), o = this.resizeEnabled(), l = this.padding(), c = this.anchorSize();
    this.find("._anchor").forEach((h) => {
      h.setAttrs({
        width: c,
        height: c,
        offsetX: c / 2,
        offsetY: c / 2,
        stroke: this.anchorStroke(),
        strokeWidth: this.anchorStrokeWidth(),
        fill: this.anchorFill(),
        cornerRadius: this.anchorCornerRadius()
      });
    }), this._batchChangeChild(".top-left", {
      x: 0,
      y: 0,
      offsetX: c / 2 + l,
      offsetY: c / 2 + l,
      visible: o && s.indexOf("top-left") >= 0
    }), this._batchChangeChild(".top-center", {
      x: n / 2,
      y: 0,
      offsetY: c / 2 + l,
      visible: o && s.indexOf("top-center") >= 0
    }), this._batchChangeChild(".top-right", {
      x: n,
      y: 0,
      offsetX: c / 2 - l,
      offsetY: c / 2 + l,
      visible: o && s.indexOf("top-right") >= 0
    }), this._batchChangeChild(".middle-left", {
      x: 0,
      y: i / 2,
      offsetX: c / 2 + l,
      visible: o && s.indexOf("middle-left") >= 0
    }), this._batchChangeChild(".middle-right", {
      x: n,
      y: i / 2,
      offsetX: c / 2 - l,
      visible: o && s.indexOf("middle-right") >= 0
    }), this._batchChangeChild(".bottom-left", {
      x: 0,
      y: i,
      offsetX: c / 2 + l,
      offsetY: c / 2 - l,
      visible: o && s.indexOf("bottom-left") >= 0
    }), this._batchChangeChild(".bottom-center", {
      x: n / 2,
      y: i,
      offsetY: c / 2 - l,
      visible: o && s.indexOf("bottom-center") >= 0
    }), this._batchChangeChild(".bottom-right", {
      x: n,
      y: i,
      offsetX: c / 2 - l,
      offsetY: c / 2 - l,
      visible: o && s.indexOf("bottom-right") >= 0
    }), this._batchChangeChild(".rotater", {
      x: n / 2,
      y: -this.rotateAnchorOffset() * W._sign(i) - l,
      visible: this.rotateEnabled()
    }), this._batchChangeChild(".back", {
      width: n,
      height: i,
      visible: this.borderEnabled(),
      stroke: this.borderStroke(),
      strokeWidth: this.borderStrokeWidth(),
      dash: this.borderDash(),
      x: 0,
      y: 0
    }), (e = this.getLayer()) === null || e === void 0 || e.batchDraw();
  }
  isTransforming() {
    return this._transforming;
  }
  stopTransform() {
    if (this._transforming) {
      this._removeEvents();
      var e = this.findOne("." + this._movingAnchorName);
      e && e.stopDrag();
    }
  }
  destroy() {
    return this.getStage() && this._cursorChange && this.getStage().content && (this.getStage().content.style.cursor = ""), Ka.prototype.destroy.call(this), this.detach(), this._removeEvents(), this;
  }
  toObject() {
    return ue.prototype.toObject.call(this);
  }
  clone(e) {
    var r = ue.prototype.clone.call(this, e);
    return r;
  }
  getClientRect() {
    return this.nodes().length > 0 ? super.getClientRect() : { x: 0, y: 0, width: 0, height: 0 };
  }
}
function kA(t) {
  return t instanceof Array || W.warn("enabledAnchors value should be an array"), t instanceof Array && t.forEach(function(e) {
    El.indexOf(e) === -1 && W.warn("Unknown anchor name: " + e + ". Available names are: " + El.join(", "));
  }), t || [];
}
tt.prototype.className = "Transformer";
St(tt);
I.addGetterSetter(tt, "enabledAnchors", El, kA);
I.addGetterSetter(tt, "flipEnabled", !0, hn());
I.addGetterSetter(tt, "resizeEnabled", !0);
I.addGetterSetter(tt, "anchorSize", 10, le());
I.addGetterSetter(tt, "rotateEnabled", !0);
I.addGetterSetter(tt, "rotationSnaps", []);
I.addGetterSetter(tt, "rotateAnchorOffset", 50, le());
I.addGetterSetter(tt, "rotationSnapTolerance", 5, le());
I.addGetterSetter(tt, "borderEnabled", !0);
I.addGetterSetter(tt, "anchorStroke", "rgb(0, 161, 255)");
I.addGetterSetter(tt, "anchorStrokeWidth", 1, le());
I.addGetterSetter(tt, "anchorFill", "white");
I.addGetterSetter(tt, "anchorCornerRadius", 0, le());
I.addGetterSetter(tt, "borderStroke", "rgb(0, 161, 255)");
I.addGetterSetter(tt, "borderStrokeWidth", 1, le());
I.addGetterSetter(tt, "borderDash");
I.addGetterSetter(tt, "keepRatio", !0);
I.addGetterSetter(tt, "centeredScaling", !1);
I.addGetterSetter(tt, "ignoreStroke", !1);
I.addGetterSetter(tt, "padding", 0, le());
I.addGetterSetter(tt, "node");
I.addGetterSetter(tt, "nodes");
I.addGetterSetter(tt, "boundBoxFunc");
I.addGetterSetter(tt, "anchorDragBoundFunc");
I.addGetterSetter(tt, "shouldOverdrawWholeArea", !1);
I.addGetterSetter(tt, "useSingleNodeRotation", !0);
I.backCompat(tt, {
  lineEnabled: "borderEnabled",
  rotateHandlerOffset: "rotateAnchorOffset",
  enabledHandlers: "enabledAnchors"
});
class ni extends se {
  _sceneFunc(e) {
    e.beginPath(), e.arc(0, 0, this.radius(), 0, ve.getAngle(this.angle()), this.clockwise()), e.lineTo(0, 0), e.closePath(), e.fillStrokeShape(this);
  }
  getWidth() {
    return this.radius() * 2;
  }
  getHeight() {
    return this.radius() * 2;
  }
  setWidth(e) {
    this.radius(e / 2);
  }
  setHeight(e) {
    this.radius(e / 2);
  }
}
ni.prototype.className = "Wedge";
ni.prototype._centroid = !0;
ni.prototype._attrsAffectingSize = ["radius"];
St(ni);
I.addGetterSetter(ni, "radius", 0, le());
I.addGetterSetter(ni, "angle", 0, le());
I.addGetterSetter(ni, "clockwise", !1);
I.backCompat(ni, {
  angleDeg: "angle",
  getAngleDeg: "getAngle",
  setAngleDeg: "setAngle"
});
function ev() {
  this.r = 0, this.g = 0, this.b = 0, this.a = 0, this.next = null;
}
var IA = [
  512,
  512,
  456,
  512,
  328,
  456,
  335,
  512,
  405,
  328,
  271,
  456,
  388,
  335,
  292,
  512,
  454,
  405,
  364,
  328,
  298,
  271,
  496,
  456,
  420,
  388,
  360,
  335,
  312,
  292,
  273,
  512,
  482,
  454,
  428,
  405,
  383,
  364,
  345,
  328,
  312,
  298,
  284,
  271,
  259,
  496,
  475,
  456,
  437,
  420,
  404,
  388,
  374,
  360,
  347,
  335,
  323,
  312,
  302,
  292,
  282,
  273,
  265,
  512,
  497,
  482,
  468,
  454,
  441,
  428,
  417,
  405,
  394,
  383,
  373,
  364,
  354,
  345,
  337,
  328,
  320,
  312,
  305,
  298,
  291,
  284,
  278,
  271,
  265,
  259,
  507,
  496,
  485,
  475,
  465,
  456,
  446,
  437,
  428,
  420,
  412,
  404,
  396,
  388,
  381,
  374,
  367,
  360,
  354,
  347,
  341,
  335,
  329,
  323,
  318,
  312,
  307,
  302,
  297,
  292,
  287,
  282,
  278,
  273,
  269,
  265,
  261,
  512,
  505,
  497,
  489,
  482,
  475,
  468,
  461,
  454,
  447,
  441,
  435,
  428,
  422,
  417,
  411,
  405,
  399,
  394,
  389,
  383,
  378,
  373,
  368,
  364,
  359,
  354,
  350,
  345,
  341,
  337,
  332,
  328,
  324,
  320,
  316,
  312,
  309,
  305,
  301,
  298,
  294,
  291,
  287,
  284,
  281,
  278,
  274,
  271,
  268,
  265,
  262,
  259,
  257,
  507,
  501,
  496,
  491,
  485,
  480,
  475,
  470,
  465,
  460,
  456,
  451,
  446,
  442,
  437,
  433,
  428,
  424,
  420,
  416,
  412,
  408,
  404,
  400,
  396,
  392,
  388,
  385,
  381,
  377,
  374,
  370,
  367,
  363,
  360,
  357,
  354,
  350,
  347,
  344,
  341,
  338,
  335,
  332,
  329,
  326,
  323,
  320,
  318,
  315,
  312,
  310,
  307,
  304,
  302,
  299,
  297,
  294,
  292,
  289,
  287,
  285,
  282,
  280,
  278,
  275,
  273,
  271,
  269,
  267,
  265,
  263,
  261,
  259
], LA = [
  9,
  11,
  12,
  13,
  13,
  14,
  14,
  15,
  15,
  15,
  15,
  16,
  16,
  16,
  16,
  17,
  17,
  17,
  17,
  17,
  17,
  17,
  18,
  18,
  18,
  18,
  18,
  18,
  18,
  18,
  18,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  19,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  20,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  21,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  22,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  23,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24,
  24
];
function DA(t, e) {
  var r = t.data, n = t.width, i = t.height, s, o, l, c, h, d, g, p, y, v, _, b, C, w, O, k, P, M, U, V, Y, X, Z, ce, $ = e + e + 1, R = n - 1, F = i - 1, H = e + 1, re = H * (H + 1) / 2, ae = new ev(), ne = null, oe = ae, de = null, Re = null, Ce = IA[e], rt = LA[e];
  for (l = 1; l < $; l++)
    oe = oe.next = new ev(), l === H && (ne = oe);
  for (oe.next = ae, g = d = 0, o = 0; o < i; o++) {
    for (k = P = M = U = p = y = v = _ = 0, b = H * (V = r[d]), C = H * (Y = r[d + 1]), w = H * (X = r[d + 2]), O = H * (Z = r[d + 3]), p += re * V, y += re * Y, v += re * X, _ += re * Z, oe = ae, l = 0; l < H; l++)
      oe.r = V, oe.g = Y, oe.b = X, oe.a = Z, oe = oe.next;
    for (l = 1; l < H; l++)
      c = d + ((R < l ? R : l) << 2), p += (oe.r = V = r[c]) * (ce = H - l), y += (oe.g = Y = r[c + 1]) * ce, v += (oe.b = X = r[c + 2]) * ce, _ += (oe.a = Z = r[c + 3]) * ce, k += V, P += Y, M += X, U += Z, oe = oe.next;
    for (de = ae, Re = ne, s = 0; s < n; s++)
      r[d + 3] = Z = _ * Ce >> rt, Z !== 0 ? (Z = 255 / Z, r[d] = (p * Ce >> rt) * Z, r[d + 1] = (y * Ce >> rt) * Z, r[d + 2] = (v * Ce >> rt) * Z) : r[d] = r[d + 1] = r[d + 2] = 0, p -= b, y -= C, v -= w, _ -= O, b -= de.r, C -= de.g, w -= de.b, O -= de.a, c = g + ((c = s + e + 1) < R ? c : R) << 2, k += de.r = r[c], P += de.g = r[c + 1], M += de.b = r[c + 2], U += de.a = r[c + 3], p += k, y += P, v += M, _ += U, de = de.next, b += V = Re.r, C += Y = Re.g, w += X = Re.b, O += Z = Re.a, k -= V, P -= Y, M -= X, U -= Z, Re = Re.next, d += 4;
    g += n;
  }
  for (s = 0; s < n; s++) {
    for (P = M = U = k = y = v = _ = p = 0, d = s << 2, b = H * (V = r[d]), C = H * (Y = r[d + 1]), w = H * (X = r[d + 2]), O = H * (Z = r[d + 3]), p += re * V, y += re * Y, v += re * X, _ += re * Z, oe = ae, l = 0; l < H; l++)
      oe.r = V, oe.g = Y, oe.b = X, oe.a = Z, oe = oe.next;
    for (h = n, l = 1; l <= e; l++)
      d = h + s << 2, p += (oe.r = V = r[d]) * (ce = H - l), y += (oe.g = Y = r[d + 1]) * ce, v += (oe.b = X = r[d + 2]) * ce, _ += (oe.a = Z = r[d + 3]) * ce, k += V, P += Y, M += X, U += Z, oe = oe.next, l < F && (h += n);
    for (d = s, de = ae, Re = ne, o = 0; o < i; o++)
      c = d << 2, r[c + 3] = Z = _ * Ce >> rt, Z > 0 ? (Z = 255 / Z, r[c] = (p * Ce >> rt) * Z, r[c + 1] = (y * Ce >> rt) * Z, r[c + 2] = (v * Ce >> rt) * Z) : r[c] = r[c + 1] = r[c + 2] = 0, p -= b, y -= C, v -= w, _ -= O, b -= de.r, C -= de.g, w -= de.b, O -= de.a, c = s + ((c = o + H) < F ? c : F) * n << 2, p += k += de.r = r[c], y += P += de.g = r[c + 1], v += M += de.b = r[c + 2], _ += U += de.a = r[c + 3], de = de.next, b += V = Re.r, C += Y = Re.g, w += X = Re.b, O += Z = Re.a, k -= V, P -= Y, M -= X, U -= Z, Re = Re.next, d += n;
  }
}
const MA = function(e) {
  var r = Math.round(this.blurRadius());
  r > 0 && DA(e, r);
};
I.addGetterSetter(ue, "blurRadius", 0, le(), I.afterSetFilter);
const NA = function(t) {
  var e = this.brightness() * 255, r = t.data, n = r.length, i;
  for (i = 0; i < n; i += 4)
    r[i] += e, r[i + 1] += e, r[i + 2] += e;
};
I.addGetterSetter(ue, "brightness", 0, le(), I.afterSetFilter);
const BA = function(t) {
  var e = Math.pow((this.contrast() + 100) / 100, 2), r = t.data, n = r.length, i = 150, s = 150, o = 150, l;
  for (l = 0; l < n; l += 4)
    i = r[l], s = r[l + 1], o = r[l + 2], i /= 255, i -= 0.5, i *= e, i += 0.5, i *= 255, s /= 255, s -= 0.5, s *= e, s += 0.5, s *= 255, o /= 255, o -= 0.5, o *= e, o += 0.5, o *= 255, i = i < 0 ? 0 : i > 255 ? 255 : i, s = s < 0 ? 0 : s > 255 ? 255 : s, o = o < 0 ? 0 : o > 255 ? 255 : o, r[l] = i, r[l + 1] = s, r[l + 2] = o;
};
I.addGetterSetter(ue, "contrast", 0, le(), I.afterSetFilter);
const FA = function(t) {
  var e = this.embossStrength() * 10, r = this.embossWhiteLevel() * 255, n = this.embossDirection(), i = this.embossBlend(), s = 0, o = 0, l = t.data, c = t.width, h = t.height, d = c * 4, g = h;
  switch (n) {
    case "top-left":
      s = -1, o = -1;
      break;
    case "top":
      s = -1, o = 0;
      break;
    case "top-right":
      s = -1, o = 1;
      break;
    case "right":
      s = 0, o = 1;
      break;
    case "bottom-right":
      s = 1, o = 1;
      break;
    case "bottom":
      s = 1, o = 0;
      break;
    case "bottom-left":
      s = 1, o = -1;
      break;
    case "left":
      s = 0, o = -1;
      break;
    default:
      W.error("Unknown emboss direction: " + n);
  }
  do {
    var p = (g - 1) * d, y = s;
    g + y < 1 && (y = 0), g + y > h && (y = 0);
    var v = (g - 1 + y) * c * 4, _ = c;
    do {
      var b = p + (_ - 1) * 4, C = o;
      _ + C < 1 && (C = 0), _ + C > c && (C = 0);
      var w = v + (_ - 1 + C) * 4, O = l[b] - l[w], k = l[b + 1] - l[w + 1], P = l[b + 2] - l[w + 2], M = O, U = M > 0 ? M : -M, V = k > 0 ? k : -k, Y = P > 0 ? P : -P;
      if (V > U && (M = k), Y > U && (M = P), M *= e, i) {
        var X = l[b] + M, Z = l[b + 1] + M, ce = l[b + 2] + M;
        l[b] = X > 255 ? 255 : X < 0 ? 0 : X, l[b + 1] = Z > 255 ? 255 : Z < 0 ? 0 : Z, l[b + 2] = ce > 255 ? 255 : ce < 0 ? 0 : ce;
      } else {
        var $ = r - M;
        $ < 0 ? $ = 0 : $ > 255 && ($ = 255), l[b] = l[b + 1] = l[b + 2] = $;
      }
    } while (--_);
  } while (--g);
};
I.addGetterSetter(ue, "embossStrength", 0.5, le(), I.afterSetFilter);
I.addGetterSetter(ue, "embossWhiteLevel", 0.5, le(), I.afterSetFilter);
I.addGetterSetter(ue, "embossDirection", "top-left", null, I.afterSetFilter);
I.addGetterSetter(ue, "embossBlend", !1, null, I.afterSetFilter);
function Rf(t, e, r, n, i) {
  var s = r - e, o = i - n, l;
  return s === 0 ? n + o / 2 : o === 0 ? n : (l = (t - e) / s, l = o * l + n, l);
}
const GA = function(t) {
  var e = t.data, r = e.length, n = e[0], i = n, s, o = e[1], l = o, c, h = e[2], d = h, g, p, y = this.enhance();
  if (y !== 0) {
    for (p = 0; p < r; p += 4)
      s = e[p + 0], s < n ? n = s : s > i && (i = s), c = e[p + 1], c < o ? o = c : c > l && (l = c), g = e[p + 2], g < h ? h = g : g > d && (d = g);
    i === n && (i = 255, n = 0), l === o && (l = 255, o = 0), d === h && (d = 255, h = 0);
    var v, _, b, C, w, O, k, P, M;
    for (y > 0 ? (_ = i + y * (255 - i), b = n - y * (n - 0), w = l + y * (255 - l), O = o - y * (o - 0), P = d + y * (255 - d), M = h - y * (h - 0)) : (v = (i + n) * 0.5, _ = i + y * (i - v), b = n + y * (n - v), C = (l + o) * 0.5, w = l + y * (l - C), O = o + y * (o - C), k = (d + h) * 0.5, P = d + y * (d - k), M = h + y * (h - k)), p = 0; p < r; p += 4)
      e[p + 0] = Rf(e[p + 0], n, i, b, _), e[p + 1] = Rf(e[p + 1], o, l, O, w), e[p + 2] = Rf(e[p + 2], h, d, M, P);
  }
};
I.addGetterSetter(ue, "enhance", 0, le(), I.afterSetFilter);
const jA = function(t) {
  var e = t.data, r = e.length, n, i;
  for (n = 0; n < r; n += 4)
    i = 0.34 * e[n] + 0.5 * e[n + 1] + 0.16 * e[n + 2], e[n] = i, e[n + 1] = i, e[n + 2] = i;
};
I.addGetterSetter(ue, "hue", 0, le(), I.afterSetFilter);
I.addGetterSetter(ue, "saturation", 0, le(), I.afterSetFilter);
I.addGetterSetter(ue, "luminance", 0, le(), I.afterSetFilter);
const $A = function(t) {
  var e = t.data, r = e.length, n = 1, i = Math.pow(2, this.saturation()), s = Math.abs(this.hue() + 360) % 360, o = this.luminance() * 127, l, c = n * i * Math.cos(s * Math.PI / 180), h = n * i * Math.sin(s * Math.PI / 180), d = 0.299 * n + 0.701 * c + 0.167 * h, g = 0.587 * n - 0.587 * c + 0.33 * h, p = 0.114 * n - 0.114 * c - 0.497 * h, y = 0.299 * n - 0.299 * c - 0.328 * h, v = 0.587 * n + 0.413 * c + 0.035 * h, _ = 0.114 * n - 0.114 * c + 0.293 * h, b = 0.299 * n - 0.3 * c + 1.25 * h, C = 0.587 * n - 0.586 * c - 1.05 * h, w = 0.114 * n + 0.886 * c - 0.2 * h, O, k, P, M;
  for (l = 0; l < r; l += 4)
    O = e[l + 0], k = e[l + 1], P = e[l + 2], M = e[l + 3], e[l + 0] = d * O + g * k + p * P + o, e[l + 1] = y * O + v * k + _ * P + o, e[l + 2] = b * O + C * k + w * P + o, e[l + 3] = M;
}, WA = function(t) {
  var e = t.data, r = e.length, n = Math.pow(2, this.value()), i = Math.pow(2, this.saturation()), s = Math.abs(this.hue() + 360) % 360, o, l = n * i * Math.cos(s * Math.PI / 180), c = n * i * Math.sin(s * Math.PI / 180), h = 0.299 * n + 0.701 * l + 0.167 * c, d = 0.587 * n - 0.587 * l + 0.33 * c, g = 0.114 * n - 0.114 * l - 0.497 * c, p = 0.299 * n - 0.299 * l - 0.328 * c, y = 0.587 * n + 0.413 * l + 0.035 * c, v = 0.114 * n - 0.114 * l + 0.293 * c, _ = 0.299 * n - 0.3 * l + 1.25 * c, b = 0.587 * n - 0.586 * l - 1.05 * c, C = 0.114 * n + 0.886 * l - 0.2 * c, w, O, k, P;
  for (o = 0; o < r; o += 4)
    w = e[o + 0], O = e[o + 1], k = e[o + 2], P = e[o + 3], e[o + 0] = h * w + d * O + g * k, e[o + 1] = p * w + y * O + v * k, e[o + 2] = _ * w + b * O + C * k, e[o + 3] = P;
};
I.addGetterSetter(ue, "hue", 0, le(), I.afterSetFilter);
I.addGetterSetter(ue, "saturation", 0, le(), I.afterSetFilter);
I.addGetterSetter(ue, "value", 0, le(), I.afterSetFilter);
const UA = function(t) {
  var e = t.data, r = e.length, n;
  for (n = 0; n < r; n += 4)
    e[n] = 255 - e[n], e[n + 1] = 255 - e[n + 1], e[n + 2] = 255 - e[n + 2];
};
var HA = function(t, e, r) {
  var n = t.data, i = e.data, s = t.width, o = t.height, l = r.polarCenterX || s / 2, c = r.polarCenterY || o / 2, h, d, g, p = 0, y = 0, v = 0, _ = 0, b, C = Math.sqrt(l * l + c * c);
  d = s - l, g = o - c, b = Math.sqrt(d * d + g * g), C = b > C ? b : C;
  var w = o, O = s, k, P, M = 360 / O * Math.PI / 180, U, V;
  for (P = 0; P < O; P += 1)
    for (U = Math.sin(P * M), V = Math.cos(P * M), k = 0; k < w; k += 1)
      d = Math.floor(l + C * k / w * V), g = Math.floor(c + C * k / w * U), h = (g * s + d) * 4, p = n[h + 0], y = n[h + 1], v = n[h + 2], _ = n[h + 3], h = (P + k * s) * 4, i[h + 0] = p, i[h + 1] = y, i[h + 2] = v, i[h + 3] = _;
}, zA = function(t, e, r) {
  var n = t.data, i = e.data, s = t.width, o = t.height, l = r.polarCenterX || s / 2, c = r.polarCenterY || o / 2, h, d, g, p, y, v = 0, _ = 0, b = 0, C = 0, w, O = Math.sqrt(l * l + c * c);
  d = s - l, g = o - c, w = Math.sqrt(d * d + g * g), O = w > O ? w : O;
  var k = o, P = s, M, U, V = r.polarRotation || 0, Y, X;
  for (d = 0; d < s; d += 1)
    for (g = 0; g < o; g += 1)
      p = d - l, y = g - c, M = Math.sqrt(p * p + y * y) * k / O, U = (Math.atan2(y, p) * 180 / Math.PI + 360 + V) % 360, U = U * P / 360, Y = Math.floor(U), X = Math.floor(M), h = (X * s + Y) * 4, v = n[h + 0], _ = n[h + 1], b = n[h + 2], C = n[h + 3], h = (g * s + d) * 4, i[h + 0] = v, i[h + 1] = _, i[h + 2] = b, i[h + 3] = C;
};
const VA = function(t) {
  var e = t.width, r = t.height, n, i, s, o, l, c, h, d, g, p, y = Math.round(this.kaleidoscopePower()), v = Math.round(this.kaleidoscopeAngle()), _ = Math.floor(e * (v % 360) / 360);
  if (!(y < 1)) {
    var b = W.createCanvasElement();
    b.width = e, b.height = r;
    var C = b.getContext("2d").getImageData(0, 0, e, r);
    W.releaseCanvas(b), HA(t, C, {
      polarCenterX: e / 2,
      polarCenterY: r / 2
    });
    for (var w = e / Math.pow(2, y); w <= 8; )
      w = w * 2, y -= 1;
    w = Math.ceil(w);
    var O = w, k = 0, P = O, M = 1;
    for (_ + w > e && (k = O, P = 0, M = -1), i = 0; i < r; i += 1)
      for (n = k; n !== P; n += M)
        s = Math.round(n + _) % e, g = (e * i + s) * 4, l = C.data[g + 0], c = C.data[g + 1], h = C.data[g + 2], d = C.data[g + 3], p = (e * i + n) * 4, C.data[p + 0] = l, C.data[p + 1] = c, C.data[p + 2] = h, C.data[p + 3] = d;
    for (i = 0; i < r; i += 1)
      for (O = Math.floor(w), o = 0; o < y; o += 1) {
        for (n = 0; n < O + 1; n += 1)
          g = (e * i + n) * 4, l = C.data[g + 0], c = C.data[g + 1], h = C.data[g + 2], d = C.data[g + 3], p = (e * i + O * 2 - n - 1) * 4, C.data[p + 0] = l, C.data[p + 1] = c, C.data[p + 2] = h, C.data[p + 3] = d;
        O *= 2;
      }
    zA(C, t, { polarRotation: 0 });
  }
};
I.addGetterSetter(ue, "kaleidoscopePower", 2, le(), I.afterSetFilter);
I.addGetterSetter(ue, "kaleidoscopeAngle", 0, le(), I.afterSetFilter);
function Hu(t, e, r) {
  var n = (r * t.width + e) * 4, i = [];
  return i.push(t.data[n++], t.data[n++], t.data[n++], t.data[n++]), i;
}
function Us(t, e) {
  return Math.sqrt(Math.pow(t[0] - e[0], 2) + Math.pow(t[1] - e[1], 2) + Math.pow(t[2] - e[2], 2));
}
function qA(t) {
  for (var e = [0, 0, 0], r = 0; r < t.length; r++)
    e[0] += t[r][0], e[1] += t[r][1], e[2] += t[r][2];
  return e[0] /= t.length, e[1] /= t.length, e[2] /= t.length, e;
}
function YA(t, e) {
  var r = Hu(t, 0, 0), n = Hu(t, t.width - 1, 0), i = Hu(t, 0, t.height - 1), s = Hu(t, t.width - 1, t.height - 1), o = e || 10;
  if (Us(r, n) < o && Us(n, s) < o && Us(s, i) < o && Us(i, r) < o) {
    for (var l = qA([n, r, s, i]), c = [], h = 0; h < t.width * t.height; h++) {
      var d = Us(l, [
        t.data[h * 4],
        t.data[h * 4 + 1],
        t.data[h * 4 + 2]
      ]);
      c[h] = d < o ? 0 : 255;
    }
    return c;
  }
}
function KA(t, e) {
  for (var r = 0; r < t.width * t.height; r++)
    t.data[4 * r + 3] = e[r];
}
function XA(t, e, r) {
  for (var n = [1, 1, 1, 1, 0, 1, 1, 1, 1], i = Math.round(Math.sqrt(n.length)), s = Math.floor(i / 2), o = [], l = 0; l < r; l++)
    for (var c = 0; c < e; c++) {
      for (var h = l * e + c, d = 0, g = 0; g < i; g++)
        for (var p = 0; p < i; p++) {
          var y = l + g - s, v = c + p - s;
          if (y >= 0 && y < r && v >= 0 && v < e) {
            var _ = y * e + v, b = n[g * i + p];
            d += t[_] * b;
          }
        }
      o[h] = d === 255 * 8 ? 255 : 0;
    }
  return o;
}
function JA(t, e, r) {
  for (var n = [1, 1, 1, 1, 1, 1, 1, 1, 1], i = Math.round(Math.sqrt(n.length)), s = Math.floor(i / 2), o = [], l = 0; l < r; l++)
    for (var c = 0; c < e; c++) {
      for (var h = l * e + c, d = 0, g = 0; g < i; g++)
        for (var p = 0; p < i; p++) {
          var y = l + g - s, v = c + p - s;
          if (y >= 0 && y < r && v >= 0 && v < e) {
            var _ = y * e + v, b = n[g * i + p];
            d += t[_] * b;
          }
        }
      o[h] = d >= 255 * 4 ? 255 : 0;
    }
  return o;
}
function QA(t, e, r) {
  for (var n = [0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111], i = Math.round(Math.sqrt(n.length)), s = Math.floor(i / 2), o = [], l = 0; l < r; l++)
    for (var c = 0; c < e; c++) {
      for (var h = l * e + c, d = 0, g = 0; g < i; g++)
        for (var p = 0; p < i; p++) {
          var y = l + g - s, v = c + p - s;
          if (y >= 0 && y < r && v >= 0 && v < e) {
            var _ = y * e + v, b = n[g * i + p];
            d += t[_] * b;
          }
        }
      o[h] = d;
    }
  return o;
}
const ZA = function(t) {
  var e = this.threshold(), r = YA(t, e);
  return r && (r = XA(r, t.width, t.height), r = JA(r, t.width, t.height), r = QA(r, t.width, t.height), KA(t, r)), t;
};
I.addGetterSetter(ue, "threshold", 0, le(), I.afterSetFilter);
const ek = function(t) {
  var e = this.noise() * 255, r = t.data, n = r.length, i = e / 2, s;
  for (s = 0; s < n; s += 4)
    r[s + 0] += i - 2 * i * Math.random(), r[s + 1] += i - 2 * i * Math.random(), r[s + 2] += i - 2 * i * Math.random();
};
I.addGetterSetter(ue, "noise", 0.2, le(), I.afterSetFilter);
const tk = function(t) {
  var e = Math.ceil(this.pixelSize()), r = t.width, n = t.height, i, s, o, l, c, h, d, g = Math.ceil(r / e), p = Math.ceil(n / e), y, v, _, b, C, w, O, k = t.data;
  if (e <= 0) {
    W.error("pixelSize value can not be <= 0");
    return;
  }
  for (C = 0; C < g; C += 1)
    for (w = 0; w < p; w += 1) {
      for (l = 0, c = 0, h = 0, d = 0, y = C * e, v = y + e, _ = w * e, b = _ + e, O = 0, i = y; i < v; i += 1)
        if (!(i >= r))
          for (s = _; s < b; s += 1)
            s >= n || (o = (r * s + i) * 4, l += k[o + 0], c += k[o + 1], h += k[o + 2], d += k[o + 3], O += 1);
      for (l = l / O, c = c / O, h = h / O, d = d / O, i = y; i < v; i += 1)
        if (!(i >= r))
          for (s = _; s < b; s += 1)
            s >= n || (o = (r * s + i) * 4, k[o + 0] = l, k[o + 1] = c, k[o + 2] = h, k[o + 3] = d);
    }
};
I.addGetterSetter(ue, "pixelSize", 8, le(), I.afterSetFilter);
const rk = function(t) {
  var e = Math.round(this.levels() * 254) + 1, r = t.data, n = r.length, i = 255 / e, s;
  for (s = 0; s < n; s += 1)
    r[s] = Math.floor(r[s] / i) * i;
};
I.addGetterSetter(ue, "levels", 0.5, le(), I.afterSetFilter);
const nk = function(t) {
  var e = t.data, r = e.length, n = this.red(), i = this.green(), s = this.blue(), o, l;
  for (o = 0; o < r; o += 4)
    l = (0.34 * e[o] + 0.5 * e[o + 1] + 0.16 * e[o + 2]) / 255, e[o] = l * n, e[o + 1] = l * i, e[o + 2] = l * s, e[o + 3] = e[o + 3];
};
I.addGetterSetter(ue, "red", 0, function(t) {
  return this._filterUpToDate = !1, t > 255 ? 255 : t < 0 ? 0 : Math.round(t);
});
I.addGetterSetter(ue, "green", 0, function(t) {
  return this._filterUpToDate = !1, t > 255 ? 255 : t < 0 ? 0 : Math.round(t);
});
I.addGetterSetter(ue, "blue", 0, n0, I.afterSetFilter);
const ik = function(t) {
  var e = t.data, r = e.length, n = this.red(), i = this.green(), s = this.blue(), o = this.alpha(), l, c;
  for (l = 0; l < r; l += 4)
    c = 1 - o, e[l] = n * o + e[l] * c, e[l + 1] = i * o + e[l + 1] * c, e[l + 2] = s * o + e[l + 2] * c;
};
I.addGetterSetter(ue, "red", 0, function(t) {
  return this._filterUpToDate = !1, t > 255 ? 255 : t < 0 ? 0 : Math.round(t);
});
I.addGetterSetter(ue, "green", 0, function(t) {
  return this._filterUpToDate = !1, t > 255 ? 255 : t < 0 ? 0 : Math.round(t);
});
I.addGetterSetter(ue, "blue", 0, n0, I.afterSetFilter);
I.addGetterSetter(ue, "alpha", 1, function(t) {
  return this._filterUpToDate = !1, t > 1 ? 1 : t < 0 ? 0 : t;
});
const ak = function(t) {
  var e = t.data, r = e.length, n, i, s, o;
  for (n = 0; n < r; n += 4)
    i = e[n + 0], s = e[n + 1], o = e[n + 2], e[n + 0] = Math.min(255, i * 0.393 + s * 0.769 + o * 0.189), e[n + 1] = Math.min(255, i * 0.349 + s * 0.686 + o * 0.168), e[n + 2] = Math.min(255, i * 0.272 + s * 0.534 + o * 0.131);
}, sk = function(t) {
  var e = t.data, r = t.width, n = t.height, i = r * 4, s = n;
  do {
    var o = (s - 1) * i, l = r;
    do {
      var c = o + (l - 1) * 4, h = e[c], d = e[c + 1], g = e[c + 2];
      h > 127 && (h = 255 - h), d > 127 && (d = 255 - d), g > 127 && (g = 255 - g), e[c] = h, e[c + 1] = d, e[c + 2] = g;
    } while (--l);
  } while (--s);
}, ok = function(t) {
  var e = this.threshold() * 255, r = t.data, n = r.length, i;
  for (i = 0; i < n; i += 1)
    r[i] = r[i] < e ? 0 : 255;
};
I.addGetterSetter(ue, "threshold", 0.5, le(), I.afterSetFilter);
const Hs = Vm.Util._assign(Vm, {
  Arc: ti,
  Arrow: ea,
  Circle: ts,
  Ellipse: Pi,
  Image: In,
  Label: Gh,
  Tag: ta,
  Line: ri,
  Path: st,
  Rect: bo,
  RegularPolygon: ra,
  Ring: na,
  Sprite: Ln,
  Star: Ri,
  Text: _t,
  TextPath: jt,
  Transformer: tt,
  Wedge: ni,
  Filters: {
    Blur: MA,
    Brighten: NA,
    Contrast: BA,
    Emboss: FA,
    Enhance: GA,
    Grayscale: jA,
    HSL: $A,
    HSV: WA,
    Invert: UA,
    Kaleidoscope: VA,
    Mask: ZA,
    Noise: ek,
    Pixelate: tk,
    Posterize: rk,
    RGB: nk,
    RGBA: ik,
    Sepia: ak,
    Solarize: sk,
    Threshold: ok
  }
}), uk = (t, e) => {
  const { width: r, height: n } = e, i = document.createElement("div"), s = new Hs.Stage({
    container: i,
    width: r,
    height: n
  }), o = new Hs.Layer(), l = new Hs.Layer();
  o.add(
    new Hs.Rect({
      ...e,
      fill: "white"
    })
  ), t.forEach(
    (h) => l.add(
      new Hs.Line({
        points: h.points,
        stroke: "black",
        strokeWidth: h.strokeWidth * 2,
        tension: 0,
        lineCap: "round",
        lineJoin: "round",
        shadowForStrokeEnabled: !1,
        globalCompositeOperation: h.tool === "brush" ? "source-over" : "destination-out"
      })
    )
  ), s.add(o), s.add(l);
  const c = s.toDataURL({ ...e });
  return i.remove(), c;
};
let M0 = null, N0 = null;
const GB = (t) => {
  M0 = t;
}, lk = () => M0, jB = (t) => {
  N0 = t;
}, $B = () => N0, ck = (t) => {
  const e = window.open("");
  e && t.forEach((r) => {
    const n = new Image();
    n.src = r.base64, e.document.write(r.caption), e.document.write("</br>"), e.document.write(n.outerHTML), e.document.write("</br></br>");
  });
}, fk = (t, e) => Math.floor(Math.random() * (e - t + 1) + t), hk = (t) => {
  const e = lk(), {
    generationMode: r,
    generationState: n,
    postprocessingState: i,
    canvasState: s,
    systemState: o
  } = t, {
    codeformerFidelity: l,
    facetoolStrength: c,
    facetoolType: h,
    hiresFix: d,
    hiresStrength: g,
    shouldRunESRGAN: p,
    shouldRunFacetool: y,
    upscalingLevel: v,
    upscalingStrength: _,
    upscalingDenoising: b
  } = i, {
    cfgScale: C,
    height: w,
    img2imgStrength: O,
    infillMethod: k,
    initialImage: P,
    iterations: M,
    perlin: U,
    prompt: V,
    negativePrompt: Y,
    sampler: X,
    seamBlur: Z,
    seamless: ce,
    seamSize: $,
    seamSteps: R,
    seamStrength: F,
    seed: H,
    seedWeights: re,
    shouldFitToWidthHeight: ae,
    shouldGenerateVariations: ne,
    shouldRandomizeSeed: oe,
    steps: de,
    threshold: Re,
    tileSize: Ce,
    variationAmount: rt,
    width: xt,
    shouldUseSymmetry: be,
    horizontalSymmetrySteps: At,
    verticalSymmetrySteps: ot
  } = n, {
    shouldDisplayInProgressType: wt,
    saveIntermediatesInterval: kt,
    enableImageDebugging: dt
  } = o, te = {
    prompt: V,
    iterations: M,
    steps: de,
    cfg_scale: C,
    threshold: Re,
    perlin: U,
    height: w,
    width: xt,
    sampler_name: X,
    seed: H,
    progress_images: wt === "full-res",
    progress_latents: wt === "latents",
    save_intermediates: kt,
    generation_mode: r,
    init_mask: ""
  };
  let $t = !1, fe = !1;
  if (Y !== "" && (te.prompt = `${V} [${Y}]`), te.seed = oe ? fk(XP, JP) : H, be && (At > 0 && (te.h_symmetry_time_pct = Math.max(
    0,
    Math.min(1, At / de)
  )), ot > 0 && (te.v_symmetry_time_pct = Math.max(
    0,
    Math.min(1, ot / de)
  ))), r === "txt2img" && (te.hires_fix = d, d && (te.strength = g)), ["txt2img", "img2img"].includes(r) && (te.seamless = ce, p && ($t = {
    level: v,
    denoise_str: b,
    strength: _
  }), y && (fe = {
    type: h,
    strength: c
  }, h === "codeformer" && (fe.codeformer_fidelity = l))), r === "img2img" && P && (te.init_img = typeof P == "string" ? P : P.url, te.strength = O, te.fit = ae), r === "unifiedCanvas" && e) {
    const {
      layerState: { objects: Ot },
      boundingBoxCoordinates: sr,
      boundingBoxDimensions: Dt,
      stageScale: dn,
      isMaskEnabled: gn,
      shouldPreserveMaskedArea: gr,
      boundingBoxScaleMethod: Mt,
      scaledBoundingBoxDimensions: qt
    } = s, yt = {
      ...sr,
      ...Dt
    }, Dn = uk(
      gn ? Ot.filter(py) : [],
      yt
    );
    te.init_mask = Dn, te.fit = !1, te.strength = O, te.invert_mask = gr, te.bounding_box = yt;
    const Sr = e.scale();
    e.scale({
      x: 1 / dn,
      y: 1 / dn
    });
    const Mn = e.getAbsolutePosition(), er = e.toDataURL({
      x: yt.x + Mn.x,
      y: yt.y + Mn.y,
      width: yt.width,
      height: yt.height
    });
    dt && ck([
      { base64: Dn, caption: "mask sent as init_mask" },
      { base64: er, caption: "image sent as init_img" }
    ]), e.scale(Sr), te.init_img = er, te.progress_images = !1, Mt !== "none" && (te.inpaint_width = qt.width, te.inpaint_height = qt.height), te.seam_size = $, te.seam_blur = Z, te.seam_strength = F, te.seam_steps = R, te.tile_size = Ce, te.infill_method = k, te.force_outpaint = !1;
  }
  return ne ? (te.variation_amount = rt, re && (te.with_variations = vO(re))) : te.variation_amount = 0, dt && (te.enable_image_debugging = dt), {
    generationParameters: te,
    esrganParameters: $t,
    facetoolParameters: fe
  };
};
var dk = /d{1,4}|D{3,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|W{1,2}|[LlopSZN]|"[^"]*"|'[^']*'/g, gk = /\b(?:[A-Z]{1,3}[A-Z][TC])(?:[-+]\d{4})?|((?:Australian )?(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time)\b/g, pk = /[^-+\dA-Z]/g;
function zt(t, e, r, n) {
  if (arguments.length === 1 && typeof t == "string" && !/\d/.test(t) && (e = t, t = void 0), t = t || t === 0 ? t : new Date(), t instanceof Date || (t = new Date(t)), isNaN(t))
    throw TypeError("Invalid date");
  e = String(tv[e] || e || tv.default);
  var i = e.slice(0, 4);
  (i === "UTC:" || i === "GMT:") && (e = e.slice(4), r = !0, i === "GMT:" && (n = !0));
  var s = function() {
    return r ? "getUTC" : "get";
  }, o = function() {
    return t[s() + "Date"]();
  }, l = function() {
    return t[s() + "Day"]();
  }, c = function() {
    return t[s() + "Month"]();
  }, h = function() {
    return t[s() + "FullYear"]();
  }, d = function() {
    return t[s() + "Hours"]();
  }, g = function() {
    return t[s() + "Minutes"]();
  }, p = function() {
    return t[s() + "Seconds"]();
  }, y = function() {
    return t[s() + "Milliseconds"]();
  }, v = function() {
    return r ? 0 : t.getTimezoneOffset();
  }, _ = function() {
    return mk(t);
  }, b = function() {
    return vk(t);
  }, C = { d: function() {
    return o();
  }, dd: function() {
    return kr(o());
  }, ddd: function() {
    return br.dayNames[l()];
  }, DDD: function() {
    return rv({ y: h(), m: c(), d: o(), _: s(), dayName: br.dayNames[l()], short: !0 });
  }, dddd: function() {
    return br.dayNames[l() + 7];
  }, DDDD: function() {
    return rv({ y: h(), m: c(), d: o(), _: s(), dayName: br.dayNames[l() + 7] });
  }, m: function() {
    return c() + 1;
  }, mm: function() {
    return kr(c() + 1);
  }, mmm: function() {
    return br.monthNames[c()];
  }, mmmm: function() {
    return br.monthNames[c() + 12];
  }, yy: function() {
    return String(h()).slice(2);
  }, yyyy: function() {
    return kr(h(), 4);
  }, h: function() {
    return d() % 12 || 12;
  }, hh: function() {
    return kr(d() % 12 || 12);
  }, H: function() {
    return d();
  }, HH: function() {
    return kr(d());
  }, M: function() {
    return g();
  }, MM: function() {
    return kr(g());
  }, s: function() {
    return p();
  }, ss: function() {
    return kr(p());
  }, l: function() {
    return kr(y(), 3);
  }, L: function() {
    return kr(Math.floor(y() / 10));
  }, t: function() {
    return d() < 12 ? br.timeNames[0] : br.timeNames[1];
  }, tt: function() {
    return d() < 12 ? br.timeNames[2] : br.timeNames[3];
  }, T: function() {
    return d() < 12 ? br.timeNames[4] : br.timeNames[5];
  }, TT: function() {
    return d() < 12 ? br.timeNames[6] : br.timeNames[7];
  }, Z: function() {
    return n ? "GMT" : r ? "UTC" : yk(t);
  }, o: function() {
    return (v() > 0 ? "-" : "+") + kr(Math.floor(Math.abs(v()) / 60) * 100 + Math.abs(v()) % 60, 4);
  }, p: function() {
    return (v() > 0 ? "-" : "+") + kr(Math.floor(Math.abs(v()) / 60), 2) + ":" + kr(Math.floor(Math.abs(v()) % 60), 2);
  }, S: function() {
    return ["th", "st", "nd", "rd"][o() % 10 > 3 ? 0 : (o() % 100 - o() % 10 != 10) * o() % 10];
  }, W: function() {
    return _();
  }, WW: function() {
    return kr(_());
  }, N: function() {
    return b();
  } };
  return e.replace(dk, function(w) {
    return w in C ? C[w]() : w.slice(1, w.length - 1);
  });
}
var tv = { default: "ddd mmm dd yyyy HH:MM:ss", shortDate: "m/d/yy", paddedShortDate: "mm/dd/yyyy", mediumDate: "mmm d, yyyy", longDate: "mmmm d, yyyy", fullDate: "dddd, mmmm d, yyyy", shortTime: "h:MM TT", mediumTime: "h:MM:ss TT", longTime: "h:MM:ss TT Z", isoDate: "yyyy-mm-dd", isoTime: "HH:MM:ss", isoDateTime: "yyyy-mm-dd'T'HH:MM:sso", isoUtcDateTime: "UTC:yyyy-mm-dd'T'HH:MM:ss'Z'", expiresHeaderFormat: "ddd, dd mmm yyyy HH:MM:ss Z" }, br = { dayNames: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], timeNames: ["a", "p", "am", "pm", "A", "P", "AM", "PM"] }, kr = function(e) {
  var r = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : 2;
  return String(e).padStart(r, "0");
}, rv = function(e) {
  var r = e.y, n = e.m, i = e.d, s = e._, o = e.dayName, l = e.short, c = l === void 0 ? !1 : l, h = new Date(), d = new Date();
  d.setDate(d[s + "Date"]() - 1);
  var g = new Date();
  g.setDate(g[s + "Date"]() + 1);
  var p = function() {
    return h[s + "Date"]();
  }, y = function() {
    return h[s + "Month"]();
  }, v = function() {
    return h[s + "FullYear"]();
  }, _ = function() {
    return d[s + "Date"]();
  }, b = function() {
    return d[s + "Month"]();
  }, C = function() {
    return d[s + "FullYear"]();
  }, w = function() {
    return g[s + "Date"]();
  }, O = function() {
    return g[s + "Month"]();
  }, k = function() {
    return g[s + "FullYear"]();
  };
  return v() === r && y() === n && p() === i ? c ? "Tdy" : "Today" : C() === r && b() === n && _() === i ? c ? "Ysd" : "Yesterday" : k() === r && O() === n && w() === i ? c ? "Tmw" : "Tomorrow" : o;
}, mk = function(e) {
  var r = new Date(e.getFullYear(), e.getMonth(), e.getDate());
  r.setDate(r.getDate() - (r.getDay() + 6) % 7 + 3);
  var n = new Date(r.getFullYear(), 0, 4);
  n.setDate(n.getDate() - (n.getDay() + 6) % 7 + 3);
  var i = r.getTimezoneOffset() - n.getTimezoneOffset();
  r.setHours(r.getHours() - i);
  var s = (r - n) / (864e5 * 7);
  return 1 + Math.floor(s);
}, vk = function(e) {
  var r = e.getDay();
  return r === 0 && (r = 7), r;
}, yk = function(e) {
  return (String(e).match(gk) || [""]).pop().replace(pk, "").replace(/GMT\+0000/g, "UTC");
};
const bk = (t, e) => {
  const { dispatch: r, getState: n } = t;
  return {
    emitGenerateImage: (i) => {
      r(En(!0));
      const s = n(), {
        generation: o,
        postprocessing: l,
        system: c,
        canvas: h
      } = s, d = {
        generationMode: i,
        generationState: o,
        postprocessingState: l,
        canvasState: h,
        systemState: c
      };
      r(rP());
      const { generationParameters: g, esrganParameters: p, facetoolParameters: y } = hk(d);
      e.emit(
        "generateImage",
        g,
        p,
        y
      ), g.init_mask && (g.init_mask = g.init_mask.substr(0, 64).concat("...")), g.init_img && (g.init_img = g.init_img.substr(0, 64).concat("...")), r(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Image generation requested: ${JSON.stringify({
            ...g,
            ...p,
            ...y
          })}`
        })
      );
    },
    emitRunESRGAN: (i) => {
      r(En(!0));
      const {
        postprocessing: {
          upscalingLevel: s,
          upscalingDenoising: o,
          upscalingStrength: l
        }
      } = n(), c = {
        upscale: [s, o, l]
      };
      e.emit("runPostprocessing", i, {
        type: "esrgan",
        ...c
      }), r(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `ESRGAN upscale requested: ${JSON.stringify({
            file: i.url,
            ...c
          })}`
        })
      );
    },
    emitRunFacetool: (i) => {
      r(En(!0));
      const {
        postprocessing: { facetoolType: s, facetoolStrength: o, codeformerFidelity: l }
      } = n(), c = {
        facetool_strength: o
      };
      s === "codeformer" && (c.codeformer_fidelity = l), e.emit("runPostprocessing", i, {
        type: s,
        ...c
      }), r(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Face restoration (${s}) requested: ${JSON.stringify(
            {
              file: i.url,
              ...c
            }
          )}`
        })
      );
    },
    emitDeleteImage: (i) => {
      const { url: s, uuid: o, category: l, thumbnail: c } = i;
      r(yy(i)), e.emit("deleteImage", s, c, o, l);
    },
    emitRequestImages: (i) => {
      const s = n().gallery, { earliest_mtime: o } = s.categories[i];
      e.emit("requestImages", i, o);
    },
    emitRequestNewImages: (i) => {
      const s = n().gallery, { latest_mtime: o } = s.categories[i];
      e.emit("requestLatestImages", i, o);
    },
    emitCancelProcessing: () => {
      e.emit("cancel");
    },
    emitRequestSystemConfig: () => {
      e.emit("requestSystemConfig");
    },
    emitSearchForModels: (i) => {
      e.emit("searchForModels", i);
    },
    emitAddNewModel: (i) => {
      e.emit("addNewModel", i);
    },
    emitDeleteModel: (i) => {
      e.emit("deleteModel", i);
    },
    emitConvertToDiffusers: (i) => {
      r(eP()), e.emit("convertToDiffusers", i);
    },
    emitMergeDiffusersModels: (i) => {
      r(tP()), e.emit("mergeDiffusersModels", i);
    },
    emitRequestModelChange: (i) => {
      r(ZT()), e.emit("requestModelChange", i);
    },
    emitSaveStagingAreaImageToGallery: (i) => {
      e.emit("requestSaveStagingAreaImageToGallery", i);
    },
    emitRequestEmptyTempFolder: () => {
      e.emit("requestEmptyTempFolder");
    }
  };
};
let zu;
const _k = new Uint8Array(16);
function Sk() {
  if (!zu && (zu = typeof crypto < "u" && crypto.getRandomValues && crypto.getRandomValues.bind(crypto), !zu))
    throw new Error("crypto.getRandomValues() not supported. See https://github.com/uuidjs/uuid#getrandomvalues-not-supported");
  return zu(_k);
}
const Jt = [];
for (let t = 0; t < 256; ++t)
  Jt.push((t + 256).toString(16).slice(1));
function wk(t, e = 0) {
  return (Jt[t[e + 0]] + Jt[t[e + 1]] + Jt[t[e + 2]] + Jt[t[e + 3]] + "-" + Jt[t[e + 4]] + Jt[t[e + 5]] + "-" + Jt[t[e + 6]] + Jt[t[e + 7]] + "-" + Jt[t[e + 8]] + Jt[t[e + 9]] + "-" + Jt[t[e + 10]] + Jt[t[e + 11]] + Jt[t[e + 12]] + Jt[t[e + 13]] + Jt[t[e + 14]] + Jt[t[e + 15]]).toLowerCase();
}
const Ek = typeof crypto < "u" && crypto.randomUUID && crypto.randomUUID.bind(crypto), nv = {
  randomUUID: Ek
};
function Vu(t, e, r) {
  if (nv.randomUUID && !e && !t)
    return nv.randomUUID();
  t = t || {};
  const n = t.random || (t.rng || Sk)();
  if (n[6] = n[6] & 15 | 64, n[8] = n[8] & 63 | 128, e) {
    r = r || 0;
    for (let i = 0; i < 16; ++i)
      e[r + i] = n[i];
    return e;
  }
  return wk(n);
}
const WB = vt(
  "socketio/generateImage"
), UB = vt("socketio/runESRGAN"), HB = vt("socketio/runFacetool"), zB = vt("socketio/deleteImage"), iv = vt(
  "socketio/requestImages"
), av = vt(
  "socketio/requestNewImages"
), VB = vt(
  "socketio/cancelProcessing"
), Ck = vt(
  "socketio/requestSystemConfig"
), qB = vt("socketio/searchForModels"), YB = vt("socketio/addNewModel"), KB = vt("socketio/deleteModel"), XB = vt(
  "socketio/convertToDiffusers"
), JB = vt(
  "socketio/mergeDiffusersModels"
), QB = vt(
  "socketio/requestModelChange"
), ZB = vt(
  "socketio/saveStagingAreaImageToGallery"
), e5 = vt(
  "socketio/requestEmptyTempFolder"
), xk = (t) => {
  const { dispatch: e, getState: r } = t;
  return {
    /**
     * Callback to run when we receive a 'connect' event.
     */
    onConnect: () => {
      try {
        e(wm(!0)), e(Ba(De.t("common.statusConnected"))), e(Ck());
        const n = r().gallery;
        n.categories.result.latest_mtime ? e(av("result")) : e(iv("result")), n.categories.user.latest_mtime ? e(av("user")) : e(iv("user"));
      } catch (n) {
        console.error(n);
      }
    },
    /**
     * Callback to run when we receive a 'disconnect' event.
     */
    onDisconnect: () => {
      try {
        e(wm(!1)), e(Ba(De.t("common.statusDisconnected"))), e(
          Ht({
            timestamp: zt(new Date(), "isoDateTime"),
            message: "Disconnected from server",
            level: "warning"
          })
        );
      } catch (n) {
        console.error(n);
      }
    },
    /**
     * Callback to run when we receive a 'generationResult' event.
     */
    onGenerationResult: (n) => {
      try {
        const i = r(), { activeTab: s } = i.ui, { shouldLoopback: o } = i.postprocessing, { boundingBox: l, generationMode: c, ...h } = n, d = {
          uuid: Vu(),
          ...h
        };
        if (["txt2img", "img2img"].includes(c) && e(
          Pu({
            category: "result",
            image: { ...d, category: "result" }
          })
        ), c === "unifiedCanvas" && n.boundingBox) {
          const { boundingBox: g } = n;
          e(
            aO({
              image: { ...d, category: "temp" },
              boundingBox: g
            })
          ), i.canvas.shouldAutoSave && e(
            Pu({
              image: { ...d, category: "result" },
              category: "result"
            })
          );
        }
        if (o)
          switch (Gy[s]) {
            case "img2img": {
              e(SO(d));
              break;
            }
          }
        e(yf()), e(
          Ht({
            timestamp: zt(new Date(), "isoDateTime"),
            message: `Image generated: ${n.url}`
          })
        );
      } catch (i) {
        console.error(i);
      }
    },
    /**
     * Callback to run when we receive a 'intermediateResult' event.
     */
    onIntermediateResult: (n) => {
      try {
        e(
          lO({
            uuid: Vu(),
            ...n,
            category: "result"
          })
        ), n.isBase64 || e(
          Ht({
            timestamp: zt(new Date(), "isoDateTime"),
            message: `Intermediate image generated: ${n.url}`
          })
        );
      } catch (i) {
        console.error(i);
      }
    },
    /**
     * Callback to run when we receive an 'esrganResult' event.
     */
    onPostprocessingResult: (n) => {
      try {
        e(
          Pu({
            category: "result",
            image: {
              uuid: Vu(),
              ...n,
              category: "result"
            }
          })
        ), e(
          Ht({
            timestamp: zt(new Date(), "isoDateTime"),
            message: `Postprocessed: ${n.url}`
          })
        );
      } catch (i) {
        console.error(i);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     * TODO: Add additional progress phases
     */
    onProgressUpdate: (n) => {
      try {
        e(En(!0)), e(XT(n));
      } catch (i) {
        console.error(i);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     */
    onError: (n) => {
      const { message: i, additionalData: s } = n;
      try {
        e(
          Ht({
            timestamp: zt(new Date(), "isoDateTime"),
            message: `Server error: ${i}`,
            level: "error"
          })
        ), e(Em()), e(yf());
      } catch (o) {
        console.error(o);
      }
    },
    /**
     * Callback to run when we receive a 'galleryImages' event.
     */
    onGalleryImages: (n) => {
      const { images: i, areMoreImagesAvailable: s, category: o } = n, l = i.map((c) => ({
        uuid: Vu(),
        ...c
      }));
      e(
        uO({
          images: l,
          areMoreImagesAvailable: s,
          category: o
        })
      ), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Loaded ${i.length} images`
        })
      );
    },
    /**
     * Callback to run when we receive a 'processingCanceled' event.
     */
    onProcessingCanceled: () => {
      e(QT());
      const { intermediateImage: n } = r().gallery;
      n && (n.isBase64 || (e(
        Pu({
          category: "result",
          image: n
        })
      ), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Intermediate image saved: ${n.url}`
        })
      )), e(yf())), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: "Processing canceled",
          level: "warning"
        })
      );
    },
    /**
     * Callback to run when we receive a 'imageDeleted' event.
     */
    onImageDeleted: (n) => {
      const { url: i } = n;
      e(yy(n));
      const {
        generation: { initialImage: s, maskPath: o }
      } = r();
      (s === i || (s == null ? void 0 : s.url) === i) && e(bO()), o === i && e(wO("")), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Image deleted: ${i}`
        })
      );
    },
    onSystemConfig: (n) => {
      e(JT(n)), n.infill_methods.includes("patchmatch") || e(_O(n.infill_methods[0]));
    },
    onFoundModels: (n) => {
      const { search_folder: i, found_models: s } = n;
      e(nP(i)), e(iP(s));
    },
    onNewModelAdded: (n) => {
      const { new_model_name: i, model_list: s, update: o } = n;
      e(Fa(s)), e(En(!1)), e(Ba(De.t("modelManager.modelAdded"))), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Model Added: ${i}`,
          level: "info"
        })
      ), e(
        Fs({
          title: o ? `${De.t("modelManager.modelUpdated")}: ${i}` : `${De.t("modelManager.modelAdded")}: ${i}`,
          status: "success",
          duration: 2500,
          isClosable: !0
        })
      );
    },
    onModelDeleted: (n) => {
      const { deleted_model_name: i, model_list: s } = n;
      e(Fa(s)), e(En(!1)), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `${De.t(
            "modelManager.modelAdded"
          )}: ${i}`,
          level: "info"
        })
      ), e(
        Fs({
          title: `${De.t(
            "modelManager.modelEntryDeleted"
          )}: ${i}`,
          status: "success",
          duration: 2500,
          isClosable: !0
        })
      );
    },
    onModelConverted: (n) => {
      const { converted_model_name: i, model_list: s } = n;
      e(Fa(s)), e(Ba(De.t("common.statusModelConverted"))), e(En(!1)), e(ku(!0)), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Model converted: ${i}`,
          level: "info"
        })
      ), e(
        Fs({
          title: `${De.t(
            "modelManager.modelConverted"
          )}: ${i}`,
          status: "success",
          duration: 2500,
          isClosable: !0
        })
      );
    },
    onModelsMerged: (n) => {
      const { merged_models: i, merged_model_name: s, model_list: o } = n;
      e(Fa(o)), e(Ba(De.t("common.statusMergedModels"))), e(En(!1)), e(ku(!0)), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Models merged: ${i}`,
          level: "info"
        })
      ), e(
        Fs({
          title: `${De.t("modelManager.modelsMerged")}: ${s}`,
          status: "success",
          duration: 2500,
          isClosable: !0
        })
      );
    },
    onModelChanged: (n) => {
      const { model_name: i, model_list: s } = n;
      e(Fa(s)), e(Ba(De.t("common.statusModelChanged"))), e(En(!1)), e(ku(!0)), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Model changed: ${i}`,
          level: "info"
        })
      );
    },
    onModelChangeFailed: (n) => {
      const { model_name: i, model_list: s } = n;
      e(Fa(s)), e(En(!1)), e(ku(!0)), e(Em()), e(
        Ht({
          timestamp: zt(new Date(), "isoDateTime"),
          message: `Model change failed: ${i}`,
          level: "error"
        })
      );
    },
    onTempFolderEmptied: () => {
      e(
        Fs({
          title: De.t("toast.tempFoldersEmptied"),
          status: "success",
          duration: 2500,
          isClosable: !0
        })
      );
    }
  };
}, Ok = () => {
  const { origin: t } = new URL(window.location.href), e = rl(t, {
    timeout: 6e4,
    path: `${window.location.pathname}socket.io`
  });
  let r = !1;
  return (i) => (s) => (o) => {
    const {
      onConnect: l,
      onDisconnect: c,
      onError: h,
      onPostprocessingResult: d,
      onGenerationResult: g,
      onIntermediateResult: p,
      onProgressUpdate: y,
      onGalleryImages: v,
      onProcessingCanceled: _,
      onImageDeleted: b,
      onSystemConfig: C,
      onModelChanged: w,
      onFoundModels: O,
      onNewModelAdded: k,
      onModelDeleted: P,
      onModelConverted: M,
      onModelsMerged: U,
      onModelChangeFailed: V,
      onTempFolderEmptied: Y
    } = xk(i), {
      emitGenerateImage: X,
      emitRunESRGAN: Z,
      emitRunFacetool: ce,
      emitDeleteImage: $,
      emitRequestImages: R,
      emitRequestNewImages: F,
      emitCancelProcessing: H,
      emitRequestSystemConfig: re,
      emitSearchForModels: ae,
      emitAddNewModel: ne,
      emitDeleteModel: oe,
      emitConvertToDiffusers: de,
      emitMergeDiffusersModels: Re,
      emitRequestModelChange: Ce,
      emitSaveStagingAreaImageToGallery: rt,
      emitRequestEmptyTempFolder: xt
    } = bk(i, e);
    switch (r || (e.on("connect", () => l()), e.on("disconnect", () => c()), e.on("error", (be) => h(be)), e.on(
      "generationResult",
      (be) => g(be)
    ), e.on(
      "postprocessingResult",
      (be) => d(be)
    ), e.on(
      "intermediateResult",
      (be) => p(be)
    ), e.on(
      "progressUpdate",
      (be) => y(be)
    ), e.on(
      "galleryImages",
      (be) => v(be)
    ), e.on("processingCanceled", () => {
      _();
    }), e.on("imageDeleted", (be) => {
      b(be);
    }), e.on("systemConfig", (be) => {
      C(be);
    }), e.on("foundModels", (be) => {
      O(be);
    }), e.on("newModelAdded", (be) => {
      k(be);
    }), e.on("modelDeleted", (be) => {
      P(be);
    }), e.on("modelConverted", (be) => {
      M(be);
    }), e.on("modelsMerged", (be) => {
      U(be);
    }), e.on("modelChanged", (be) => {
      w(be);
    }), e.on("modelChangeFailed", (be) => {
      V(be);
    }), e.on("tempFolderEmptied", () => {
      Y();
    }), r = !0), o.type) {
      case "socketio/generateImage": {
        X(o.payload);
        break;
      }
      case "socketio/runESRGAN": {
        Z(o.payload);
        break;
      }
      case "socketio/runFacetool": {
        ce(o.payload);
        break;
      }
      case "socketio/deleteImage": {
        $(o.payload);
        break;
      }
      case "socketio/requestImages": {
        R(o.payload);
        break;
      }
      case "socketio/requestNewImages": {
        F(o.payload);
        break;
      }
      case "socketio/cancelProcessing": {
        H();
        break;
      }
      case "socketio/requestSystemConfig": {
        re();
        break;
      }
      case "socketio/searchForModels": {
        ae(o.payload);
        break;
      }
      case "socketio/addNewModel": {
        ne(o.payload);
        break;
      }
      case "socketio/deleteModel": {
        oe(o.payload);
        break;
      }
      case "socketio/convertToDiffusers": {
        de(o.payload);
        break;
      }
      case "socketio/mergeDiffusersModels": {
        Re(o.payload);
        break;
      }
      case "socketio/requestModelChange": {
        Ce(o.payload);
        break;
      }
      case "socketio/saveStagingAreaImageToGallery": {
        rt(o.payload);
        break;
      }
      case "socketio/requestEmptyTempFolder": {
        xt();
        break;
      }
    }
    s(o);
  };
}, Tk = [
  "cursorPosition",
  "isCanvasInitialized",
  "doesCanvasNeedScaling"
].map((t) => `canvas.${t}`), Pk = [
  "currentIteration",
  "currentStatus",
  "currentStep",
  "isCancelable",
  "isConnected",
  "isESRGANAvailable",
  "isGFPGANAvailable",
  "isProcessing",
  "socketId",
  "totalIterations",
  "totalSteps",
  "openModel",
  "cancelOptions.cancelAfter"
].map((t) => `system.${t}`), Rk = [
  "categories",
  "currentCategory",
  "currentImage",
  "currentImageUuid",
  "shouldAutoSwitchToNewImages",
  "shouldHoldGalleryOpen",
  "intermediateImage"
].map((t) => `gallery.${t}`), B0 = Xv({
  generation: EO,
  postprocessing: xO,
  gallery: cO,
  system: aP,
  canvas: sO,
  ui: uP,
  lightbox: dO
}), Ak = dy.getPersistConfig({
  key: "root",
  storage: hy,
  rootReducer: B0,
  blacklist: [...Tk, ...Pk, ...Rk],
  debounce: 300
}), kk = G2(Ak, B0), F0 = p2({
  reducer: kk,
  middleware: (t) => t({
    immutableCheck: !1,
    serializableCheck: !1
  }).concat(Ok()),
  devTools: {
    // Uncommenting these very rapidly called actions makes the redux dev tools output much more readable
    actionsDenylist: [
      "canvas/setCursorPosition",
      "canvas/setStageCoordinates",
      "canvas/setStageScale",
      "canvas/setIsDrawing",
      "canvas/setBoundingBoxCoordinates",
      "canvas/setBoundingBoxDimensions",
      "canvas/setIsDrawing",
      "canvas/addPointToCurrentLine"
    ]
  }
}), Ik = z2(F0);
function Lk(t) {
  if (t.sheet)
    return t.sheet;
  for (var e = 0; e < document.styleSheets.length; e++)
    if (document.styleSheets[e].ownerNode === t)
      return document.styleSheets[e];
}
function Dk(t) {
  var e = document.createElement("style");
  return e.setAttribute("data-emotion", t.key), t.nonce !== void 0 && e.setAttribute("nonce", t.nonce), e.appendChild(document.createTextNode("")), e.setAttribute("data-s", ""), e;
}
var Mk = /* @__PURE__ */ function() {
  function t(r) {
    var n = this;
    this._insertTag = function(i) {
      var s;
      n.tags.length === 0 ? n.insertionPoint ? s = n.insertionPoint.nextSibling : n.prepend ? s = n.container.firstChild : s = n.before : s = n.tags[n.tags.length - 1].nextSibling, n.container.insertBefore(i, s), n.tags.push(i);
    }, this.isSpeedy = r.speedy === void 0 ? process.env.NODE_ENV === "production" : r.speedy, this.tags = [], this.ctr = 0, this.nonce = r.nonce, this.key = r.key, this.container = r.container, this.prepend = r.prepend, this.insertionPoint = r.insertionPoint, this.before = null;
  }
  var e = t.prototype;
  return e.hydrate = function(n) {
    n.forEach(this._insertTag);
  }, e.insert = function(n) {
    this.ctr % (this.isSpeedy ? 65e3 : 1) === 0 && this._insertTag(Dk(this));
    var i = this.tags[this.tags.length - 1];
    if (process.env.NODE_ENV !== "production") {
      var s = n.charCodeAt(0) === 64 && n.charCodeAt(1) === 105;
      s && this._alreadyInsertedOrderInsensitiveRule && console.error(`You're attempting to insert the following rule:
` + n + "\n\n`@import` rules must be before all other types of rules in a stylesheet but other rules have already been inserted. Please ensure that `@import` rules are before all other rules."), this._alreadyInsertedOrderInsensitiveRule = this._alreadyInsertedOrderInsensitiveRule || !s;
    }
    if (this.isSpeedy) {
      var o = Lk(i);
      try {
        o.insertRule(n, o.cssRules.length);
      } catch (l) {
        process.env.NODE_ENV !== "production" && !/:(-moz-placeholder|-moz-focus-inner|-moz-focusring|-ms-input-placeholder|-moz-read-write|-moz-read-only|-ms-clear|-ms-expand|-ms-reveal){/.test(n) && console.error('There was a problem inserting the following rule: "' + n + '"', l);
      }
    } else
      i.appendChild(document.createTextNode(n));
    this.ctr++;
  }, e.flush = function() {
    this.tags.forEach(function(n) {
      return n.parentNode && n.parentNode.removeChild(n);
    }), this.tags = [], this.ctr = 0, process.env.NODE_ENV !== "production" && (this._alreadyInsertedOrderInsensitiveRule = !1);
  }, t;
}(), ir = "-ms-", Cl = "-moz-", Ye = "-webkit-", jh = "comm", $h = "rule", Wh = "decl", Nk = "@import", G0 = "@keyframes", Bk = Math.abs, Nl = String.fromCharCode, Fk = Object.assign;
function Gk(t, e) {
  return Zt(t, 0) ^ 45 ? (((e << 2 ^ Zt(t, 0)) << 2 ^ Zt(t, 1)) << 2 ^ Zt(t, 2)) << 2 ^ Zt(t, 3) : 0;
}
function j0(t) {
  return t.trim();
}
function jk(t, e) {
  return (t = e.exec(t)) ? t[0] : t;
}
function et(t, e, r) {
  return t.replace(e, r);
}
function ch(t, e) {
  return t.indexOf(e);
}
function Zt(t, e) {
  return t.charCodeAt(e) | 0;
}
function go(t, e, r) {
  return t.slice(e, r);
}
function Cn(t) {
  return t.length;
}
function Uh(t) {
  return t.length;
}
function qu(t, e) {
  return e.push(t), t;
}
function $k(t, e) {
  return t.map(e).join("");
}
var Bl = 1, Xa = 1, $0 = 0, _r = 0, It = 0, rs = "";
function Fl(t, e, r, n, i, s, o) {
  return { value: t, root: e, parent: r, type: n, props: i, children: s, line: Bl, column: Xa, length: o, return: "" };
}
function zs(t, e) {
  return Fk(Fl("", null, null, "", null, null, 0), t, { length: -t.length }, e);
}
function Wk() {
  return It;
}
function Uk() {
  return It = _r > 0 ? Zt(rs, --_r) : 0, Xa--, It === 10 && (Xa = 1, Bl--), It;
}
function Dr() {
  return It = _r < $0 ? Zt(rs, _r++) : 0, Xa++, It === 10 && (Xa = 1, Bl++), It;
}
function Rn() {
  return Zt(rs, _r);
}
function sl() {
  return _r;
}
function _o(t, e) {
  return go(rs, t, e);
}
function po(t) {
  switch (t) {
    case 0:
    case 9:
    case 10:
    case 13:
    case 32:
      return 5;
    case 33:
    case 43:
    case 44:
    case 47:
    case 62:
    case 64:
    case 126:
    case 59:
    case 123:
    case 125:
      return 4;
    case 58:
      return 3;
    case 34:
    case 39:
    case 40:
    case 91:
      return 2;
    case 41:
    case 93:
      return 1;
  }
  return 0;
}
function W0(t) {
  return Bl = Xa = 1, $0 = Cn(rs = t), _r = 0, [];
}
function U0(t) {
  return rs = "", t;
}
function ol(t) {
  return j0(_o(_r - 1, fh(t === 91 ? t + 2 : t === 40 ? t + 1 : t)));
}
function Hk(t) {
  for (; (It = Rn()) && It < 33; )
    Dr();
  return po(t) > 2 || po(It) > 3 ? "" : " ";
}
function zk(t, e) {
  for (; --e && Dr() && !(It < 48 || It > 102 || It > 57 && It < 65 || It > 70 && It < 97); )
    ;
  return _o(t, sl() + (e < 6 && Rn() == 32 && Dr() == 32));
}
function fh(t) {
  for (; Dr(); )
    switch (It) {
      case t:
        return _r;
      case 34:
      case 39:
        t !== 34 && t !== 39 && fh(It);
        break;
      case 40:
        t === 41 && fh(t);
        break;
      case 92:
        Dr();
        break;
    }
  return _r;
}
function Vk(t, e) {
  for (; Dr() && t + It !== 47 + 10; )
    if (t + It === 42 + 42 && Rn() === 47)
      break;
  return "/*" + _o(e, _r - 1) + "*" + Nl(t === 47 ? t : Dr());
}
function qk(t) {
  for (; !po(Rn()); )
    Dr();
  return _o(t, _r);
}
function Yk(t) {
  return U0(ul("", null, null, null, [""], t = W0(t), 0, [0], t));
}
function ul(t, e, r, n, i, s, o, l, c) {
  for (var h = 0, d = 0, g = o, p = 0, y = 0, v = 0, _ = 1, b = 1, C = 1, w = 0, O = "", k = i, P = s, M = n, U = O; b; )
    switch (v = w, w = Dr()) {
      case 40:
        if (v != 108 && Zt(U, g - 1) == 58) {
          ch(U += et(ol(w), "&", "&\f"), "&\f") != -1 && (C = -1);
          break;
        }
      case 34:
      case 39:
      case 91:
        U += ol(w);
        break;
      case 9:
      case 10:
      case 13:
      case 32:
        U += Hk(v);
        break;
      case 92:
        U += zk(sl() - 1, 7);
        continue;
      case 47:
        switch (Rn()) {
          case 42:
          case 47:
            qu(Kk(Vk(Dr(), sl()), e, r), c);
            break;
          default:
            U += "/";
        }
        break;
      case 123 * _:
        l[h++] = Cn(U) * C;
      case 125 * _:
      case 59:
      case 0:
        switch (w) {
          case 0:
          case 125:
            b = 0;
          case 59 + d:
            y > 0 && Cn(U) - g && qu(y > 32 ? ov(U + ";", n, r, g - 1) : ov(et(U, " ", "") + ";", n, r, g - 2), c);
            break;
          case 59:
            U += ";";
          default:
            if (qu(M = sv(U, e, r, h, d, i, l, O, k = [], P = [], g), s), w === 123)
              if (d === 0)
                ul(U, e, M, M, k, s, g, l, P);
              else
                switch (p === 99 && Zt(U, 3) === 110 ? 100 : p) {
                  case 100:
                  case 109:
                  case 115:
                    ul(t, M, M, n && qu(sv(t, M, M, 0, 0, i, l, O, i, k = [], g), P), i, P, g, l, n ? k : P);
                    break;
                  default:
                    ul(U, M, M, M, [""], P, 0, l, P);
                }
        }
        h = d = y = 0, _ = C = 1, O = U = "", g = o;
        break;
      case 58:
        g = 1 + Cn(U), y = v;
      default:
        if (_ < 1) {
          if (w == 123)
            --_;
          else if (w == 125 && _++ == 0 && Uk() == 125)
            continue;
        }
        switch (U += Nl(w), w * _) {
          case 38:
            C = d > 0 ? 1 : (U += "\f", -1);
            break;
          case 44:
            l[h++] = (Cn(U) - 1) * C, C = 1;
            break;
          case 64:
            Rn() === 45 && (U += ol(Dr())), p = Rn(), d = g = Cn(O = U += qk(sl())), w++;
            break;
          case 45:
            v === 45 && Cn(U) == 2 && (_ = 0);
        }
    }
  return s;
}
function sv(t, e, r, n, i, s, o, l, c, h, d) {
  for (var g = i - 1, p = i === 0 ? s : [""], y = Uh(p), v = 0, _ = 0, b = 0; v < n; ++v)
    for (var C = 0, w = go(t, g + 1, g = Bk(_ = o[v])), O = t; C < y; ++C)
      (O = j0(_ > 0 ? p[C] + " " + w : et(w, /&\f/g, p[C]))) && (c[b++] = O);
  return Fl(t, e, r, i === 0 ? $h : l, c, h, d);
}
function Kk(t, e, r) {
  return Fl(t, e, r, jh, Nl(Wk()), go(t, 2, -2), 0);
}
function ov(t, e, r, n) {
  return Fl(t, e, r, Wh, go(t, 0, n), go(t, n + 1, -1), n);
}
function qa(t, e) {
  for (var r = "", n = Uh(t), i = 0; i < n; i++)
    r += e(t[i], i, t, e) || "";
  return r;
}
function Xk(t, e, r, n) {
  switch (t.type) {
    case Nk:
    case Wh:
      return t.return = t.return || t.value;
    case jh:
      return "";
    case G0:
      return t.return = t.value + "{" + qa(t.children, n) + "}";
    case $h:
      t.value = t.props.join(",");
  }
  return Cn(r = qa(t.children, n)) ? t.return = t.value + "{" + r + "}" : "";
}
function Jk(t) {
  var e = Uh(t);
  return function(r, n, i, s) {
    for (var o = "", l = 0; l < e; l++)
      o += t[l](r, n, i, s) || "";
    return o;
  };
}
function Qk(t) {
  return function(e) {
    e.root || (e = e.return) && t(e);
  };
}
var uv = function(e) {
  var r = /* @__PURE__ */ new WeakMap();
  return function(n) {
    if (r.has(n))
      return r.get(n);
    var i = e(n);
    return r.set(n, i), i;
  };
};
function H0(t) {
  var e = /* @__PURE__ */ Object.create(null);
  return function(r) {
    return e[r] === void 0 && (e[r] = t(r)), e[r];
  };
}
var Zk = function(e, r, n) {
  for (var i = 0, s = 0; i = s, s = Rn(), i === 38 && s === 12 && (r[n] = 1), !po(s); )
    Dr();
  return _o(e, _r);
}, eI = function(e, r) {
  var n = -1, i = 44;
  do
    switch (po(i)) {
      case 0:
        i === 38 && Rn() === 12 && (r[n] = 1), e[n] += Zk(_r - 1, r, n);
        break;
      case 2:
        e[n] += ol(i);
        break;
      case 4:
        if (i === 44) {
          e[++n] = Rn() === 58 ? "&\f" : "", r[n] = e[n].length;
          break;
        }
      default:
        e[n] += Nl(i);
    }
  while (i = Dr());
  return e;
}, tI = function(e, r) {
  return U0(eI(W0(e), r));
}, lv = /* @__PURE__ */ new WeakMap(), rI = function(e) {
  if (!(e.type !== "rule" || !e.parent || // positive .length indicates that this rule contains pseudo
  // negative .length indicates that this rule has been already prefixed
  e.length < 1)) {
    for (var r = e.value, n = e.parent, i = e.column === n.column && e.line === n.line; n.type !== "rule"; )
      if (n = n.parent, !n)
        return;
    if (!(e.props.length === 1 && r.charCodeAt(0) !== 58 && !lv.get(n)) && !i) {
      lv.set(e, !0);
      for (var s = [], o = tI(r, s), l = n.props, c = 0, h = 0; c < o.length; c++)
        for (var d = 0; d < l.length; d++, h++)
          e.props[h] = s[c] ? o[c].replace(/&\f/g, l[d]) : l[d] + " " + o[c];
    }
  }
}, nI = function(e) {
  if (e.type === "decl") {
    var r = e.value;
    // charcode for l
    r.charCodeAt(0) === 108 && // charcode for b
    r.charCodeAt(2) === 98 && (e.return = "", e.value = "");
  }
}, iI = "emotion-disable-server-rendering-unsafe-selector-warning-please-do-not-use-this-the-warning-exists-for-a-reason", aI = function(e) {
  return e.type === "comm" && e.children.indexOf(iI) > -1;
}, sI = function(e) {
  return function(r, n, i) {
    if (!(r.type !== "rule" || e.compat)) {
      var s = r.value.match(/(:first|:nth|:nth-last)-child/g);
      if (s) {
        for (var o = r.parent === i[0], l = o ? i[0].children : (
          // global rule at the root level
          i
        ), c = l.length - 1; c >= 0; c--) {
          var h = l[c];
          if (h.line < r.line)
            break;
          if (h.column < r.column) {
            if (aI(h))
              return;
            break;
          }
        }
        s.forEach(function(d) {
          console.error('The pseudo class "' + d + '" is potentially unsafe when doing server-side rendering. Try changing it to "' + d.split("-child")[0] + '-of-type".');
        });
      }
    }
  };
}, z0 = function(e) {
  return e.type.charCodeAt(1) === 105 && e.type.charCodeAt(0) === 64;
}, oI = function(e, r) {
  for (var n = e - 1; n >= 0; n--)
    if (!z0(r[n]))
      return !0;
  return !1;
}, cv = function(e) {
  e.type = "", e.value = "", e.return = "", e.children = "", e.props = "";
}, uI = function(e, r, n) {
  z0(e) && (e.parent ? (console.error("`@import` rules can't be nested inside other rules. Please move it to the top level and put it before regular rules. Keep in mind that they can only be used within global styles."), cv(e)) : oI(r, n) && (console.error("`@import` rules can't be after other rules. Please put your `@import` rules before your other rules."), cv(e)));
};
function V0(t, e) {
  switch (Gk(t, e)) {
    case 5103:
      return Ye + "print-" + t + t;
    case 5737:
    case 4201:
    case 3177:
    case 3433:
    case 1641:
    case 4457:
    case 2921:
    case 5572:
    case 6356:
    case 5844:
    case 3191:
    case 6645:
    case 3005:
    case 6391:
    case 5879:
    case 5623:
    case 6135:
    case 4599:
    case 4855:
    case 4215:
    case 6389:
    case 5109:
    case 5365:
    case 5621:
    case 3829:
      return Ye + t + t;
    case 5349:
    case 4246:
    case 4810:
    case 6968:
    case 2756:
      return Ye + t + Cl + t + ir + t + t;
    case 6828:
    case 4268:
      return Ye + t + ir + t + t;
    case 6165:
      return Ye + t + ir + "flex-" + t + t;
    case 5187:
      return Ye + t + et(t, /(\w+).+(:[^]+)/, Ye + "box-$1$2" + ir + "flex-$1$2") + t;
    case 5443:
      return Ye + t + ir + "flex-item-" + et(t, /flex-|-self/, "") + t;
    case 4675:
      return Ye + t + ir + "flex-line-pack" + et(t, /align-content|flex-|-self/, "") + t;
    case 5548:
      return Ye + t + ir + et(t, "shrink", "negative") + t;
    case 5292:
      return Ye + t + ir + et(t, "basis", "preferred-size") + t;
    case 6060:
      return Ye + "box-" + et(t, "-grow", "") + Ye + t + ir + et(t, "grow", "positive") + t;
    case 4554:
      return Ye + et(t, /([^-])(transform)/g, "$1" + Ye + "$2") + t;
    case 6187:
      return et(et(et(t, /(zoom-|grab)/, Ye + "$1"), /(image-set)/, Ye + "$1"), t, "") + t;
    case 5495:
    case 3959:
      return et(t, /(image-set\([^]*)/, Ye + "$1$`$1");
    case 4968:
      return et(et(t, /(.+:)(flex-)?(.*)/, Ye + "box-pack:$3" + ir + "flex-pack:$3"), /s.+-b[^;]+/, "justify") + Ye + t + t;
    case 4095:
    case 3583:
    case 4068:
    case 2532:
      return et(t, /(.+)-inline(.+)/, Ye + "$1$2") + t;
    case 8116:
    case 7059:
    case 5753:
    case 5535:
    case 5445:
    case 5701:
    case 4933:
    case 4677:
    case 5533:
    case 5789:
    case 5021:
    case 4765:
      if (Cn(t) - 1 - e > 6)
        switch (Zt(t, e + 1)) {
          case 109:
            if (Zt(t, e + 4) !== 45)
              break;
          case 102:
            return et(t, /(.+:)(.+)-([^]+)/, "$1" + Ye + "$2-$3$1" + Cl + (Zt(t, e + 3) == 108 ? "$3" : "$2-$3")) + t;
          case 115:
            return ~ch(t, "stretch") ? V0(et(t, "stretch", "fill-available"), e) + t : t;
        }
      break;
    case 4949:
      if (Zt(t, e + 1) !== 115)
        break;
    case 6444:
      switch (Zt(t, Cn(t) - 3 - (~ch(t, "!important") && 10))) {
        case 107:
          return et(t, ":", ":" + Ye) + t;
        case 101:
          return et(t, /(.+:)([^;!]+)(;|!.+)?/, "$1" + Ye + (Zt(t, 14) === 45 ? "inline-" : "") + "box$3$1" + Ye + "$2$3$1" + ir + "$2box$3") + t;
      }
      break;
    case 5936:
      switch (Zt(t, e + 11)) {
        case 114:
          return Ye + t + ir + et(t, /[svh]\w+-[tblr]{2}/, "tb") + t;
        case 108:
          return Ye + t + ir + et(t, /[svh]\w+-[tblr]{2}/, "tb-rl") + t;
        case 45:
          return Ye + t + ir + et(t, /[svh]\w+-[tblr]{2}/, "lr") + t;
      }
      return Ye + t + ir + t + t;
  }
  return t;
}
var lI = function(e, r, n, i) {
  if (e.length > -1 && !e.return)
    switch (e.type) {
      case Wh:
        e.return = V0(e.value, e.length);
        break;
      case G0:
        return qa([zs(e, {
          value: et(e.value, "@", "@" + Ye)
        })], i);
      case $h:
        if (e.length)
          return $k(e.props, function(s) {
            switch (jk(s, /(::plac\w+|:read-\w+)/)) {
              case ":read-only":
              case ":read-write":
                return qa([zs(e, {
                  props: [et(s, /:(read-\w+)/, ":" + Cl + "$1")]
                })], i);
              case "::placeholder":
                return qa([zs(e, {
                  props: [et(s, /:(plac\w+)/, ":" + Ye + "input-$1")]
                }), zs(e, {
                  props: [et(s, /:(plac\w+)/, ":" + Cl + "$1")]
                }), zs(e, {
                  props: [et(s, /:(plac\w+)/, ir + "input-$1")]
                })], i);
            }
            return "";
          });
    }
}, cI = [lI], fI = function(e) {
  var r = e.key;
  if (process.env.NODE_ENV !== "production" && !r)
    throw new Error(`You have to configure \`key\` for your cache. Please make sure it's unique (and not equal to 'css') as it's used for linking styles to your cache.
If multiple caches share the same key they might "fight" for each other's style elements.`);
  if (r === "css") {
    var n = document.querySelectorAll("style[data-emotion]:not([data-s])");
    Array.prototype.forEach.call(n, function(_) {
      var b = _.getAttribute("data-emotion");
      b.indexOf(" ") !== -1 && (document.head.appendChild(_), _.setAttribute("data-s", ""));
    });
  }
  var i = e.stylisPlugins || cI;
  if (process.env.NODE_ENV !== "production" && /[^a-z-]/.test(r))
    throw new Error('Emotion key must only contain lower case alphabetical characters and - but "' + r + '" was passed');
  var s = {}, o, l = [];
  o = e.container || document.head, Array.prototype.forEach.call(
    // this means we will ignore elements which don't have a space in them which
    // means that the style elements we're looking at are only Emotion 11 server-rendered style elements
    document.querySelectorAll('style[data-emotion^="' + r + ' "]'),
    function(_) {
      for (var b = _.getAttribute("data-emotion").split(" "), C = 1; C < b.length; C++)
        s[b[C]] = !0;
      l.push(_);
    }
  );
  var c, h = [rI, nI];
  process.env.NODE_ENV !== "production" && h.push(sI({
    get compat() {
      return v.compat;
    }
  }), uI);
  {
    var d, g = [Xk, process.env.NODE_ENV !== "production" ? function(_) {
      _.root || (_.return ? d.insert(_.return) : _.value && _.type !== jh && d.insert(_.value + "{}"));
    } : Qk(function(_) {
      d.insert(_);
    })], p = Jk(h.concat(i, g)), y = function(b) {
      return qa(Yk(b), p);
    };
    c = function(b, C, w, O) {
      d = w, process.env.NODE_ENV !== "production" && C.map !== void 0 && (d = {
        insert: function(P) {
          w.insert(P + C.map);
        }
      }), y(b ? b + "{" + C.styles + "}" : C.styles), O && (v.inserted[C.name] = !0);
    };
  }
  var v = {
    key: r,
    sheet: new Mk({
      key: r,
      container: o,
      nonce: e.nonce,
      speedy: e.speedy,
      prepend: e.prepend,
      insertionPoint: e.insertionPoint
    }),
    nonce: e.nonce,
    inserted: s,
    registered: {},
    insert: c
  };
  return v.sheet.hydrate(l), v;
}, hI = !0;
function Hh(t, e, r) {
  var n = "";
  return r.split(" ").forEach(function(i) {
    t[i] !== void 0 ? e.push(t[i] + ";") : n += i + " ";
  }), n;
}
var Gl = function(e, r, n) {
  var i = e.key + "-" + r.name;
  // we only need to add the styles to the registered cache if the
  // class name could be used further down
  // the tree but if it's a string tag, we know it won't
  // so we don't have to add it to registered cache.
  // this improves memory usage since we can avoid storing the whole style string
  (n === !1 || // we need to always store it if we're in compat mode and
  // in node since emotion-server relies on whether a style is in
  // the registered cache to know whether a style is global or not
  // also, note that this check will be dead code eliminated in the browser
  hI === !1) && e.registered[i] === void 0 && (e.registered[i] = r.styles);
}, jl = function(e, r, n) {
  Gl(e, r, n);
  var i = e.key + "-" + r.name;
  if (e.inserted[r.name] === void 0) {
    var s = r;
    do
      e.insert(r === s ? "." + i : "", s, e.sheet, !0), s = s.next;
    while (s !== void 0);
  }
};
function dI(t) {
  for (var e = 0, r, n = 0, i = t.length; i >= 4; ++n, i -= 4)
    r = t.charCodeAt(n) & 255 | (t.charCodeAt(++n) & 255) << 8 | (t.charCodeAt(++n) & 255) << 16 | (t.charCodeAt(++n) & 255) << 24, r = /* Math.imul(k, m): */
    (r & 65535) * 1540483477 + ((r >>> 16) * 59797 << 16), r ^= /* k >>> r: */
    r >>> 24, e = /* Math.imul(k, m): */
    (r & 65535) * 1540483477 + ((r >>> 16) * 59797 << 16) ^ /* Math.imul(h, m): */
    (e & 65535) * 1540483477 + ((e >>> 16) * 59797 << 16);
  switch (i) {
    case 3:
      e ^= (t.charCodeAt(n + 2) & 255) << 16;
    case 2:
      e ^= (t.charCodeAt(n + 1) & 255) << 8;
    case 1:
      e ^= t.charCodeAt(n) & 255, e = /* Math.imul(h, m): */
      (e & 65535) * 1540483477 + ((e >>> 16) * 59797 << 16);
  }
  return e ^= e >>> 13, e = /* Math.imul(h, m): */
  (e & 65535) * 1540483477 + ((e >>> 16) * 59797 << 16), ((e ^ e >>> 15) >>> 0).toString(36);
}
var gI = {
  animationIterationCount: 1,
  borderImageOutset: 1,
  borderImageSlice: 1,
  borderImageWidth: 1,
  boxFlex: 1,
  boxFlexGroup: 1,
  boxOrdinalGroup: 1,
  columnCount: 1,
  columns: 1,
  flex: 1,
  flexGrow: 1,
  flexPositive: 1,
  flexShrink: 1,
  flexNegative: 1,
  flexOrder: 1,
  gridRow: 1,
  gridRowEnd: 1,
  gridRowSpan: 1,
  gridRowStart: 1,
  gridColumn: 1,
  gridColumnEnd: 1,
  gridColumnSpan: 1,
  gridColumnStart: 1,
  msGridRow: 1,
  msGridRowSpan: 1,
  msGridColumn: 1,
  msGridColumnSpan: 1,
  fontWeight: 1,
  lineHeight: 1,
  opacity: 1,
  order: 1,
  orphans: 1,
  tabSize: 1,
  widows: 1,
  zIndex: 1,
  zoom: 1,
  WebkitLineClamp: 1,
  // SVG-related properties
  fillOpacity: 1,
  floodOpacity: 1,
  stopOpacity: 1,
  strokeDasharray: 1,
  strokeDashoffset: 1,
  strokeMiterlimit: 1,
  strokeOpacity: 1,
  strokeWidth: 1
}, fv = `You have illegal escape sequence in your template literal, most likely inside content's property value.
Because you write your CSS inside a JavaScript string you actually have to do double escaping, so for example "content: '\\00d7';" should become "content: '\\\\00d7';".
You can read more about this here:
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals#ES2018_revision_of_illegal_escape_sequences`, pI = "You have passed in falsy value as style object's key (can happen when in example you pass unexported component as computed key).", mI = /[A-Z]|^ms/g, q0 = /_EMO_([^_]+?)_([^]*?)_EMO_/g, zh = function(e) {
  return e.charCodeAt(1) === 45;
}, hv = function(e) {
  return e != null && typeof e != "boolean";
}, Af = /* @__PURE__ */ H0(function(t) {
  return zh(t) ? t : t.replace(mI, "-$&").toLowerCase();
}), xl = function(e, r) {
  switch (e) {
    case "animation":
    case "animationName":
      if (typeof r == "string")
        return r.replace(q0, function(n, i, s) {
          return on = {
            name: i,
            styles: s,
            next: on
          }, i;
        });
  }
  return gI[e] !== 1 && !zh(e) && typeof r == "number" && r !== 0 ? r + "px" : r;
};
if (process.env.NODE_ENV !== "production") {
  var vI = /(var|attr|counters?|url|element|(((repeating-)?(linear|radial))|conic)-gradient)\(|(no-)?(open|close)-quote/, yI = ["normal", "none", "initial", "inherit", "unset"], bI = xl, _I = /^-ms-/, SI = /-(.)/g, dv = {};
  xl = function(e, r) {
    if (e === "content" && (typeof r != "string" || yI.indexOf(r) === -1 && !vI.test(r) && (r.charAt(0) !== r.charAt(r.length - 1) || r.charAt(0) !== '"' && r.charAt(0) !== "'")))
      throw new Error("You seem to be using a value for 'content' without quotes, try replacing it with `content: '\"" + r + "\"'`");
    var n = bI(e, r);
    return n !== "" && !zh(e) && e.indexOf("-") !== -1 && dv[e] === void 0 && (dv[e] = !0, console.error("Using kebab-case for css properties in objects is not supported. Did you mean " + e.replace(_I, "ms-").replace(SI, function(i, s) {
      return s.toUpperCase();
    }) + "?")), n;
  };
}
var Y0 = "Component selectors can only be used in conjunction with @emotion/babel-plugin, the swc Emotion plugin, or another Emotion-aware compiler transform.";
function mo(t, e, r) {
  if (r == null)
    return "";
  if (r.__emotion_styles !== void 0) {
    if (process.env.NODE_ENV !== "production" && r.toString() === "NO_COMPONENT_SELECTOR")
      throw new Error(Y0);
    return r;
  }
  switch (typeof r) {
    case "boolean":
      return "";
    case "object": {
      if (r.anim === 1)
        return on = {
          name: r.name,
          styles: r.styles,
          next: on
        }, r.name;
      if (r.styles !== void 0) {
        var n = r.next;
        if (n !== void 0)
          for (; n !== void 0; )
            on = {
              name: n.name,
              styles: n.styles,
              next: on
            }, n = n.next;
        var i = r.styles + ";";
        return process.env.NODE_ENV !== "production" && r.map !== void 0 && (i += r.map), i;
      }
      return wI(t, e, r);
    }
    case "function": {
      if (t !== void 0) {
        var s = on, o = r(t);
        return on = s, mo(t, e, o);
      } else
        process.env.NODE_ENV !== "production" && console.error("Functions that are interpolated in css calls will be stringified.\nIf you want to have a css call based on props, create a function that returns a css call like this\nlet dynamicStyle = (props) => css`color: ${props.color}`\nIt can be called directly with props or interpolated in a styled call like this\nlet SomeComponent = styled('div')`${dynamicStyle}`");
      break;
    }
    case "string":
      if (process.env.NODE_ENV !== "production") {
        var l = [], c = r.replace(q0, function(d, g, p) {
          var y = "animation" + l.length;
          return l.push("const " + y + " = keyframes`" + p.replace(/^@keyframes animation-\w+/, "") + "`"), "${" + y + "}";
        });
        l.length && console.error("`keyframes` output got interpolated into plain string, please wrap it with `css`.\n\nInstead of doing this:\n\n" + [].concat(l, ["`" + c + "`"]).join(`
`) + `

You should wrap it with \`css\` like this:

` + ("css`" + c + "`"));
      }
      break;
  }
  if (e == null)
    return r;
  var h = e[r];
  return h !== void 0 ? h : r;
}
function wI(t, e, r) {
  var n = "";
  if (Array.isArray(r))
    for (var i = 0; i < r.length; i++)
      n += mo(t, e, r[i]) + ";";
  else
    for (var s in r) {
      var o = r[s];
      if (typeof o != "object")
        e != null && e[o] !== void 0 ? n += s + "{" + e[o] + "}" : hv(o) && (n += Af(s) + ":" + xl(s, o) + ";");
      else {
        if (s === "NO_COMPONENT_SELECTOR" && process.env.NODE_ENV !== "production")
          throw new Error(Y0);
        if (Array.isArray(o) && typeof o[0] == "string" && (e == null || e[o[0]] === void 0))
          for (var l = 0; l < o.length; l++)
            hv(o[l]) && (n += Af(s) + ":" + xl(s, o[l]) + ";");
        else {
          var c = mo(t, e, o);
          switch (s) {
            case "animation":
            case "animationName": {
              n += Af(s) + ":" + c + ";";
              break;
            }
            default:
              process.env.NODE_ENV !== "production" && s === "undefined" && console.error(pI), n += s + "{" + c + "}";
          }
        }
      }
    }
  return n;
}
var gv = /label:\s*([^\s;\n{]+)\s*(;|$)/g, K0;
process.env.NODE_ENV !== "production" && (K0 = /\/\*#\ssourceMappingURL=data:application\/json;\S+\s+\*\//g);
var on, Ja = function(e, r, n) {
  if (e.length === 1 && typeof e[0] == "object" && e[0] !== null && e[0].styles !== void 0)
    return e[0];
  var i = !0, s = "";
  on = void 0;
  var o = e[0];
  o == null || o.raw === void 0 ? (i = !1, s += mo(n, r, o)) : (process.env.NODE_ENV !== "production" && o[0] === void 0 && console.error(fv), s += o[0]);
  for (var l = 1; l < e.length; l++)
    s += mo(n, r, e[l]), i && (process.env.NODE_ENV !== "production" && o[l] === void 0 && console.error(fv), s += o[l]);
  var c;
  process.env.NODE_ENV !== "production" && (s = s.replace(K0, function(p) {
    return c = p, "";
  })), gv.lastIndex = 0;
  for (var h = "", d; (d = gv.exec(s)) !== null; )
    h += "-" + // $FlowFixMe we know it's not null
    d[1];
  var g = dI(s) + h;
  return process.env.NODE_ENV !== "production" ? {
    name: g,
    styles: s,
    map: c,
    next: on,
    toString: function() {
      return "You have tried to stringify object returned from `css` function. It isn't supposed to be used directly (e.g. as value of the `className` prop), but rather handed to emotion so it can handle it (e.g. as value of `css` prop).";
    }
  } : {
    name: g,
    styles: s,
    next: on
  };
}, EI = function(e) {
  return e();
}, X0 = ep["useInsertionEffect"] ? ep["useInsertionEffect"] : !1, Vh = X0 || EI, pv = X0 || Mv, CI = {}.hasOwnProperty, qh = /* @__PURE__ */ vo(
  // we're doing this to avoid preconstruct's dead code elimination in this one case
  // because this module is primarily intended for the browser and node
  // but it's also required in react native and similar environments sometimes
  // and we could have a special build just for that
  // but this is much easier and the native packages
  // might use a different theme context in the future anyway
  typeof HTMLElement < "u" ? /* @__PURE__ */ fI({
    key: "css"
  }) : null
);
process.env.NODE_ENV !== "production" && (qh.displayName = "EmotionCacheContext");
qh.Provider;
var $l = function(e) {
  return /* @__PURE__ */ Nv(function(r, n) {
    var i = cn(qh);
    return e(r, i, n);
  });
}, Oi = /* @__PURE__ */ vo({});
process.env.NODE_ENV !== "production" && (Oi.displayName = "EmotionThemeContext");
var xI = function(e, r) {
  if (typeof r == "function") {
    var n = r(e);
    if (process.env.NODE_ENV !== "production" && (n == null || typeof n != "object" || Array.isArray(n)))
      throw new Error("[ThemeProvider] Please return an object from your theme function, i.e. theme={() => ({})}!");
    return n;
  }
  if (process.env.NODE_ENV !== "production" && (r == null || typeof r != "object" || Array.isArray(r)))
    throw new Error("[ThemeProvider] Please make your theme prop a plain object");
  return dl({}, e, r);
}, OI = /* @__PURE__ */ uv(function(t) {
  return uv(function(e) {
    return xI(t, e);
  });
}), t5 = function(e) {
  var r = cn(Oi);
  return e.theme !== r && (r = OI(r)(e.theme)), /* @__PURE__ */ Qn(Oi.Provider, {
    value: r
  }, e.children);
}, mv = "__EMOTION_TYPE_PLEASE_DO_NOT_USE__", vv = "__EMOTION_LABEL_PLEASE_DO_NOT_USE__", TI = function(e) {
  var r = e.cache, n = e.serialized, i = e.isStringTag;
  return Gl(r, n, i), Vh(function() {
    return jl(r, n, i);
  }), null;
}, PI = /* @__PURE__ */ $l(function(t, e, r) {
  var n = t.css;
  typeof n == "string" && e.registered[n] !== void 0 && (n = e.registered[n]);
  var i = t[mv], s = [n], o = "";
  typeof t.className == "string" ? o = Hh(e.registered, s, t.className) : t.className != null && (o = t.className + " ");
  var l = Ja(s, void 0, cn(Oi));
  if (process.env.NODE_ENV !== "production" && l.name.indexOf("-") === -1) {
    var c = t[vv];
    c && (l = Ja([l, "label:" + c + ";"]));
  }
  o += e.key + "-" + l.name;
  var h = {};
  for (var d in t)
    CI.call(t, d) && d !== "css" && d !== mv && (process.env.NODE_ENV === "production" || d !== vv) && (h[d] = t[d]);
  return h.ref = r, h.className = o, /* @__PURE__ */ Qn(vh, null, /* @__PURE__ */ Qn(TI, {
    cache: e,
    serialized: l,
    isStringTag: typeof i == "string"
  }), /* @__PURE__ */ Qn(i, h));
});
process.env.NODE_ENV !== "production" && (PI.displayName = "EmotionCssPropInternal");
var RI = {
  name: "@emotion/react",
  version: "11.10.6",
  main: "dist/emotion-react.cjs.js",
  module: "dist/emotion-react.esm.js",
  browser: {
    "./dist/emotion-react.esm.js": "./dist/emotion-react.browser.esm.js"
  },
  exports: {
    ".": {
      module: {
        worker: "./dist/emotion-react.worker.esm.js",
        browser: "./dist/emotion-react.browser.esm.js",
        default: "./dist/emotion-react.esm.js"
      },
      default: "./dist/emotion-react.cjs.js"
    },
    "./jsx-runtime": {
      module: {
        worker: "./jsx-runtime/dist/emotion-react-jsx-runtime.worker.esm.js",
        browser: "./jsx-runtime/dist/emotion-react-jsx-runtime.browser.esm.js",
        default: "./jsx-runtime/dist/emotion-react-jsx-runtime.esm.js"
      },
      default: "./jsx-runtime/dist/emotion-react-jsx-runtime.cjs.js"
    },
    "./_isolated-hnrs": {
      module: {
        worker: "./_isolated-hnrs/dist/emotion-react-_isolated-hnrs.worker.esm.js",
        browser: "./_isolated-hnrs/dist/emotion-react-_isolated-hnrs.browser.esm.js",
        default: "./_isolated-hnrs/dist/emotion-react-_isolated-hnrs.esm.js"
      },
      default: "./_isolated-hnrs/dist/emotion-react-_isolated-hnrs.cjs.js"
    },
    "./jsx-dev-runtime": {
      module: {
        worker: "./jsx-dev-runtime/dist/emotion-react-jsx-dev-runtime.worker.esm.js",
        browser: "./jsx-dev-runtime/dist/emotion-react-jsx-dev-runtime.browser.esm.js",
        default: "./jsx-dev-runtime/dist/emotion-react-jsx-dev-runtime.esm.js"
      },
      default: "./jsx-dev-runtime/dist/emotion-react-jsx-dev-runtime.cjs.js"
    },
    "./package.json": "./package.json",
    "./types/css-prop": "./types/css-prop.d.ts",
    "./macro": "./macro.js"
  },
  types: "types/index.d.ts",
  files: [
    "src",
    "dist",
    "jsx-runtime",
    "jsx-dev-runtime",
    "_isolated-hnrs",
    "types/*.d.ts",
    "macro.js",
    "macro.d.ts",
    "macro.js.flow"
  ],
  sideEffects: !1,
  author: "Emotion Contributors",
  license: "MIT",
  scripts: {
    "test:typescript": "dtslint types"
  },
  dependencies: {
    "@babel/runtime": "^7.18.3",
    "@emotion/babel-plugin": "^11.10.6",
    "@emotion/cache": "^11.10.5",
    "@emotion/serialize": "^1.1.1",
    "@emotion/use-insertion-effect-with-fallbacks": "^1.0.0",
    "@emotion/utils": "^1.2.0",
    "@emotion/weak-memoize": "^0.3.0",
    "hoist-non-react-statics": "^3.3.1"
  },
  peerDependencies: {
    react: ">=16.8.0"
  },
  peerDependenciesMeta: {
    "@types/react": {
      optional: !0
    }
  },
  devDependencies: {
    "@definitelytyped/dtslint": "0.0.112",
    "@emotion/css": "11.10.6",
    "@emotion/css-prettifier": "1.1.1",
    "@emotion/server": "11.10.0",
    "@emotion/styled": "11.10.6",
    "html-tag-names": "^1.1.2",
    react: "16.14.0",
    "svg-tag-names": "^1.1.1",
    typescript: "^4.5.5"
  },
  repository: "https://github.com/emotion-js/emotion/tree/main/packages/react",
  publishConfig: {
    access: "public"
  },
  "umd:main": "dist/emotion-react.umd.min.js",
  preconstruct: {
    entrypoints: [
      "./index.js",
      "./jsx-runtime.js",
      "./jsx-dev-runtime.js",
      "./_isolated-hnrs.js"
    ],
    umdName: "emotionReact",
    exports: {
      envConditions: [
        "browser",
        "worker"
      ],
      extra: {
        "./types/css-prop": "./types/css-prop.d.ts",
        "./macro": "./macro.js"
      }
    }
  }
}, yv = !1, AI = /* @__PURE__ */ $l(function(t, e) {
  process.env.NODE_ENV !== "production" && !yv && // check for className as well since the user is
  // probably using the custom createElement which
  // means it will be turned into a className prop
  // $FlowFixMe I don't really want to add it to the type since it shouldn't be used
  (t.className || t.css) && (console.error("It looks like you're using the css prop on Global, did you mean to use the styles prop instead?"), yv = !0);
  var r = t.styles, n = Ja([r], void 0, cn(Oi)), i = so();
  return pv(function() {
    var s = e.key + "-global", o = new e.sheet.constructor({
      key: s,
      nonce: e.sheet.nonce,
      container: e.sheet.container,
      speedy: e.sheet.isSpeedy
    }), l = !1, c = document.querySelector('style[data-emotion="' + s + " " + n.name + '"]');
    return e.sheet.tags.length && (o.before = e.sheet.tags[0]), c !== null && (l = !0, c.setAttribute("data-emotion", s), o.hydrate([c])), i.current = [o, l], function() {
      o.flush();
    };
  }, [e]), pv(function() {
    var s = i.current, o = s[0], l = s[1];
    if (l) {
      s[1] = !1;
      return;
    }
    if (n.next !== void 0 && jl(e, n.next, !0), o.tags.length) {
      var c = o.tags[o.tags.length - 1].nextElementSibling;
      o.before = c, o.flush();
    }
    e.insert("", n, o, !1);
  }, [e, n.name]), null;
});
process.env.NODE_ENV !== "production" && (AI.displayName = "EmotionGlobal");
function kI() {
  for (var t = arguments.length, e = new Array(t), r = 0; r < t; r++)
    e[r] = arguments[r];
  return Ja(e);
}
var II = function() {
  var e = kI.apply(void 0, arguments), r = "animation-" + e.name;
  return {
    name: r,
    styles: "@keyframes " + r + "{" + e.styles + "}",
    anim: 1,
    toString: function() {
      return "_EMO_" + this.name + "_" + this.styles + "_EMO_";
    }
  };
}, LI = function t(e) {
  for (var r = e.length, n = 0, i = ""; n < r; n++) {
    var s = e[n];
    if (s != null) {
      var o = void 0;
      switch (typeof s) {
        case "boolean":
          break;
        case "object": {
          if (Array.isArray(s))
            o = t(s);
          else {
            process.env.NODE_ENV !== "production" && s.styles !== void 0 && s.name !== void 0 && console.error("You have passed styles created with `css` from `@emotion/react` package to the `cx`.\n`cx` is meant to compose class names (strings) so you should convert those styles to a class name by passing them to the `css` received from <ClassNames/> component."), o = "";
            for (var l in s)
              s[l] && l && (o && (o += " "), o += l);
          }
          break;
        }
        default:
          o = s;
      }
      o && (i && (i += " "), i += o);
    }
  }
  return i;
};
function DI(t, e, r) {
  var n = [], i = Hh(t, n, r);
  return n.length < 2 ? r : i + e(n);
}
var MI = function(e) {
  var r = e.cache, n = e.serializedArr;
  return Vh(function() {
    for (var i = 0; i < n.length; i++)
      jl(r, n[i], !1);
  }), null;
}, NI = /* @__PURE__ */ $l(function(t, e) {
  var r = !1, n = [], i = function() {
    if (r && process.env.NODE_ENV !== "production")
      throw new Error("css can only be used during render");
    for (var h = arguments.length, d = new Array(h), g = 0; g < h; g++)
      d[g] = arguments[g];
    var p = Ja(d, e.registered);
    return n.push(p), Gl(e, p, !1), e.key + "-" + p.name;
  }, s = function() {
    if (r && process.env.NODE_ENV !== "production")
      throw new Error("cx can only be used during render");
    for (var h = arguments.length, d = new Array(h), g = 0; g < h; g++)
      d[g] = arguments[g];
    return DI(e.registered, i, LI(d));
  }, o = {
    css: i,
    cx: s,
    theme: cn(Oi)
  }, l = t.children(o);
  return r = !0, /* @__PURE__ */ Qn(vh, null, /* @__PURE__ */ Qn(MI, {
    cache: e,
    serializedArr: n
  }), l);
});
process.env.NODE_ENV !== "production" && (NI.displayName = "EmotionClassNames");
if (process.env.NODE_ENV !== "production") {
  var bv = !0, BI = typeof jest < "u" || typeof vi < "u";
  if (bv && !BI) {
    var _v = (
      // $FlowIgnore
      typeof globalThis < "u" ? globalThis : bv ? window : global
    ), Sv = "__EMOTION_REACT_" + RI.version.split(".")[0] + "__";
    _v[Sv] && console.warn("You are loading @emotion/react when it is already loaded. Running multiple instances may cause problems. This can happen if multiple versions are used, or if multiple builds of the same version are used."), _v[Sv] = !0;
  }
}
function FI() {
  const t = cn(
    Oi
  );
  if (!t)
    throw Error(
      "useTheme: `theme` is undefined. Seems you forgot to wrap your app in `<ChakraProvider />` or `<ThemeProvider />`"
    );
  return t;
}
var J0 = vo({});
J0.displayName = "ColorModeContext";
function Q0() {
  const t = cn(J0);
  if (t === void 0)
    throw new Error("useColorMode must be used within a ColorModeProvider");
  return t;
}
function GI() {
  const t = Q0(), e = FI();
  return { ...t, theme: e };
}
var Z0 = (...t) => t.filter(Boolean).join(" ");
function jI() {
  return process.env.NODE_ENV !== "production";
}
function un(t) {
  const e = typeof t;
  return t != null && (e === "object" || e === "function") && !Array.isArray(t);
}
var r5 = (t) => {
  const { condition: e, message: r } = t;
  e && jI() && console.warn(r);
};
function Yi(t, ...e) {
  return $I(t) ? t(...e) : t;
}
var $I = (t) => typeof t == "function", n5 = (t) => t ? "" : void 0, i5 = (t) => t ? !0 : void 0;
function a5(...t) {
  return function(r) {
    t.some((n) => (n == null || n(r), r == null ? void 0 : r.defaultPrevented));
  };
}
function s5(...t) {
  return function(r) {
    t.forEach((n) => {
      n == null || n(r);
    });
  };
}
var Ol = {}, WI = {
  get exports() {
    return Ol;
  },
  set exports(t) {
    Ol = t;
  }
};
(function(t, e) {
  var r = 200, n = "__lodash_hash_undefined__", i = 800, s = 16, o = 9007199254740991, l = "[object Arguments]", c = "[object Array]", h = "[object AsyncFunction]", d = "[object Boolean]", g = "[object Date]", p = "[object Error]", y = "[object Function]", v = "[object GeneratorFunction]", _ = "[object Map]", b = "[object Number]", C = "[object Null]", w = "[object Object]", O = "[object Proxy]", k = "[object RegExp]", P = "[object Set]", M = "[object String]", U = "[object Undefined]", V = "[object WeakMap]", Y = "[object ArrayBuffer]", X = "[object DataView]", Z = "[object Float32Array]", ce = "[object Float64Array]", $ = "[object Int8Array]", R = "[object Int16Array]", F = "[object Int32Array]", H = "[object Uint8Array]", re = "[object Uint8ClampedArray]", ae = "[object Uint16Array]", ne = "[object Uint32Array]", oe = /[\\^$.*+?()[\]{}|]/g, de = /^\[object .+?Constructor\]$/, Re = /^(?:0|[1-9]\d*)$/, Ce = {};
  Ce[Z] = Ce[ce] = Ce[$] = Ce[R] = Ce[F] = Ce[H] = Ce[re] = Ce[ae] = Ce[ne] = !0, Ce[l] = Ce[c] = Ce[Y] = Ce[d] = Ce[X] = Ce[g] = Ce[p] = Ce[y] = Ce[_] = Ce[b] = Ce[w] = Ce[k] = Ce[P] = Ce[M] = Ce[V] = !1;
  var rt = typeof dr == "object" && dr && dr.Object === Object && dr, xt = typeof self == "object" && self && self.Object === Object && self, be = rt || xt || Function("return this")(), At = e && !e.nodeType && e, ot = At && !0 && t && !t.nodeType && t, wt = ot && ot.exports === At, kt = wt && rt.process, dt = function() {
    try {
      var T = ot && ot.require && ot.require("util").types;
      return T || kt && kt.binding && kt.binding("util");
    } catch {
    }
  }(), te = dt && dt.isTypedArray;
  function $t(T, N, z) {
    switch (z.length) {
      case 0:
        return T.call(N);
      case 1:
        return T.call(N, z[0]);
      case 2:
        return T.call(N, z[0], z[1]);
      case 3:
        return T.call(N, z[0], z[1], z[2]);
    }
    return T.apply(N, z);
  }
  function fe(T, N) {
    for (var z = -1, pe = Array(T); ++z < T; )
      pe[z] = N(z);
    return pe;
  }
  function Ot(T) {
    return function(N) {
      return T(N);
    };
  }
  function sr(T, N) {
    return T == null ? void 0 : T[N];
  }
  function Dt(T, N) {
    return function(z) {
      return T(N(z));
    };
  }
  var dn = Array.prototype, gn = Function.prototype, gr = Object.prototype, Mt = be["__core-js_shared__"], qt = gn.toString, yt = gr.hasOwnProperty, Dn = function() {
    var T = /[^.]+$/.exec(Mt && Mt.keys && Mt.keys.IE_PROTO || "");
    return T ? "Symbol(src)_1." + T : "";
  }(), Sr = gr.toString, Mn = qt.call(Object), er = RegExp(
    "^" + qt.call(yt).replace(oe, "\\$&").replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, "$1.*?") + "$"
  ), wr = wt ? be.Buffer : void 0, Nn = be.Symbol, Bn = be.Uint8Array, pn = wr ? wr.allocUnsafe : void 0, ii = Dt(Object.getPrototypeOf, Object), ai = Object.create, Ai = gr.propertyIsEnumerable, ki = dn.splice, Er = Nn ? Nn.toStringTag : void 0, mn = function() {
    try {
      var T = ga(Object, "defineProperty");
      return T({}, "", {}), T;
    } catch {
    }
  }(), ns = wr ? wr.isBuffer : void 0, si = Math.max, ia = Date.now, Nr = ga(be, "Map"), Br = ga(Object, "create"), Ii = function() {
    function T() {
    }
    return function(N) {
      if (!Qr(N))
        return {};
      if (ai)
        return ai(N);
      T.prototype = N;
      var z = new T();
      return T.prototype = void 0, z;
    };
  }();
  function Fr(T) {
    var N = -1, z = T == null ? 0 : T.length;
    for (this.clear(); ++N < z; ) {
      var pe = T[N];
      this.set(pe[0], pe[1]);
    }
  }
  function is() {
    this.__data__ = Br ? Br(null) : {}, this.size = 0;
  }
  function aa(T) {
    var N = this.has(T) && delete this.__data__[T];
    return this.size -= N ? 1 : 0, N;
  }
  function sa(T) {
    var N = this.__data__;
    if (Br) {
      var z = N[T];
      return z === n ? void 0 : z;
    }
    return yt.call(N, T) ? N[T] : void 0;
  }
  function oa(T) {
    var N = this.__data__;
    return Br ? N[T] !== void 0 : yt.call(N, T);
  }
  function ua(T, N) {
    var z = this.__data__;
    return this.size += this.has(T) ? 0 : 1, z[T] = Br && N === void 0 ? n : N, this;
  }
  Fr.prototype.clear = is, Fr.prototype.delete = aa, Fr.prototype.get = sa, Fr.prototype.has = oa, Fr.prototype.set = ua;
  function Cr(T) {
    var N = -1, z = T == null ? 0 : T.length;
    for (this.clear(); ++N < z; ) {
      var pe = T[N];
      this.set(pe[0], pe[1]);
    }
  }
  function Li() {
    this.__data__ = [], this.size = 0;
  }
  function la(T) {
    var N = this.__data__, z = Bt(N, T);
    if (z < 0)
      return !1;
    var pe = N.length - 1;
    return z == pe ? N.pop() : ki.call(N, z, 1), --this.size, !0;
  }
  function Di(T) {
    var N = this.__data__, z = Bt(N, T);
    return z < 0 ? void 0 : N[z][1];
  }
  function as(T) {
    return Bt(this.__data__, T) > -1;
  }
  function ss(T, N) {
    var z = this.__data__, pe = Bt(z, T);
    return pe < 0 ? (++this.size, z.push([T, N])) : z[pe][1] = N, this;
  }
  Cr.prototype.clear = Li, Cr.prototype.delete = la, Cr.prototype.get = Di, Cr.prototype.has = as, Cr.prototype.set = ss;
  function vn(T) {
    var N = -1, z = T == null ? 0 : T.length;
    for (this.clear(); ++N < z; ) {
      var pe = T[N];
      this.set(pe[0], pe[1]);
    }
  }
  function L() {
    this.size = 0, this.__data__ = {
      hash: new Fr(),
      map: new (Nr || Cr)(),
      string: new Fr()
    };
  }
  function ie(T) {
    var N = da(this, T).delete(T);
    return this.size -= N ? 1 : 0, N;
  }
  function he(T) {
    return da(this, T).get(T);
  }
  function Se(T) {
    return da(this, T).has(T);
  }
  function je(T, N) {
    var z = da(this, T), pe = z.size;
    return z.set(T, N), this.size += z.size == pe ? 0 : 1, this;
  }
  vn.prototype.clear = L, vn.prototype.delete = ie, vn.prototype.get = he, vn.prototype.has = Se, vn.prototype.set = je;
  function Ne(T) {
    var N = this.__data__ = new Cr(T);
    this.size = N.size;
  }
  function Me() {
    this.__data__ = new Cr(), this.size = 0;
  }
  function Ae(T) {
    var N = this.__data__, z = N.delete(T);
    return this.size = N.size, z;
  }
  function Nt(T) {
    return this.__data__.get(T);
  }
  function ft(T) {
    return this.__data__.has(T);
  }
  function gt(T, N) {
    var z = this.__data__;
    if (z instanceof Cr) {
      var pe = z.__data__;
      if (!Nr || pe.length < r - 1)
        return pe.push([T, N]), this.size = ++z.size, this;
      z = this.__data__ = new vn(pe);
    }
    return z.set(T, N), this.size = z.size, this;
  }
  Ne.prototype.clear = Me, Ne.prototype.delete = Ae, Ne.prototype.get = Nt, Ne.prototype.has = ft, Ne.prototype.set = gt;
  function tr(T, N) {
    var z = gs(T), pe = !z && ds(T), Be = !z && !pe && Bo(T), nt = !z && !pe && !Be && Go(T), Te = z || pe || Be || nt, Ie = Te ? fe(T.length, String) : [], He = Ie.length;
    for (var ur in T)
      (N || yt.call(T, ur)) && !(Te && // Safari 9 has enumerable `arguments.length` in strict mode.
      (ur == "length" || // Node.js 0.10 has enumerable non-index properties on buffers.
      Be && (ur == "offset" || ur == "parent") || // PhantomJS 2 has enumerable non-index properties on typed arrays.
      nt && (ur == "buffer" || ur == "byteLength" || ur == "byteOffset") || // Skip index properties.
      Ao(ur, He))) && Ie.push(ur);
    return Ie;
  }
  function or(T, N, z) {
    (z !== void 0 && !ma(T[N], z) || z === void 0 && !(N in T)) && os(T, N, z);
  }
  function ca(T, N, z) {
    var pe = T[N];
    (!(yt.call(T, N) && ma(pe, z)) || z === void 0 && !(N in T)) && os(T, N, z);
  }
  function Bt(T, N) {
    for (var z = T.length; z--; )
      if (ma(T[z][0], N))
        return z;
    return -1;
  }
  function os(T, N, z) {
    N == "__proto__" && mn ? mn(T, N, {
      configurable: !0,
      enumerable: !0,
      value: z,
      writable: !0
    }) : T[N] = z;
  }
  var Hl = Po();
  function fa(T) {
    return T == null ? T === void 0 ? U : C : Er && Er in Object(T) ? Ro(T) : Lo(T);
  }
  function us(T) {
    return Mi(T) && fa(T) == l;
  }
  function So(T) {
    if (!Qr(T) || hs(T))
      return !1;
    var N = ps(T) ? er : de;
    return N.test(No(T));
  }
  function wo(T) {
    return Mi(T) && Fo(T.length) && !!Ce[fa(T)];
  }
  function zl(T) {
    if (!Qr(T))
      return Io(T);
    var N = Fn(T), z = [];
    for (var pe in T)
      pe == "constructor" && (N || !yt.call(T, pe)) || z.push(pe);
    return z;
  }
  function Eo(T, N, z, pe, Be) {
    T !== N && Hl(N, function(nt, Te) {
      if (Be || (Be = new Ne()), Qr(nt))
        Vl(T, N, Te, z, Eo, pe, Be);
      else {
        var Ie = pe ? pe(pa(T, Te), nt, Te + "", T, N, Be) : void 0;
        Ie === void 0 && (Ie = nt), or(T, Te, Ie);
      }
    }, jo);
  }
  function Vl(T, N, z, pe, Be, nt, Te) {
    var Ie = pa(T, z), He = pa(N, z), ur = Te.get(He);
    if (ur) {
      or(T, z, ur);
      return;
    }
    var Yt = nt ? nt(Ie, He, z + "", T, N, Te) : void 0, Wt = Yt === void 0;
    if (Wt) {
      var ya = gs(He), ba = !ya && Bo(He), ms = !ya && !ba && Go(He);
      Yt = He, ya || ba || ms ? gs(Ie) ? Yt = Ie : Xl(Ie) ? Yt = Yl(Ie) : ba ? (Wt = !1, Yt = xo(He, !0)) : ms ? (Wt = !1, Yt = ls(He, !0)) : Yt = [] : Jl(He) || ds(He) ? (Yt = Ie, ds(Ie) ? Yt = Ql(Ie) : (!Qr(Ie) || ps(Ie)) && (Yt = cs(He))) : Wt = !1;
    }
    Wt && (Te.set(He, Yt), Be(Yt, He, pe, nt, Te), Te.delete(He)), or(T, z, Yt);
  }
  function ql(T, N) {
    return Do(Kl(T, N, $o), T + "");
  }
  var Co = mn ? function(T, N) {
    return mn(T, "toString", {
      configurable: !0,
      enumerable: !1,
      value: at(N),
      writable: !0
    });
  } : $o;
  function xo(T, N) {
    if (N)
      return T.slice();
    var z = T.length, pe = pn ? pn(z) : new T.constructor(z);
    return T.copy(pe), pe;
  }
  function Oo(T) {
    var N = new T.constructor(T.byteLength);
    return new Bn(N).set(new Bn(T)), N;
  }
  function ls(T, N) {
    var z = N ? Oo(T.buffer) : T.buffer;
    return new T.constructor(z, T.byteOffset, T.length);
  }
  function Yl(T, N) {
    var z = -1, pe = T.length;
    for (N || (N = Array(pe)); ++z < pe; )
      N[z] = T[z];
    return N;
  }
  function To(T, N, z, pe) {
    var Be = !z;
    z || (z = {});
    for (var nt = -1, Te = N.length; ++nt < Te; ) {
      var Ie = N[nt], He = pe ? pe(z[Ie], T[Ie], Ie, z, T) : void 0;
      He === void 0 && (He = T[Ie]), Be ? os(z, Ie, He) : ca(z, Ie, He);
    }
    return z;
  }
  function ha(T) {
    return ql(function(N, z) {
      var pe = -1, Be = z.length, nt = Be > 1 ? z[Be - 1] : void 0, Te = Be > 2 ? z[2] : void 0;
      for (nt = T.length > 3 && typeof nt == "function" ? (Be--, nt) : void 0, Te && ko(z[0], z[1], Te) && (nt = Be < 3 ? void 0 : nt, Be = 1), N = Object(N); ++pe < Be; ) {
        var Ie = z[pe];
        Ie && T(N, Ie, pe, nt);
      }
      return N;
    });
  }
  function Po(T) {
    return function(N, z, pe) {
      for (var Be = -1, nt = Object(N), Te = pe(N), Ie = Te.length; Ie--; ) {
        var He = Te[T ? Ie : ++Be];
        if (z(nt[He], He, nt) === !1)
          break;
      }
      return N;
    };
  }
  function da(T, N) {
    var z = T.__data__;
    return fs(N) ? z[typeof N == "string" ? "string" : "hash"] : z.map;
  }
  function ga(T, N) {
    var z = sr(T, N);
    return So(z) ? z : void 0;
  }
  function Ro(T) {
    var N = yt.call(T, Er), z = T[Er];
    try {
      T[Er] = void 0;
      var pe = !0;
    } catch {
    }
    var Be = Sr.call(T);
    return pe && (N ? T[Er] = z : delete T[Er]), Be;
  }
  function cs(T) {
    return typeof T.constructor == "function" && !Fn(T) ? Ii(ii(T)) : {};
  }
  function Ao(T, N) {
    var z = typeof T;
    return N = N ?? o, !!N && (z == "number" || z != "symbol" && Re.test(T)) && T > -1 && T % 1 == 0 && T < N;
  }
  function ko(T, N, z) {
    if (!Qr(z))
      return !1;
    var pe = typeof N;
    return (pe == "number" ? va(z) && Ao(N, z.length) : pe == "string" && N in z) ? ma(z[N], T) : !1;
  }
  function fs(T) {
    var N = typeof T;
    return N == "string" || N == "number" || N == "symbol" || N == "boolean" ? T !== "__proto__" : T === null;
  }
  function hs(T) {
    return !!Dn && Dn in T;
  }
  function Fn(T) {
    var N = T && T.constructor, z = typeof N == "function" && N.prototype || gr;
    return T === z;
  }
  function Io(T) {
    var N = [];
    if (T != null)
      for (var z in Object(T))
        N.push(z);
    return N;
  }
  function Lo(T) {
    return Sr.call(T);
  }
  function Kl(T, N, z) {
    return N = si(N === void 0 ? T.length - 1 : N, 0), function() {
      for (var pe = arguments, Be = -1, nt = si(pe.length - N, 0), Te = Array(nt); ++Be < nt; )
        Te[Be] = pe[N + Be];
      Be = -1;
      for (var Ie = Array(N + 1); ++Be < N; )
        Ie[Be] = pe[Be];
      return Ie[N] = z(Te), $t(T, this, Ie);
    };
  }
  function pa(T, N) {
    if (!(N === "constructor" && typeof T[N] == "function") && N != "__proto__")
      return T[N];
  }
  var Do = Mo(Co);
  function Mo(T) {
    var N = 0, z = 0;
    return function() {
      var pe = ia(), Be = s - (pe - z);
      if (z = pe, Be > 0) {
        if (++N >= i)
          return arguments[0];
      } else
        N = 0;
      return T.apply(void 0, arguments);
    };
  }
  function No(T) {
    if (T != null) {
      try {
        return qt.call(T);
      } catch {
      }
      try {
        return T + "";
      } catch {
      }
    }
    return "";
  }
  function ma(T, N) {
    return T === N || T !== T && N !== N;
  }
  var ds = us(function() {
    return arguments;
  }()) ? us : function(T) {
    return Mi(T) && yt.call(T, "callee") && !Ai.call(T, "callee");
  }, gs = Array.isArray;
  function va(T) {
    return T != null && Fo(T.length) && !ps(T);
  }
  function Xl(T) {
    return Mi(T) && va(T);
  }
  var Bo = ns || Zl;
  function ps(T) {
    if (!Qr(T))
      return !1;
    var N = fa(T);
    return N == y || N == v || N == h || N == O;
  }
  function Fo(T) {
    return typeof T == "number" && T > -1 && T % 1 == 0 && T <= o;
  }
  function Qr(T) {
    var N = typeof T;
    return T != null && (N == "object" || N == "function");
  }
  function Mi(T) {
    return T != null && typeof T == "object";
  }
  function Jl(T) {
    if (!Mi(T) || fa(T) != w)
      return !1;
    var N = ii(T);
    if (N === null)
      return !0;
    var z = yt.call(N, "constructor") && N.constructor;
    return typeof z == "function" && z instanceof z && qt.call(z) == Mn;
  }
  var Go = te ? Ot(te) : wo;
  function Ql(T) {
    return To(T, jo(T));
  }
  function jo(T) {
    return va(T) ? tr(T, !0) : zl(T);
  }
  var ut = ha(function(T, N, z, pe) {
    Eo(T, N, z, pe);
  });
  function at(T) {
    return function() {
      return T;
    };
  }
  function $o(T) {
    return T;
  }
  function Zl() {
    return !1;
  }
  t.exports = ut;
})(WI, Ol);
const On = Ol;
var UI = (t) => /!(important)?$/.test(t), wv = (t) => typeof t == "string" ? t.replace(/!(important)?$/, "").trim() : t, HI = (t, e) => (r) => {
  const n = String(e), i = UI(n), s = wv(n), o = t ? `${t}.${s}` : s;
  let l = un(r.__cssMap) && o in r.__cssMap ? r.__cssMap[o].varRef : e;
  return l = wv(l), i ? `${l} !important` : l;
};
function Yh(t) {
  const { scale: e, transform: r, compose: n } = t;
  return (s, o) => {
    var l;
    const c = HI(e, s)(o);
    let h = (l = r == null ? void 0 : r(c, o)) != null ? l : c;
    return n && (h = n(h, o)), h;
  };
}
var Yu = (...t) => (e) => t.reduce((r, n) => n(r), e);
function an(t, e) {
  return (r) => {
    const n = { property: r, scale: t };
    return n.transform = Yh({
      scale: t,
      transform: e
    }), n;
  };
}
var zI = ({ rtl: t, ltr: e }) => (r) => r.direction === "rtl" ? t : e;
function VI(t) {
  const { property: e, scale: r, transform: n } = t;
  return {
    scale: r,
    property: zI(e),
    transform: r ? Yh({
      scale: r,
      compose: n
    }) : n
  };
}
var eb = [
  "rotate(var(--chakra-rotate, 0))",
  "scaleX(var(--chakra-scale-x, 1))",
  "scaleY(var(--chakra-scale-y, 1))",
  "skewX(var(--chakra-skew-x, 0))",
  "skewY(var(--chakra-skew-y, 0))"
];
function qI() {
  return [
    "translateX(var(--chakra-translate-x, 0))",
    "translateY(var(--chakra-translate-y, 0))",
    ...eb
  ].join(" ");
}
function YI() {
  return [
    "translate3d(var(--chakra-translate-x, 0), var(--chakra-translate-y, 0), 0)",
    ...eb
  ].join(" ");
}
var KI = {
  "--chakra-blur": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-brightness": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-contrast": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-grayscale": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-hue-rotate": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-invert": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-saturate": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-sepia": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-drop-shadow": "var(--chakra-empty,/*!*/ /*!*/)",
  filter: [
    "var(--chakra-blur)",
    "var(--chakra-brightness)",
    "var(--chakra-contrast)",
    "var(--chakra-grayscale)",
    "var(--chakra-hue-rotate)",
    "var(--chakra-invert)",
    "var(--chakra-saturate)",
    "var(--chakra-sepia)",
    "var(--chakra-drop-shadow)"
  ].join(" ")
}, XI = {
  backdropFilter: [
    "var(--chakra-backdrop-blur)",
    "var(--chakra-backdrop-brightness)",
    "var(--chakra-backdrop-contrast)",
    "var(--chakra-backdrop-grayscale)",
    "var(--chakra-backdrop-hue-rotate)",
    "var(--chakra-backdrop-invert)",
    "var(--chakra-backdrop-opacity)",
    "var(--chakra-backdrop-saturate)",
    "var(--chakra-backdrop-sepia)"
  ].join(" "),
  "--chakra-backdrop-blur": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-brightness": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-contrast": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-grayscale": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-hue-rotate": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-invert": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-opacity": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-saturate": "var(--chakra-empty,/*!*/ /*!*/)",
  "--chakra-backdrop-sepia": "var(--chakra-empty,/*!*/ /*!*/)"
};
function JI(t) {
  return {
    "--chakra-ring-offset-shadow": "var(--chakra-ring-inset) 0 0 0 var(--chakra-ring-offset-width) var(--chakra-ring-offset-color)",
    "--chakra-ring-shadow": "var(--chakra-ring-inset) 0 0 0 calc(var(--chakra-ring-width) + var(--chakra-ring-offset-width)) var(--chakra-ring-color)",
    "--chakra-ring-width": t,
    boxShadow: [
      "var(--chakra-ring-offset-shadow)",
      "var(--chakra-ring-shadow)",
      "var(--chakra-shadow, 0 0 #0000)"
    ].join(", ")
  };
}
var QI = {
  "row-reverse": {
    space: "--chakra-space-x-reverse",
    divide: "--chakra-divide-x-reverse"
  },
  "column-reverse": {
    space: "--chakra-space-y-reverse",
    divide: "--chakra-divide-y-reverse"
  }
}, hh = {
  "to-t": "to top",
  "to-tr": "to top right",
  "to-r": "to right",
  "to-br": "to bottom right",
  "to-b": "to bottom",
  "to-bl": "to bottom left",
  "to-l": "to left",
  "to-tl": "to top left"
}, ZI = new Set(Object.values(hh)), tb = /* @__PURE__ */ new Set([
  "none",
  "-moz-initial",
  "inherit",
  "initial",
  "revert",
  "unset"
]), eL = (t) => t.trim();
function tL(t, e) {
  if (t == null || tb.has(t))
    return t;
  const n = /(^[a-z-A-Z]+)\((.*)\)/g.exec(t), i = n == null ? void 0 : n[1], s = n == null ? void 0 : n[2];
  if (!i || !s)
    return t;
  const o = i.includes("-gradient") ? i : `${i}-gradient`, [l, ...c] = s.split(",").map(eL).filter(Boolean);
  if ((c == null ? void 0 : c.length) === 0)
    return t;
  const h = l in hh ? hh[l] : l;
  c.unshift(h);
  const d = c.map((g) => {
    if (ZI.has(g))
      return g;
    const p = g.indexOf(" "), [y, v] = p !== -1 ? [g.substr(0, p), g.substr(p + 1)] : [g], _ = rb(v) ? v : v && v.split(" "), b = `colors.${y}`, C = b in e.__cssMap ? e.__cssMap[b].varRef : y;
    return _ ? [
      C,
      ...Array.isArray(_) ? _ : [_]
    ].join(" ") : C;
  });
  return `${o}(${d.join(", ")})`;
}
var rb = (t) => typeof t == "string" && t.includes("(") && t.includes(")"), rL = (t, e) => tL(t, e ?? {});
function nL(t) {
  return /^var\(--.+\)$/.test(t);
}
var iL = (t) => {
  const e = parseFloat(t.toString()), r = t.toString().replace(String(e), "");
  return { unitless: !r, value: e, unit: r };
}, wn = (t) => (e) => `${t}(${e})`, Ue = {
  filter(t) {
    return t !== "auto" ? t : KI;
  },
  backdropFilter(t) {
    return t !== "auto" ? t : XI;
  },
  ring(t) {
    return JI(Ue.px(t));
  },
  bgClip(t) {
    return t === "text" ? { color: "transparent", backgroundClip: "text" } : { backgroundClip: t };
  },
  transform(t) {
    return t === "auto" ? qI() : t === "auto-gpu" ? YI() : t;
  },
  vh(t) {
    return t === "$100vh" ? "var(--chakra-vh)" : t;
  },
  px(t) {
    if (t == null)
      return t;
    const { unitless: e } = iL(t);
    return e || typeof t == "number" ? `${t}px` : t;
  },
  fraction(t) {
    return typeof t != "number" || t > 1 ? t : `${t * 100}%`;
  },
  float(t, e) {
    const r = { left: "right", right: "left" };
    return e.direction === "rtl" ? r[t] : t;
  },
  degree(t) {
    if (nL(t) || t == null)
      return t;
    const e = typeof t == "string" && !t.endsWith("deg");
    return typeof t == "number" || e ? `${t}deg` : t;
  },
  gradient: rL,
  blur: wn("blur"),
  opacity: wn("opacity"),
  brightness: wn("brightness"),
  contrast: wn("contrast"),
  dropShadow: wn("drop-shadow"),
  grayscale: wn("grayscale"),
  hueRotate: wn("hue-rotate"),
  invert: wn("invert"),
  saturate: wn("saturate"),
  sepia: wn("sepia"),
  bgImage(t) {
    return t == null || rb(t) || tb.has(t) ? t : `url(${t})`;
  },
  outline(t) {
    const e = String(t) === "0" || String(t) === "none";
    return t !== null && e ? { outline: "2px solid transparent", outlineOffset: "2px" } : { outline: t };
  },
  flexDirection(t) {
    var e;
    const { space: r, divide: n } = (e = QI[t]) != null ? e : {}, i = { flexDirection: t };
    return r && (i[r] = 1), n && (i[n] = 1), i;
  }
}, B = {
  borderWidths: an("borderWidths"),
  borderStyles: an("borderStyles"),
  colors: an("colors"),
  borders: an("borders"),
  radii: an("radii", Ue.px),
  space: an("space", Yu(Ue.vh, Ue.px)),
  spaceT: an("space", Yu(Ue.vh, Ue.px)),
  degreeT(t) {
    return { property: t, transform: Ue.degree };
  },
  prop(t, e, r) {
    return {
      property: t,
      scale: e,
      ...e && {
        transform: Yh({ scale: e, transform: r })
      }
    };
  },
  propT(t, e) {
    return { property: t, transform: e };
  },
  sizes: an("sizes", Yu(Ue.vh, Ue.px)),
  sizesT: an("sizes", Yu(Ue.vh, Ue.fraction)),
  shadows: an("shadows"),
  logical: VI,
  blur: an("blur", Ue.blur)
}, ll = {
  background: B.colors("background"),
  backgroundColor: B.colors("backgroundColor"),
  backgroundImage: B.propT("backgroundImage", Ue.bgImage),
  backgroundSize: !0,
  backgroundPosition: !0,
  backgroundRepeat: !0,
  backgroundAttachment: !0,
  backgroundClip: { transform: Ue.bgClip },
  bgSize: B.prop("backgroundSize"),
  bgPosition: B.prop("backgroundPosition"),
  bg: B.colors("background"),
  bgColor: B.colors("backgroundColor"),
  bgPos: B.prop("backgroundPosition"),
  bgRepeat: B.prop("backgroundRepeat"),
  bgAttachment: B.prop("backgroundAttachment"),
  bgGradient: B.propT("backgroundImage", Ue.gradient),
  bgClip: { transform: Ue.bgClip }
};
Object.assign(ll, {
  bgImage: ll.backgroundImage,
  bgImg: ll.backgroundImage
});
var qe = {
  border: B.borders("border"),
  borderWidth: B.borderWidths("borderWidth"),
  borderStyle: B.borderStyles("borderStyle"),
  borderColor: B.colors("borderColor"),
  borderRadius: B.radii("borderRadius"),
  borderTop: B.borders("borderTop"),
  borderBlockStart: B.borders("borderBlockStart"),
  borderTopLeftRadius: B.radii("borderTopLeftRadius"),
  borderStartStartRadius: B.logical({
    scale: "radii",
    property: {
      ltr: "borderTopLeftRadius",
      rtl: "borderTopRightRadius"
    }
  }),
  borderEndStartRadius: B.logical({
    scale: "radii",
    property: {
      ltr: "borderBottomLeftRadius",
      rtl: "borderBottomRightRadius"
    }
  }),
  borderTopRightRadius: B.radii("borderTopRightRadius"),
  borderStartEndRadius: B.logical({
    scale: "radii",
    property: {
      ltr: "borderTopRightRadius",
      rtl: "borderTopLeftRadius"
    }
  }),
  borderEndEndRadius: B.logical({
    scale: "radii",
    property: {
      ltr: "borderBottomRightRadius",
      rtl: "borderBottomLeftRadius"
    }
  }),
  borderRight: B.borders("borderRight"),
  borderInlineEnd: B.borders("borderInlineEnd"),
  borderBottom: B.borders("borderBottom"),
  borderBlockEnd: B.borders("borderBlockEnd"),
  borderBottomLeftRadius: B.radii("borderBottomLeftRadius"),
  borderBottomRightRadius: B.radii("borderBottomRightRadius"),
  borderLeft: B.borders("borderLeft"),
  borderInlineStart: {
    property: "borderInlineStart",
    scale: "borders"
  },
  borderInlineStartRadius: B.logical({
    scale: "radii",
    property: {
      ltr: ["borderTopLeftRadius", "borderBottomLeftRadius"],
      rtl: ["borderTopRightRadius", "borderBottomRightRadius"]
    }
  }),
  borderInlineEndRadius: B.logical({
    scale: "radii",
    property: {
      ltr: ["borderTopRightRadius", "borderBottomRightRadius"],
      rtl: ["borderTopLeftRadius", "borderBottomLeftRadius"]
    }
  }),
  borderX: B.borders(["borderLeft", "borderRight"]),
  borderInline: B.borders("borderInline"),
  borderY: B.borders(["borderTop", "borderBottom"]),
  borderBlock: B.borders("borderBlock"),
  borderTopWidth: B.borderWidths("borderTopWidth"),
  borderBlockStartWidth: B.borderWidths("borderBlockStartWidth"),
  borderTopColor: B.colors("borderTopColor"),
  borderBlockStartColor: B.colors("borderBlockStartColor"),
  borderTopStyle: B.borderStyles("borderTopStyle"),
  borderBlockStartStyle: B.borderStyles("borderBlockStartStyle"),
  borderBottomWidth: B.borderWidths("borderBottomWidth"),
  borderBlockEndWidth: B.borderWidths("borderBlockEndWidth"),
  borderBottomColor: B.colors("borderBottomColor"),
  borderBlockEndColor: B.colors("borderBlockEndColor"),
  borderBottomStyle: B.borderStyles("borderBottomStyle"),
  borderBlockEndStyle: B.borderStyles("borderBlockEndStyle"),
  borderLeftWidth: B.borderWidths("borderLeftWidth"),
  borderInlineStartWidth: B.borderWidths("borderInlineStartWidth"),
  borderLeftColor: B.colors("borderLeftColor"),
  borderInlineStartColor: B.colors("borderInlineStartColor"),
  borderLeftStyle: B.borderStyles("borderLeftStyle"),
  borderInlineStartStyle: B.borderStyles("borderInlineStartStyle"),
  borderRightWidth: B.borderWidths("borderRightWidth"),
  borderInlineEndWidth: B.borderWidths("borderInlineEndWidth"),
  borderRightColor: B.colors("borderRightColor"),
  borderInlineEndColor: B.colors("borderInlineEndColor"),
  borderRightStyle: B.borderStyles("borderRightStyle"),
  borderInlineEndStyle: B.borderStyles("borderInlineEndStyle"),
  borderTopRadius: B.radii(["borderTopLeftRadius", "borderTopRightRadius"]),
  borderBottomRadius: B.radii([
    "borderBottomLeftRadius",
    "borderBottomRightRadius"
  ]),
  borderLeftRadius: B.radii(["borderTopLeftRadius", "borderBottomLeftRadius"]),
  borderRightRadius: B.radii([
    "borderTopRightRadius",
    "borderBottomRightRadius"
  ])
};
Object.assign(qe, {
  rounded: qe.borderRadius,
  roundedTop: qe.borderTopRadius,
  roundedTopLeft: qe.borderTopLeftRadius,
  roundedTopRight: qe.borderTopRightRadius,
  roundedTopStart: qe.borderStartStartRadius,
  roundedTopEnd: qe.borderStartEndRadius,
  roundedBottom: qe.borderBottomRadius,
  roundedBottomLeft: qe.borderBottomLeftRadius,
  roundedBottomRight: qe.borderBottomRightRadius,
  roundedBottomStart: qe.borderEndStartRadius,
  roundedBottomEnd: qe.borderEndEndRadius,
  roundedLeft: qe.borderLeftRadius,
  roundedRight: qe.borderRightRadius,
  roundedStart: qe.borderInlineStartRadius,
  roundedEnd: qe.borderInlineEndRadius,
  borderStart: qe.borderInlineStart,
  borderEnd: qe.borderInlineEnd,
  borderTopStartRadius: qe.borderStartStartRadius,
  borderTopEndRadius: qe.borderStartEndRadius,
  borderBottomStartRadius: qe.borderEndStartRadius,
  borderBottomEndRadius: qe.borderEndEndRadius,
  borderStartRadius: qe.borderInlineStartRadius,
  borderEndRadius: qe.borderInlineEndRadius,
  borderStartWidth: qe.borderInlineStartWidth,
  borderEndWidth: qe.borderInlineEndWidth,
  borderStartColor: qe.borderInlineStartColor,
  borderEndColor: qe.borderInlineEndColor,
  borderStartStyle: qe.borderInlineStartStyle,
  borderEndStyle: qe.borderInlineEndStyle
});
var aL = {
  color: B.colors("color"),
  textColor: B.colors("color"),
  fill: B.colors("fill"),
  stroke: B.colors("stroke")
}, dh = {
  boxShadow: B.shadows("boxShadow"),
  mixBlendMode: !0,
  blendMode: B.prop("mixBlendMode"),
  backgroundBlendMode: !0,
  bgBlendMode: B.prop("backgroundBlendMode"),
  opacity: !0
};
Object.assign(dh, {
  shadow: dh.boxShadow
});
var sL = {
  filter: { transform: Ue.filter },
  blur: B.blur("--chakra-blur"),
  brightness: B.propT("--chakra-brightness", Ue.brightness),
  contrast: B.propT("--chakra-contrast", Ue.contrast),
  hueRotate: B.degreeT("--chakra-hue-rotate"),
  invert: B.propT("--chakra-invert", Ue.invert),
  saturate: B.propT("--chakra-saturate", Ue.saturate),
  dropShadow: B.propT("--chakra-drop-shadow", Ue.dropShadow),
  backdropFilter: { transform: Ue.backdropFilter },
  backdropBlur: B.blur("--chakra-backdrop-blur"),
  backdropBrightness: B.propT(
    "--chakra-backdrop-brightness",
    Ue.brightness
  ),
  backdropContrast: B.propT("--chakra-backdrop-contrast", Ue.contrast),
  backdropHueRotate: B.degreeT("--chakra-backdrop-hue-rotate"),
  backdropInvert: B.propT("--chakra-backdrop-invert", Ue.invert),
  backdropSaturate: B.propT("--chakra-backdrop-saturate", Ue.saturate)
}, Tl = {
  alignItems: !0,
  alignContent: !0,
  justifyItems: !0,
  justifyContent: !0,
  flexWrap: !0,
  flexDirection: { transform: Ue.flexDirection },
  flex: !0,
  flexFlow: !0,
  flexGrow: !0,
  flexShrink: !0,
  flexBasis: B.sizes("flexBasis"),
  justifySelf: !0,
  alignSelf: !0,
  order: !0,
  placeItems: !0,
  placeContent: !0,
  placeSelf: !0,
  gap: B.space("gap"),
  rowGap: B.space("rowGap"),
  columnGap: B.space("columnGap")
};
Object.assign(Tl, {
  flexDir: Tl.flexDirection
});
var nb = {
  gridGap: B.space("gridGap"),
  gridColumnGap: B.space("gridColumnGap"),
  gridRowGap: B.space("gridRowGap"),
  gridColumn: !0,
  gridRow: !0,
  gridAutoFlow: !0,
  gridAutoColumns: !0,
  gridColumnStart: !0,
  gridColumnEnd: !0,
  gridRowStart: !0,
  gridRowEnd: !0,
  gridAutoRows: !0,
  gridTemplate: !0,
  gridTemplateColumns: !0,
  gridTemplateRows: !0,
  gridTemplateAreas: !0,
  gridArea: !0
}, oL = {
  appearance: !0,
  cursor: !0,
  resize: !0,
  userSelect: !0,
  pointerEvents: !0,
  outline: { transform: Ue.outline },
  outlineOffset: !0,
  outlineColor: B.colors("outlineColor")
}, Vr = {
  width: B.sizesT("width"),
  inlineSize: B.sizesT("inlineSize"),
  height: B.sizes("height"),
  blockSize: B.sizes("blockSize"),
  boxSize: B.sizes(["width", "height"]),
  minWidth: B.sizes("minWidth"),
  minInlineSize: B.sizes("minInlineSize"),
  minHeight: B.sizes("minHeight"),
  minBlockSize: B.sizes("minBlockSize"),
  maxWidth: B.sizes("maxWidth"),
  maxInlineSize: B.sizes("maxInlineSize"),
  maxHeight: B.sizes("maxHeight"),
  maxBlockSize: B.sizes("maxBlockSize"),
  overflow: !0,
  overflowX: !0,
  overflowY: !0,
  overscrollBehavior: !0,
  overscrollBehaviorX: !0,
  overscrollBehaviorY: !0,
  display: !0,
  hideFrom: {
    scale: "breakpoints",
    transform: (t, e) => {
      var r, n, i;
      return { [(i = (n = (r = e.__breakpoints) == null ? void 0 : r.get(t)) == null ? void 0 : n.minWQuery) != null ? i : `@media screen and (min-width: ${t})`]: { display: "none" } };
    }
  },
  hideBelow: {
    scale: "breakpoints",
    transform: (t, e) => {
      var r, n, i;
      return { [(i = (n = (r = e.__breakpoints) == null ? void 0 : r.get(t)) == null ? void 0 : n.maxWQuery) != null ? i : `@media screen and (max-width: ${t})`]: { display: "none" } };
    }
  },
  verticalAlign: !0,
  boxSizing: !0,
  boxDecorationBreak: !0,
  float: B.propT("float", Ue.float),
  objectFit: !0,
  objectPosition: !0,
  visibility: !0,
  isolation: !0
};
Object.assign(Vr, {
  w: Vr.width,
  h: Vr.height,
  minW: Vr.minWidth,
  maxW: Vr.maxWidth,
  minH: Vr.minHeight,
  maxH: Vr.maxHeight,
  overscroll: Vr.overscrollBehavior,
  overscrollX: Vr.overscrollBehaviorX,
  overscrollY: Vr.overscrollBehaviorY
});
var uL = {
  listStyleType: !0,
  listStylePosition: !0,
  listStylePos: B.prop("listStylePosition"),
  listStyleImage: !0,
  listStyleImg: B.prop("listStyleImage")
};
function lL(t, e, r, n) {
  const i = typeof e == "string" ? e.split(".") : [e];
  for (n = 0; n < i.length && t; n += 1)
    t = t[i[n]];
  return t === void 0 ? r : t;
}
var cL = (t) => {
  const e = /* @__PURE__ */ new WeakMap();
  return (n, i, s, o) => {
    if (typeof n > "u")
      return t(n, i, s);
    e.has(n) || e.set(n, /* @__PURE__ */ new Map());
    const l = e.get(n);
    if (l.has(i))
      return l.get(i);
    const c = t(n, i, s, o);
    return l.set(i, c), c;
  };
}, fL = cL(lL), hL = {
  border: "0px",
  clip: "rect(0, 0, 0, 0)",
  width: "1px",
  height: "1px",
  margin: "-1px",
  padding: "0px",
  overflow: "hidden",
  whiteSpace: "nowrap",
  position: "absolute"
}, dL = {
  position: "static",
  width: "auto",
  height: "auto",
  clip: "auto",
  padding: "0",
  margin: "0",
  overflow: "visible",
  whiteSpace: "normal"
}, kf = (t, e, r) => {
  const n = {}, i = fL(t, e, {});
  for (const s in i)
    s in r && r[s] != null || (n[s] = i[s]);
  return n;
}, gL = {
  srOnly: {
    transform(t) {
      return t === !0 ? hL : t === "focusable" ? dL : {};
    }
  },
  layerStyle: {
    processResult: !0,
    transform: (t, e, r) => kf(e, `layerStyles.${t}`, r)
  },
  textStyle: {
    processResult: !0,
    transform: (t, e, r) => kf(e, `textStyles.${t}`, r)
  },
  apply: {
    processResult: !0,
    transform: (t, e, r) => kf(e, t, r)
  }
}, ao = {
  position: !0,
  pos: B.prop("position"),
  zIndex: B.prop("zIndex", "zIndices"),
  inset: B.spaceT("inset"),
  insetX: B.spaceT(["left", "right"]),
  insetInline: B.spaceT("insetInline"),
  insetY: B.spaceT(["top", "bottom"]),
  insetBlock: B.spaceT("insetBlock"),
  top: B.spaceT("top"),
  insetBlockStart: B.spaceT("insetBlockStart"),
  bottom: B.spaceT("bottom"),
  insetBlockEnd: B.spaceT("insetBlockEnd"),
  left: B.spaceT("left"),
  insetInlineStart: B.logical({
    scale: "space",
    property: { ltr: "left", rtl: "right" }
  }),
  right: B.spaceT("right"),
  insetInlineEnd: B.logical({
    scale: "space",
    property: { ltr: "right", rtl: "left" }
  })
};
Object.assign(ao, {
  insetStart: ao.insetInlineStart,
  insetEnd: ao.insetInlineEnd
});
var pL = {
  ring: { transform: Ue.ring },
  ringColor: B.colors("--chakra-ring-color"),
  ringOffset: B.prop("--chakra-ring-offset-width"),
  ringOffsetColor: B.colors("--chakra-ring-offset-color"),
  ringInset: B.prop("--chakra-ring-inset")
}, ht = {
  margin: B.spaceT("margin"),
  marginTop: B.spaceT("marginTop"),
  marginBlockStart: B.spaceT("marginBlockStart"),
  marginRight: B.spaceT("marginRight"),
  marginInlineEnd: B.spaceT("marginInlineEnd"),
  marginBottom: B.spaceT("marginBottom"),
  marginBlockEnd: B.spaceT("marginBlockEnd"),
  marginLeft: B.spaceT("marginLeft"),
  marginInlineStart: B.spaceT("marginInlineStart"),
  marginX: B.spaceT(["marginInlineStart", "marginInlineEnd"]),
  marginInline: B.spaceT("marginInline"),
  marginY: B.spaceT(["marginTop", "marginBottom"]),
  marginBlock: B.spaceT("marginBlock"),
  padding: B.space("padding"),
  paddingTop: B.space("paddingTop"),
  paddingBlockStart: B.space("paddingBlockStart"),
  paddingRight: B.space("paddingRight"),
  paddingBottom: B.space("paddingBottom"),
  paddingBlockEnd: B.space("paddingBlockEnd"),
  paddingLeft: B.space("paddingLeft"),
  paddingInlineStart: B.space("paddingInlineStart"),
  paddingInlineEnd: B.space("paddingInlineEnd"),
  paddingX: B.space(["paddingInlineStart", "paddingInlineEnd"]),
  paddingInline: B.space("paddingInline"),
  paddingY: B.space(["paddingTop", "paddingBottom"]),
  paddingBlock: B.space("paddingBlock")
};
Object.assign(ht, {
  m: ht.margin,
  mt: ht.marginTop,
  mr: ht.marginRight,
  me: ht.marginInlineEnd,
  marginEnd: ht.marginInlineEnd,
  mb: ht.marginBottom,
  ml: ht.marginLeft,
  ms: ht.marginInlineStart,
  marginStart: ht.marginInlineStart,
  mx: ht.marginX,
  my: ht.marginY,
  p: ht.padding,
  pt: ht.paddingTop,
  py: ht.paddingY,
  px: ht.paddingX,
  pb: ht.paddingBottom,
  pl: ht.paddingLeft,
  ps: ht.paddingInlineStart,
  paddingStart: ht.paddingInlineStart,
  pr: ht.paddingRight,
  pe: ht.paddingInlineEnd,
  paddingEnd: ht.paddingInlineEnd
});
var mL = {
  textDecorationColor: B.colors("textDecorationColor"),
  textDecoration: !0,
  textDecor: { property: "textDecoration" },
  textDecorationLine: !0,
  textDecorationStyle: !0,
  textDecorationThickness: !0,
  textUnderlineOffset: !0,
  textShadow: B.shadows("textShadow")
}, vL = {
  clipPath: !0,
  transform: B.propT("transform", Ue.transform),
  transformOrigin: !0,
  translateX: B.spaceT("--chakra-translate-x"),
  translateY: B.spaceT("--chakra-translate-y"),
  skewX: B.degreeT("--chakra-skew-x"),
  skewY: B.degreeT("--chakra-skew-y"),
  scaleX: B.prop("--chakra-scale-x"),
  scaleY: B.prop("--chakra-scale-y"),
  scale: B.prop(["--chakra-scale-x", "--chakra-scale-y"]),
  rotate: B.degreeT("--chakra-rotate")
}, yL = {
  transition: !0,
  transitionDelay: !0,
  animation: !0,
  willChange: !0,
  transitionDuration: B.prop("transitionDuration", "transition.duration"),
  transitionProperty: B.prop("transitionProperty", "transition.property"),
  transitionTimingFunction: B.prop(
    "transitionTimingFunction",
    "transition.easing"
  )
}, bL = {
  fontFamily: B.prop("fontFamily", "fonts"),
  fontSize: B.prop("fontSize", "fontSizes", Ue.px),
  fontWeight: B.prop("fontWeight", "fontWeights"),
  lineHeight: B.prop("lineHeight", "lineHeights"),
  letterSpacing: B.prop("letterSpacing", "letterSpacings"),
  textAlign: !0,
  fontStyle: !0,
  textIndent: !0,
  wordBreak: !0,
  overflowWrap: !0,
  textOverflow: !0,
  textTransform: !0,
  whiteSpace: !0,
  isTruncated: {
    transform(t) {
      if (t === !0)
        return {
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap"
        };
    }
  },
  noOfLines: {
    static: {
      overflow: "hidden",
      textOverflow: "ellipsis",
      display: "-webkit-box",
      WebkitBoxOrient: "vertical",
      WebkitLineClamp: "var(--chakra-line-clamp)"
    },
    property: "--chakra-line-clamp"
  }
}, _L = {
  scrollBehavior: !0,
  scrollSnapAlign: !0,
  scrollSnapStop: !0,
  scrollSnapType: !0,
  scrollMargin: B.spaceT("scrollMargin"),
  scrollMarginTop: B.spaceT("scrollMarginTop"),
  scrollMarginBottom: B.spaceT("scrollMarginBottom"),
  scrollMarginLeft: B.spaceT("scrollMarginLeft"),
  scrollMarginRight: B.spaceT("scrollMarginRight"),
  scrollMarginX: B.spaceT(["scrollMarginLeft", "scrollMarginRight"]),
  scrollMarginY: B.spaceT(["scrollMarginTop", "scrollMarginBottom"]),
  scrollPadding: B.spaceT("scrollPadding"),
  scrollPaddingTop: B.spaceT("scrollPaddingTop"),
  scrollPaddingBottom: B.spaceT("scrollPaddingBottom"),
  scrollPaddingLeft: B.spaceT("scrollPaddingLeft"),
  scrollPaddingRight: B.spaceT("scrollPaddingRight"),
  scrollPaddingX: B.spaceT(["scrollPaddingLeft", "scrollPaddingRight"]),
  scrollPaddingY: B.spaceT(["scrollPaddingTop", "scrollPaddingBottom"])
};
function ib(t) {
  return un(t) && t.reference ? t.reference : String(t);
}
var Wl = (t, ...e) => e.map(ib).join(` ${t} `).replace(/calc/g, ""), Ev = (...t) => `calc(${Wl("+", ...t)})`, Cv = (...t) => `calc(${Wl("-", ...t)})`, gh = (...t) => `calc(${Wl("*", ...t)})`, xv = (...t) => `calc(${Wl("/", ...t)})`, Ov = (t) => {
  const e = ib(t);
  return e != null && !Number.isNaN(parseFloat(e)) ? String(e).startsWith("-") ? String(e).slice(1) : `-${e}` : gh(e, -1);
}, qi = Object.assign(
  (t) => ({
    add: (...e) => qi(Ev(t, ...e)),
    subtract: (...e) => qi(Cv(t, ...e)),
    multiply: (...e) => qi(gh(t, ...e)),
    divide: (...e) => qi(xv(t, ...e)),
    negate: () => qi(Ov(t)),
    toString: () => t.toString()
  }),
  {
    add: Ev,
    subtract: Cv,
    multiply: gh,
    divide: xv,
    negate: Ov
  }
);
function SL(t, e = "-") {
  return t.replace(/\s+/g, e);
}
function wL(t) {
  const e = SL(t.toString());
  return CL(EL(e));
}
function EL(t) {
  return t.includes("\\.") ? t : !Number.isInteger(parseFloat(t.toString())) ? t.replace(".", "\\.") : t;
}
function CL(t) {
  return t.replace(/[!-,/:-@[-^`{-~]/g, "\\$&");
}
function xL(t, e = "") {
  return [e, t].filter(Boolean).join("-");
}
function OL(t, e) {
  return `var(${t}${e ? `, ${e}` : ""})`;
}
function TL(t, e = "") {
  return wL(`--${xL(t, e)}`);
}
function PL(t, e, r) {
  const n = TL(t, r);
  return {
    variable: n,
    reference: OL(n, e)
  };
}
function RL(t) {
  const e = t == null ? 0 : t.length;
  return e ? t[e - 1] : void 0;
}
function AL(t) {
  const e = parseFloat(t.toString()), r = t.toString().replace(String(e), "");
  return { unitless: !r, value: e, unit: r };
}
function ph(t) {
  if (t == null)
    return t;
  const { unitless: e } = AL(t);
  return e || typeof t == "number" ? `${t}px` : t;
}
var ab = (t, e) => parseInt(t[1], 10) > parseInt(e[1], 10) ? 1 : -1, Kh = (t) => Object.fromEntries(Object.entries(t).sort(ab));
function Tv(t) {
  const e = Kh(t);
  return Object.assign(Object.values(e), e);
}
function kL(t) {
  const e = Object.keys(Kh(t));
  return new Set(e);
}
function Pv(t) {
  var e;
  if (!t)
    return t;
  t = (e = ph(t)) != null ? e : t;
  const r = -0.02;
  return typeof t == "number" ? `${t + r}` : t.replace(/(\d+\.?\d*)/u, (n) => `${parseFloat(n) + r}`);
}
function Zs(t, e) {
  const r = ["@media screen"];
  return t && r.push("and", `(min-width: ${ph(t)})`), e && r.push("and", `(max-width: ${ph(e)})`), r.join(" ");
}
function IL(t) {
  var e;
  if (!t)
    return null;
  t.base = (e = t.base) != null ? e : "0px";
  const r = Tv(t), n = Object.entries(t).sort(ab).map(([o, l], c, h) => {
    var d;
    let [, g] = (d = h[c + 1]) != null ? d : [];
    return g = parseFloat(g) > 0 ? Pv(g) : void 0, {
      _minW: Pv(l),
      breakpoint: o,
      minW: l,
      maxW: g,
      maxWQuery: Zs(null, g),
      minWQuery: Zs(l),
      minMaxQuery: Zs(l, g)
    };
  }), i = kL(t), s = Array.from(i.values());
  return {
    keys: i,
    normalized: r,
    isResponsive(o) {
      const l = Object.keys(o);
      return l.length > 0 && l.every((c) => i.has(c));
    },
    asObject: Kh(t),
    asArray: Tv(t),
    details: n,
    get(o) {
      return n.find((l) => l.breakpoint === o);
    },
    media: [
      null,
      ...r.map((o) => Zs(o)).slice(1)
    ],
    toArrayValue(o) {
      if (!un(o))
        throw new Error("toArrayValue: value must be an object");
      const l = s.map((c) => {
        var h;
        return (h = o[c]) != null ? h : null;
      });
      for (; RL(l) === null; )
        l.pop();
      return l;
    },
    toObjectValue(o) {
      if (!Array.isArray(o))
        throw new Error("toObjectValue: value must be an array");
      return o.reduce((l, c, h) => {
        const d = s[h];
        return d != null && c != null && (l[d] = c), l;
      }, {});
    }
  };
}
var Xt = {
  hover: (t, e) => `${t}:hover ${e}, ${t}[data-hover] ${e}`,
  focus: (t, e) => `${t}:focus ${e}, ${t}[data-focus] ${e}`,
  focusVisible: (t, e) => `${t}:focus-visible ${e}`,
  focusWithin: (t, e) => `${t}:focus-within ${e}`,
  active: (t, e) => `${t}:active ${e}, ${t}[data-active] ${e}`,
  disabled: (t, e) => `${t}:disabled ${e}, ${t}[data-disabled] ${e}`,
  invalid: (t, e) => `${t}:invalid ${e}, ${t}[data-invalid] ${e}`,
  checked: (t, e) => `${t}:checked ${e}, ${t}[data-checked] ${e}`,
  indeterminate: (t, e) => `${t}:indeterminate ${e}, ${t}[aria-checked=mixed] ${e}, ${t}[data-indeterminate] ${e}`,
  readOnly: (t, e) => `${t}:read-only ${e}, ${t}[readonly] ${e}, ${t}[data-read-only] ${e}`,
  expanded: (t, e) => `${t}:read-only ${e}, ${t}[aria-expanded=true] ${e}, ${t}[data-expanded] ${e}`,
  placeholderShown: (t, e) => `${t}:placeholder-shown ${e}`
}, mi = (t) => sb((e) => t(e, "&"), "[role=group]", "[data-group]", ".group"), Xn = (t) => sb((e) => t(e, "~ &"), "[data-peer]", ".peer"), sb = (t, ...e) => e.map(t).join(", "), Ul = {
  _hover: "&:hover, &[data-hover]",
  _active: "&:active, &[data-active]",
  _focus: "&:focus, &[data-focus]",
  _highlighted: "&[data-highlighted]",
  _focusWithin: "&:focus-within",
  _focusVisible: "&:focus-visible, &[data-focus-visible]",
  _disabled: "&:disabled, &[disabled], &[aria-disabled=true], &[data-disabled]",
  _readOnly: "&[aria-readonly=true], &[readonly], &[data-readonly]",
  _before: "&::before",
  _after: "&::after",
  _empty: "&:empty",
  _expanded: "&[aria-expanded=true], &[data-expanded]",
  _checked: "&[aria-checked=true], &[data-checked]",
  _grabbed: "&[aria-grabbed=true], &[data-grabbed]",
  _pressed: "&[aria-pressed=true], &[data-pressed]",
  _invalid: "&[aria-invalid=true], &[data-invalid]",
  _valid: "&[data-valid], &[data-state=valid]",
  _loading: "&[data-loading], &[aria-busy=true]",
  _selected: "&[aria-selected=true], &[data-selected]",
  _hidden: "&[hidden], &[data-hidden]",
  _autofill: "&:-webkit-autofill",
  _even: "&:nth-of-type(even)",
  _odd: "&:nth-of-type(odd)",
  _first: "&:first-of-type",
  _last: "&:last-of-type",
  _notFirst: "&:not(:first-of-type)",
  _notLast: "&:not(:last-of-type)",
  _visited: "&:visited",
  _activeLink: "&[aria-current=page]",
  _activeStep: "&[aria-current=step]",
  _indeterminate: "&:indeterminate, &[aria-checked=mixed], &[data-indeterminate]",
  _groupHover: mi(Xt.hover),
  _peerHover: Xn(Xt.hover),
  _groupFocus: mi(Xt.focus),
  _peerFocus: Xn(Xt.focus),
  _groupFocusVisible: mi(Xt.focusVisible),
  _peerFocusVisible: Xn(Xt.focusVisible),
  _groupActive: mi(Xt.active),
  _peerActive: Xn(Xt.active),
  _groupDisabled: mi(Xt.disabled),
  _peerDisabled: Xn(Xt.disabled),
  _groupInvalid: mi(Xt.invalid),
  _peerInvalid: Xn(Xt.invalid),
  _groupChecked: mi(Xt.checked),
  _peerChecked: Xn(Xt.checked),
  _groupFocusWithin: mi(Xt.focusWithin),
  _peerFocusWithin: Xn(Xt.focusWithin),
  _peerPlaceholderShown: Xn(Xt.placeholderShown),
  _placeholder: "&::placeholder",
  _placeholderShown: "&:placeholder-shown",
  _fullScreen: "&:fullscreen",
  _selection: "&::selection",
  _rtl: "[dir=rtl] &, &[dir=rtl]",
  _ltr: "[dir=ltr] &, &[dir=ltr]",
  _mediaDark: "@media (prefers-color-scheme: dark)",
  _mediaReduceMotion: "@media (prefers-reduced-motion: reduce)",
  _dark: ".chakra-ui-dark &:not([data-theme]),[data-theme=dark] &:not([data-theme]),&[data-theme=dark]",
  _light: ".chakra-ui-light &:not([data-theme]),[data-theme=light] &:not([data-theme]),&[data-theme=light]"
}, LL = Object.keys(
  Ul
);
function Rv(t, e) {
  return PL(String(t).replace(/\./g, "-"), void 0, e);
}
function DL(t, e) {
  let r = {};
  const n = {};
  for (const [i, s] of Object.entries(t)) {
    const { isSemantic: o, value: l } = s, { variable: c, reference: h } = Rv(i, e == null ? void 0 : e.cssVarPrefix);
    if (!o) {
      if (i.startsWith("space")) {
        const p = i.split("."), [y, ...v] = p, _ = `${y}.-${v.join(".")}`, b = qi.negate(l), C = qi.negate(h);
        n[_] = {
          value: b,
          var: c,
          varRef: C
        };
      }
      r[c] = l, n[i] = {
        value: l,
        var: c,
        varRef: h
      };
      continue;
    }
    const d = (p) => {
      const v = [String(i).split(".")[0], p].join(".");
      if (!t[v])
        return p;
      const { reference: b } = Rv(v, e == null ? void 0 : e.cssVarPrefix);
      return b;
    }, g = un(l) ? l : { default: l };
    r = On(
      r,
      Object.entries(g).reduce(
        (p, [y, v]) => {
          var _, b;
          const C = d(v);
          if (y === "default")
            return p[c] = C, p;
          const w = (b = (_ = Ul) == null ? void 0 : _[y]) != null ? b : y;
          return p[w] = { [c]: C }, p;
        },
        {}
      )
    ), n[i] = {
      value: h,
      var: c,
      varRef: h
    };
  }
  return {
    cssVars: r,
    cssMap: n
  };
}
function ML(t, e = []) {
  const r = Object.assign({}, t);
  for (const n of e)
    n in r && delete r[n];
  return r;
}
function NL(t, e) {
  const r = {};
  for (const n of e)
    n in t && (r[n] = t[n]);
  return r;
}
var BL = [
  "colors",
  "borders",
  "borderWidths",
  "borderStyles",
  "fonts",
  "fontSizes",
  "fontWeights",
  "letterSpacings",
  "lineHeights",
  "radii",
  "space",
  "shadows",
  "sizes",
  "zIndices",
  "transition",
  "blur",
  "breakpoints"
];
function FL(t) {
  return NL(t, BL);
}
function GL(t) {
  return t.semanticTokens;
}
function jL(t) {
  const { __cssMap: e, __cssVars: r, __breakpoints: n, ...i } = t;
  return i;
}
function $L({
  tokens: t,
  semanticTokens: e
}) {
  var r, n;
  const i = Object.entries((r = mh(t)) != null ? r : {}).map(
    ([o, l]) => [o, { isSemantic: !1, value: l }]
  ), s = Object.entries(
    (n = mh(e, 1)) != null ? n : {}
  ).map(([o, l]) => [o, { isSemantic: !0, value: l }]);
  return Object.fromEntries([
    ...i,
    ...s
  ]);
}
function mh(t, e = 1 / 0) {
  return !un(t) && !Array.isArray(t) || !e ? t : Object.entries(t).reduce((r, [n, i]) => (un(i) || Array.isArray(i) ? Object.entries(mh(i, e - 1)).forEach(
    ([s, o]) => {
      r[`${n}.${s}`] = o;
    }
  ) : r[n] = i, r), {});
}
function o5(t) {
  var e;
  const r = jL(t), n = FL(r), i = GL(r), s = $L({ tokens: n, semanticTokens: i }), o = (e = r.config) == null ? void 0 : e.cssVarPrefix, {
    cssMap: l,
    cssVars: c
  } = DL(s, { cssVarPrefix: o });
  return Object.assign(r, {
    __cssVars: { ...{
      "--chakra-ring-inset": "var(--chakra-empty,/*!*/ /*!*/)",
      "--chakra-ring-offset-width": "0px",
      "--chakra-ring-offset-color": "#fff",
      "--chakra-ring-color": "rgba(66, 153, 225, 0.6)",
      "--chakra-ring-offset-shadow": "0 0 #0000",
      "--chakra-ring-shadow": "0 0 #0000",
      "--chakra-space-x-reverse": "0",
      "--chakra-space-y-reverse": "0"
    }, ...c },
    __cssMap: l,
    __breakpoints: IL(r.breakpoints)
  }), r;
}
var Xh = On(
  {},
  ll,
  qe,
  aL,
  Tl,
  Vr,
  sL,
  pL,
  oL,
  nb,
  gL,
  ao,
  dh,
  ht,
  _L,
  bL,
  mL,
  vL,
  uL,
  yL
), WL = Object.assign({}, ht, Vr, Tl, nb, ao), u5 = Object.keys(
  WL
), UL = [...Object.keys(Xh), ...LL], HL = { ...Xh, ...Ul }, zL = (t) => t in HL, VL = (t) => (e) => {
  if (!e.__breakpoints)
    return t;
  const { isResponsive: r, toArrayValue: n, media: i } = e.__breakpoints, s = {};
  for (const o in t) {
    let l = Yi(t[o], e);
    if (l == null)
      continue;
    if (l = un(l) && r(l) ? n(l) : l, !Array.isArray(l)) {
      s[o] = l;
      continue;
    }
    const c = l.slice(0, i.length).length;
    for (let h = 0; h < c; h += 1) {
      const d = i == null ? void 0 : i[h];
      if (!d) {
        s[o] = l[h];
        continue;
      }
      s[d] = s[d] || {}, l[h] != null && (s[d][o] = l[h]);
    }
  }
  return s;
};
function qL(t) {
  const e = [];
  let r = "", n = !1;
  for (let i = 0; i < t.length; i++) {
    const s = t[i];
    s === "(" ? (n = !0, r += s) : s === ")" ? (n = !1, r += s) : s === "," && !n ? (e.push(r), r = "") : r += s;
  }
  return r = r.trim(), r && e.push(r), e;
}
function YL(t) {
  return /^var\(--.+\)$/.test(t);
}
var KL = (t, e) => t.startsWith("--") && typeof e == "string" && !YL(e), XL = (t, e) => {
  var r, n;
  if (e == null)
    return e;
  const i = (c) => {
    var h, d;
    return (d = (h = t.__cssMap) == null ? void 0 : h[c]) == null ? void 0 : d.varRef;
  }, s = (c) => {
    var h;
    return (h = i(c)) != null ? h : c;
  }, [o, l] = qL(e);
  return e = (n = (r = i(o)) != null ? r : s(l)) != null ? n : s(e), e;
};
function JL(t) {
  const { configs: e = {}, pseudos: r = {}, theme: n } = t, i = (s, o = !1) => {
    var l, c, h;
    const d = Yi(s, n), g = VL(d)(n);
    let p = {};
    for (let y in g) {
      const v = g[y];
      let _ = Yi(v, n);
      y in r && (y = r[y]), KL(y, _) && (_ = XL(n, _));
      let b = e[y];
      if (b === !0 && (b = { property: y }), un(_)) {
        p[y] = (l = p[y]) != null ? l : {}, p[y] = On(
          {},
          p[y],
          i(_, !0)
        );
        continue;
      }
      let C = (h = (c = b == null ? void 0 : b.transform) == null ? void 0 : c.call(b, _, n, d)) != null ? h : _;
      C = b != null && b.processResult ? i(C, !0) : C;
      const w = Yi(b == null ? void 0 : b.property, n);
      if (!o && (b != null && b.static)) {
        const O = Yi(b.static, n);
        p = On({}, p, O);
      }
      if (w && Array.isArray(w)) {
        for (const O of w)
          p[O] = C;
        continue;
      }
      if (w) {
        w === "&" && un(C) ? p = On({}, p, C) : p[w] = C;
        continue;
      }
      if (un(C)) {
        p = On({}, p, C);
        continue;
      }
      p[y] = C;
    }
    return p;
  };
  return i;
}
var QL = (t) => (e) => JL({
  theme: e,
  pseudos: Ul,
  configs: Xh
})(t);
function l5(t) {
  return t;
}
function c5(t) {
  return t;
}
function f5(t) {
  return {
    definePartsStyle(e) {
      return e;
    },
    defineMultiStyleConfig(e) {
      return { parts: t, ...e };
    }
  };
}
function ZL(t, e) {
  if (Array.isArray(t))
    return t;
  if (un(t))
    return e(t);
  if (t != null)
    return [t];
}
function eD(t, e) {
  for (let r = e + 1; r < t.length; r++)
    if (t[r] != null)
      return r;
  return -1;
}
function tD(t) {
  const e = t.__breakpoints;
  return function(n, i, s, o) {
    var l, c;
    if (!e)
      return;
    const h = {}, d = ZL(s, e.toArrayValue);
    if (!d)
      return h;
    const g = d.length, p = g === 1, y = !!n.parts;
    for (let v = 0; v < g; v++) {
      const _ = e.details[v], b = e.details[eD(d, v)], C = Zs(_.minW, b == null ? void 0 : b._minW), w = Yi((l = n[i]) == null ? void 0 : l[d[v]], o);
      if (w) {
        if (y) {
          (c = n.parts) == null || c.forEach((O) => {
            On(h, {
              [O]: p ? w[O] : { [C]: w[O] }
            });
          });
          continue;
        }
        if (!y) {
          p ? On(h, w) : h[C] = w;
          continue;
        }
        h[C] = w;
      }
    }
    return h;
  };
}
function rD(t) {
  return (e) => {
    var r;
    const { variant: n, size: i, theme: s } = e, o = tD(s);
    return On(
      {},
      Yi((r = t.baseStyle) != null ? r : {}, e),
      o(t, "sizes", i, e),
      o(t, "variants", n, e)
    );
  };
}
function h5(t, e, r) {
  var n, i, s;
  return (s = (i = (n = t.__cssMap) == null ? void 0 : n[`${e}.${r}`]) == null ? void 0 : i.varRef) != null ? s : r;
}
function ob(t) {
  return ML(t, ["styleConfig", "size", "variant", "colorScheme"]);
}
function nD(t, e) {
  const r = {};
  return Object.keys(t).forEach((n) => {
    e.includes(n) || (r[n] = t[n]);
  }), r;
}
function iD(t, e, r, n) {
  const i = typeof e == "string" ? e.split(".") : [e];
  for (n = 0; n < i.length && t; n += 1)
    t = t[i[n]];
  return t === void 0 ? r : t;
}
var aD = (t) => {
  const e = /* @__PURE__ */ new WeakMap();
  return (n, i, s, o) => {
    if (typeof n > "u")
      return t(n, i, s);
    e.has(n) || e.set(n, /* @__PURE__ */ new Map());
    const l = e.get(n);
    if (l.has(i))
      return l.get(i);
    const c = t(n, i, s, o);
    return l.set(i, c), c;
  };
}, sD = aD(iD);
function ub(t, e) {
  const r = {};
  return Object.keys(t).forEach((n) => {
    const i = t[n];
    e(i, n, t) && (r[n] = i);
  }), r;
}
var lb = (t) => ub(t, (e) => e != null);
function oD(t) {
  return typeof t == "function";
}
process.env.NODE_ENV;
process.env.NODE_ENV;
function uD(t, ...e) {
  return oD(t) ? t(...e) : t;
}
function d5(...t) {
  return function(r) {
    t.some((n) => (n == null || n(r), r == null ? void 0 : r.defaultPrevented));
  };
}
var lD = typeof Element < "u", cD = typeof Map == "function", fD = typeof Set == "function", hD = typeof ArrayBuffer == "function" && !!ArrayBuffer.isView;
function cl(t, e) {
  if (t === e)
    return !0;
  if (t && e && typeof t == "object" && typeof e == "object") {
    if (t.constructor !== e.constructor)
      return !1;
    var r, n, i;
    if (Array.isArray(t)) {
      if (r = t.length, r != e.length)
        return !1;
      for (n = r; n-- !== 0; )
        if (!cl(t[n], e[n]))
          return !1;
      return !0;
    }
    var s;
    if (cD && t instanceof Map && e instanceof Map) {
      if (t.size !== e.size)
        return !1;
      for (s = t.entries(); !(n = s.next()).done; )
        if (!e.has(n.value[0]))
          return !1;
      for (s = t.entries(); !(n = s.next()).done; )
        if (!cl(n.value[1], e.get(n.value[0])))
          return !1;
      return !0;
    }
    if (fD && t instanceof Set && e instanceof Set) {
      if (t.size !== e.size)
        return !1;
      for (s = t.entries(); !(n = s.next()).done; )
        if (!e.has(n.value[0]))
          return !1;
      return !0;
    }
    if (hD && ArrayBuffer.isView(t) && ArrayBuffer.isView(e)) {
      if (r = t.length, r != e.length)
        return !1;
      for (n = r; n-- !== 0; )
        if (t[n] !== e[n])
          return !1;
      return !0;
    }
    if (t.constructor === RegExp)
      return t.source === e.source && t.flags === e.flags;
    if (t.valueOf !== Object.prototype.valueOf)
      return t.valueOf() === e.valueOf();
    if (t.toString !== Object.prototype.toString)
      return t.toString() === e.toString();
    if (i = Object.keys(t), r = i.length, r !== Object.keys(e).length)
      return !1;
    for (n = r; n-- !== 0; )
      if (!Object.prototype.hasOwnProperty.call(e, i[n]))
        return !1;
    if (lD && t instanceof Element)
      return !1;
    for (n = r; n-- !== 0; )
      if (!((i[n] === "_owner" || i[n] === "__v" || i[n] === "__o") && t.$$typeof) && !cl(t[i[n]], e[i[n]]))
        return !1;
    return !0;
  }
  return t !== t && e !== e;
}
var dD = function(e, r) {
  try {
    return cl(e, r);
  } catch (n) {
    if ((n.message || "").match(/stack|recursion/i))
      return console.warn("react-fast-compare cannot handle circular refs"), !1;
    throw n;
  }
};
function cb(t, e = {}) {
  var r;
  const { styleConfig: n, ...i } = e, { theme: s, colorMode: o } = GI(), l = t ? sD(s, `components.${t}`) : void 0, c = n || l, h = On(
    { theme: s, colorMode: o },
    (r = c == null ? void 0 : c.defaultProps) != null ? r : {},
    lb(nD(i, ["children"]))
  ), d = so({});
  if (c) {
    const p = rD(c)(h);
    dD(d.current, p) || (d.current = p);
  }
  return d.current;
}
function fb(t, e = {}) {
  return cb(t, e);
}
function g5(t, e = {}) {
  return cb(t, e);
}
var gD = /* @__PURE__ */ new Set([
  ...UL,
  "textStyle",
  "layerStyle",
  "apply",
  "noOfLines",
  "focusBorderColor",
  "errorBorderColor",
  "as",
  "__css",
  "css",
  "sx"
]), pD = /* @__PURE__ */ new Set([
  "htmlWidth",
  "htmlHeight",
  "htmlSize",
  "htmlTranslate"
]);
function mD(t) {
  return pD.has(t) || !gD.has(t);
}
function vD(t) {
  const e = Object.assign({}, t);
  for (let r in e)
    e[r] === void 0 && delete e[r];
  return e;
}
function yD(t, ...e) {
  if (t == null)
    throw new TypeError("Cannot convert undefined or null to object");
  const r = { ...t };
  for (const n of e)
    if (n != null)
      for (const i in n)
        Object.prototype.hasOwnProperty.call(n, i) && (i in r && delete r[i], r[i] = n[i]);
  return r;
}
var bD = /^((children|dangerouslySetInnerHTML|key|ref|autoFocus|defaultValue|defaultChecked|innerHTML|suppressContentEditableWarning|suppressHydrationWarning|valueLink|abbr|accept|acceptCharset|accessKey|action|allow|allowUserMedia|allowPaymentRequest|allowFullScreen|allowTransparency|alt|async|autoComplete|autoPlay|capture|cellPadding|cellSpacing|challenge|charSet|checked|cite|classID|className|cols|colSpan|content|contentEditable|contextMenu|controls|controlsList|coords|crossOrigin|data|dateTime|decoding|default|defer|dir|disabled|disablePictureInPicture|download|draggable|encType|enterKeyHint|form|formAction|formEncType|formMethod|formNoValidate|formTarget|frameBorder|headers|height|hidden|high|href|hrefLang|htmlFor|httpEquiv|id|inputMode|integrity|is|keyParams|keyType|kind|label|lang|list|loading|loop|low|marginHeight|marginWidth|max|maxLength|media|mediaGroup|method|min|minLength|multiple|muted|name|nonce|noValidate|open|optimum|pattern|placeholder|playsInline|poster|preload|profile|radioGroup|readOnly|referrerPolicy|rel|required|reversed|role|rows|rowSpan|sandbox|scope|scoped|scrolling|seamless|selected|shape|size|sizes|slot|span|spellCheck|src|srcDoc|srcLang|srcSet|start|step|style|summary|tabIndex|target|title|translate|type|useMap|value|width|wmode|wrap|about|datatype|inlist|prefix|property|resource|typeof|vocab|autoCapitalize|autoCorrect|autoSave|color|incremental|fallback|inert|itemProp|itemScope|itemType|itemID|itemRef|on|option|results|security|unselectable|accentHeight|accumulate|additive|alignmentBaseline|allowReorder|alphabetic|amplitude|arabicForm|ascent|attributeName|attributeType|autoReverse|azimuth|baseFrequency|baselineShift|baseProfile|bbox|begin|bias|by|calcMode|capHeight|clip|clipPathUnits|clipPath|clipRule|colorInterpolation|colorInterpolationFilters|colorProfile|colorRendering|contentScriptType|contentStyleType|cursor|cx|cy|d|decelerate|descent|diffuseConstant|direction|display|divisor|dominantBaseline|dur|dx|dy|edgeMode|elevation|enableBackground|end|exponent|externalResourcesRequired|fill|fillOpacity|fillRule|filter|filterRes|filterUnits|floodColor|floodOpacity|focusable|fontFamily|fontSize|fontSizeAdjust|fontStretch|fontStyle|fontVariant|fontWeight|format|from|fr|fx|fy|g1|g2|glyphName|glyphOrientationHorizontal|glyphOrientationVertical|glyphRef|gradientTransform|gradientUnits|hanging|horizAdvX|horizOriginX|ideographic|imageRendering|in|in2|intercept|k|k1|k2|k3|k4|kernelMatrix|kernelUnitLength|kerning|keyPoints|keySplines|keyTimes|lengthAdjust|letterSpacing|lightingColor|limitingConeAngle|local|markerEnd|markerMid|markerStart|markerHeight|markerUnits|markerWidth|mask|maskContentUnits|maskUnits|mathematical|mode|numOctaves|offset|opacity|operator|order|orient|orientation|origin|overflow|overlinePosition|overlineThickness|panose1|paintOrder|pathLength|patternContentUnits|patternTransform|patternUnits|pointerEvents|points|pointsAtX|pointsAtY|pointsAtZ|preserveAlpha|preserveAspectRatio|primitiveUnits|r|radius|refX|refY|renderingIntent|repeatCount|repeatDur|requiredExtensions|requiredFeatures|restart|result|rotate|rx|ry|scale|seed|shapeRendering|slope|spacing|specularConstant|specularExponent|speed|spreadMethod|startOffset|stdDeviation|stemh|stemv|stitchTiles|stopColor|stopOpacity|strikethroughPosition|strikethroughThickness|string|stroke|strokeDasharray|strokeDashoffset|strokeLinecap|strokeLinejoin|strokeMiterlimit|strokeOpacity|strokeWidth|surfaceScale|systemLanguage|tableValues|targetX|targetY|textAnchor|textDecoration|textRendering|textLength|to|transform|u1|u2|underlinePosition|underlineThickness|unicode|unicodeBidi|unicodeRange|unitsPerEm|vAlphabetic|vHanging|vIdeographic|vMathematical|values|vectorEffect|version|vertAdvY|vertOriginX|vertOriginY|viewBox|viewTarget|visibility|widths|wordSpacing|writingMode|x|xHeight|x1|x2|xChannelSelector|xlinkActuate|xlinkArcrole|xlinkHref|xlinkRole|xlinkShow|xlinkTitle|xlinkType|xmlBase|xmlns|xmlnsXlink|xmlLang|xmlSpace|y|y1|y2|yChannelSelector|z|zoomAndPan|for|class|autofocus)|(([Dd][Aa][Tt][Aa]|[Aa][Rr][Ii][Aa]|x)-.*))$/, _D = /* @__PURE__ */ H0(
  function(t) {
    return bD.test(t) || t.charCodeAt(0) === 111 && t.charCodeAt(1) === 110 && t.charCodeAt(2) < 91;
  }
  /* Z+1 */
), SD = _D, wD = function(e) {
  return e !== "theme";
}, Av = function(e) {
  return typeof e == "string" && // 96 is one less than the char code
  // for "a" so this is checking that
  // it's a lowercase character
  e.charCodeAt(0) > 96 ? SD : wD;
}, kv = function(e, r, n) {
  var i;
  if (r) {
    var s = r.shouldForwardProp;
    i = e.__emotion_forwardProp && s ? function(o) {
      return e.__emotion_forwardProp(o) && s(o);
    } : s;
  }
  return typeof i != "function" && n && (i = e.__emotion_forwardProp), i;
}, Iv = `You have illegal escape sequence in your template literal, most likely inside content's property value.
Because you write your CSS inside a JavaScript string you actually have to do double escaping, so for example "content: '\\00d7';" should become "content: '\\\\00d7';".
You can read more about this here:
https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals#ES2018_revision_of_illegal_escape_sequences`, ED = function(e) {
  var r = e.cache, n = e.serialized, i = e.isStringTag;
  return Gl(r, n, i), Vh(function() {
    return jl(r, n, i);
  }), null;
}, CD = function t(e, r) {
  if (process.env.NODE_ENV !== "production" && e === void 0)
    throw new Error(`You are trying to create a styled element with an undefined component.
You may have forgotten to import it.`);
  var n = e.__emotion_real === e, i = n && e.__emotion_base || e, s, o;
  r !== void 0 && (s = r.label, o = r.target);
  var l = kv(e, r, n), c = l || Av(i), h = !c("as");
  return function() {
    var d = arguments, g = n && e.__emotion_styles !== void 0 ? e.__emotion_styles.slice(0) : [];
    if (s !== void 0 && g.push("label:" + s + ";"), d[0] == null || d[0].raw === void 0)
      g.push.apply(g, d);
    else {
      process.env.NODE_ENV !== "production" && d[0][0] === void 0 && console.error(Iv), g.push(d[0][0]);
      for (var p = d.length, y = 1; y < p; y++)
        process.env.NODE_ENV !== "production" && d[0][y] === void 0 && console.error(Iv), g.push(d[y], d[0][y]);
    }
    var v = $l(function(_, b, C) {
      var w = h && _.as || i, O = "", k = [], P = _;
      if (_.theme == null) {
        P = {};
        for (var M in _)
          P[M] = _[M];
        P.theme = cn(Oi);
      }
      typeof _.className == "string" ? O = Hh(b.registered, k, _.className) : _.className != null && (O = _.className + " ");
      var U = Ja(g.concat(k), b.registered, P);
      O += b.key + "-" + U.name, o !== void 0 && (O += " " + o);
      var V = h && l === void 0 ? Av(w) : c, Y = {};
      for (var X in _)
        h && X === "as" || // $FlowFixMe
        V(X) && (Y[X] = _[X]);
      return Y.className = O, Y.ref = C, /* @__PURE__ */ Qn(vh, null, /* @__PURE__ */ Qn(ED, {
        cache: b,
        serialized: U,
        isStringTag: typeof w == "string"
      }), /* @__PURE__ */ Qn(w, Y));
    });
    return v.displayName = s !== void 0 ? s : "Styled(" + (typeof i == "string" ? i : i.displayName || i.name || "Component") + ")", v.defaultProps = e.defaultProps, v.__emotion_real = v, v.__emotion_base = i, v.__emotion_styles = g, v.__emotion_forwardProp = l, Object.defineProperty(v, "toString", {
      value: function() {
        return o === void 0 && process.env.NODE_ENV !== "production" ? "NO_COMPONENT_SELECTOR" : "." + o;
      }
    }), v.withComponent = function(_, b) {
      return t(_, dl({}, r, b, {
        shouldForwardProp: kv(v, b, !0)
      })).apply(void 0, g);
    }, v;
  };
}, xD = [
  "a",
  "abbr",
  "address",
  "area",
  "article",
  "aside",
  "audio",
  "b",
  "base",
  "bdi",
  "bdo",
  "big",
  "blockquote",
  "body",
  "br",
  "button",
  "canvas",
  "caption",
  "cite",
  "code",
  "col",
  "colgroup",
  "data",
  "datalist",
  "dd",
  "del",
  "details",
  "dfn",
  "dialog",
  "div",
  "dl",
  "dt",
  "em",
  "embed",
  "fieldset",
  "figcaption",
  "figure",
  "footer",
  "form",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "head",
  "header",
  "hgroup",
  "hr",
  "html",
  "i",
  "iframe",
  "img",
  "input",
  "ins",
  "kbd",
  "keygen",
  "label",
  "legend",
  "li",
  "link",
  "main",
  "map",
  "mark",
  "marquee",
  "menu",
  "menuitem",
  "meta",
  "meter",
  "nav",
  "noscript",
  "object",
  "ol",
  "optgroup",
  "option",
  "output",
  "p",
  "param",
  "picture",
  "pre",
  "progress",
  "q",
  "rp",
  "rt",
  "ruby",
  "s",
  "samp",
  "script",
  "section",
  "select",
  "small",
  "source",
  "span",
  "strong",
  "style",
  "sub",
  "summary",
  "sup",
  "table",
  "tbody",
  "td",
  "textarea",
  "tfoot",
  "th",
  "thead",
  "time",
  "title",
  "tr",
  "track",
  "u",
  "ul",
  "var",
  "video",
  "wbr",
  // SVG
  "circle",
  "clipPath",
  "defs",
  "ellipse",
  "foreignObject",
  "g",
  "image",
  "line",
  "linearGradient",
  "mask",
  "path",
  "pattern",
  "polygon",
  "polyline",
  "radialGradient",
  "rect",
  "stop",
  "svg",
  "text",
  "tspan"
], Pl = CD.bind();
xD.forEach(function(t) {
  Pl[t] = Pl(t);
});
var Lv, OD = (Lv = Pl.default) != null ? Lv : Pl, TD = ({ baseStyle: t }) => (e) => {
  const { theme: r, css: n, __css: i, sx: s, ...o } = e, l = ub(o, (g, p) => zL(p)), c = uD(t, e), h = yD(
    {},
    i,
    c,
    lb(l),
    s
  ), d = QL(h)(e.theme);
  return n ? [d, n] : d;
};
function If(t, e) {
  const { baseStyle: r, ...n } = e ?? {};
  n.shouldForwardProp || (n.shouldForwardProp = mD);
  const i = TD({ baseStyle: r }), s = OD(
    t,
    n
  )(i);
  return ln.forwardRef(function(c, h) {
    const { colorMode: d, forced: g } = Q0();
    return ln.createElement(s, {
      ref: h,
      "data-theme": g ? d : void 0,
      ...c
    });
  });
}
function PD() {
  const t = /* @__PURE__ */ new Map();
  return new Proxy(If, {
    apply(e, r, n) {
      return If(...n);
    },
    get(e, r) {
      return t.has(r) || t.set(r, If(r)), t.get(r);
    }
  });
}
var Rl = PD();
function Jh(t) {
  return Nv(t);
}
var RD = II({
  "0%": {
    transform: "rotate(0deg)"
  },
  "100%": {
    transform: "rotate(360deg)"
  }
}), hb = Jh((t, e) => {
  const r = fb("Spinner", t), {
    label: n = "Loading...",
    thickness: i = "2px",
    speed: s = "0.45s",
    emptyColor: o = "transparent",
    className: l,
    ...c
  } = ob(t), h = Z0("chakra-spinner", l), d = {
    display: "inline-block",
    borderColor: "currentColor",
    borderStyle: "solid",
    borderRadius: "99999px",
    borderWidth: i,
    borderBottomColor: o,
    borderLeftColor: o,
    animation: `${RD} ${s} linear infinite`,
    ...r
  };
  return /* @__PURE__ */ Qt.jsx(
    Rl.div,
    {
      ref: e,
      __css: d,
      className: h,
      ...c,
      children: n && /* @__PURE__ */ Qt.jsx(Rl.span, { srOnly: !0, children: n })
    }
  );
});
hb.displayName = "Spinner";
var db = Jh(function(e, r) {
  const n = fb("Text", e), { className: i, align: s, decoration: o, casing: l, ...c } = ob(e), h = vD({
    textAlign: e.align,
    textDecoration: e.decoration,
    textTransform: e.casing
  });
  return /* @__PURE__ */ Qt.jsx(
    Rl.p,
    {
      ref: r,
      className: Z0("chakra-text", e.className),
      ...h,
      ...c,
      __css: n
    }
  );
});
db.displayName = "Text";
var gb = Jh(function(e, r) {
  const { direction: n, align: i, justify: s, wrap: o, basis: l, grow: c, shrink: h, ...d } = e, g = {
    display: "flex",
    flexDirection: n,
    alignItems: i,
    justifyContent: s,
    flexWrap: o,
    flexBasis: l,
    flexGrow: c,
    flexShrink: h
  };
  return /* @__PURE__ */ Qt.jsx(Rl.div, { ref: r, __css: g, ...d });
});
gb.displayName = "Flex";
const Dv = (t) => {
  const { t: e } = YT(), { showText: r = !1, text: n = e("common.loadingInvokeAI") } = t;
  return /* @__PURE__ */ Qt.jsxs(
    gb,
    {
      width: "100vw",
      height: "100vh",
      alignItems: "center",
      justifyContent: "center",
      bg: "#121212",
      flexDirection: "column",
      rowGap: 4,
      children: [
        /* @__PURE__ */ Qt.jsx(hb, { color: "grey", w: "5rem", h: "5rem" }),
        r && /* @__PURE__ */ Qt.jsx(
          db,
          {
            color: "grey",
            fontWeight: "semibold",
            fontFamily: "'Inter', sans-serif",
            children: n
          }
        )
      ]
    }
  );
}, AD = Bv(() => import("./App-d3a51b95.mjs")), kD = Bv(() => import("./ThemeLocaleProvider-e939cd27.mjs"));
function p5() {
  return /* @__PURE__ */ Qt.jsx(ln.StrictMode, { children: /* @__PURE__ */ Qt.jsx(Ox, { store: F0, children: /* @__PURE__ */ Qt.jsx(Wv, { loading: /* @__PURE__ */ Qt.jsx(Dv, {}), persistor: Ik, children: /* @__PURE__ */ Qt.jsx(ln.Suspense, { fallback: /* @__PURE__ */ Qt.jsx(Dv, { showText: !0 }), children: /* @__PURE__ */ Qt.jsx(kD, { children: /* @__PURE__ */ Qt.jsx(AD, {}) }) }) }) }) });
}
export {
  MN as $,
  Vu as A,
  Pu as B,
  SM as C,
  SO as D,
  YT as E,
  gb as F,
  NB as G,
  JN as H,
  XN as I,
  KN as J,
  iB as K,
  aB as L,
  nB as M,
  jp as N,
  zB as O,
  lB as P,
  Zu as Q,
  gO as R,
  hb as S,
  db as T,
  MB as U,
  TN as V,
  PN as W,
  gN as X,
  _B as Y,
  bN as Z,
  dl as _,
  Z0 as a,
  DD as a$,
  UB as a0,
  HB as a1,
  OB as a2,
  _M as a3,
  lM as a4,
  yN as a5,
  rN as a6,
  sN as a7,
  lN as a8,
  hN as a9,
  mO as aA,
  pN as aB,
  WB as aC,
  xB as aD,
  bO as aE,
  rB as aF,
  VN as aG,
  qN as aH,
  zN as aI,
  GN as aJ,
  fk as aK,
  JP as aL,
  XP as aM,
  tB as aN,
  FN as aO,
  UN as aP,
  DB as aQ,
  CN as aR,
  IB as aS,
  kB as aT,
  LB as aU,
  cB as aV,
  eB as aW,
  wB as aX,
  EB as aY,
  CB as aZ,
  ZN as a_,
  oN as aa,
  fN as ab,
  uN as ac,
  cN as ad,
  dN as ae,
  aN as af,
  iv as ag,
  iN as ah,
  nN as ai,
  $N as aj,
  ON as ak,
  RN as al,
  jN as am,
  SN as an,
  $p as ao,
  NN as ap,
  kN as aq,
  QN as ar,
  HN as as,
  wN as at,
  wO as au,
  EN as av,
  BN as aw,
  VB as ax,
  bB as ay,
  yB as az,
  a5 as b,
  e5 as b$,
  ve as b0,
  Vm as b1,
  PM as b2,
  YM as b3,
  qM as b4,
  oM as b5,
  FM as b6,
  XM as b7,
  $B as b8,
  JD as b9,
  hM as bA,
  Tu as bB,
  Si as bC,
  RM as bD,
  TM as bE,
  CM as bF,
  GD as bG,
  jD as bH,
  GB as bI,
  jB as bJ,
  vM as bK,
  cM as bL,
  aM as bM,
  ZM as bN,
  mB as bO,
  ku as bP,
  En as bQ,
  Fs as bR,
  De as bS,
  IM as bT,
  Ba as bU,
  $D as bV,
  AM as bW,
  sM as bX,
  uM as bY,
  qD as bZ,
  VD as b_,
  EM as ba,
  Hs as bb,
  yM as bc,
  QD as bd,
  lk as be,
  wM as bf,
  YD as bg,
  bM as bh,
  KD as bi,
  rM as bj,
  ND as bk,
  BD as bl,
  FD as bm,
  KM as bn,
  py as bo,
  UD as bp,
  HD as bq,
  zD as br,
  VM as bs,
  iM as bt,
  nM as bu,
  ZD as bv,
  zM as bw,
  ZB as bx,
  eM as by,
  fM as bz,
  Rl as c,
  XD as c0,
  HM as c1,
  LM as c2,
  DM as c3,
  $M as c4,
  mM as c5,
  pM as c6,
  kM as c7,
  tN as c8,
  BM as c9,
  fB as cA,
  TB as cB,
  RB as cC,
  gB as cD,
  Ik as cE,
  dB as cF,
  SB as cG,
  pB as cH,
  AI as cI,
  J0 as cJ,
  l5 as cK,
  f5 as cL,
  PL as cM,
  c5 as cN,
  On as cO,
  o5 as cP,
  t5 as cQ,
  Q0 as cR,
  sD as cS,
  QL as cT,
  oo as cU,
  lx as cV,
  MD as cW,
  p5 as cX,
  MM as ca,
  UM as cb,
  WD as cc,
  _O as cd,
  WN as ce,
  eN as cf,
  gM as cg,
  AN as ch,
  IN as ci,
  LN as cj,
  DN as ck,
  oB as cl,
  hB as cm,
  XB as cn,
  YB as co,
  qB as cp,
  PB as cq,
  nP as cr,
  iP as cs,
  AB as ct,
  QB as cu,
  vB as cv,
  KB as cw,
  JB as cx,
  BB as cy,
  sB as cz,
  g5 as d,
  fb as e,
  Jh as f,
  n5 as g,
  i5 as h,
  s5 as i,
  Qt as j,
  II as k,
  Hf as l,
  fn as m,
  uD as n,
  ob as o,
  d5 as p,
  un as q,
  Df as r,
  FI as s,
  Yi as t,
  GI as u,
  u5 as v,
  r5 as w,
  h5 as x,
  Gy as y,
  Ct as z
};
