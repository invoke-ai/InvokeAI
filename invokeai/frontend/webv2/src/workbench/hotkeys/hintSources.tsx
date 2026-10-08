import type { ReactNode } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { createContext, use, useCallback, useSyncExternalStore } from 'react';

/** Commands use their registered bindings; gestures belong to the surface that implements them. */
export type ShortcutHint =
  | { commandId: string; labelKey?: string }
  | { labelKey: string; parts: readonly string[]; pointerKey?: string };

export interface ShortcutHintSnapshot {
  titleKey: string;
  hints: readonly ShortcutHint[];
}

export interface ShortcutHintSource {
  projectId: string;
  instanceId: string;
  getSnapshot(): ShortcutHintSnapshot;
  subscribe(listener: () => void): () => void;
}

export const createShortcutHintSources = () => {
  const sources = new Map<string, ShortcutHintSource>();
  const listeners = new Set<() => void>();
  const notify = () => listeners.forEach((listener) => listener());
  // The page itself owns nothing a guide could describe or return focus to; popovers fall back to their trigger.
  const toFocusOwner = (element: Element | null): Element | null =>
    element === document.body || element === document.documentElement ? null : element;
  let focusElement: Element | null = typeof document === 'undefined' ? null : toFocusOwner(document.activeElement);
  let clearedElement: Element | null = null;
  const focusListeners = new Set<() => void>();
  const updateFocus = (event?: FocusEvent) => {
    const active = document.activeElement;
    if (!event && active === clearedElement) {
      return;
    }
    if (!active?.closest('[data-workbench-focus-preserve]')) {
      focusElement = toFocusOwner(active);
      focusListeners.forEach((listener) => listener());
    }
  };
  // Leaving for the body (a background click, blur()) fires focusout without a focusin; the element that will own
  // focus is only known once the focus update settles, while focus moving to another element reports it via focusin.
  const settleFocus = (event: FocusEvent) => {
    if (!event.relatedTarget) {
      queueMicrotask(() => updateFocus(event));
    }
  };
  return {
    focus: {
      clear: () => {
        clearedElement = typeof document === 'undefined' ? null : document.activeElement;
        focusElement = null;
        focusListeners.forEach((listener) => listener());
      },
      getSnapshot: () => (focusElement?.isConnected ? focusElement : null),
      subscribe: (listener: () => void): (() => void) => {
        if (focusListeners.size === 0) {
          document.addEventListener('focusin', updateFocus);
          document.addEventListener('focusout', settleFocus);
          updateFocus();
        }
        focusListeners.add(listener);
        return () => {
          focusListeners.delete(listener);
          if (focusListeners.size === 0) {
            document.removeEventListener('focusin', updateFocus);
            document.removeEventListener('focusout', settleFocus);
          }
        };
      },
    },
    get: (projectId: string, instanceId: string): ShortcutHintSource | null => {
      const source = sources.get(instanceId);
      return source?.projectId === projectId ? source : null;
    },
    register: (source: ShortcutHintSource): (() => void) => {
      sources.set(source.instanceId, source);
      const unsubscribe = source.subscribe(notify);
      notify();
      return () => {
        unsubscribe();
        if (sources.get(source.instanceId) === source) {
          sources.delete(source.instanceId);
          notify();
        }
      };
    },
    subscribe: (listener: () => void): (() => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const HintSourcesContext = createContext<ReturnType<typeof createShortcutHintSources> | null>(null);

export const ShortcutHintSourcesProvider = ({
  children,
  sources,
}: {
  children: ReactNode;
  sources: ReturnType<typeof createShortcutHintSources>;
}) => {
  return <HintSourcesContext value={sources}>{children}</HintSourcesContext>;
};

const getRegistrationSnapshot = () => null;
const subscribeToNothing = () => () => {};

/** Source identity belongs to the caller's keyed owner lifetime, not changing render state. */
export const useRegisterShortcutHintSource = (source: ShortcutHintSource | null): void => {
  const sources = use(HintSourcesContext);
  useMountEffect(() => (source && sources ? sources.register(source) : undefined));
};

export const useShortcutHintSource = (projectId: string, instanceId: string | null): ShortcutHintSnapshot | null => {
  const sources = use(HintSourcesContext);
  const getSnapshot = useCallback(
    () => (instanceId ? (sources?.get(projectId, instanceId)?.getSnapshot() ?? null) : null),
    [instanceId, projectId, sources]
  );
  return useSyncExternalStore(sources?.subscribe ?? subscribeToNothing, getSnapshot, getSnapshot);
};

/** Shared by compact and expanded presentations so opening the helper preserves its editing context. */
export const useShortcutFocusElement = (): Element | null => {
  const focus = use(HintSourcesContext)?.focus;
  return useSyncExternalStore(focus?.subscribe ?? subscribeToNothing, focus?.getSnapshot ?? getRegistrationSnapshot);
};

export const useShortcutFocusRestore = (): (() => HTMLElement | null) => {
  const focus = use(HintSourcesContext)?.focus;
  return useCallback(() => {
    const element = focus?.getSnapshot();
    return element instanceof HTMLElement ? element : null;
  }, [focus]);
};
