import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FocusEvent, PointerEvent, ReactNode } from 'react';

import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { getModalOpenGeneration } from '@platform/ui/modalPresence';
import { createContext, use, useCallback, useSyncExternalStore } from 'react';

import { useWorkbenchPreferenceSelector } from './settings/store';

export type WorkbenchFocusTarget =
  | { kind: 'region'; region: WidgetRegion }
  | { instanceId: WidgetInstanceId; kind: 'floating' };

type FocusDirection = 'left' | 'right' | 'up' | 'down';

/** Transient and never persisted; fenced to the project and account it was set under. The center is the default. */
export interface WorkbenchFocusController {
  /** A pointer press anywhere but a pending focus move's destination abandons that move. */
  activate(target: WorkbenchFocusTarget, options?: { byPointer?: boolean; projectId?: string }): FocusActivation;
  clear(): void;
  focusAdjacentRegion(direction: FocusDirection): void;
  focusFloating(instanceId: WidgetInstanceId): void;
  /** An instance id waits for that widget's visible frame, even while a hidden instance of its type is mounted. */
  focusRegion(region: WidgetRegion, typeId?: string, instanceId?: WidgetInstanceId, ready?: Promise<unknown>): void;
  /** Forgetting also keeps a re-floated instance (preset, undo) from taking focus back unasked. */
  forgetClosedWindow(): void;
  /** Never null inside the provider; the nullable type is shared with readers outside it. */
  getTarget(): WorkbenchFocusTarget | null;
  subscribe(listener: () => void): () => void;
}

export type FocusActivation = 'activated' | 'refused' | 'unchanged';

export interface WorkbenchFocusScope {
  getProjectId(): string;
  isFloating(instanceId: WidgetInstanceId): boolean;
}

/** How long a focus move waits for its widget to show: a lazy widget, or the window chunk, has to load first. */
const FOCUS_MOVE_WAIT_MS = 1000;
/** How long the move holds after it lands: a closing menu or dialog hands focus back to its trigger on the way out. */
const FOCUS_MOVE_SETTLE_MS = 500;

// One object, so external-store snapshots of the default stay stable between reads.
const CENTER_TARGET: WorkbenchFocusTarget = Object.freeze({ kind: 'region', region: 'center' });

const isSameTarget = (left: WorkbenchFocusTarget | null, right: WorkbenchFocusTarget): boolean =>
  left !== null &&
  (left.kind === 'region'
    ? right.kind === 'region' && left.region === right.region
    : right.kind === 'floating' && left.instanceId === right.instanceId);

const visibleRegionContainers = (): HTMLElement[] =>
  [...document.querySelectorAll<HTMLElement>('[data-focus-region]')].filter(
    (container) => container.getClientRects().length > 0 && getComputedStyle(container).visibility === 'visible'
  );

export const createWorkbenchFocusController = ({
  getProjectId,
  isFloating,
}: WorkbenchFocusScope): WorkbenchFocusController => {
  let entry: { owner: AccountScope; projectId: string; target: WorkbenchFocusTarget } | null = null;
  // Only the latest move runs: a control that opens two widgets means the second one.
  let focusMove = 0;
  let moveTarget: WorkbenchFocusTarget | null = null;
  const listeners = new Set<() => void>();
  const notify = () => {
    for (const listener of listeners) {
      listener();
    }
  };
  // The one place a target is judged; covers the instant between a project/account change and the owner's clear.
  const getTarget = (): WorkbenchFocusTarget =>
    entry &&
    entry.projectId === getProjectId() &&
    isAccountScopeCurrent(entry.owner) &&
    (entry.target.kind === 'region' || isFloating(entry.target.instanceId))
      ? entry.target
      : CENTER_TARGET;

  /** Focus that starts inside is still watched: the opener can sit in the view the open replaces. */
  const moveFocus = (
    target: WorkbenchFocusTarget,
    findContainer: () => HTMLElement | null,
    ready?: Promise<unknown>
  ): void => {
    const opener = document.activeElement;
    const move = ++focusMove;
    const projectId = getProjectId();
    const owner = captureAccountScope();
    const modalGeneration = getModalOpenGeneration();
    const isCurrent = () =>
      move === focusMove &&
      modalGeneration === getModalOpenGeneration() &&
      projectId === getProjectId() &&
      isAccountScopeCurrent(owner);
    let deadline: number;

    moveTarget = target;

    const settle = (container: HTMLElement, until: number) => {
      if (!isCurrent() || !container.isConnected) {
        return;
      }

      const active = document.activeElement;

      if (!container.contains(active) && (active === opener || active === document.body || active === null)) {
        container.focus({ preventScroll: true });
      }
      if (performance.now() < until) {
        requestAnimationFrame(() => settle(container, until));
      }
    };

    const attempt = () => {
      if (!isCurrent()) {
        return;
      }

      const container = findContainer();

      if (container) {
        if (!container.hasAttribute('tabindex')) {
          container.tabIndex = -1;
        }
        settle(container, performance.now() + FOCUS_MOVE_SETTLE_MS);
        return;
      }

      if (performance.now() < deadline) {
        requestAnimationFrame(attempt);
      }
    };

    const start = () => {
      if (!isCurrent()) {
        return;
      }
      deadline = performance.now() + FOCUS_MOVE_WAIT_MS;
      requestAnimationFrame(attempt);
    };
    if (ready) {
      void ready.then(start, start);
    } else {
      start();
    }
  };

  const focusRegion: WorkbenchFocusController['focusRegion'] = (region, typeId, instanceId, ready) =>
    moveFocus(
      { kind: 'region', region },
      () => {
        // Side frames are replaced when their lazy view loads; hidden kept-alive instances may still be mounted.
        for (const container of visibleRegionContainers()) {
          if (
            container.dataset.focusRegion !== region ||
            (region !== 'center' && container.querySelector('[data-widget-loading]'))
          ) {
            continue;
          }
          if (instanceId !== undefined) {
            const selector = `[data-hotkey-widget-instance-id="${CSS.escape(instanceId)}"]`;
            if (
              [container, ...container.querySelectorAll<HTMLElement>(selector)].some(
                (element) => element.matches(selector) && element.getClientRects().length > 0
              )
            ) {
              return container;
            }
            continue;
          }
          if (typeId === undefined || container.querySelector(`[data-hotkey-widget-type-id="${CSS.escape(typeId)}"]`)) {
            return container;
          }
        }

        return null;
      },
      ready
    );

  return {
    activate: (target, { byPointer = false, projectId = getProjectId() } = {}) => {
      if (projectId !== getProjectId()) {
        return 'refused';
      }
      if (byPointer && !isSameTarget(moveTarget, target)) {
        focusMove += 1;
        moveTarget = null;
      }
      if (isSameTarget(getTarget(), target)) {
        return 'unchanged';
      }

      entry = { owner: captureAccountScope(), projectId, target };
      notify();

      return 'activated';
    },
    clear: () => {
      focusMove += 1;
      moveTarget = null;
      if (entry) {
        entry = null;
        notify();
      }
    },
    focusAdjacentRegion: (direction) => {
      const regions = visibleRegionContainers().map((container) => ({
        container,
        rect: container.getBoundingClientRect(),
      }));
      const target = getTarget();
      const origin =
        target.kind === 'floating'
          ? document.querySelector<HTMLElement>(`[data-floating-window="${CSS.escape(target.instanceId)}"]`)
          : regions.find(({ container }) => container.dataset.focusRegion === target.region)?.container;

      if (!origin) {
        const center = regions.find(({ container }) => container.dataset.focusRegion === 'center');
        if (center) {
          focusRegion('center');
        }
        return;
      }

      const originRect = origin.getBoundingClientRect();
      const horizontal = direction === 'left' || direction === 'right';
      const forward = direction === 'right' || direction === 'down';
      const start = horizontal ? 'left' : 'top';
      const end = horizontal ? 'right' : 'bottom';
      const crossStart = horizontal ? 'top' : 'left';
      const crossEnd = horizontal ? 'bottom' : 'right';
      // A window can overlap several docks: use its center on the movement axis.
      const originStart = target.kind === 'floating' ? (originRect[start] + originRect[end]) / 2 : originRect[start];
      const originEnd = target.kind === 'floating' ? originStart : originRect[end];
      const candidates = regions.flatMap(({ container, rect }) => {
        const gap = forward ? rect[start] - originEnd : originStart - rect[end];
        if (container === origin || gap < -1) {
          return [];
        }
        const aligned =
          Math.min(rect[crossEnd], originRect[crossEnd]) > Math.max(rect[crossStart], originRect[crossStart]);
        const offset = Math.abs(rect[crossStart] + rect[crossEnd] - originRect[crossStart] - originRect[crossEnd]) / 2;
        return [{ aligned, container, gap, offset }];
      });
      candidates.sort(
        (a, b) =>
          Number(b.aligned) - Number(a.aligned) ||
          (a.aligned ? a.gap - b.gap || a.offset - b.offset : a.gap + 2 * a.offset - b.gap - 2 * b.offset)
      );
      // With no dock beyond the window, return to the dock under its center rather than trapping focus.
      const next =
        candidates[0]?.container ??
        (target.kind === 'floating'
          ? regions.find(
              ({ rect }) =>
                rect.left <= originRect.left + originRect.width / 2 &&
                rect.right >= originRect.left + originRect.width / 2 &&
                rect.top <= originRect.top + originRect.height / 2 &&
                rect.bottom >= originRect.top + originRect.height / 2
            )?.container
          : undefined);
      if (next) {
        focusRegion(next.dataset.focusRegion as WidgetRegion);
      }
    },
    focusFloating: (instanceId) =>
      moveFocus({ instanceId, kind: 'floating' }, () =>
        document.querySelector<HTMLElement>(`[data-floating-window="${CSS.escape(instanceId)}"]`)
      ),
    focusRegion,
    forgetClosedWindow: () => {
      if (entry?.target.kind === 'floating' && !isFloating(entry.target.instanceId)) {
        entry = null;
        notify();
      }
    },
    getTarget,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const FocusRegionContext = createContext<WorkbenchFocusController | null>(null);

const highlightBase = {
  borderRadius: 'md',
  content: '""',
  opacity: 0,
  pointerEvents: 'none',
  position: 'absolute',
  // Above resize handles, so the outline runs through the middle of their grips.
  zIndex: 4,
} as const;

const fade = (property: string) =>
  `${property} var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease`;

/** Each stroke paints its 1px from one snapped edge; the side it grows toward picks the anchoring CSS edge. */
const STROKES = {
  inside: {
    hidden: { border: '1px solid', borderColor: 'transparent', transition: fade('border-color') },
    shown: { borderColor: 'accent.solid' },
  },
  outside: {
    hidden: { outline: '1px solid', outlineColor: 'transparent', transition: fade('outline-color') },
    shown: { outlineColor: 'accent.solid' },
  },
} as const;

const regionHighlight = (inset: string, stroke: keyof typeof STROKES) => ({
  '&[data-highlighted="true"]::after': { ...STROKES[stroke].shown, opacity: 1 },
  // The region outline stands in for the browser's frame focus indicator.
  '&[data-highlighted="true"]:focus-visible, &[data-highlighted="true"] [data-hotkey-widget-instance-id]:focus-visible':
    {
      outline: 'none',
    },
  '&::after': { ...highlightBase, ...STROKES[stroke].hidden, inset },
});

// Neighbours draw their shared edge on the same 1px divider column, anchored to one CSS edge, so fractional zoom
// cannot snap them onto different device pixels.
const HIGHLIGHT_STYLES = {
  bottom: regionHighlight('0 0 -1px 0', 'inside'),
  center: regionHighlight('0 -1px', 'inside'),
  left: regionHighlight('1px 1px 1px 0', 'outside'),
  right: regionHighlight('1px 0 1px 1px', 'outside'),
} satisfies Record<WidgetRegion, unknown>;

export const FocusRegionProvider = ({
  children,
  controller,
}: {
  children: ReactNode;
  controller: WorkbenchFocusController;
}) => <FocusRegionContext value={controller}>{children}</FocusRegionContext>;

const subscribeToNothing = (): (() => void) => () => {};

/** `select` returns a primitive so a component re-renders only when its own answer changes. */
const useFocusSelector = <Selected extends string | boolean | null>(
  select: (target: WorkbenchFocusTarget | null) => Selected
): Selected => {
  const controller = use(FocusRegionContext);
  const getSnapshot = () => select(controller?.getTarget() ?? null);

  return useSyncExternalStore(controller?.subscribe ?? subscribeToNothing, getSnapshot, getSnapshot);
};

const useShowsFocusHighlight = (): boolean =>
  useWorkbenchPreferenceSelector((preferences) => preferences.showFocusRegionHighlight);

type WorkbenchFocusMoves = Pick<WorkbenchFocusController, 'focusAdjacentRegion' | 'focusFloating' | 'focusRegion'>;

const NO_FOCUS_MOVES: WorkbenchFocusMoves = {
  focusAdjacentRegion: () => {},
  focusFloating: () => {},
  focusRegion: () => {},
};

/** Outside the workbench shell the moves do nothing. */
export const useWorkbenchFocus = (): WorkbenchFocusMoves => use(FocusRegionContext) ?? NO_FOCUS_MOVES;

export const useWorkbenchFocusTarget = (): (() => WorkbenchFocusTarget | null) => {
  const controller = use(FocusRegionContext);

  if (!controller) {
    throw new Error('useWorkbenchFocusTarget must be used within a FocusRegionProvider.');
  }

  return controller.getTarget;
};

export const useCurrentWorkbenchFocusTarget = (): WorkbenchFocusTarget | null => {
  const controller = use(FocusRegionContext);
  return useSyncExternalStore(controller?.subscribe ?? subscribeToNothing, () => controller?.getTarget() ?? null);
};

/** Borders under the outline hide while it shows, so a shared edge draws one line at any display scale. */
export const useHighlightedRegion = (): WidgetRegion | null => {
  const region = useFocusSelector((target) => (target?.kind === 'region' ? target.region : null));
  const showsHighlight = useShowsFocusHighlight();

  return showsHighlight ? region : null;
};

export const useFocusRegionProps = (region: WidgetRegion) => {
  const controller = use(FocusRegionContext);

  if (!controller) {
    throw new Error('useFocusRegionProps must be used within a FocusRegionProvider.');
  }

  const isHighlighted = useHighlightedRegion() === region;

  return {
    css: HIGHLIGHT_STYLES[region],
    'data-focus-region': region,
    'data-highlighted': isHighlighted,
    onFocusCapture: (event: FocusEvent<HTMLElement>) => {
      if (event.target.closest('[data-workbench-focus-preserve]')) {
        return;
      }
      controller.activate({ kind: 'region', region });
    },
    onPointerDownCapture: (event: PointerEvent<HTMLElement>) => {
      if (event.target instanceof Element && event.target.closest('[data-workbench-focus-preserve]')) {
        return;
      }
      const container = event.currentTarget;

      controller.activate({ kind: 'region', region }, { byPointer: true });
      // A press that takes no focus of its own would leave the keys in a floating window; pull focus out of it.
      if (
        document.activeElement?.closest('[data-floating-window]') &&
        event.target instanceof Node &&
        container.contains(event.target)
      ) {
        if (!container.hasAttribute('tabindex')) {
          container.tabIndex = -1;
        }
        container.focus({ preventScroll: true });
      }
    },
    position: 'relative' as const,
  };
};

/** Hover does not activate; a window whose project left the screen is refused, so a late event cannot raise it. */
export const useFloatingWindowFocus = (
  instanceId: WidgetInstanceId,
  projectId: string
): { activate: (options?: { byPointer?: boolean }) => FocusActivation; isHighlighted: boolean } => {
  const controller = use(FocusRegionContext);
  const isActive = useFocusSelector((target) => target?.kind === 'floating' && target.instanceId === instanceId);
  const showsHighlight = useShowsFocusHighlight();
  const activate = useCallback(
    ({ byPointer }: { byPointer?: boolean } = {}) =>
      controller?.activate({ instanceId, kind: 'floating' }, { byPointer, projectId }) ?? 'activated',
    [controller, instanceId, projectId]
  );

  return { activate, isHighlighted: isActive && showsHighlight };
};

/**
 * Gives focus to `getTarget()` when a focused node leaves the document. React also runs ref cleanups on a node that
 * stays, so removal is checked a microtask later.
 */
export const releaseFocusOnRemoval = (
  node: HTMLElement,
  getTarget: () => HTMLElement | null = () => node.closest<HTMLElement>('[role="region"]')
): (() => void) => {
  return () => {
    if (document.activeElement !== node) {
      return;
    }

    // Resolved while the node is still in place: a region found through it cannot be found once it is gone.
    const target = getTarget();

    queueMicrotask(() => {
      const { activeElement } = document;

      if (!node.isConnected && (activeElement === null || activeElement === document.body)) {
        target?.focus({ preventScroll: true });
      }
    });
  };
};
