import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FocusEvent, PointerEvent, ReactNode } from 'react';

import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { createContext, use, useCallback, useSyncExternalStore } from 'react';

import { useWorkbenchPreferenceSelector } from './settings/store';

/** What holds workbench focus: a docked region, or one floating window. */
export type WorkbenchFocusTarget =
  | { kind: 'region'; region: WidgetRegion }
  | { instanceId: WidgetInstanceId; kind: 'floating' };

type FocusDirection = 'left' | 'right' | 'up' | 'down';

/**
 * Transient workbench focus, owned by one provider above the shell and the hotkey runtime. It is never persisted:
 * the outline, the hotkey target, and focus moves all read it, and it is fenced to the project and account it was
 * set under.
 */
export interface WorkbenchFocusController {
  /**
   * Record what the user focused: `'activated'` when the target changed, `'unchanged'` when it already held
   * focus, `'refused'` for an event from a project that is no longer on screen. A pointer press on anything but a
   * pending focus move's destination also abandons that move: the user went somewhere else on purpose.
   */
  activate(target: WorkbenchFocusTarget, options?: { byPointer?: boolean; projectId?: string }): FocusActivation;
  /** Forget the target and abandon any focus move still waiting for its widget to show. */
  clear(): void;
  /** Focus the nearest visible docked region in this direction, without changing its selected widget. */
  focusAdjacentRegion(direction: FocusDirection): void;
  /** Move keyboard focus into a floating window once it shows; focus arriving there activates it. */
  focusFloating(instanceId: WidgetInstanceId): void;
  /**
   * Move keyboard focus into the region a control just opened a widget in, once that widget shows. Without a
   * widget, focus moves into the region as it stands. An instance id waits for that widget's visible frame,
   * including when another instance of the same type is still mounted but hidden.
   */
  focusRegion(region: WidgetRegion, typeId?: string, instanceId?: WidgetInstanceId): void;
  /**
   * Forget a window target whose window has docked or closed. It already reads as null; forgetting it keeps the
   * same instance from taking focus back without being focused, should a preset or undo float it again.
   */
  forgetClosedWindow(): void;
  /** The focus target, or null once it no longer describes something on screen in this project and account. */
  getTarget(): WorkbenchFocusTarget | null;
  subscribe(listener: () => void): () => void;
}

export type FocusActivation = 'activated' | 'refused' | 'unchanged';

/** What the controller has to know about the workbench to keep its target honest. */
export interface WorkbenchFocusScope {
  getProjectId(): string;
  /** Whether the instance floats in the project on screen; a window that docked or closed holds no focus. */
  isFloating(instanceId: WidgetInstanceId): boolean;
}

/** How long a focus move waits for its widget to show: a lazy widget, or the window chunk, has to load first. */
const FOCUS_MOVE_WAIT_MS = 1000;
/** How long the move holds after it lands: a closing menu or dialog hands focus back to its trigger on the way out. */
const FOCUS_MOVE_SETTLE_MS = 500;

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
  // The one place a target is judged. The owner clears eagerly on project and account changes, which is what
  // notifies subscribers and stops moves, and forgets a closed window on the next workbench change; this read-time
  // check covers the instant between such a change and the owner's response.
  const getTarget = (): WorkbenchFocusTarget | null =>
    entry &&
    entry.projectId === getProjectId() &&
    isAccountScopeCurrent(entry.owner) &&
    (entry.target.kind === 'region' || isFloating(entry.target.instanceId))
      ? entry.target
      : null;

  /**
   * Leaves focus alone while it is inside the container, and gives up if the container never shows, the project or
   * account changes first, or the user presses somewhere else. Focus that starts inside is still watched: the opener
   * can sit in the view the open replaces, which drops its focus a frame later.
   */
  const moveFocus = (target: WorkbenchFocusTarget, findContainer: () => HTMLElement | null): void => {
    // The control that asked, where a closing menu or dialog would put focus back.
    const opener = document.activeElement;
    const move = ++focusMove;
    const projectId = getProjectId();
    const owner = captureAccountScope();
    const isCurrent = () => move === focusMove && projectId === getProjectId() && isAccountScopeCurrent(owner);
    const deadline = performance.now() + FOCUS_MOVE_WAIT_MS;

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

    requestAnimationFrame(attempt);
  };

  const focusRegion: WorkbenchFocusController['focusRegion'] = (region, typeId, instanceId) =>
    moveFocus({ kind: 'region', region }, () => {
      // Side frames are replaced when their lazy view loads; center's owning frame stays mounted.
      // Kept-alive panels and instances of the same type may also still be mounted but hidden.
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
    });

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
      // Four docked regions at most; read their actual layout only on a navigation command.
      const regions = visibleRegionContainers().map((container) => ({
        container,
        rect: container.getBoundingClientRect(),
      }));
      const target = getTarget();
      const origin =
        target?.kind === 'floating'
          ? document.querySelector<HTMLElement>(`[data-floating-window="${CSS.escape(target.instanceId)}"]`)
          : regions.find(({ container }) => container.dataset.focusRegion === target?.region)?.container;

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
      // A window can overlap several docks. Use its center on the movement axis so its outer edges cannot
      // exclude all of them; docked regions still navigate from edge to edge.
      const originStart = target?.kind === 'floating' ? (originRect[start] + originRect[end]) / 2 : originRect[start];
      const originEnd = target?.kind === 'floating' ? originStart : originRect[end];
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
      // With no dock beyond the window in this direction (for example, all side panels closed), return to
      // the dock underneath its center instead of trapping keyboard focus in the window.
      const next =
        candidates[0]?.container ??
        (target?.kind === 'floating'
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
  border: '1px solid',
  borderColor: 'transparent',
  borderRadius: 'md',
  content: '""',
  opacity: 0,
  pointerEvents: 'none',
  position: 'absolute',
  transition: 'border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease',
  // Above resize handles, so the outline runs through the middle of their grips.
  zIndex: 4,
} as const;

const regionHighlight = (inset: string) => ({
  '&[data-highlighted="true"]::after': { borderColor: 'accent.solid', opacity: 1 },
  // The region outline represents frame focus. Controls keep their own rings, and disabling the highlight
  // restores the browser's frame indicator as well.
  '&[data-highlighted="true"]:focus-visible, &[data-highlighted="true"] [data-hotkey-widget-instance-id]:focus-visible':
    {
      outline: 'none',
    },
  '&::after': { ...highlightBase, inset },
});

// Sideways edges sit on the neighbouring divider or rail border rather than beside it. A panel's own divider is
// already its frame's edge; top and bottom stay inside because the workbench row clips vertical overflow.
const HIGHLIGHT_STYLES = {
  bottom: regionHighlight('0 0 -1px 0'),
  center: regionHighlight('0 -1px'),
  left: regionHighlight('0 0 0 -1px'),
  right: regionHighlight('0 -1px 0 0'),
} satisfies Record<WidgetRegion, unknown>;

/** Provides the workbench's focus controller (see `WorkbenchFocusProvider`) to everything below it. */
export const FocusRegionProvider = ({
  children,
  controller,
}: {
  children: ReactNode;
  controller: WorkbenchFocusController;
}) => <FocusRegionContext value={controller}>{children}</FocusRegionContext>;

const subscribeToNothing = (): (() => void) => () => {};

/**
 * Subscribe to one fact about the focus target. `select` returns a primitive, so a component re-renders only
 * when its own answer changes — not every window and region on every activation.
 */
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

/**
 * Focus moves for controls that open, float, or dock a widget. Some of those controls also render outside the
 * workbench shell, where there is nothing to move focus between and the moves do nothing.
 */
export const useWorkbenchFocus = (): WorkbenchFocusMoves => use(FocusRegionContext) ?? NO_FOCUS_MOVES;

/** Reads the focus target at call time, for the hotkey runtime. It has no meaning outside the provider. */
export const useWorkbenchFocusTarget = (): (() => WorkbenchFocusTarget | null) => {
  const controller = use(FocusRegionContext);

  if (!controller) {
    throw new Error('useWorkbenchFocusTarget must be used within a FocusRegionProvider.');
  }

  return controller.getTarget;
};

/** Live target for contextual guidance; unlike the highlight it does not depend on appearance preferences. */
export const useCurrentWorkbenchFocusTarget = (): WorkbenchFocusTarget | null => {
  const controller = use(FocusRegionContext);
  return useSyncExternalStore(controller?.subscribe ?? subscribeToNothing, () => controller?.getTarget() ?? null);
};

/**
 * The region whose outline is showing, if any. Borders the outline is drawn over hide while it shows, so a shared
 * edge draws one line at any display scale. No region is outlined while a floating window holds focus.
 */
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
      // A press that takes no focus of its own — a resize handle, the canvas — would leave the keys in a floating
      // window while this region shows as focused. Pull focus out of the window; a focusable target then takes
      // it as usual.
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

/**
 * Focus for one floating window. Pointer-down and keyboard focus activate it — hover does not — and `activate`
 * reports what happened: a window whose project has left the screen is refused, so a late event cannot raise a
 * stale one.
 */
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
