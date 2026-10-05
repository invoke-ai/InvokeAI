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
  /** Move keyboard focus into a floating window once it shows; focus arriving there activates it. */
  focusFloating(instanceId: WidgetInstanceId): void;
  /**
   * Move keyboard focus into the region a control just opened a widget in, once that widget shows. Without a
   * widget, focus moves into the region as it stands.
   */
  focusRegion(region: WidgetRegion, typeId?: string): void;
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
    focusFloating: (instanceId) =>
      moveFocus({ instanceId, kind: 'floating' }, () =>
        document.querySelector<HTMLElement>(`[data-floating-window="${CSS.escape(instanceId)}"]`)
      ),
    focusRegion: (region, typeId) =>
      moveFocus({ kind: 'region', region }, () => {
        // A side region keeps the panels it showed before mounted but hidden, each in its own region frame; only
        // the frame on screen can take focus.
        for (const container of document.querySelectorAll<HTMLElement>(`[data-focus-region="${region}"]`)) {
          if (
            container.getClientRects().length > 0 &&
            (typeId === undefined || container.querySelector(`[data-hotkey-widget-type-id="${CSS.escape(typeId)}"]`))
          ) {
            return container;
          }
        }

        return null;
      }),
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

type WorkbenchFocusMoves = Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion'>;

const NO_FOCUS_MOVES: WorkbenchFocusMoves = { focusFloating: () => {}, focusRegion: () => {} };

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
    onFocusCapture: (_event: FocusEvent<HTMLElement>) => {
      controller.activate({ kind: 'region', region });
    },
    onPointerDownCapture: (event: PointerEvent<HTMLElement>) => {
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
