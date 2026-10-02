import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceId } from '@workbench/widgetContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { FocusEvent, PointerEvent, ReactNode } from 'react';

import { captureAccountScope, isAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { createContext, use, useCallback, useState, useSyncExternalStore } from 'react';

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
  /** Record what the user focused. Returns false for an event from a project that is no longer on screen. */
  activate(target: WorkbenchFocusTarget, projectId?: string): boolean;
  /** Forget the target and abandon any focus move still waiting for its widget to show. */
  clear(): void;
  /** Move keyboard focus into a floating window once it shows; focus arriving there activates it. */
  focusFloating(instanceId: WidgetInstanceId): void;
  /** Move keyboard focus into the region a control just opened a widget in, once that widget shows. */
  focusRegion(region: WidgetRegion, typeId: string): void;
  /** The focus target, or null once the project or account it was set under is no longer the current one. */
  getTarget(): WorkbenchFocusTarget | null;
  subscribe(listener: () => void): () => void;
}

/** A lazy widget mounts within a few frames of opening; stop looking after about half a second. */
const FOCUS_MOVE_FRAME_BUDGET = 30;
/** Frames to keep the move after it lands: a closing menu or dialog hands focus back to its trigger on the way out. */
const FOCUS_MOVE_SETTLE_FRAMES = 30;

const isSameTarget = (left: WorkbenchFocusTarget | null, right: WorkbenchFocusTarget): boolean =>
  left !== null &&
  (left.kind === 'region'
    ? right.kind === 'region' && left.region === right.region
    : right.kind === 'floating' && left.instanceId === right.instanceId);

export const createWorkbenchFocusController = (getProjectId: () => string): WorkbenchFocusController => {
  let entry: { owner: AccountScope; projectId: string; target: WorkbenchFocusTarget } | null = null;
  // Only the latest move runs: a control that opens two widgets means the second one.
  let focusMove = 0;
  const listeners = new Set<() => void>();
  const notify = () => {
    for (const listener of listeners) {
      listener();
    }
  };
  const getTarget = (): WorkbenchFocusTarget | null =>
    entry && entry.projectId === getProjectId() && isAccountScopeCurrent(entry.owner) ? entry.target : null;

  /**
   * Leaves focus alone when it is already inside the container, and gives up if the container never shows or the
   * project or account changes first.
   */
  const moveFocus = (findContainer: () => HTMLElement | null): void => {
    // The control that asked, where a closing menu or dialog would put focus back.
    const opener = document.activeElement;
    const move = ++focusMove;
    const projectId = getProjectId();
    const owner = captureAccountScope();
    const isCurrent = () => move === focusMove && projectId === getProjectId() && isAccountScopeCurrent(owner);
    let frames = 0;

    const settle = (container: HTMLElement, remaining: number) => {
      if (!isCurrent() || !container.isConnected) {
        return;
      }

      const active = document.activeElement;

      if (!container.contains(active) && (active === opener || active === document.body || active === null)) {
        container.focus({ preventScroll: true });
      }
      if (remaining > 0) {
        requestAnimationFrame(() => settle(container, remaining - 1));
      }
    };

    const attempt = () => {
      if (!isCurrent()) {
        return;
      }

      const container = findContainer();

      if (container) {
        if (!container.contains(document.activeElement)) {
          if (!container.hasAttribute('tabindex')) {
            container.tabIndex = -1;
          }
          settle(container, FOCUS_MOVE_SETTLE_FRAMES);
        }
        return;
      }

      frames += 1;
      if (frames < FOCUS_MOVE_FRAME_BUDGET) {
        requestAnimationFrame(attempt);
      }
    };

    requestAnimationFrame(attempt);
  };

  return {
    activate: (target, projectId = getProjectId()) => {
      if (projectId !== getProjectId()) {
        return false;
      }
      if (!isSameTarget(getTarget(), target)) {
        entry = { owner: captureAccountScope(), projectId, target };
        notify();
      }
      return true;
    },
    clear: () => {
      focusMove += 1;
      if (entry) {
        entry = null;
        notify();
      }
    },
    focusFloating: (instanceId) =>
      moveFocus(() => document.querySelector<HTMLElement>(`[data-floating-window="${CSS.escape(instanceId)}"]`)),
    focusRegion: (region, typeId) =>
      moveFocus(() => {
        // A side region keeps the panels it showed before mounted but hidden, each in its own region frame; only
        // the frame on screen can take focus.
        for (const container of document.querySelectorAll<HTMLElement>(`[data-focus-region="${region}"]`)) {
          if (
            container.getClientRects().length > 0 &&
            container.querySelector(`[data-hotkey-widget-type-id="${CSS.escape(typeId)}"]`)
          ) {
            return container;
          }
        }

        return null;
      }),
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

/**
 * Provides workbench focus to everything below it. The workbench passes the controller it owns (see
 * `WorkbenchFocusProvider`); without one the provider keeps its own, which is all an isolated subtree needs.
 */
export const FocusRegionProvider = ({
  children,
  controller,
}: {
  children: ReactNode;
  controller?: WorkbenchFocusController;
}) => {
  const [ownController] = useState(() => controller ?? createWorkbenchFocusController(() => ''));

  return <FocusRegionContext value={controller ?? ownController}>{children}</FocusRegionContext>;
};

const NO_FOCUS_TARGET = (): null => null;
const subscribeToNothing = (): (() => void) => () => {};

const useFocusTarget = (): WorkbenchFocusTarget | null => {
  const controller = use(FocusRegionContext);

  return useSyncExternalStore(
    controller?.subscribe ?? subscribeToNothing,
    controller?.getTarget ?? NO_FOCUS_TARGET,
    controller?.getTarget ?? NO_FOCUS_TARGET
  );
};

const useShowsFocusHighlight = (): boolean =>
  useWorkbenchPreferenceSelector((preferences) => preferences.showFocusRegionHighlight);

const NO_FOCUS_MOVES: Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion' | 'getTarget'> = {
  focusFloating: () => {},
  focusRegion: () => {},
  getTarget: NO_FOCUS_TARGET,
};

/**
 * Focus moves and the current target, for controls that open, float, or dock a widget and for the hotkey runtime.
 * Outside a provider there is nothing to move focus between, so the moves do nothing.
 */
export const useWorkbenchFocus = (): Pick<WorkbenchFocusController, 'focusFloating' | 'focusRegion' | 'getTarget'> =>
  use(FocusRegionContext) ?? NO_FOCUS_MOVES;

/**
 * The region whose outline is showing, if any. Borders the outline is drawn over hide while it shows, so a shared
 * edge draws one line at any display scale. No region is outlined while a floating window holds focus.
 */
export const useHighlightedRegion = (): WidgetRegion | null => {
  const target = useFocusTarget();
  const showsHighlight = useShowsFocusHighlight();

  return showsHighlight && target?.kind === 'region' ? target.region : null;
};

export const useFocusRegionProps = (region: WidgetRegion) => {
  const controller = use(FocusRegionContext);

  if (!controller) {
    throw new Error('useFocusRegionProps must be used within a FocusRegionProvider.');
  }

  const isHighlighted = useHighlightedRegion() === region;
  const activate = () => controller.activate({ kind: 'region', region });

  return {
    css: HIGHLIGHT_STYLES[region],
    'data-focus-region': region,
    'data-highlighted': isHighlighted,
    onFocusCapture: (_event: FocusEvent<HTMLElement>) => activate(),
    onPointerDownCapture: (_event: PointerEvent<HTMLElement>) => activate(),
    position: 'relative' as const,
  };
};

/**
 * Focus for one floating window. Pointer-down and keyboard focus activate it — hover does not — and `activate`
 * reports whether the window still belongs to the project on screen, so a late event cannot raise a stale one.
 */
export const useFloatingWindowFocus = (
  instanceId: WidgetInstanceId,
  projectId: string
): { activate: () => boolean; isActive: boolean; isHighlighted: boolean } => {
  const controller = use(FocusRegionContext);
  const target = useFocusTarget();
  const showsHighlight = useShowsFocusHighlight();
  const isActive = target?.kind === 'floating' && target.instanceId === instanceId;
  const activate = useCallback(
    () => controller?.activate({ instanceId, kind: 'floating' }, projectId) ?? true,
    [controller, instanceId, projectId]
  );

  return {
    activate,
    isActive,
    isHighlighted: isActive && showsHighlight,
  };
};
