import { Tooltip as ChakraTooltip, Portal } from '@chakra-ui/react';
import {
  cloneElement,
  createContext,
  isValidElement,
  use,
  useCallback,
  useId,
  useMemo,
  useReducer,
  useRef,
  type ComponentProps,
  type PointerEvent,
  type SyntheticEvent,
  type ReactElement,
  type ReactNode,
  type Ref,
  type RefObject,
} from 'react';

type TooltipTriggerProps = Omit<ComponentProps<typeof ChakraTooltip.Trigger>, 'children'>;

const ROOT_PROP_NAMES = new Set([
  'closeDelay',
  'closeOnClick',
  'closeOnEscape',
  'closeOnPointerDown',
  'closeOnScroll',
  'defaultOpen',
  'disabled',
  'ids',
  'immediate',
  'interactive',
  'lazyMount',
  'onOpenChange',
  'open',
  'openDelay',
  'positioning',
  'present',
  'unmountOnExit',
]);

const splitTooltipProps = (props: Record<string, unknown>) => {
  const rootProps: Record<string, unknown> = {};
  const triggerProps: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(props)) {
    if (ROOT_PROP_NAMES.has(key)) {
      rootProps[key] = value;
    } else {
      triggerProps[key] = value;
    }
  }

  return { rootProps, triggerProps };
};

const setRef = <T,>(ref: Ref<T> | undefined, value: T | null): void => {
  if (typeof ref === 'function') {
    ref(value);
    return;
  }

  if (ref) {
    ref.current = value;
  }
};

const composeRefs =
  <T,>(...refs: (Ref<T> | undefined)[]): Ref<T> =>
  (value) => {
    for (const ref of refs) {
      setRef(ref, value);
    }
  };

type TooltipChildProps = Record<string, unknown> & { ref?: Ref<HTMLButtonElement> };

type GatedEvent = 'onFocus' | 'onPointerMove' | 'onPointerOver';
type GatedHandlers = Partial<Record<GatedEvent, (event: SyntheticEvent) => void>>;

/**
 * Trigger handlers that run the trigger's own handlers, then hide the event from the tooltip when `isHidden` says so.
 * Zag's trigger handlers skip an event whose default is prevented, and run after the child's own (Ark merges the
 * child's handlers first). None of these events has a default action, and focus cannot be cancelled natively, so only
 * the tooltip notices.
 *
 * This is how a tip is kept shut, rather than Zag's `disabled` or a controlled `open` alone. While another tooltip is
 * showing, Zag opens a hovered trigger at once whatever the controlled prop says, and the tip then stays painted with
 * no prop change left to close it. `disabled` also hides the pointer leave, leaving Zag's record that the pointer
 * opened the tip standing, so the next hover after a re-enable shows nothing.
 */
const gateTriggerEvents = (
  own: readonly (GatedHandlers | undefined)[],
  isHidden: (event: SyntheticEvent) => boolean,
  events: readonly GatedEvent[]
): GatedHandlers => {
  const handlers: GatedHandlers = {};
  for (const name of events) {
    handlers[name] = (event) => {
      for (const props of own) {
        props?.[name]?.(event);
      }
      if (isHidden(event)) {
        event.preventDefault();
      }
    };
  }
  return handlers;
};

const POINTER_EVENTS = ['onPointerMove', 'onPointerOver'] as const;
const ARRIVAL_EVENTS = ['onFocus', 'onPointerMove', 'onPointerOver'] as const;
const alwaysHidden = () => true;

type TooltipPlacement = NonNullable<NonNullable<ChakraTooltip.RootProps['positioning']>['placement']>;

export interface TooltipProps extends ChakraTooltip.RootProps {
  showArrow?: boolean;
  portalled?: boolean;
  portalRef?: RefObject<HTMLElement | null>;
  content: ReactNode;
  contentRef?: Ref<HTMLDivElement>;
  contentProps?: ChakraTooltip.ContentProps;
  /** Shorthand for positioning.placement; an explicit positioning.placement wins. */
  placement?: TooltipPlacement;
  ref?: Ref<HTMLButtonElement>;
  triggerProps?: TooltipTriggerProps;
  /**
   * `disabled` normally unmounts the tooltip and leaves the bare trigger. This keeps it mounted and shut instead, for a
   * tooltip that toggles while its trigger is in use, such as one that keeps quiet while its menu is open: unmounting
   * would remount the trigger, dropping its focus or pointer capture. Use it, never a bare `open={false}`; see
   * `gateTriggerEvents`.
   */
  keepsRootWhenDisabled?: boolean;
}

/** Pass the same ids to Menu/Popover and Tooltip; competing trigger IDs leave the popup without an anchor. */
export const useTooltipTriggerIds = (): { trigger: string } => {
  const trigger = useId();

  return useMemo(() => ({ trigger }), [trigger]);
};

export const Tooltip = (props: TooltipProps) => {
  const {
    showArrow = true,
    children,
    disabled,
    keepsRootWhenDisabled = false,
    portalled = true,
    content,
    contentProps,
    contentRef,
    placement,
    portalRef,
    ref,
    triggerProps,
    ...rest
  } = props;
  const { rootProps, triggerProps: triggerPassthroughProps } = splitTooltipProps(rest);

  if (placement) {
    rootProps.positioning = { placement, ...(rootProps.positioning as object | undefined) };
  }
  const { ref: explicitTriggerRef, ...explicitTriggerProps } = triggerProps ?? {};
  const mergedTriggerProps = { ...triggerPassthroughProps, ...explicitTriggerProps };
  const hasTriggerPassthrough = ref || explicitTriggerRef || Object.keys(mergedTriggerProps).length > 0;
  const childRef = isValidElement<TooltipChildProps>(children) ? children.props.ref : undefined;
  // Stable while the refs are, so the child's ref is not detached and attached again on every commit.
  const composedRef = useMemo(
    // eslint-disable-next-line react/refs -- composes the refs, never reads them
    () => composeRefs(childRef, explicitTriggerRef, ref),
    [childRef, explicitTriggerRef, ref]
  );
  const isKeptShut = !!disabled && keepsRootWhenDisabled;
  const childProps = isValidElement<TooltipChildProps>(children) ? (children.props as GatedHandlers) : undefined;
  const triggerChild =
    (hasTriggerPassthrough || isKeptShut) && isValidElement<TooltipChildProps>(children)
      ? cloneElement(children as ReactElement<TooltipChildProps>, {
          ...mergedTriggerProps,
          ...(isKeptShut
            ? gateTriggerEvents([childProps, mergedTriggerProps as GatedHandlers], alwaysHidden, POINTER_EVENTS)
            : {}),
          ref: composedRef,
        })
      : children;

  if (disabled && !keepsRootWhenDisabled) {
    return triggerChild;
  }

  // Shut through the controlled prop, which closes a tip already showing; focus opens wait for it, and the gate above
  // keeps the pointer from opening around it.
  return (
    <ChakraTooltip.Root {...(rootProps as ChakraTooltip.RootProps)} {...(isKeptShut ? { open: false } : {})}>
      <ChakraTooltip.Trigger asChild>{triggerChild}</ChakraTooltip.Trigger>
      <Portal disabled={!portalled} container={portalRef}>
        <ChakraTooltip.Positioner>
          <ChakraTooltip.Content ref={contentRef} {...contentProps}>
            {showArrow && (
              <ChakraTooltip.Arrow>
                <ChakraTooltip.ArrowTip />
              </ChakraTooltip.Arrow>
            )}
            {content}
          </ChakraTooltip.Content>
        </ChakraTooltip.Positioner>
      </Portal>
    </ChakraTooltip.Root>
  );
};

export interface ClippedTooltipProps extends Omit<
  TooltipProps,
  'defaultOpen' | 'disabled' | 'keepsRootWhenDisabled' | 'onOpenChange' | 'open'
> {
  /** Whether the trigger hides some of the text the tooltip shows, such as a truncated name. */
  isClipped: () => boolean;
}

/**
 * A tooltip that shows the full text of a clipped label, and nothing while the label shows it all. Measured each time
 * the pointer or focus arrives on the trigger, so a resize costs no render; a label that does not hide anything keeps
 * the arrival from the tooltip (see `gateTriggerEvents`).
 */
export const ClippedTooltip = ({ children, isClipped, triggerProps, ...props }: ClippedTooltipProps) => {
  const isClippedRef = useRef(false);
  const childProps = isValidElement(children) ? (children.props as GatedHandlers) : undefined;
  const childOnFocus = childProps?.onFocus;
  const childOnPointerMove = childProps?.onPointerMove;
  const childOnPointerOver = childProps?.onPointerOver;
  // Pointer moves follow the pointerover that measured; focus and pointerover arrive, so they measure.
  const isHidden = useCallback(
    (event: SyntheticEvent) => {
      if (event.type !== 'pointermove') {
        isClippedRef.current = isClipped();
      }
      return !isClippedRef.current;
    },
    [isClipped]
  );
  const gatedTriggerProps = useMemo<TooltipTriggerProps>(() => {
    const own: GatedHandlers = {
      onFocus: childOnFocus,
      onPointerMove: childOnPointerMove,
      onPointerOver: childOnPointerOver,
    };

    return {
      ...triggerProps,
      // eslint-disable-next-line react/refs -- isHidden reads the ref only when an event arrives
      ...gateTriggerEvents([own, triggerProps as GatedHandlers | undefined], isHidden, ARRIVAL_EVENTS),
    } as TooltipTriggerProps;
  }, [childOnFocus, childOnPointerMove, childOnPointerOver, isHidden, triggerProps]);

  return (
    <Tooltip {...props} triggerProps={gatedTriggerProps}>
      {children}
    </Tooltip>
  );
};

interface TooltipGroupProps {
  children: ReactNode;
  closeOnScroll?: boolean;
  content: ReactNode;
  /** Each trigger keeps its own element id (a tab's, say); the group anchors to that id rather than replacing it. */
  getTriggerId: (value: string) => string;
  /**
   * Whether the tip shows for the trigger with this value. A trigger that is not enabled when focus or the pointer
   * arrives shows nothing until they arrive again, and one that stops being enabled closes the tip.
   */
  isEnabled: (value: string) => boolean;
  showArrow?: boolean;
}

interface TooltipGroupState {
  /** The trigger focus or the pointer is on, as the tooltip reports it. */
  value: string | null;
  /** The tooltip asked to show and has not asked to hide since. */
  isRequested: boolean;
}

type TooltipGroupAction = { type: 'trigger'; value: string | null } | { type: 'open' } | { type: 'close' };

const reduceTooltipGroup = (state: TooltipGroupState, action: TooltipGroupAction): TooltipGroupState => {
  switch (action.type) {
    case 'trigger':
      return state.value === action.value ? state : { ...state, value: action.value };
    case 'open':
      return state.isRequested ? state : { ...state, isRequested: true };
    case 'close':
      return state.isRequested ? { ...state, isRequested: false } : state;
  }
};

interface TooltipGroupContextValue {
  /** The trigger the shown tip describes, if any. */
  describedValue: string | null;
  isEnabled: (value: string) => boolean;
}

const TooltipGroupContext = createContext<TooltipGroupContextValue>({ describedValue: null, isEnabled: () => false });

/**
 * One tooltip shared by a group of triggers, such as the tabs of a strip, so moving focus or the pointer between them
 * hands the tip over. Separate tooltips close and open through deferred events, and a focus move made inside a key
 * handler (roving tab focus) can deliver the closing one's release after the arriving one opened, closing it again.
 *
 * The tip is controlled: the tooltip only asks to open and close, reporting the trigger and the open in an order that
 * depends on how it got there. A request on a trigger that is not enabled ends at the next render, and so does one
 * whose trigger stops being enabled: the tooltip then stays closed and ignores the blur or pointer leave that would
 * otherwise end the request, and a request left standing would open the tip later on a trigger nobody is on.
 *
 * Refusing works for every open but one: while another tooltip is showing, Zag opens a hovered trigger at once,
 * whatever the controlled prop says, and a refused open leaves the tip painted with no prop change to close it. So a
 * trigger that is not enabled keeps the pointer from Zag altogether (see TooltipGroupTrigger).
 */
const TooltipGroupRoot = ({
  children,
  closeOnScroll = true,
  content,
  getTriggerId,
  isEnabled,
  showArrow = true,
}: TooltipGroupProps) => {
  const [state, dispatch] = useReducer(reduceTooltipGroup, { isRequested: false, value: null });
  const isOnDisabledTrigger = state.value !== null && !isEnabled(state.value);
  const isShown = state.isRequested && state.value !== null && !isOnDisabledTrigger;

  // Adjusted during render (React's pattern for state derived from changing input). A request with no trigger yet is
  // left alone: the trigger may be reported just after the open.
  if (state.isRequested && isOnDisabledTrigger) {
    dispatch({ type: 'close' });
  }

  const ids = useMemo(() => ({ trigger: (triggerValue?: string) => getTriggerId(triggerValue ?? '') }), [getTriggerId]);
  const handleOpenChange = useCallback(
    (details: { open: boolean }) => dispatch({ type: details.open ? 'open' : 'close' }),
    []
  );
  const handleTriggerValueChange = useCallback(
    (details: { value: string | null }) => dispatch({ type: 'trigger', value: details.value }),
    []
  );
  const describedValue = isShown ? state.value : null;
  const contextValue = useMemo(() => ({ describedValue, isEnabled }), [describedValue, isEnabled]);

  return (
    <ChakraTooltip.Root
      closeOnScroll={closeOnScroll}
      ids={ids}
      open={isShown}
      onOpenChange={handleOpenChange}
      onTriggerValueChange={handleTriggerValueChange}
    >
      <TooltipGroupContext value={contextValue}>{children}</TooltipGroupContext>
      <Portal>
        <ChakraTooltip.Positioner>
          <ChakraTooltip.Content>
            {showArrow && (
              <ChakraTooltip.Arrow>
                <ChakraTooltip.ArrowTip />
              </ChakraTooltip.Arrow>
            )}
            {content}
          </ChakraTooltip.Content>
        </ChakraTooltip.Positioner>
      </Portal>
    </ChakraTooltip.Root>
  );
};

// zag's prop merge keeps a value over `undefined`, so `null` is what removes the description (React renders nothing).
const NOT_DESCRIBED = { 'aria-describedby': null } as unknown as { 'aria-describedby'?: string };

// Zag's trigger handlers skip an event whose default is prevented; pointerover and pointermove have no default action.
const HIDE_POINTER_FROM_TOOLTIP = {
  onPointerMoveCapture: (event: PointerEvent) => event.preventDefault(),
  onPointerOverCapture: (event: PointerEvent) => event.preventDefault(),
};

/**
 * The tooltip describes every trigger in a group while open; only the one it shows on keeps the description. A trigger
 * that is not enabled hides the pointer from the tooltip, so it can neither open on it nor hand the tip to it. Focus
 * needs no such guard: from closed, Zag waits for the controlled prop; from open, it switches the trigger at once and
 * the group's reducer closes the tip at the next render.
 */
const TooltipGroupTrigger = ({ children, value }: { children: ReactElement; value: string }) => {
  const { describedValue, isEnabled } = use(TooltipGroupContext);

  return (
    <ChakraTooltip.Trigger
      {...(describedValue === value ? {} : NOT_DESCRIBED)}
      {...(isEnabled(value) ? {} : HIDE_POINTER_FROM_TOOLTIP)}
      asChild
      value={value}
    >
      {children}
    </ChakraTooltip.Trigger>
  );
};

export const TooltipGroup = { Root: TooltipGroupRoot, Trigger: TooltipGroupTrigger };
