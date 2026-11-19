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
 * Zag skips trigger events whose default is prevented; this keeps a tip shut where `disabled` or `open` fail:
 * Zag opens a hovered trigger at once while another tip shows, and `disabled` swallows the pointer leave.
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
  placement?: TooltipPlacement;
  ref?: Ref<HTMLButtonElement>;
  triggerProps?: TooltipTriggerProps;
  /** Keeps the tooltip mounted and shut instead of unmounting, which would drop the trigger's focus or capture. */
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

  // Shut through the controlled prop; the gate above keeps the pointer from opening around it.
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
  isClipped: () => boolean;
}

/** Measured as the pointer or focus arrives, so a resize costs no render. */
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
  /** A trigger not enabled on arrival shows nothing until the next one; one that stops being enabled closes the tip. */
  isEnabled: (value: string) => boolean;
  showArrow?: boolean;
}

interface TooltipGroupState {
  value: string | null;
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
  describedValue: string | null;
  isEnabled: (value: string) => boolean;
}

const TooltipGroupContext = createContext<TooltipGroupContextValue>({ describedValue: null, isEnabled: () => false });

/**
 * One tooltip shared across triggers: separate ones close via deferred events, and roving focus can close the new one.
 * A disabled trigger keeps the pointer from Zag entirely (see TooltipGroupTrigger).
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

  // Derived state adjusted during render; a request with no trigger yet is left alone.
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

/** Only the shown trigger keeps the description; a disabled trigger hides the pointer, while focus needs no guard. */
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
