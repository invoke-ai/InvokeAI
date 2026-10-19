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
  type ComponentProps,
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
  const triggerChild =
    hasTriggerPassthrough && isValidElement<TooltipChildProps>(children)
      ? cloneElement(children as ReactElement<TooltipChildProps>, {
          ...mergedTriggerProps,
          // eslint-disable-next-line react/refs
          ref: composeRefs(children.props.ref, explicitTriggerRef, ref),
        })
      : children;

  if (disabled) {
    return triggerChild;
  }

  return (
    <ChakraTooltip.Root {...(rootProps as ChakraTooltip.RootProps)}>
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

const TooltipGroupContext = createContext<string | null>(null);

/**
 * One tooltip shared by a group of triggers, such as the tabs of a strip, so moving focus or the pointer between them
 * hands the tip over. Separate tooltips close and open through deferred events, and a focus move made inside a key
 * handler (roving tab focus) can deliver the closing one's release after the arriving one opened, closing it again.
 *
 * The tip is controlled: the tooltip only asks to open and close, reporting the trigger and the open in an order that
 * depends on how it got there. A request on a trigger that is not enabled ends at the next render, and so does one
 * whose trigger stops being enabled: the tooltip then stays closed and ignores the blur or pointer leave that would
 * otherwise end the request, and a request left standing would open the tip later on a trigger nobody is on.
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

  return (
    <ChakraTooltip.Root
      closeOnScroll={closeOnScroll}
      ids={ids}
      open={isShown}
      onOpenChange={handleOpenChange}
      onTriggerValueChange={handleTriggerValueChange}
    >
      <TooltipGroupContext value={isShown ? state.value : null}>{children}</TooltipGroupContext>
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

/** The tooltip describes every trigger in a group while open; only the one it shows on keeps the description. */
const TooltipGroupTrigger = ({ children, value }: { children: ReactElement; value: string }) => {
  const describedValue = use(TooltipGroupContext);

  return (
    <ChakraTooltip.Trigger {...(describedValue === value ? {} : NOT_DESCRIBED)} asChild value={value}>
      {children}
    </ChakraTooltip.Trigger>
  );
};

export const TooltipGroup = { Root: TooltipGroupRoot, Trigger: TooltipGroupTrigger };
