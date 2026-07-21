/* eslint-disable react/react-compiler */
import { Tooltip as ChakraTooltip, Portal } from '@chakra-ui/react';
import {
  cloneElement,
  isValidElement,
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
  /**
   * Which trigger edge the tooltip sits on, centered against it — `top`/
   * `bottom` center horizontally, `left`/`right` center vertically; `-start`/
   * `-end` suffixes align to an edge instead. Shorthand for
   * `positioning.placement` (an explicit `positioning.placement` wins).
   */
  placement?: TooltipPlacement;
  ref?: Ref<HTMLButtonElement>;
  triggerProps?: TooltipTriggerProps;
}

/**
 * Workbench tooltip. Chrome comes from the `tooltip` slot-recipe override in
 * `theme/recipes.ts`, so this wrapper only provides the trigger/portal
 * structure and the `content` convenience API.
 */
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
