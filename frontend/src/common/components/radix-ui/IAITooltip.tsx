import * as Tooltip from '@radix-ui/react-tooltip';
import { ReactNode } from 'react';

type IAITooltipProps = Tooltip.TooltipProps & {
  trigger: ReactNode;
  children: ReactNode;
  triggerProps?: Tooltip.TooltipTriggerProps;
  contentProps?: Tooltip.TooltipContentProps;
  arrowProps?: Tooltip.TooltipArrowProps;
};

const IAITooltip = (props: IAITooltipProps) => {
  const { trigger, children, triggerProps, contentProps, arrowProps, ...rest } =
    props;

  return (
    <Tooltip.Provider>
      <Tooltip.Root {...rest} delayDuration={0}>
        <Tooltip.Trigger {...triggerProps}>{trigger}</Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            {...contentProps}
            onPointerDownOutside={(e: any) => {e.preventDefault()}}
            className="invokeai__tooltip-content"
          >
            <Tooltip.Arrow {...arrowProps} className="invokeai__tooltip-arrow" />
            {children}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
};

export default IAITooltip;
