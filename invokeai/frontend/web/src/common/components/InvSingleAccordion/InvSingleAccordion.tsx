import { InvAccordionButton } from 'common/components/InvAccordion/InvAccordionButton';
import {
  InvAccordion,
  InvAccordionItem,
  InvAccordionPanel,
} from 'common/components/InvAccordion/wrapper';
import { memo, useCallback } from 'react';

import type { InvSingleAccordionProps } from './types';

export const InvSingleAccordion = memo((props: InvSingleAccordionProps) => {
  const handleAccordionClick = useCallback(
    (isExpanded: boolean) => {
      props.onClick && props.onClick(isExpanded);
    },
    [props]
  );

  return (
    <InvAccordion
      allowToggle
      defaultIndex={props.defaultIsOpen ? 0 : undefined}
    >
      <InvAccordionItem>
        {({ isExpanded }) => (
          <>
            <InvAccordionButton
              badges={props.badges}
              onClick={handleAccordionClick.bind(null, isExpanded)}
            >
              {props.label}
            </InvAccordionButton>
            <InvAccordionPanel>{props.children}</InvAccordionPanel>
          </>
        )}
      </InvAccordionItem>
    </InvAccordion>
  );
});

InvSingleAccordion.displayName = 'InvSingleAccordion';
