import { InvAccordionButton } from 'common/components/InvAccordion/InvAccordionButton';
import {
  InvAccordion,
  InvAccordionItem,
  InvAccordionPanel,
} from 'common/components/InvAccordion/wrapper';
import { memo, useCallback } from 'react';

import type { InvSingleAccordionProps } from './types';
import { singleAccordionExpanded } from '../../../features/parameters/store/actions';
import { useAppDispatch } from '../../../app/store/storeHooks';

export const InvSingleAccordion = memo((props: InvSingleAccordionProps) => {
  const dispatch = useAppDispatch();
  const handleAccordionClick = useCallback(
    (isOpen: boolean) => {
      if (props.id) {
        dispatch(singleAccordionExpanded({ id: props.id, isOpen }));
      }
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
