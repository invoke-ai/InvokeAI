import { Accordion } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { Feature } from 'app/features';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { tabMap } from 'features/ui/store/tabMap';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { openAccordionItemsChanged } from 'features/ui/store/uiSlice';
import { filter, map } from 'lodash';
import { ReactNode, useCallback } from 'react';
import InvokeAccordionItem from './AccordionItems/InvokeAccordionItem';

const parametersAccordionSelector = createSelector(
  [uiSelector, systemSelector],
  (uiSlice, system) => {
    const {
      activeTab,
      openLinearAccordionItems,
      openUnifiedCanvasAccordionItems,
    } = uiSlice;

    const { disabledFeatures } = system;

    let openAccordions: number[] = [];

    if (tabMap[activeTab] === 'linear') {
      openAccordions = openLinearAccordionItems;
    }

    if (tabMap[activeTab] === 'unifiedCanvas') {
      openAccordions = openUnifiedCanvasAccordionItems;
    }

    return {
      openAccordions,
      disabledFeatures,
    };
  }
);

export type ParametersAccordionItem = {
  name: string;
  header: string;
  content: ReactNode;
  feature?: Feature;
  additionalHeaderComponents?: ReactNode;
};

export type ParametersAccordionItems = {
  [parametersAccordionKey: string]: ParametersAccordionItem;
};

type ParametersAccordionProps = {
  accordionItems: ParametersAccordionItems;
};

/**
 * Main container for generation and processing parameters.
 */
const ParametersAccordion = ({ accordionItems }: ParametersAccordionProps) => {
  const { openAccordions, disabledFeatures } = useAppSelector(
    parametersAccordionSelector
  );

  const dispatch = useAppDispatch();

  const handleChangeAccordionState = (openAccordions: number | number[]) => {
    dispatch(
      openAccordionItemsChanged(
        Array.isArray(openAccordions) ? openAccordions : [openAccordions]
      )
    );
  };

  // Render function for accordion items
  const renderAccordionItems = useCallback(
    () =>
      map(accordionItems, (accordionItem) => (
        <InvokeAccordionItem
          key={accordionItem.name}
          accordionItem={accordionItem}
        />
      )),
    [accordionItems]
  );

  return (
    <Accordion
      defaultIndex={openAccordions}
      allowMultiple
      onChange={handleChangeAccordionState}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      {renderAccordionItems()}
    </Accordion>
  );
};

export default ParametersAccordion;
