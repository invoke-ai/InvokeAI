import { Accordion } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setOpenAccordions } from 'features/system/store/systemSlice';
import { ReactElement } from 'react';
import InvokeAccordionItem, {
  InvokeAccordionItemProps,
} from './AccordionItems/InvokeAccordionItem';

type ParametersAccordionType = {
  [parametersAccordionKey: string]: InvokeAccordionItemProps;
};

type ParametersAccordionsType = {
  accordionInfo: ParametersAccordionType;
};

/**
 * Main container for generation and processing parameters.
 */
const ParametersAccordion = (props: ParametersAccordionsType) => {
  const { accordionInfo } = props;

  const openAccordions = useAppSelector(
    (state: RootState) => state.system.openAccordions
  );

  const dispatch = useAppDispatch();

  /**
   * Stores accordion state in redux so preferred UI setup is retained.
   */
  const handleChangeAccordionState = (openAccordions: number | number[]) =>
    dispatch(setOpenAccordions(openAccordions));

  const renderAccordions = () => {
    const accordionsToRender: ReactElement[] = [];
    if (accordionInfo) {
      Object.keys(accordionInfo).forEach((key) => {
        const { header, feature, content, additionalHeaderComponents } =
          accordionInfo[key];
        accordionsToRender.push(
          <InvokeAccordionItem
            key={key}
            header={header}
            feature={feature}
            content={content}
            additionalHeaderComponents={additionalHeaderComponents}
          />
        );
      });
    }
    return accordionsToRender;
  };

  return (
    <Accordion
      defaultIndex={openAccordions}
      allowMultiple
      reduceMotion
      onChange={handleChangeAccordionState}
    >
      {renderAccordions()}
    </Accordion>
  );
};

export default ParametersAccordion;
