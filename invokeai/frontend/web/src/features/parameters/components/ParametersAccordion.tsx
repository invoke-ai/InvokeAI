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

  const { system, ui } = useAppSelector((state: RootState) => state);

  const { openAccordions } = system;
  const { disabledParameterPanels } = ui;

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

        // do not render if panel is disabled in global state
        if (disabledParameterPanels.indexOf(key) === -1) {
          accordionsToRender.push(
            <InvokeAccordionItem
              key={key}
              header={header}
              feature={feature}
              content={content}
              additionalHeaderComponents={additionalHeaderComponents}
            />
          );
        }
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
      sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      {renderAccordions()}
    </Accordion>
  );
};

export default ParametersAccordion;
