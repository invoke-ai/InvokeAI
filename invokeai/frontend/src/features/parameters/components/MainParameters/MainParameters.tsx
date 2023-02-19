import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { useTranslation } from 'react-i18next';
import ParametersAccordion from '../ParametersAccordion';
import MainCFGScale from './MainCFGScale';
import MainHeight from './MainHeight';
import MainIterations from './MainIterations';
import MainSampler from './MainSampler';
import MainSteps from './MainSteps';
import MainWidth from './MainWidth';

export const inputWidth = 'auto';

export default function MainSettings() {
  const { t } = useTranslation();

  const shouldUseSliders = useAppSelector(
    (state: RootState) => state.ui.shouldUseSliders
  );

  const accordionItems = {
    main: {
      header: `${t('parameters.general')}`,
      feature: undefined,
      content: shouldUseSliders ? (
        <Flex flexDir="column" rowGap={2}>
          <MainIterations />
          <MainSteps />
          <MainCFGScale />
          <MainWidth />
          <MainHeight />
          <MainSampler />
        </Flex>
      ) : (
        <Flex flexDirection="column" rowGap={2}>
          <Flex gap={2}>
            <MainIterations />
            <MainSteps />
            <MainCFGScale />
          </Flex>
          <Flex>
            <MainWidth />
            <MainHeight />
            <MainSampler />
          </Flex>
        </Flex>
      ),
    },
  };
  return <ParametersAccordion accordionInfo={accordionItems} />;
}
