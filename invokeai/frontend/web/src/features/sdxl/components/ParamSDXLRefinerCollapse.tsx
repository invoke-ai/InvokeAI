import { Flex, Text } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';
import ParamSDXLRefinerCFGScale from './SDXLRefiner/ParamSDXLRefinerCFGScale';
import ParamSDXLRefinerModelSelect from './SDXLRefiner/ParamSDXLRefinerModelSelect';
import ParamSDXLRefinerNegativeAestheticScore from './SDXLRefiner/ParamSDXLRefinerNegativeAestheticScore';
import ParamSDXLRefinerPositiveAestheticScore from './SDXLRefiner/ParamSDXLRefinerPositiveAestheticScore';
import ParamSDXLRefinerScheduler from './SDXLRefiner/ParamSDXLRefinerScheduler';
import ParamSDXLRefinerStart from './SDXLRefiner/ParamSDXLRefinerStart';
import ParamSDXLRefinerSteps from './SDXLRefiner/ParamSDXLRefinerSteps';
import ParamUseSDXLRefiner from './SDXLRefiner/ParamUseSDXLRefiner';

const selector = createMemoizedSelector(stateSelector, (state) => {
  const { shouldUseSDXLRefiner } = state.sdxl;
  const { shouldUseSliders } = state.ui;
  return {
    activeLabel: shouldUseSDXLRefiner ? 'Enabled' : undefined,
    shouldUseSliders,
  };
});

const ParamSDXLRefinerCollapse = () => {
  const { activeLabel, shouldUseSliders } = useAppSelector(selector);
  const { t } = useTranslation();
  const isRefinerAvailable = useIsRefinerAvailable();

  if (!isRefinerAvailable) {
    return (
      <IAICollapse label={t('sdxl.refiner')} activeLabel={activeLabel}>
        <Flex sx={{ justifyContent: 'center', p: 2 }}>
          <Text sx={{ fontSize: 'sm', color: 'base.500', _dark: 'base.700' }}>
            {t('models.noRefinerModelsInstalled')}
          </Text>
        </Flex>
      </IAICollapse>
    );
  }

  return (
    <IAICollapse label={t('sdxl.refiner')} activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamUseSDXLRefiner />
        <ParamSDXLRefinerModelSelect />
        <Flex gap={2} flexDirection={shouldUseSliders ? 'column' : 'row'}>
          <ParamSDXLRefinerSteps />
          <ParamSDXLRefinerCFGScale />
        </Flex>
        <ParamSDXLRefinerScheduler />
        <ParamSDXLRefinerPositiveAestheticScore />
        <ParamSDXLRefinerNegativeAestheticScore />
        <ParamSDXLRefinerStart />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSDXLRefinerCollapse);
