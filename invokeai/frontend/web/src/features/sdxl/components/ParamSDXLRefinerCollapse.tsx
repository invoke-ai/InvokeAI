import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import ParamSDXLRefinerAestheticScore from './SDXLRefiner/ParamSDXLRefinerAestheticScore';
import ParamSDXLRefinerCFGScale from './SDXLRefiner/ParamSDXLRefinerCFGScale';
import ParamSDXLRefinerModelSelect from './SDXLRefiner/ParamSDXLRefinerModelSelect';
import ParamSDXLRefinerScheduler from './SDXLRefiner/ParamSDXLRefinerScheduler';
import ParamSDXLRefinerStart from './SDXLRefiner/ParamSDXLRefinerStart';
import ParamSDXLRefinerSteps from './SDXLRefiner/ParamSDXLRefinerSteps';
import ParamUseSDXLRefiner from './SDXLRefiner/ParamUseSDXLRefiner';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { shouldUseSDXLRefiner } = state.sdxl;
    const { shouldUseSliders } = state.ui;
    return {
      activeLabel: shouldUseSDXLRefiner ? 'Enabled' : undefined,
      shouldUseSliders,
    };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerCollapse = () => {
  const { activeLabel, shouldUseSliders } = useAppSelector(selector);

  return (
    <IAICollapse label="Refiner" activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamUseSDXLRefiner />
        <ParamSDXLRefinerModelSelect />
        <Flex gap={2} flexDirection={shouldUseSliders ? 'column' : 'row'}>
          <ParamSDXLRefinerSteps />
          <ParamSDXLRefinerCFGScale />
        </Flex>
        <ParamSDXLRefinerScheduler />
        <ParamSDXLRefinerAestheticScore />
        <ParamSDXLRefinerStart />
      </Flex>
    </IAICollapse>
  );
};

export default ParamSDXLRefinerCollapse;
