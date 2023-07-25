import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import ParamRefinerModelSelect from './SDXLRefiner/ParamRefinerModelSelect';
import ParamSDXLRefinerSteps from './SDXLRefiner/ParamSDXLRefinerSteps';
import ParamUseSDXLRefiner from './SDXLRefiner/ParamUseSDXLRefiner';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { shouldUseSDXLRefiner } = state.sdxl;
    return { activeLabel: shouldUseSDXLRefiner ? 'Enabled' : undefined };
  },
  defaultSelectorOptions
);

const ParamSDXLRefinerCollapse = () => {
  const { activeLabel } = useAppSelector(selector);

  return (
    <IAICollapse label="Refiner" activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <ParamUseSDXLRefiner />
        <ParamRefinerModelSelect />
        <ParamSDXLRefinerSteps />
      </Flex>
    </IAICollapse>
  );
};

export default ParamSDXLRefinerCollapse;
