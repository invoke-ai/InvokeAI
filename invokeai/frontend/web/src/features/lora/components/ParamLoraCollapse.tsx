import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICollapse from 'common/components/IAICollapse';
import { size } from 'lodash-es';
import { memo } from 'react';
import { useFeatureStatus } from '../../system/hooks/useFeatureStatus';
import ParamLoraList from './ParamLoraList';
import ParamLoRASelect from './ParamLoraSelect';

const selector = createSelector(
  stateSelector,
  (state) => {
    const loraCount = size(state.lora.loras);
    return {
      activeLabel: loraCount > 0 ? `${loraCount} Active` : undefined,
    };
  },
  defaultSelectorOptions
);

const ParamLoraCollapse = () => {
  const { activeLabel } = useAppSelector(selector);

  const isLoraEnabled = useFeatureStatus('lora').isFeatureEnabled;

  if (!isLoraEnabled) {
    return null;
  }

  return (
    <IAICollapse label="LoRA" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamLoRASelect />
        <ParamLoraList />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamLoraCollapse);
