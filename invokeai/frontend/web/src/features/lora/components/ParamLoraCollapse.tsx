import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { size } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamLoraList from './ParamLoraList';
import ParamLoRASelect from './ParamLoraSelect';

const selector = createMemoizedSelector(stateSelector, (state) => {
  const loraCount = size(state.lora.loras);
  return {
    activeLabel: loraCount > 0 ? `${loraCount} Active` : undefined,
  };
});

const ParamLoraCollapse = () => {
  const { t } = useTranslation();
  const { activeLabel } = useAppSelector(selector);

  const isLoraEnabled = useFeatureStatus('lora').isFeatureEnabled;

  if (!isLoraEnabled) {
    return null;
  }

  return (
    <IAICollapse label={t('modelManager.loraModels')} activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamLoRASelect />
        <ParamLoraList />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamLoraCollapse);
