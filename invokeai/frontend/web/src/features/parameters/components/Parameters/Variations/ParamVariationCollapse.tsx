import ParamVariationWeights from './ParamVariationWeights';
import ParamVariationAmount from './ParamVariationAmount';
import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RootState } from 'app/store/store';
import { setShouldGenerateVariations } from 'features/parameters/store/generationSlice';
import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

const ParamVariationCollapse = () => {
  const { t } = useTranslation();
  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.generation.shouldGenerateVariations
  );

  const isVariationEnabled = useFeatureStatus('variation').isFeatureEnabled;

  const dispatch = useAppDispatch();

  const handleToggle = () =>
    dispatch(setShouldGenerateVariations(!shouldGenerateVariations));

  if (!isVariationEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('parameters.variations')}
      isOpen={shouldGenerateVariations}
      onToggle={handleToggle}
      withSwitch
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamVariationAmount />
        <ParamVariationWeights />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamVariationCollapse);
