import { memo } from 'react';
import { Flex } from '@chakra-ui/react';
import ParamSymmetryHorizontal from './ParamSymmetryHorizontal';
import ParamSymmetryVertical from './ParamSymmetryVertical';

import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldUseSymmetry } from 'features/parameters/store/generationSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';

const ParamSymmetryCollapse = () => {
  const { t } = useTranslation();
  const shouldUseSymmetry = useAppSelector(
    (state: RootState) => state.generation.shouldUseSymmetry
  );

  const isSymmetryEnabled = useFeatureStatus('symmetry').isFeatureEnabled;

  const dispatch = useAppDispatch();

  const handleToggle = () => dispatch(setShouldUseSymmetry(!shouldUseSymmetry));

  if (!isSymmetryEnabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('parameters.symmetry')}
      isOpen={shouldUseSymmetry}
      onToggle={handleToggle}
      withSwitch
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamSymmetryHorizontal />
        <ParamSymmetryVertical />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSymmetryCollapse);
