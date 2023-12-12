import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamSymmetryHorizontal from './ParamSymmetryHorizontal';
import ParamSymmetryToggle from './ParamSymmetryToggle';
import ParamSymmetryVertical from './ParamSymmetryVertical';

const selector = createMemoizedSelector(stateSelector, (state) => ({
  activeLabel: state.generation.shouldUseSymmetry ? 'Enabled' : undefined,
}));

const ParamSymmetryCollapse = () => {
  const { t } = useTranslation();
  const { activeLabel } = useAppSelector(selector);

  const isSymmetryEnabled = useFeatureStatus('symmetry').isFeatureEnabled;

  if (!isSymmetryEnabled) {
    return null;
  }

  return (
    <IAICollapse label={t('parameters.symmetry')} activeLabel={activeLabel}>
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamSymmetryToggle />
        <ParamSymmetryHorizontal />
        <ParamSymmetryVertical />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamSymmetryCollapse);
