import { Flex } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { RootState } from 'app/store/store';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { ParamHiresStrength } from './ParamHiresStrength';
import { setHiresFix } from 'features/parameters/store/postprocessingSlice';

const ParamHiresCollapse = () => {
  const { t } = useTranslation();
  const hiresFix = useAppSelector(
    (state: RootState) => state.postprocessing.hiresFix
  );

  const dispatch = useAppDispatch();

  const handleToggle = () => dispatch(setHiresFix(!hiresFix));

  return (
    <IAICollapse
      label={t('parameters.hiresOptim')}
      isOpen={hiresFix}
      onToggle={handleToggle}
      withSwitch
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamHiresStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamHiresCollapse);
