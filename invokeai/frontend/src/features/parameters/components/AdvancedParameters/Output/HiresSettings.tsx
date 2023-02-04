import { Flex } from '@chakra-ui/react';
import { ChangeEvent } from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHiresFix } from 'features/parameters/store/postprocessingSlice';
import { useTranslation } from 'react-i18next';

/**
 * Hires Fix Toggle
 */
const HiresSettings = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector((state: RootState) => state.postprocessing.hiresFix);

  const { t } = useTranslation();

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label={t('parameters:hiresOptim')}
        fontSize={'md'}
        isChecked={hiresFix}
        onChange={handleChangeHiresFix}
      />
    </Flex>
  );
};

export default HiresSettings;
