import { Flex } from '@chakra-ui/react';
import { ChangeEvent } from 'react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHiresFix } from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

/**
 * Hires Fix Toggle
 */
const HiresOptions = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector((state: RootState) => state.options.hiresFix);

  const { t } = useTranslation();

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label={t('options:hiresOptim')}
        fontSize={'md'}
        isChecked={hiresFix}
        onChange={handleChangeHiresFix}
      />
    </Flex>
  );
};

export default HiresOptions;
