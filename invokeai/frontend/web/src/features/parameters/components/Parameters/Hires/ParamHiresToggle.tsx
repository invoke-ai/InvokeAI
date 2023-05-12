import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setHiresFix } from 'features/parameters/store/postprocessingSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Hires Fix Toggle
 */
export const ParamHiresToggle = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector(
    (state: RootState) => state.postprocessing.hiresFix
  );

  const { t } = useTranslation();

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));

  return (
    <IAISwitch
      label={t('parameters.hiresOptim')}
      fontSize="md"
      isChecked={hiresFix}
      onChange={handleChangeHiresFix}
    />
  );
};
