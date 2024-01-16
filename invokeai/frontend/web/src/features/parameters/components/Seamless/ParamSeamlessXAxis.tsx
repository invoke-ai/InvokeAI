import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { setSeamlessXAxis } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamSeamlessXAxis = () => {
  const { t } = useTranslation();
  const seamlessXAxis = useAppSelector((s) => s.generation.seamlessXAxis);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessXAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <InvControl label={t('parameters.seamlessXAxis')}>
      <InvSwitch isChecked={seamlessXAxis} onChange={handleChange} />
    </InvControl>
  );
};

export default memo(ParamSeamlessXAxis);
