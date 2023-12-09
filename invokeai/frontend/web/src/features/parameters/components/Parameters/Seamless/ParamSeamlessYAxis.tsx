import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setSeamlessYAxis } from 'features/parameters/store/generationSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(stateSelector, ({ generation }) => {
  const { seamlessYAxis } = generation;

  return { seamlessYAxis };
});

const ParamSeamlessYAxis = () => {
  const { t } = useTranslation();
  const { seamlessYAxis } = useAppSelector(selector);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessYAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <IAISwitch
      label={t('parameters.seamlessYAxis')}
      aria-label={t('parameters.seamlessYAxis')}
      isChecked={seamlessYAxis}
      onChange={handleChange}
    />
  );
};

export default memo(ParamSeamlessYAxis);
