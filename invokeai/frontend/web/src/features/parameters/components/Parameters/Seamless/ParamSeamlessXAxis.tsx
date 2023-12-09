import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setSeamlessXAxis } from 'features/parameters/store/generationSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(stateSelector, ({ generation }) => {
  const { seamlessXAxis } = generation;

  return { seamlessXAxis };
});

const ParamSeamlessXAxis = () => {
  const { t } = useTranslation();
  const { seamlessXAxis } = useAppSelector(selector);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessXAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <IAISwitch
      label={t('parameters.seamlessXAxis')}
      aria-label={t('parameters.seamlessXAxis')}
      isChecked={seamlessXAxis}
      onChange={handleChange}
    />
  );
};

export default memo(ParamSeamlessXAxis);
