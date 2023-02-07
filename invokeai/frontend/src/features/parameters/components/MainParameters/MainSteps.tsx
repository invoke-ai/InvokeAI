import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import { setSteps } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

export default function MainSteps() {
  const dispatch = useAppDispatch();
  const steps = useAppSelector((state: RootState) => state.generation.steps);
  const { t } = useTranslation();

  const handleChangeSteps = (v: number) => dispatch(setSteps(v));

  return (
    <IAINumberInput
      label={t('parameters:steps')}
      min={1}
      max={9999}
      step={1}
      onChange={handleChangeSteps}
      value={steps}
      width="auto"
      styleClass="main-settings-block"
      textAlign="center"
    />
  );
}
