import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAISelectDataType } from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setCanvasCoherenceMode } from 'features/parameters/store/generationSlice';
import { CanvasCoherenceModeParam } from 'features/parameters/types/parameterSchemas';

import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const coherenceModeSelectData: IAISelectDataType[] = [
  { label: 'Unmasked', value: 'unmasked' },
  { label: 'Mask', value: 'mask' },
  { label: 'Mask Edge', value: 'edge' },
];

const ParamCanvasCoherenceMode = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceMode = useAppSelector(
    (state: RootState) => state.generation.canvasCoherenceMode
  );
  const { t } = useTranslation();

  const handleCoherenceModeChange = (v: string | null) => {
    if (!v) {
      return;
    }

    dispatch(setCanvasCoherenceMode(v as CanvasCoherenceModeParam));
  };

  return (
    <IAIMantineSelect
      label={t('parameters.coherenceMode')}
      data={coherenceModeSelectData}
      value={canvasCoherenceMode}
      onChange={handleCoherenceModeChange}
    />
  );
};

export default memo(ParamCanvasCoherenceMode);
