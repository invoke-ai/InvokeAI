import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import { IAISelectDataType } from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setCanvasCoherenceMode } from 'features/parameters/store/generationSlice';
import { ParameterCanvasCoherenceMode } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const ParamCanvasCoherenceMode = () => {
  const dispatch = useAppDispatch();
  const canvasCoherenceMode = useAppSelector(
    (state: RootState) => state.generation.canvasCoherenceMode
  );
  const { t } = useTranslation();

  const coherenceModeSelectData: IAISelectDataType[] = useMemo(
    () => [
      { label: t('parameters.unmasked'), value: 'unmasked' },
      { label: t('unifiedCanvas.mask'), value: 'mask' },
      { label: t('parameters.maskEdge'), value: 'edge' },
    ],
    [t]
  );

  const handleCoherenceModeChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      dispatch(setCanvasCoherenceMode(v as ParameterCanvasCoherenceMode));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="compositingCoherenceMode">
      <IAIMantineSelect
        label={t('parameters.coherenceMode')}
        data={coherenceModeSelectData}
        value={canvasCoherenceMode}
        onChange={handleCoherenceModeChange}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamCanvasCoherenceMode);
