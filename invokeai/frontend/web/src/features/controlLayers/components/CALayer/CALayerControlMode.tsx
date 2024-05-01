import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { caLayerControlModeChanged, selectCALayer } from 'features/controlLayers/store/controlLayersSlice';
import { isControlMode } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const CALayerControlMode = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const controlMode = useAppSelector((s) => {
    const ca = selectCALayer(s.controlLayers.present, layerId).controlAdapter;
    assert(ca.type === 'controlnet');
    return ca.controlMode;
  });

  const CONTROL_MODE_DATA = useMemo(
    () => [
      { label: t('controlnet.balanced'), value: 'balanced' },
      { label: t('controlnet.prompt'), value: 'more_prompt' },
      { label: t('controlnet.control'), value: 'more_control' },
      { label: t('controlnet.megaControl'), value: 'unbalanced' },
    ],
    [t]
  );

  const handleControlModeChange = useCallback<ComboboxOnChange>(
    (v) => {
      assert(isControlMode(v?.value));
      dispatch(
        caLayerControlModeChanged({
          layerId,
          controlMode: v.value,
        })
      );
    },
    [layerId, dispatch]
  );

  const value = useMemo(
    () => CONTROL_MODE_DATA.filter((o) => o.value === controlMode)[0],
    [CONTROL_MODE_DATA, controlMode]
  );

  if (!controlMode) {
    return null;
  }

  return (
    <FormControl>
      <InformationalPopover feature="controlNetControlMode">
        <FormLabel m={0}>{t('controlnet.control')}</FormLabel>
      </InformationalPopover>
      <Combobox
        value={value}
        options={CONTROL_MODE_DATA}
        onChange={handleControlModeChange}
        isClearable={false}
        isSearchable={false}
      />
    </FormControl>
  );
});

CALayerControlMode.displayName = 'CALayerControlMode';
