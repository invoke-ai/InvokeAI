import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { ControlMode } from 'features/controlLayers/store/types';
import { isControlMode } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ControlNetModelConfig } from 'services/api/types';
import { assert } from 'tsafe';

type Props = {
  controlMode: ControlMode;
  onChange: (controlMode: ControlMode) => void;
  model: ControlNetModelConfig | null;
};

export const ControlLayerControlAdapterControlMode = memo(({ controlMode, onChange, model }: Props) => {
  const { t } = useTranslation();
  
  const CONTROL_MODE_DATA = useMemo(() => {
    // Show BRIA-specific control modes if a BRIA model is selected
    if (model?.base === 'bria') {
      return [
        { label: t('controlLayers.controlMode.depth'), value: 'depth' },
        { label: t('controlLayers.controlMode.canny'), value: 'canny' },
        { label: t('controlLayers.controlMode.colorgrid'), value: 'colorgrid' },
        { label: t('controlLayers.controlMode.recolor'), value: 'recolor' },
        { label: t('controlLayers.controlMode.tile'), value: 'tile' },
        { label: t('controlLayers.controlMode.pose'), value: 'pose' },
      ];
    }
    // Show standard control modes for other models
    return [
      { label: t('controlLayers.controlMode.balanced'), value: 'balanced' },
      { label: t('controlLayers.controlMode.prompt'), value: 'more_prompt' },
      { label: t('controlLayers.controlMode.control'), value: 'more_control' },
      { label: t('controlLayers.controlMode.megaControl'), value: 'unbalanced' },
    ];
  }, [t, model?.base]);

  const handleControlModeChange = useCallback<ComboboxOnChange>(
    (v) => {
      assert(isControlMode(v?.value));
      onChange(v.value);
    },
    [onChange]
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
        <FormLabel m={0}>{t('controlLayers.controlMode.controlMode')}</FormLabel>
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

ControlLayerControlAdapterControlMode.displayName = 'ControlLayerControlAdapterControlMode';
