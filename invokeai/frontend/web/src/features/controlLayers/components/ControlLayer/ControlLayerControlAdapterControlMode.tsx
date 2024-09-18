import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { ControlModeV2 } from 'features/controlLayers/store/types';
import { isControlModeV2 } from 'features/controlLayers/store/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  controlMode: ControlModeV2;
  onChange: (controlMode: ControlModeV2) => void;
};

export const ControlLayerControlAdapterControlMode = memo(({ controlMode, onChange }: Props) => {
  const { t } = useTranslation();
  const CONTROL_MODE_DATA = useMemo(
    () => [
      { label: t('controlLayers.controlMode.balanced'), value: 'balanced' },
      { label: t('controlLayers.controlMode.prompt'), value: 'more_prompt' },
      { label: t('controlLayers.controlMode.control'), value: 'more_control' },
      { label: t('controlLayers.controlMode.megaControl'), value: 'unbalanced' },
    ],
    [t]
  );

  const handleControlModeChange = useCallback<ComboboxOnChange>(
    (v) => {
      assert(isControlModeV2(v?.value));
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
