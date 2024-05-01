import type { ComboboxOnChange } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { ControlMode } from 'features/controlLayers/util/controlAdapters';
import { isControlMode } from 'features/controlLayers/util/controlAdapters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type Props = {
  controlMode: ControlMode;
  onChange: (controlMode: ControlMode) => void;
};

export const ControlAdapterControlModeSelect = memo(({ controlMode, onChange }: Props) => {
  const { t } = useTranslation();
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

ControlAdapterControlModeSelect.displayName = 'ControlAdapterControlModeSelect';
