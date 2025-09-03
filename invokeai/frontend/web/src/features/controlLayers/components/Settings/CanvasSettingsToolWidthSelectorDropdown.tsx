import type { ComboboxOnChange, ComboboxOption } from '@invoke-ai/ui-library';
import { Combobox, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import type { ToolWidthSelector } from 'features/controlLayers/store/canvasSettingsSlice';
import {
  selectToolWidthSelector,
  settingsToolWidthSelectorChanged,
  zToolWidthSelector,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const isToolWidthSelector = (v: unknown): v is ToolWidthSelector => zToolWidthSelector.safeParse(v).success;

export const CanvasSettingsToolWidthSelectorDropdown = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const toolWidthSelector = useAppSelector(selectToolWidthSelector);

  const OPTIONS: ComboboxOption[] = useMemo(
    () => [
      { value: 'dropDown', label: t('controlLayers.toolWidthSelectorDropDown') },
      { value: 'slider', label: t('controlLayers.toolWidthSelectorSlider') },
    ],
    [t]
  );

  const value = useMemo(() => {
    return OPTIONS.find((o) => o.value === toolWidthSelector) || OPTIONS[0];
  }, [toolWidthSelector, OPTIONS]);

  const onChange = useCallback<ComboboxOnChange>(
    (option) => {
      if (!isToolWidthSelector(option?.value) || option.value === toolWidthSelector) {
        return;
      }
      dispatch(settingsToolWidthSelectorChanged(option.value));
    },
    [toolWidthSelector, dispatch]
  );

  return (
    <FormControl>
      <FormLabel m={0} flexGrow={1}>
        {t('controlLayers.toolWidthSelector')}
      </FormLabel>
      <Combobox isSearchable={false} value={value} options={OPTIONS} onChange={onChange} />
    </FormControl>
  );
});

CanvasSettingsToolWidthSelectorDropdown.displayName = 'CanvasSettingsToolWidthSelectorDropdown';
