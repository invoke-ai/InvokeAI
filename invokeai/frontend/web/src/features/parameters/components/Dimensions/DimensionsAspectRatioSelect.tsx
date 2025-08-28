import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  aspectRatioIdChanged,
  selectAspectRatioID,
  selectIsChatGPT4o,
  selectIsFluxKontext,
  selectIsGemini2_5,
  selectIsImagen3,
  selectIsImagen4,
} from 'features/controlLayers/store/paramsSlice';
import {
  isAspectRatioID,
  zAspectRatioID,
  zChatGPT4oAspectRatioID,
  zFluxKontextAspectRatioID,
  zGemini2_5AspectRatioID,
  zImagen3AspectRatioID,
  zVeo3AspectRatioID,
} from 'features/controlLayers/store/types';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const DimensionsAspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatioID);
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isChatGPT4o = useAppSelector(selectIsChatGPT4o);
  const isImagen4 = useAppSelector(selectIsImagen4);
  const isFluxKontext = useAppSelector(selectIsFluxKontext);
  const isGemini2_5 = useAppSelector(selectIsGemini2_5);
  const activeTab = useAppSelector(selectActiveTab);
  const options = useMemo(() => {
    // Imagen3 and ChatGPT4o have different aspect ratio options, and do not support freeform sizes
    if (isImagen3 || isImagen4) {
      return zImagen3AspectRatioID.options;
    }
    if (isChatGPT4o) {
      return zChatGPT4oAspectRatioID.options;
    }
    if (isFluxKontext) {
      return zFluxKontextAspectRatioID.options;
    }
    if (isGemini2_5) {
      return zGemini2_5AspectRatioID.options;
    }
    if (activeTab === 'video') {
      return zVeo3AspectRatioID.options;
    }
    // All other models
    return zAspectRatioID.options;
  }, [isImagen3, isChatGPT4o, isImagen4, isFluxKontext, activeTab, isGemini2_5]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isAspectRatioID(e.target.value)) {
        return;
      }
      dispatch(aspectRatioIdChanged({ id: e.target.value }));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Select size="sm" value={id} onChange={onChange} cursor="pointer" iconSize="0.75rem" icon={<PiCaretDownBold />}>
        {options.map((ratio) => (
          <option key={ratio} value={ratio}>
            {ratio}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

DimensionsAspectRatioSelect.displayName = 'DimensionsAspectRatioSelect';
