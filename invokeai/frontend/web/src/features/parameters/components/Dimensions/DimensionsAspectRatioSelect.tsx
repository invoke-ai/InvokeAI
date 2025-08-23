import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  aspectRatioIdChanged,
  selectAspectRatioID,
  selectIsChatGPT4o,
  selectIsFluxKontext,
  selectIsImagen3,
  selectIsImagen4,
} from 'features/controlLayers/store/paramsSlice';
import {
  isAspectRatioID,
  zAspectRatioID,
  zChatGPT4oAspectRatioID,
  zFluxKontextAspectRatioID,
  zImagen3AspectRatioID,
  zRunwayAspectRatioID,
  zVeo3AspectRatioID,
} from 'features/controlLayers/store/types';
import { selectIsRunway, selectIsVeo3 } from 'features/parameters/store/videoSlice';
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
  const isVeo3 = useAppSelector(selectIsVeo3);
  const isRunway = useAppSelector(selectIsRunway);
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
    if (isVeo3) {
      return zVeo3AspectRatioID.options;
    }
    if (isRunway) {
      return zRunwayAspectRatioID.options;
    }
    // All other models
    return zAspectRatioID.options;
  }, [isImagen3, isChatGPT4o, isImagen4, isFluxKontext, isVeo3, isRunway]);

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
