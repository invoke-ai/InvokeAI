import { FormControl, FormLabel, Select } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { bboxAspectRatioIdChanged } from 'features/controlLayers/store/canvasInstanceSlice';
import { useCanvasIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  selectIsChatGPT4o,
  selectIsFluxKontext,
  selectIsImagen3,
  selectIsImagen4,
} from 'features/controlLayers/store/paramsSlice';
import { selectAspectRatioID } from 'features/controlLayers/store/selectors';
import {
  isAspectRatioID,
  zAspectRatioID,
  zChatGPT4oAspectRatioID,
  zFluxKontextAspectRatioID,
  zImagen3AspectRatioID,
} from 'features/controlLayers/store/types';
import type { ChangeEventHandler } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const BboxAspectRatioSelect = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useAppSelector(selectAspectRatioID);
  const isStaging = useCanvasIsStaging();
  const isImagen3 = useAppSelector(selectIsImagen3);
  const isChatGPT4o = useAppSelector(selectIsChatGPT4o);
  const isImagen4 = useAppSelector(selectIsImagen4);
  const isFluxKontext = useAppSelector(selectIsFluxKontext);
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
    // All other models
    return zAspectRatioID.options;
  }, [isImagen3, isChatGPT4o, isImagen4, isFluxKontext]);

  const onChange = useCallback<ChangeEventHandler<HTMLSelectElement>>(
    (e) => {
      if (!isAspectRatioID(e.target.value)) {
        return;
      }
      dispatch(bboxAspectRatioIdChanged({ id: e.target.value }));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={isStaging}>
      <InformationalPopover feature="paramAspect">
        <FormLabel>{t('parameters.aspect')}</FormLabel>
      </InformationalPopover>
      <Select size="sm" value={id ?? 'Free'} onChange={onChange} cursor="pointer" iconSize="0.75rem" icon={<PiCaretDownBold />}>
        {options.map((ratio) => (
          <option key={ratio} value={ratio}>
            {ratio}
          </option>
        ))}
      </Select>
    </FormControl>
  );
});

BboxAspectRatioSelect.displayName = 'BboxAspectRatioSelect';
