import {
  CompositeSlider,
  FormControl,
  FormLabel,
  IconButton,
  NumberInput,
  NumberInputField,
  Popover,
  PopoverAnchor,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import { clamp } from 'lodash-es';
import type { KeyboardEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

const marks = [0, 0.25, 0.5, 0.75, 1];

export const EntityListGlobalActionBarDenoisingStrength = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const strength = useAppSelector(selectImg2imgStrength);
  const config = useAppSelector(selectImg2imgStrengthConfig);

  const [localStrength, setLocalStrength] = useState(strength);

  const onChangeSlider = useCallback(
    (value: number) => {
      dispatch(setImg2imgStrength(value));
    },
    [dispatch]
  );

  const onBlur = useCallback(() => {
    if (isNaN(Number(localStrength))) {
      setLocalStrength(config.initial);
      return;
    }
    dispatch(setImg2imgStrength(clamp(localStrength, 0, 1)));
  }, [config.initial, dispatch, localStrength]);

  const onChangeNumberInput = useCallback((valueAsString: string, valueAsNumber: number) => {
    setLocalStrength(valueAsNumber);
  }, []);

  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        onBlur();
      }
    },
    [onBlur]
  );

  useEffect(() => {
    setLocalStrength(strength);
  }, [strength]);

  return (
    <Popover>
      <FormControl w="min-content" gap={2}>
        <InformationalPopover feature="paramDenoisingStrength">
          <FormLabel m={0}>{`${t('parameters.denoisingStrength')}`}</FormLabel>
        </InformationalPopover>
        <PopoverAnchor>
          <NumberInput
            display="flex"
            alignItems="center"
            step={config.coarseStep}
            min={config.numberInputMin}
            max={config.numberInputMax}
            defaultValue={config.initial}
            value={localStrength}
            onChange={onChangeNumberInput}
            onBlur={onBlur}
            w="60px"
            onKeyDown={onKeyDown}
            clampValueOnBlur={false}
            variant="outline"
          >
            <NumberInputField paddingInlineEnd={7} _focusVisible={{ zIndex: 0 }} />
            <PopoverTrigger>
              <IconButton
                aria-label="open-slider"
                icon={<PiCaretDownBold />}
                size="sm"
                variant="link"
                position="absolute"
                insetInlineEnd={0}
                h="full"
              />
            </PopoverTrigger>
          </NumberInput>
        </PopoverAnchor>
      </FormControl>
      <PopoverContent w={200} pt={0} pb={2} px={4}>
        <PopoverArrow />
        <PopoverBody>
          <CompositeSlider
            step={config.coarseStep}
            fineStep={config.fineStep}
            min={config.sliderMin}
            max={config.sliderMax}
            defaultValue={config.initial}
            onChange={onChangeSlider}
            value={localStrength}
            marks={marks}
            alwaysShowMarks
          />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

EntityListGlobalActionBarDenoisingStrength.displayName = 'EntityListGlobalActionBarDenoisingStrength';
