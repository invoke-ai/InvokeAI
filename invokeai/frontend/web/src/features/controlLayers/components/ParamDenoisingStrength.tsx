import {
  Badge,
  Button,
  Collapse,
  CompositeNumberInput,
  CompositeSlider,
  Flex,
  FormControl,
  FormLabel,
  Icon,
  Tooltip,
} from '@invoke-ai/ui-library';
import { useAppDispatch,useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useBoolean } from 'common/hooks/useBoolean';
import { selectImg2imgStrength, setImg2imgStrength } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { selectImg2imgStrengthConfig } from 'features/system/store/configSlice';
import type { CSSProperties} from 'react';
import { useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCaretDownBold,
  PiImageBold,
  PiImageFill,
} from 'react-icons/pi';

const marks = [0, 0.5, 1];
const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0, width: '100%' };


export const ParamDenoisingStrength = () => {
  const img2imgStrength = useAppSelector(selectImg2imgStrength);
  const dispatch = useAppDispatch();
  const collapse = useBoolean(true);
  const isEnabled = useAppSelector(
    (s) => selectCanvasSlice(s).rasterLayers.entities.length > 0 && selectCanvasSlice(s).rasterLayers.entities.some(layer => layer.isEnabled)
  );

  useEffect(() => {
      collapse.set(!isEnabled)
  }, [isEnabled])


  const onChange = useCallback(
    (v: number) => {
      dispatch(setImg2imgStrength(v));
    },
    [dispatch]
  );

  console.log({collapse})

  const config = useAppSelector(selectImg2imgStrengthConfig);
  const { t } = useTranslation();
  return (
    <FormControl orientation="vertical" w="full" isDisabled={!isEnabled}>
      <Flex
        as={Button}
        onClick={collapse.toggle}
        justifyContent="space-between"
        alignItems="center"
        gap={3}
        variant="unstyled"
      >
        <Icon
          boxSize={4}
          as={PiCaretDownBold}
          transform={!collapse.isTrue ? undefined : 'rotate(-90deg)'}
          fill={isEnabled ? 'base.200' : 'base.500'}
          transitionProperty="common"
          transitionDuration="fast"
        />
        <InformationalPopover feature="paramDenoisingStrength">
          <FormLabel color="base.200">{`${t('parameters.denoisingStrength')}`} {isEnabled && collapse.isTrue ? `(${img2imgStrength})` : ""}{!isEnabled && <Badge ml={2}>No Raster Layers Enabled</Badge>}</FormLabel>
        </InformationalPopover>
      </Flex>
      <Collapse in={!collapse.isTrue} style={COLLAPSE_STYLES}>
        <Flex w="full" gap={5}>
          <Flex gap={3} w="full" alignItems="center">
            <Tooltip label="Less change">
              <Flex flexDir="row" layerStyle="second" padding={1} borderRadius="base" border="1px solid gray" borderColor="base.600" opacity={isEnabled ? "1" : "0.7"}>
                <Icon as={PiImageBold} color="base.300" />
                <Icon as={PiImageBold} color="base.300" />
              </Flex>
            </Tooltip>

            <CompositeSlider
              step={config.coarseStep}
              fineStep={config.fineStep}
              min={config.sliderMin}
              max={config.sliderMax}
              defaultValue={config.initial}
              onChange={onChange}
              value={img2imgStrength}
              marks={marks}
            />
            <Tooltip label="More change">
              <Flex flexDir="row" layerStyle="second" padding={1} borderRadius="base" border="1px solid gray" borderColor="base.600" opacity={isEnabled ? "1" : "0.7"}>
                <Icon as={PiImageBold} color="base.300" />
                <Icon as={PiImageFill} color="base.300" />
              </Flex>
            </Tooltip>
          </Flex>
          <CompositeNumberInput
            step={config.coarseStep}
            fineStep={config.fineStep}
            min={config.numberInputMin}
            max={config.numberInputMax}
            defaultValue={config.initial}
            onChange={onChange}
            value={img2imgStrength}
          />
        </Flex>
      </Collapse>
    </FormControl>
  );
};
