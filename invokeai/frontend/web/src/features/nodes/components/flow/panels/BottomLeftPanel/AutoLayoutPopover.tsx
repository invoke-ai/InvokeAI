import {
  Button,
  CompositeNumberInput,
  CompositeSlider,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Select,
} from '@invoke-ai/ui-library';
import { useReactFlow } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useAutoLayout } from 'features/nodes/hooks/useAutoLayout';
import {
  layeringStrategyChanged,
  layerSpacingChanged,
  layoutDirectionChanged,
  nodeAlignmentChanged,
  nodeSpacingChanged,
  selectLayeringStrategy,
  selectLayerSpacing,
  selectLayoutDirection,
  selectNodeAlignment,
  selectNodeSpacing,
  zLayeringStrategy,
  zLayoutDirection,
  zNodeAlignment,
} from 'features/nodes/store/workflowSettingsSlice';
import { type ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagicWandBold } from 'react-icons/pi';

const [useLayoutSettingsPopover] = buildUseBoolean(false);

export const AutoLayoutPopover = memo(() => {
  const { t } = useTranslation();
  const { fitView } = useReactFlow();
  const autoLayout = useAutoLayout();
  const dispatch = useAppDispatch();
  const popover = useLayoutSettingsPopover();
  const layeringStrategy = useAppSelector(selectLayeringStrategy);
  const nodeSpacing = useAppSelector(selectNodeSpacing);
  const layerSpacing = useAppSelector(selectLayerSpacing);
  const layoutDirection = useAppSelector(selectLayoutDirection);
  const nodeAlignment = useAppSelector(selectNodeAlignment);

  const handleLayeringStrategyChanged = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const val = zLayeringStrategy.parse(e.target.value);
      dispatch(layeringStrategyChanged(val));
    },
    [dispatch]
  );

  const handleNodeSpacingSliderChange = useCallback(
    (v: number) => {
      dispatch(nodeSpacingChanged(v));
    },
    [dispatch]
  );

  const handleNodeSpacingInputChange = useCallback(
    (v: number) => {
      dispatch(nodeSpacingChanged(v));
    },
    [dispatch]
  );

  const handleLayerSpacingSliderChange = useCallback(
    (v: number) => {
      dispatch(layerSpacingChanged(v));
    },
    [dispatch]
  );

  const handleLayerSpacingInputChange = useCallback(
    (v: number) => {
      dispatch(layerSpacingChanged(v));
    },
    [dispatch]
  );

  const handleLayoutDirectionChanged = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const val = zLayoutDirection.parse(e.target.value);
      dispatch(layoutDirectionChanged(val));
    },
    [dispatch]
  );

  const handleNodeAlignmentChanged = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const val = zNodeAlignment.parse(e.target.value);
      dispatch(nodeAlignmentChanged(val));
    },
    [dispatch]
  );

  const handleApplyAutoLayout = useCallback(() => {
    autoLayout();
    fitView({ duration: 300 });
    popover.setFalse();
  }, [autoLayout, fitView, popover]);

  return (
    <Popover isOpen={popover.isTrue} onClose={popover.setFalse} placement="top">
      <PopoverTrigger>
        <IconButton
          tooltip={t('nodes.layout.autoLayout')}
          aria-label={t('nodes.layout.autoLayout')}
          icon={<PiMagicWandBold />}
          onClick={popover.toggle}
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />

        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControl>
              <FormLabel>{t('nodes.layout.layoutDirection')}</FormLabel>
              <Select size="sm" value={layoutDirection} onChange={handleLayoutDirectionChanged}>
                <option value="LR">{t('nodes.layout.layoutDirectionRight')}</option>
                <option value="TB">{t('nodes.layout.layoutDirectionDown')}</option>
              </Select>
            </FormControl>
            <FormControl>
              <FormLabel>{t('nodes.layout.layeringStrategy')}</FormLabel>
              <Select size="sm" value={layeringStrategy} onChange={handleLayeringStrategyChanged}>
                <option value="network-simplex">{t('nodes.layout.networkSimplex')}</option>
                <option value="longest-path">{t('nodes.layout.longestPath')}</option>
              </Select>
            </FormControl>
            <FormControl>
              <FormLabel>{t('nodes.layout.alignment')}</FormLabel>
              <Select size="sm" value={nodeAlignment} onChange={handleNodeAlignmentChanged}>
                <option value="UL">{t('nodes.layout.alignmentUL')}</option>
                <option value="DL">{t('nodes.layout.alignmentDL')}</option>
                <option value="UR">{t('nodes.layout.alignmentUR')}</option>
                <option value="DR">{t('nodes.layout.alignmentDR')}</option>
              </Select>
            </FormControl>
            <Divider />
            <FormControl>
              <FormLabel>{t('nodes.layout.nodeSpacing')}</FormLabel>
              <Grid w="full" gap={2} templateColumns="1fr auto">
                <CompositeSlider min={0} max={200} value={nodeSpacing} onChange={handleNodeSpacingSliderChange} marks />
                <CompositeNumberInput
                  value={nodeSpacing}
                  min={0}
                  max={200}
                  onChange={handleNodeSpacingInputChange}
                  w={24}
                />
              </Grid>
            </FormControl>
            <FormControl>
              <FormLabel>{t('nodes.layout.layerSpacing')}</FormLabel>
              <Grid w="full" gap={2} templateColumns="1fr auto">
                <CompositeSlider
                  min={0}
                  max={200}
                  value={layerSpacing}
                  onChange={handleLayerSpacingSliderChange}
                  marks
                />
                <CompositeNumberInput
                  value={layerSpacing}
                  min={0}
                  max={200}
                  onChange={handleLayerSpacingInputChange}
                  w={24}
                />
              </Grid>
            </FormControl>
            <Divider />
            <Button w="full" onClick={handleApplyAutoLayout}>
              {t('common.apply')}
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
AutoLayoutPopover.displayName = 'AutoLayoutPopover';
