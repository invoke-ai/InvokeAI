import {
  Button,
  ButtonGroup,
  CompositeSlider,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  IconButton,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverFooter,
  PopoverTrigger,
  Select,
} from '@invoke-ai/ui-library';
import { useReactFlow } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { buildUseBoolean } from 'common/hooks/useBoolean';
import { useAutoLayout } from 'features/nodes/hooks/useAutoLayout';
import {
  type LayeringStrategy,
  layeringStrategyChanged,
  layerSpacingChanged,
  type LayoutDirection,
  layoutDirectionChanged,
  nodeAlignmentChanged,
  type NodeAlignment,
  nodeSpacingChanged,
  selectLayeringStrategy,
  selectLayerSpacing,
  selectLayoutDirection,
  selectNodeAlignment,
  selectNodeSpacing,
  selectShouldShowMinimapPanel,
  shouldShowMinimapPanelChanged,
} from 'features/nodes/store/workflowSettingsSlice';
import { type ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiFrameCornersBold,
  PiGitDiffBold,
  PiMagnifyingGlassMinusBold,
  PiMagnifyingGlassPlusBold,
  PiMapPinBold,
} from 'react-icons/pi';

const [useLayoutSettingsPopover] = buildUseBoolean(false);

const ViewportControls = () => {
  const { t } = useTranslation();
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const autoLayout = useAutoLayout();
  const dispatch = useAppDispatch();
  const popover = useLayoutSettingsPopover();
  const shouldShowMinimapPanel = useAppSelector(selectShouldShowMinimapPanel);
  const layeringStrategy = useAppSelector(selectLayeringStrategy);
  const nodeSpacing = useAppSelector(selectNodeSpacing);
  const layerSpacing = useAppSelector(selectLayerSpacing);
  const layoutDirection = useAppSelector(selectLayoutDirection);
  const nodeAlignment = useAppSelector(selectNodeAlignment);

  const handleClickedZoomIn = useCallback(() => {
    zoomIn({ duration: 300 });
  }, [zoomIn]);

  const handleClickedZoomOut = useCallback(() => {
    zoomOut({ duration: 300 });
  }, [zoomOut]);

  const handleClickedFitView = useCallback(() => {
    fitView({ duration: 300 });
  }, [fitView]);

  const handleClickedToggleMiniMapPanel = useCallback(() => {
    dispatch(shouldShowMinimapPanelChanged(!shouldShowMinimapPanel));
  }, [shouldShowMinimapPanel, dispatch]);

  const handleLayeringStrategyChanged = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      dispatch(layeringStrategyChanged(e.target.value as LayeringStrategy));
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
    (_: string, v: number) => {
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
    (_: string, v: number) => {
      dispatch(layerSpacingChanged(v));
    },
    [dispatch]
  );

  const handleLayoutDirectionChanged = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      dispatch(layoutDirectionChanged(e.target.value as LayoutDirection));
    },
    [dispatch]
  );

  const handleNodeAlignmentChanged = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value as NodeAlignment;
      dispatch(nodeAlignmentChanged(value));
    },
    [dispatch]
  );

  const handleApplyAutoLayout = useCallback(async () => {
    await autoLayout();
    fitView({ duration: 300 });
    popover.setFalse();
  }, [autoLayout, fitView, popover]);

  return (
    <ButtonGroup orientation="vertical">
      <IconButton
        tooltip={t('nodes.zoomInNodes')}
        aria-label={t('nodes.zoomInNodes')}
        onClick={handleClickedZoomIn}
        icon={<PiMagnifyingGlassPlusBold />}
      />
      <IconButton
        tooltip={t('nodes.zoomOutNodes')}
        aria-label={t('nodes.zoomOutNodes')}
        onClick={handleClickedZoomOut}
        icon={<PiMagnifyingGlassMinusBold />}
      />
      <IconButton
        tooltip={t('nodes.fitViewportNodes')}
        aria-label={t('nodes.fitViewportNodes')}
        onClick={handleClickedFitView}
        icon={<PiFrameCornersBold />}
      />
      <Popover isOpen={popover.isTrue} onClose={popover.setFalse} placement="top">
        <PopoverTrigger>
          <IconButton
            tooltip={t('nodes.layout.autoLayout')}
            aria-label={t('nodes.layout.autoLayout')}
            icon={<PiGitDiffBold />}
            onClick={popover.toggle}
          />
        </PopoverTrigger>
        <PopoverContent>
          <PopoverArrow />

          <PopoverBody>
            <Flex direction="column" gap={2}>
              <FormControl>
                <FormLabel>{t('nodes.layout.layoutDirection')}</FormLabel>
                <Select value={layoutDirection} onChange={handleLayoutDirectionChanged}>
                  <option value="LR">{t('nodes.layout.layoutDirectionRight')}</option>
                  <option value="TB">{t('nodes.layout.layoutDirectionDown')}</option>
                </Select>
              </FormControl>
              <FormControl>
                <FormLabel>{t('nodes.layout.layeringStrategy')}</FormLabel>
                <Select value={layeringStrategy} onChange={handleLayeringStrategyChanged}>
                  <option value="network-simplex">{t('nodes.layout.networkSimplex')}</option>
                  <option value="longest-path">{t('nodes.layout.longestPath')}</option>
                </Select>
              </FormControl>
              <FormControl>
                <FormLabel>{t('nodes.layout.alignment')}</FormLabel>
                <Select value={nodeAlignment} onChange={handleNodeAlignmentChanged}>
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
                  <CompositeSlider
                    min={0}
                    max={200}
                    value={nodeSpacing}
                    onChange={handleNodeSpacingSliderChange}
                    marks
                  />
                  <NumberInput
                    size="sm"
                    value={nodeSpacing}
                    min={0}
                    max={200}
                    onChange={handleNodeSpacingInputChange}
                    w={24}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
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
                  <NumberInput
                    size="sm"
                    value={layerSpacing}
                    min={0}
                    max={200}
                    onChange={handleLayerSpacingInputChange}
                    w={24}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </Grid>
              </FormControl>
            </Flex>
          </PopoverBody>
          <PopoverFooter>
            <Button w="full" onClick={handleApplyAutoLayout}>
              {t('common.apply')}
            </Button>
          </PopoverFooter>
        </PopoverContent>
      </Popover>
      {/* <Tooltip
        label={
          shouldShowFieldTypeLegend
            ? t('nodes.hideLegendNodes')
            : t('nodes.showLegendNodes')
        }
      >
        <IconButton
          aria-label="Toggle field type legend"
          isChecked={shouldShowFieldTypeLegend}
          onClick={handleClickedToggleFieldTypeLegend}
          icon={<FaInfo />}
        />
      </Tooltip> */}
      <IconButton
        tooltip={shouldShowMinimapPanel ? t('nodes.hideMinimapnodes') : t('nodes.showMinimapnodes')}
        aria-label={shouldShowMinimapPanel ? t('nodes.hideMinimapnodes') : t('nodes.showMinimapnodes')}
        isChecked={shouldShowMinimapPanel}
        onClick={handleClickedToggleMiniMapPanel}
        icon={<PiMapPinBold />}
      />
    </ButtonGroup>
  );
};

export default memo(ViewportControls);
