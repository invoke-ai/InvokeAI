import {
  Button,
  ButtonGroup,
  CompositeSlider,
  Divider,
  Flex,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverFooter,
  PopoverTrigger,
  Radio,
  RadioGroup,
  Text,
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
  type NodePlacementStrategy,
  nodePlacementStrategyChanged,
  nodeSpacingChanged,
  selectLayeringStrategy,
  selectLayerSpacing,
  selectLayoutDirection,
  selectNodePlacementStrategy,
  selectNodeSpacing,
  selectShouldShowMinimapPanel,
  shouldShowMinimapPanelChanged,
} from 'features/nodes/store/workflowSettingsSlice';
import { memo, useCallback } from 'react';
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
  const nodePlacementStrategy = useAppSelector(selectNodePlacementStrategy);
  const layeringStrategy = useAppSelector(selectLayeringStrategy);
  const nodeSpacing = useAppSelector(selectNodeSpacing);
  const layerSpacing = useAppSelector(selectLayerSpacing);
  const layoutDirection = useAppSelector(selectLayoutDirection);

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

  const handleStrategyChanged = useCallback(
    (value: NodePlacementStrategy) => {
      dispatch(nodePlacementStrategyChanged(value));
    },
    [dispatch]
  );

  const handleLayeringStrategyChanged = useCallback(
    (value: LayeringStrategy) => {
      dispatch(layeringStrategyChanged(value));
    },
    [dispatch]
  );

  const handleNodeSpacingChanged = useCallback(
    (v: number) => {
      dispatch(nodeSpacingChanged(v));
    },
    [dispatch]
  );

  const handleLayerSpacingChanged = useCallback(
    (v: number) => {
      dispatch(layerSpacingChanged(v));
    },
    [dispatch]
  );

  const handleLayoutDirectionChanged = useCallback(
    (value: LayoutDirection) => {
      dispatch(layoutDirectionChanged(value));
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
            <Flex direction="column" gap={4}>
              <Text fontWeight="semibold">{t('nodes.layout.nodePlacementStrategy')}</Text>
              <RadioGroup value={nodePlacementStrategy} onChange={handleStrategyChanged}>
                <Flex direction="column" gap={2}>
                  <Radio value="NETWORK_SIMPLEX">{t('nodes.layout.networkSimplex')}</Radio>
                  <Radio value="BRANDES_KOEPF">{t('nodes.layout.brandesKoepf')}</Radio>
                  <Radio value="LINEAR_SEGMENTS">{t('nodes.layout.linearSegments')}</Radio>
                  <Radio value="SIMPLE">{t('nodes.layout.simplePlacement')}</Radio>
                </Flex>
              </RadioGroup>
              <Divider />
              <Text fontWeight="semibold">{t('nodes.layout.layeringStrategy')}</Text>
              <RadioGroup value={layeringStrategy} onChange={handleLayeringStrategyChanged}>
                <Flex direction="column" gap={2}>
                  <Radio value="NETWORK_SIMPLEX">{t('nodes.layout.networkSimplex')}</Radio>
                  <Radio value="LONGEST_PATH">{t('nodes.layout.longestPath')}</Radio>
                  <Radio value="COFFMAN_GRAHAM">{t('nodes.layout.coffmanGraham')}</Radio>
                </Flex>
              </RadioGroup>
              <Divider />
              <Text fontWeight="semibold">{t('nodes.layout.layoutDirection')}</Text>
              <RadioGroup value={layoutDirection} onChange={handleLayoutDirectionChanged}>
                <Flex direction="column" gap={2}>
                  <Radio value="RIGHT">{t('nodes.layout.layoutDirectionRight')}</Radio>
                  <Radio value="DOWN">{t('nodes.layout.layoutDirectionDown')}</Radio>
                </Flex>
              </RadioGroup>
              <Divider />
              <Flex justifyContent="space-between" alignItems="center">
                <Text fontWeight="semibold">{t('nodes.layout.nodeSpacing')}</Text>
                <Text variant="subtext">{nodeSpacing}</Text>
              </Flex>
              <CompositeSlider min={0} max={200} value={nodeSpacing} onChange={handleNodeSpacingChanged} marks />
              <Flex justifyContent="space-between" alignItems="center">
                <Text fontWeight="semibold">{t('nodes.layout.layerSpacing')}</Text>
                <Text variant="subtext">{layerSpacing}</Text>
              </Flex>
              <CompositeSlider min={0} max={200} value={layerSpacing} onChange={handleLayerSpacingChanged} marks />
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
