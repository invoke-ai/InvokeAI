import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  // shouldShowFieldTypeLegendChanged,
  shouldShowMinimapPanelChanged,
} from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiFrameCornersBold,
  PiMagnifyingGlassMinusBold,
  PiMagnifyingGlassPlusBold,
  PiMapPinBold,
} from 'react-icons/pi';
import { useReactFlow } from 'reactflow';

const ViewportControls = () => {
  const { t } = useTranslation();
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const dispatch = useAppDispatch();
  // const shouldShowFieldTypeLegend = useAppSelector(
  //   (s) => s.nodes.shouldShowFieldTypeLegend
  // );
  const shouldShowMinimapPanel = useAppSelector((s) => s.nodes.shouldShowMinimapPanel);

  const handleClickedZoomIn = useCallback(() => {
    zoomIn();
  }, [zoomIn]);

  const handleClickedZoomOut = useCallback(() => {
    zoomOut();
  }, [zoomOut]);

  const handleClickedFitView = useCallback(() => {
    fitView();
  }, [fitView]);

  // const handleClickedToggleFieldTypeLegend = useCallback(() => {
  //   dispatch(shouldShowFieldTypeLegendChanged(!shouldShowFieldTypeLegend));
  // }, [shouldShowFieldTypeLegend, dispatch]);

  const handleClickedToggleMiniMapPanel = useCallback(() => {
    dispatch(shouldShowMinimapPanelChanged(!shouldShowMinimapPanel));
  }, [shouldShowMinimapPanel, dispatch]);

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
