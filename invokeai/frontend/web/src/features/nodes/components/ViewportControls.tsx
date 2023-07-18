import { ButtonGroup } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { memo, useCallback } from 'react';
import {
  FaCode,
  FaExpand,
  FaMinus,
  FaPlus,
  FaInfo,
  FaMapMarkerAlt,
} from 'react-icons/fa';
import { useReactFlow } from 'reactflow';
import {
  shouldShowGraphOverlayChanged,
  shouldShowFieldTypeLegendChanged,
  shouldShowMinimapPanelChanged,
} from '../store/nodesSlice';

const ViewportControls = () => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const dispatch = useAppDispatch();
  const shouldShowGraphOverlay = useAppSelector(
    (state) => state.nodes.shouldShowGraphOverlay
  );
  const shouldShowFieldTypeLegend = useAppSelector(
    (state) => state.nodes.shouldShowFieldTypeLegend
  );

  const shouldShowMinimapPanel = useAppSelector(
    (state) => state.nodes.shouldShowMinimapPanel
  );

  const handleClickedZoomIn = useCallback(() => {
    zoomIn();
  }, [zoomIn]);

  const handleClickedZoomOut = useCallback(() => {
    zoomOut();
  }, [zoomOut]);

  const handleClickedFitView = useCallback(() => {
    fitView();
  }, [fitView]);

  const handleClickedToggleGraphOverlay = useCallback(() => {
    dispatch(shouldShowGraphOverlayChanged(!shouldShowGraphOverlay));
  }, [shouldShowGraphOverlay, dispatch]);

  const handleClickedToggleFieldTypeLegend = useCallback(() => {
    dispatch(shouldShowFieldTypeLegendChanged(!shouldShowFieldTypeLegend));
  }, [shouldShowFieldTypeLegend, dispatch]);

  const handleClickedToggleMiniMapPanel = useCallback(() => {
    dispatch(shouldShowMinimapPanelChanged(!shouldShowMinimapPanel));
  }, [shouldShowMinimapPanel, dispatch]);

  return (
    <ButtonGroup isAttached orientation="vertical">
      <IAIIconButton
        onClick={handleClickedZoomIn}
        aria-label="Zoom In"
        icon={<FaPlus />}
      />
      <IAIIconButton
        onClick={handleClickedZoomOut}
        aria-label="Zoom Out"
        icon={<FaMinus />}
      />
      <IAIIconButton
        onClick={handleClickedFitView}
        aria-label="Fit to Viewport"
        icon={<FaExpand />}
      />
      <IAIIconButton
        isChecked={shouldShowGraphOverlay}
        onClick={handleClickedToggleGraphOverlay}
        aria-label="Show/Hide Graph"
        icon={<FaCode />}
      />
      <IAIIconButton
        isChecked={shouldShowFieldTypeLegend}
        onClick={handleClickedToggleFieldTypeLegend}
        aria-label="Show/Hide Field Type Legend"
        icon={<FaInfo />}
      />
      <IAIIconButton
        isChecked={shouldShowMinimapPanel}
        onClick={handleClickedToggleMiniMapPanel}
        aria-label="Show/Hide Minimap"
        icon={<FaMapMarkerAlt />}
      />
    </ButtonGroup>
  );
};

export default memo(ViewportControls);
