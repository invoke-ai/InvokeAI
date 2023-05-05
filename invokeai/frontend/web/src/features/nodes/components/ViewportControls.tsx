import { ButtonGroup } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { memo, useCallback } from 'react';
import { FaCode, FaExpand, FaMinus, FaPlus } from 'react-icons/fa';
import { useReactFlow } from 'reactflow';
import { shouldShowGraphOverlayChanged } from '../store/nodesSlice';

const ViewportControls = () => {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const dispatch = useAppDispatch();
  const shouldShowGraphOverlay = useAppSelector(
    (state) => state.nodes.shouldShowGraphOverlay
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
    </ButtonGroup>
  );
};

export default memo(ViewportControls);
