import { ButtonGroup, Tooltip } from '@chakra-ui/react';
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
import { useTranslation } from 'react-i18next';
import {
  shouldShowGraphOverlayChanged,
  shouldShowFieldTypeLegendChanged,
  shouldShowMinimapPanelChanged,
} from '../store/nodesSlice';

const ViewportControls = () => {
  const { t } = useTranslation();
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
      <Tooltip label={t('nodes.zoomInNodes')}>
        <IAIIconButton onClick={handleClickedZoomIn} icon={<FaPlus />} />
      </Tooltip>
      <Tooltip label={t('nodes.zoomOutNodes')}>
        <IAIIconButton onClick={handleClickedZoomOut} icon={<FaMinus />} />
      </Tooltip>
      <Tooltip label={t('nodes.fitViewportNodes')}>
        <IAIIconButton onClick={handleClickedFitView} icon={<FaExpand />} />
      </Tooltip>
      <Tooltip
        label={
          shouldShowGraphOverlay
            ? t('nodes.hideGraphNodes')
            : t('nodes.showGraphNodes')
        }
      >
        <IAIIconButton
          isChecked={shouldShowGraphOverlay}
          onClick={handleClickedToggleGraphOverlay}
          icon={<FaCode />}
        />
      </Tooltip>
      <Tooltip
        label={
          shouldShowFieldTypeLegend
            ? t('nodes.hideLegendNodes')
            : t('nodes.showLegendNodes')
        }
      >
        <IAIIconButton
          isChecked={shouldShowFieldTypeLegend}
          onClick={handleClickedToggleFieldTypeLegend}
          icon={<FaInfo />}
        />
      </Tooltip>
      <Tooltip
        label={
          shouldShowMinimapPanel
            ? t('nodes.hideMinimapnodes')
            : t('nodes.showMinimapnodes')
        }
      >
        <IAIIconButton
          isChecked={shouldShowMinimapPanel}
          onClick={handleClickedToggleMiniMapPanel}
          icon={<FaMapMarkerAlt />}
        />
      </Tooltip>
    </ButtonGroup>
  );
};

export default memo(ViewportControls);
