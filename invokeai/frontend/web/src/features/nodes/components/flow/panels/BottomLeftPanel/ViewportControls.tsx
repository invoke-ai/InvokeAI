import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import {
  // shouldShowFieldTypeLegendChanged,
  shouldShowMinimapPanelChanged,
} from 'features/nodes/store/nodesSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  FaExpand,
  // FaInfo,
  FaMapMarkerAlt,
} from 'react-icons/fa';
import { FaMagnifyingGlassMinus, FaMagnifyingGlassPlus } from 'react-icons/fa6';
import { useReactFlow } from 'reactflow';

const ViewportControls = () => {
  const { t } = useTranslation();
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const dispatch = useAppDispatch();
  // const shouldShowFieldTypeLegend = useAppSelector(
  //   (state) => state.nodes.shouldShowFieldTypeLegend
  // );
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

  // const handleClickedToggleFieldTypeLegend = useCallback(() => {
  //   dispatch(shouldShowFieldTypeLegendChanged(!shouldShowFieldTypeLegend));
  // }, [shouldShowFieldTypeLegend, dispatch]);

  const handleClickedToggleMiniMapPanel = useCallback(() => {
    dispatch(shouldShowMinimapPanelChanged(!shouldShowMinimapPanel));
  }, [shouldShowMinimapPanel, dispatch]);

  return (
    <InvButtonGroup orientation="vertical">
      <InvIconButton
        tooltip={t('nodes.zoomInNodes')}
        aria-label={t('nodes.zoomInNodes')}
        onClick={handleClickedZoomIn}
        icon={<FaMagnifyingGlassPlus />}
      />
      <InvIconButton
        tooltip={t('nodes.zoomOutNodes')}
        aria-label={t('nodes.zoomOutNodes')}
        onClick={handleClickedZoomOut}
        icon={<FaMagnifyingGlassMinus />}
      />
      <InvIconButton
        tooltip={t('nodes.fitViewportNodes')}
        aria-label={t('nodes.fitViewportNodes')}
        onClick={handleClickedFitView}
        icon={<FaExpand />}
      />
      {/* <InvTooltip
        label={
          shouldShowFieldTypeLegend
            ? t('nodes.hideLegendNodes')
            : t('nodes.showLegendNodes')
        }
      >
        <InvIconButton
          aria-label="Toggle field type legend"
          isChecked={shouldShowFieldTypeLegend}
          onClick={handleClickedToggleFieldTypeLegend}
          icon={<FaInfo />}
        />
      </InvTooltip> */}
      <InvIconButton
        tooltip={
          shouldShowMinimapPanel
            ? t('nodes.hideMinimapnodes')
            : t('nodes.showMinimapnodes')
        }
        aria-label={
          shouldShowMinimapPanel
            ? t('nodes.hideMinimapnodes')
            : t('nodes.showMinimapnodes')
        }
        isChecked={shouldShowMinimapPanel}
        onClick={handleClickedToggleMiniMapPanel}
        icon={<FaMapMarkerAlt />}
      />
    </InvButtonGroup>
  );
};

export default memo(ViewportControls);
