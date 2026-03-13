import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useReactFlow } from '@xyflow/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import {
  selectShouldShowMinimapPanel,
  shouldShowMinimapPanelChanged,
} from 'features/nodes/store/workflowSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiFrameCornersBold,
  PiMagnifyingGlassMinusBold,
  PiMagnifyingGlassPlusBold,
  PiMapPinBold,
} from 'react-icons/pi';

import { AutoLayoutPopover } from './AutoLayoutPopover';

const ViewportControls = () => {
  const { t } = useTranslation();
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const dispatch = useAppDispatch();
  const shouldShowMinimapPanel = useAppSelector(selectShouldShowMinimapPanel);

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

  return (
    <ButtonGroup orientation="vertical">
      <IAITooltip label={t('nodes.zoomInNodes')}>
        <IconButton
          aria-label={t('nodes.zoomInNodes')}
          onClick={handleClickedZoomIn}
          icon={<PiMagnifyingGlassPlusBold />}
        />
      </IAITooltip>
      <IAITooltip label={t('nodes.zoomOutNodes')}>
        <IconButton
          aria-label={t('nodes.zoomOutNodes')}
          onClick={handleClickedZoomOut}
          icon={<PiMagnifyingGlassMinusBold />}
        />
      </IAITooltip>
      <IAITooltip label={t('nodes.fitViewportNodes')}>
        <IconButton
          aria-label={t('nodes.fitViewportNodes')}
          onClick={handleClickedFitView}
          icon={<PiFrameCornersBold />}
        />
      </IAITooltip>
      <AutoLayoutPopover />
      <IAITooltip label={shouldShowMinimapPanel ? t('nodes.hideMinimapnodes') : t('nodes.showMinimapnodes')}>
        <IconButton
          aria-label={shouldShowMinimapPanel ? t('nodes.hideMinimapnodes') : t('nodes.showMinimapnodes')}
          isChecked={shouldShowMinimapPanel}
          onClick={handleClickedToggleMiniMapPanel}
          icon={<PiMapPinBold />}
        />
      </IAITooltip>
    </ButtonGroup>
  );
};

export default memo(ViewportControls);
