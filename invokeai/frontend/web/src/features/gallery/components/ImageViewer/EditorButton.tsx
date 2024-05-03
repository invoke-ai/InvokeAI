import { IconButton } from '@chakra-ui/react';
import { Button } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsDownUpBold } from 'react-icons/pi';

import { useImageViewer } from './useImageViewer';

// const TAB_NAME_TO_TKEY: Record<InvokeTabName, string> = {
//   generation: 'ui.tabs.generationTab',
//   canvas: 'ui.tabs.canvasTab',
//   workflows: 'ui.tabs.workflowsTab',
//   models: 'ui.tabs.modelsTab',
//   queue: 'ui.tabs.queueTab',
// };

const TAB_NAME_TO_TKEY_SHORT: Record<InvokeTabName, string> = {
  generation: 'ui.tabs.generation',
  canvas: 'ui.tabs.canvas',
  workflows: 'ui.tabs.workflows',
  models: 'ui.tabs.models',
  queue: 'ui.tabs.queue',
};

export const EditorButton = () => {
  const { t } = useTranslation();
  const { onClose } = useImageViewer();
  const activeTabName = useAppSelector(activeTabNameSelector);

  const tooltip = useMemo(
    () =>
      t('gallery.switchTo', {
        tab: activeTabName === 'generation' ? t('common.controlLayers') : t(TAB_NAME_TO_TKEY_SHORT[activeTabName]),
      }),
    [t, activeTabName]
  );

  return (
    <Button aria-label={tooltip} tooltip={tooltip} onClick={onClose} variant="outline" sx={{ display: 'flex', gap: 2 }}>
      <IconButton aria-label={tooltip} variant="ghost" size="sm" icon={<PiArrowsDownUpBold />} />
      {activeTabName === 'generation' ? t('common.controlLayers') : t(TAB_NAME_TO_TKEY_SHORT[activeTabName])}
    </Button>
  );
};
