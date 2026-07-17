import type { GenerateWidgetValues } from '@workbench/generation/types';

import { Dialog, Icon, Portal } from '@chakra-ui/react';
import { Button, CloseButton, JsonPreview } from '@workbench/components/ui';
import { isSupportedGenerateModel } from '@workbench/generation/baseGenerationPolicies';
import {
  getCurrentGenerateValues,
  getImageRecallTitle,
  RecallActionButtons,
  type ImageRecallKind,
} from '@workbench/image-actions';
import { ensureModelsLoaded, useModelsSelector } from '@workbench/models/modelsStore';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { createGenerateFormValuesSelector } from '@workbench/widgets/generate/generateFormViewModel';
import { useWidgetValuesSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { FileTextIcon, WandSparklesIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueServerItem } from './queueServerApi';

import { extractGenerationMeta } from './fieldValues';
import { buildQueueRecallValues, getQueueRecallCapabilities } from './queueRecall';

const selectGenerateRecallValues = createGenerateFormValuesSelector();

/**
 * Per-item actions for the RECENT details panel. Recall uses the shared
 * {@link RecallActionButtons} verbs (same look as the preview's metadata
 * panel): items this client submitted recall from the exact submission
 * snapshot; foreign items still offer prompts + the executed seed from the
 * session. "View JSON" opens the raw queue item in a dialog.
 */
export const QueueItemActions = ({
  item,
  localGenerateValues,
}: {
  item: QueueServerItem;
  localGenerateValues: GenerateWidgetValues | null;
}) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();
  const openWidget = useOpenWorkbenchWidget();
  const notify = useNotify();
  const [jsonOpen, setJsonOpen] = useState(false);
  const generateValues = useWidgetValuesSelector('generate', selectGenerateRecallValues);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const meta = useMemo(() => extractGenerationMeta(item), [item]);
  const capabilities = useMemo(
    () => getQueueRecallCapabilities(localGenerateValues, meta),
    [localGenerateValues, meta]
  );

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  const onRecall = useCallback(
    (kind: ImageRecallKind) => {
      const current = getCurrentGenerateValues({ generateValues, supportedModels });
      const values = buildQueueRecallValues(kind, { current, meta, snapshot: localGenerateValues });

      if (!values) {
        notify.info(getImageRecallTitle(kind), t('widgets.queue.recallUnavailable'));
        return;
      }

      dispatch({ type: 'setGenerateSettings', values });
      openWidget('generate', { preferredRegions: ['left'] });
      notify.success(getImageRecallTitle(kind), t('widgets.queue.settingsRecalledDescription'));
    },
    [dispatch, generateValues, localGenerateValues, meta, notify, openWidget, supportedModels, t]
  );

  const onSendToCanvas = useCallback(
    () => notify.info(t('widgets.queue.sendToCanvas'), t('widgets.queue.sendToCanvasComingSoon')),
    [notify, t]
  );

  const openJson = useCallback(() => setJsonOpen(true), []);
  const closeJson = useCallback(() => setJsonOpen(false), []);

  return (
    <>
      <RecallActionButtons
        capabilities={capabilities}
        disabledReason={t('widgets.queue.recallFromGallery')}
        onRecall={onRecall}
      >
        <Button disabled variant="ghost" onClick={onSendToCanvas}>
          <Icon as={WandSparklesIcon} boxSize="3" />
          {t('widgets.queue.sendToCanvas')}
        </Button>
        <Button onClick={openJson}>
          <Icon as={FileTextIcon} boxSize="3" />
          {t('common.viewJson')}
        </Button>
      </RecallActionButtons>

      <Dialog.Root open={jsonOpen} placement="center" scrollBehavior="inside" size="lg" onOpenChange={closeJson}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg">
              <Dialog.Header>
                <Dialog.Title fontSize="sm" fontWeight="700">
                  {t('widgets.queue.itemTitle', { id: item.item_id })}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <JsonPreview label={t('widgets.queue.itemJsonLabel', { id: item.item_id })} maxH="60vh" value={item} />
              </Dialog.Body>
              <Dialog.CloseTrigger asChild>
                <CloseButton color="fg.muted" size="sm" />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </>
  );
};
