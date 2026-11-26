import { Flex, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { getDocumentNode, lookupDocumentLeaf } from '@workbench/canvas-engine/api';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { useCanvasHasFloatingSelection, useTransformSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { LayerPositionRow, LayerTransformRows } from '@workbench/widgets/canvas/tool-options/layerTransformFields';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { GroupSelectedNotice } from './GroupSelectedNotice';
import { PropertiesSection } from './PropertiesSection';

interface SelectedNode {
  id: string;
  isGroup: boolean;
  name: string;
}

/** Each field commits one patch; a Transform session absorbs them until its Apply / Cancel. */
export const TransformPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();

  if (!engine) {
    return (
      <Flex align="center" color="fg.muted" fontSize="md" h="full" justify="center" p="4">
        {t('widgets.properties.noCanvas')}
      </Flex>
    );
  }
  return (
    <Scrollable h="full">
      <ConnectedTransform engine={engine} />
    </Scrollable>
  );
};

const ConnectedTransform = ({ engine }: { engine: CanvasEngineHandle }) => {
  const { t } = useTranslation();
  const session = useTransformSession(engine);
  const hasFloat = useCanvasHasFloatingSelection(engine);
  const selected = useActiveProjectSelector(
    (project): SelectedNode | null => {
      const { document } = project.canvas;
      const node = getDocumentNode(document, document.selectedLayerId);
      if (node?.type === 'group') {
        return { id: node.id, isGroup: true, name: node.name };
      }
      const leaf = node ? lookupDocumentLeaf(document, node.id) : null;
      return leaf ? { id: leaf.id, isGroup: false, name: leaf.layer.name } : null;
    },
    (a, b) => a?.id === b?.id && a?.isGroup === b?.isGroup && a?.name === b?.name
  );
  const processing = session?.processing ?? false;
  const onApply = useCallback(() => engine.layers.applyTransform(), [engine]);
  const onCancel = useCallback(() => engine.layers.cancelTransform(), [engine]);
  const pending = session !== null || hasFloat;

  return (
    <Stack gap="0">
      <PropertiesSection
        // A draft typed for one layer must not commit to the next selection.
        key={session ? 'session' : (selected?.id ?? '')}
        subtitle={session ? t('widgets.transform.session') : (selected?.name ?? t('widgets.transform.noSelection'))}
        title={t('widgets.transform.title')}
      >
        {selected?.isGroup && !session ? <GroupSelectedNotice hint={t('widgets.transform.groupSelected')} /> : null}
        <LayerPositionRow engine={engine} showLabel />
        <LayerTransformRows engine={engine} />
        {pending ? (
          <Flex gap="2" justify="flex-end">
            <Button variant="ghost" onClick={onCancel}>
              {t('common.cancel')}
            </Button>
            <Button disabled={processing} loading={processing} onClick={onApply}>
              {t('common.apply')}
            </Button>
          </Flex>
        ) : (
          <Text color="fg.muted" fontSize="xs">
            {t('widgets.transform.hint')}
          </Text>
        )}
      </PropertiesSection>
    </Stack>
  );
};
