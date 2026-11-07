import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Flex, Portal, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { useVectorEditSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

type Engine = Pick<CanvasEngineHandle, 'layers' | 'interaction'>;

export const VectorLayerSettings = ({ engine, layer }: {
  engine: Engine | null;
  layer: Extract<CanvasLayerContract, { type: 'vector' }>;
}) => {
  const { t } = useTranslation();
  const session = useVectorEditSession(engine);
  const [error, setError] = useState(false);
  const editing = session?.layerId === layer.id ? session : null;
  const api = engine?.layers.vectorEdit;
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const selected = editing && editing.selectedPoints.length > 0;
  const active = editing?.paths.find((path) => path.id === editing.activePathId);
  return (
    <Stack gap="2">
      <Text color="fg.muted" fontSize="sm">{t('widgets.layers.summary.vectorPaths', { count: editing?.paths.length ?? layer.paths.length })}</Text>
      {!editing ? (
        <Button disabled={!api || !!session || layer.paths.length === 0} onClick={() => setError(!api?.begin(layer.id))} size="sm">{label('edit')}</Button>
      ) : (
        <>
          <Flex gap="2" wrap="wrap">
            <Button onClick={() => { const result = api?.apply(); setError(!!result && result.status !== 'committed'); }} size="sm">{label('apply')}</Button>
            <Button onClick={() => api?.cancel()} size="sm" variant="outline">{label('cancel')}</Button>
            <Button onClick={() => api?.reset()} size="sm" variant="outline">{label('reset')}</Button>
          </Flex>
          <Flex gap="2">
            <Button disabled={!editing.canUndo} onClick={() => api?.undo()} size="xs" variant="outline">{label('undo')}</Button>
            <Button disabled={!editing.canRedo} onClick={() => api?.redo()} size="xs" variant="outline">{label('redo')}</Button>
          </Flex>
          <Text color="fg.muted" fontSize="sm">{label('hint')}</Text>
          <Flex gap="1" wrap="wrap">
            {(['corner', 'smooth', 'symmetric'] as const).map((type) => (
              <Button key={type} disabled={!selected} onClick={() => api?.setPointType(type)} size="xs" variant="outline">{label(type)}</Button>
            ))}
          </Flex>
          <Flex gap="1" wrap="wrap">
            <Button disabled={!active} onClick={() => api?.smooth(false)} size="xs" variant="outline">{label('smoothPath')}</Button>
            <Button disabled={!selected} onClick={() => api?.smooth(true)} size="xs" variant="outline">{label('smoothPoints')}</Button>
            <Button disabled={!api?.canSplit()} onClick={() => api?.split()} size="xs" variant="outline">{label('split')}</Button>
            <Button disabled={!api?.canJoin()} onClick={() => api?.join()} size="xs" variant="outline">{label('join')}</Button>
            <Button disabled={!active} onClick={() => api?.copyPath()} size="xs" variant="outline">{label('copy')}</Button>
            <Button disabled={!selected && !active} onClick={() => api?.deleteSelection()} size="xs" variant="outline">{label('delete')}</Button>
          </Flex>
        </>
      )}
      {error ? <Text color="fg.error" fontSize="sm">{label('unavailable')}</Text> : null}
    </Stack>
  );
};

/** Mounted independently of the selected layer so a selection change cannot hide the exit decision. */
export const VectorEditExitDialog = ({ engine }: { engine: Engine }) => {
  const { t } = useTranslation();
  const session = useVectorEditSession(engine);
  const [error, setError] = useState(false);
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const resolve = (decision: 'apply' | 'discard' | 'cancel') => {
    const result = engine.layers.vectorEdit.resolveExit(decision);
    setError(!!result && result.status !== 'committed');
  };
  return (
    <Dialog.Root open={!!session?.pendingExit} onOpenChange={({ open }) => { if (!open) { resolve('cancel'); } }}>
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header><Dialog.Title>{label('exitTitle')}</Dialog.Title></Dialog.Header>
            <Dialog.Body>
              <Text>{label('exitBody')}</Text>
              {error ? <Text color="fg.error">{label('unavailable')}</Text> : null}
            </Dialog.Body>
            <Dialog.Footer>
              <Button onClick={() => resolve('cancel')} variant="outline">{label('cancel')}</Button>
              <Button onClick={() => resolve('discard')} variant="outline">{label('discard')}</Button>
              <Button onClick={() => resolve('apply')}>{label('apply')}</Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
