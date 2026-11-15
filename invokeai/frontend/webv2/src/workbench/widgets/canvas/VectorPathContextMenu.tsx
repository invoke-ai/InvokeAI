import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Menu, Portal } from '@chakra-ui/react';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { useNotify } from '@workbench/useNotify';
import { CopyIcon, PaintBucketIcon, ScanIcon, SquareArrowOutUpRightIcon, Trash2Icon, WavesIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useVectorEditSession } from './engineStoreHooks';
import { vectorTraceErrorKey } from './vectorTraceFeedback';

export interface VectorPathMenuTarget {
  x: number;
  y: number;
  layerId: string;
  pathId: string;
}

/** Path actions use the point-tool selection and commit through document history. */
export const VectorPathContextMenu = ({
  engine,
  target,
  onClose,
}: {
  engine: Pick<CanvasEngineHandle, 'layers' | 'interaction'>;
  target: VectorPathMenuTarget | null;
  onClose(): void;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const session = useVectorEditSession(engine);
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => ({ x: target?.x ?? 0, y: target?.y ?? 0, width: 1, height: 1 }),
      placement: 'bottom-start' as const,
    }),
    [target?.x, target?.y]
  );
  const onOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        onClose();
      }
    },
    [onClose]
  );
  if (!target || !session || session.layerId !== target.layerId || session.activePathId !== target.pathId) {
    return null;
  }
  const api = engine.layers.vectorEdit;
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const run = (action: () => void) => () => {
    const live = engine.interaction.get('vectorEditSession');
    if (live && live.layerId === target.layerId && live.activePathId === target.pathId) {
      action();
    }
    onClose();
  };
  return (
    <Menu.Root lazyMount open unmountOnExit positioning={positioning} onOpenChange={onOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="14rem" py="1">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel>{label('pathTitle')}</Menu.ItemGroupLabel>
              <MenuActionItem
                value="transform-path"
                icon={ScanIcon}
                label={label('transformPath')}
                onSelect={run(api.transformPath)}
              />
              <MenuActionItem
                value="trace-path"
                icon={WavesIcon}
                label={label('tracePath')}
                disabled={!session.paths.some((path) => path.id === target.pathId && path.points.length >= 2)}
                onSelect={run(async () => {
                  const errorKey = vectorTraceErrorKey(
                    await engine.layers.traceVectorLayer(target.layerId, target.pathId)
                  );
                  if (errorKey) {
                    notify.error(label(errorKey));
                  }
                })}
              />
              <MenuActionItem
                value="fill-path"
                icon={PaintBucketIcon}
                label={label('fillPath')}
                disabled={
                  !session.paths.some((path) => path.id === target.pathId && path.isClosed && path.points.length >= 3)
                }
                onSelect={run(async () => {
                  const errorKey = vectorTraceErrorKey(
                    await engine.layers.fillVectorPath(target.layerId, target.pathId)
                  );
                  if (errorKey) {
                    notify.error(label(errorKey));
                  }
                })}
              />
              <MenuActionItem
                value="extract-path"
                icon={SquareArrowOutUpRightIcon}
                label={label('extractPath')}
                onSelect={run(api.extractPath)}
              />
              <MenuActionItem value="copy-path" icon={CopyIcon} label={label('copy')} onSelect={run(api.copyPath)} />
              <MenuActionItem
                value="delete-path"
                icon={Trash2Icon}
                label={label('deletePath')}
                tone="danger"
                onSelect={run(api.deletePath)}
              />
            </Menu.ItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
