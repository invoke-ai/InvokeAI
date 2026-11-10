import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { Menu, Portal } from '@chakra-ui/react';
import { MenuActionItem, MenuContent } from '@platform/ui/menu';
import { CopyIcon, ScanIcon, SquareArrowOutUpRightIcon, Trash2Icon, WavesIcon } from 'lucide-react';
import { useNotify } from '@workbench/useNotify';
import { useTranslation } from 'react-i18next';

import { useVectorEditSession } from './engineStoreHooks';


export interface VectorPathMenuTarget {
  x: number;
  y: number;
  layerId: string;
  pathId: string;
}

/** Path actions use the point-tool selection and commit through document history. */
export const VectorPathContextMenu = ({ engine, target, onClose }: {
  engine: Pick<CanvasEngineHandle, 'layers' | 'interaction'>;
  target: VectorPathMenuTarget | null;
  onClose(): void;
}) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const session = useVectorEditSession(engine);
  if (!target || !session || session.layerId !== target.layerId || session.activePathId !== target.pathId) { return null; }
  const api = engine.layers.vectorEdit;
  const label = (key: string) => t(`widgets.layers.vector.${key}`);
  const run = (action: () => void) => () => {
    const live = engine.interaction.get('vectorEditSession');
    if (live && live.layerId === target.layerId && live.activePathId === target.pathId) { action(); }
    onClose();
  };
  return (
    <Menu.Root lazyMount open unmountOnExit
      positioning={{ getAnchorRect: () => ({ x: target.x, y: target.y, width: 1, height: 1 }), placement: 'bottom-start' }}
      onOpenChange={({ open }) => { if (!open) { onClose(); } }}>
      <Portal><Menu.Positioner><MenuContent minW="14rem" py="1">
        <Menu.ItemGroup>
          <Menu.ItemGroupLabel>{label('pathTitle')}</Menu.ItemGroupLabel>
          <MenuActionItem value="transform-path" icon={ScanIcon} label={label('transformPath')} onSelect={run(api.transformPath)} />
          <MenuActionItem value="trace-path" icon={WavesIcon} label={label('tracePath')}
            disabled={!session.paths.some((path) => path.id === target.pathId && path.points.length >= 2)}
            onSelect={run(() => {
              if (!engine.layers.traceVectorLayer(target.layerId, target.pathId)) { notify.error(label('traceFailed')); }
            })} />
          <MenuActionItem value="extract-path" icon={SquareArrowOutUpRightIcon} label={label('extractPath')} onSelect={run(api.extractPath)} />
          <MenuActionItem value="copy-path" icon={CopyIcon} label={label('copy')} onSelect={run(api.copyPath)} />
          <MenuActionItem value="delete-path" icon={Trash2Icon} label={label('deletePath')} tone="danger" onSelect={run(api.deletePath)} />
        </Menu.ItemGroup>
      </MenuContent></Menu.Positioner></Portal>
    </Menu.Root>
  );
};
