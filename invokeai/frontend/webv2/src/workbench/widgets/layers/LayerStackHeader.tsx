import type { FeatureHintId } from '@platform/ui/hints/hintRegistry';
import type { CanvasDocumentContractV3, LayerStackKind } from '@workbench/canvas-engine/api';
import type { KeyboardEvent, MouseEvent } from 'react';

import { Box, HStack, Icon, Text, VisuallyHidden } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { FeatureHint, FeatureHintExclusion } from '@platform/ui/hints/FeatureHint';
import { ChevronDownIcon } from 'lucide-react';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerRowCommands } from './layerRowCommands';

import { anchorFromPoint } from './layerRowCommands';
import { useLayerStackActions, type LayerStackActionsEngine } from './useLayerStackActions';

const HEADER_FOCUS = { outline: '2px solid', outlineColor: 'accent.solid', outlineOffset: '-2px' };

const STACK_HINTS: Record<LayerStackKind, FeatureHintId> = {
  control: 'layerStackControl',
  inpaint_mask: 'layerStackInpaintMask',
  raster: 'layerStackRaster',
  regional_guidance: 'layerStackRegionalGuidance',
};

interface LayerStackHeaderProps {
  collapsed: boolean;
  commands: LayerRowCommands;
  document: CanvasDocumentContractV3;
  editingLocked: boolean;
  engine: LayerStackActionsEngine | null;
  focused: boolean;
  leafCount: number;
  /** The header's tree key, the id the tree addresses it by. */
  rowKey: string;
  posInSet: number;
  setSize: number;
  stack: LayerStackKind;
  /** A pinned copy is visual only; the tree's own header stays the accessible one. */
  pinned?: boolean;
}

const stopPropagation = (event: { stopPropagation: () => void }): void => event.stopPropagation();

/**
 * Stack headers own the roving tree stop; pointer-only action buttons remain keyboard-accessible through Shift+F10
 * menus.
 */
const LayerStackHeaderComponent = ({
  collapsed,
  commands,
  document,
  editingLocked,
  engine,
  focused,
  leafCount,
  pinned = false,
  posInSet,
  rowKey,
  setSize,
  stack,
}: LayerStackHeaderProps) => {
  const { t } = useTranslation();
  const actions = useLayerStackActions(stack, document, engine, editingLocked);
  const handleToggle = useCallback(
    (event: { stopPropagation: () => void }) => {
      event.stopPropagation();
      commands.toggleCollapse(stack);
    },
    [commands, stack]
  );
  const handleClick = useCallback(() => commands.toggleCollapse(stack), [commands, stack]);
  const handleFocus = useCallback(() => commands.focus(rowKey), [commands, rowKey]);
  const element = useRef<HTMLDivElement | null>(null);
  // A pressed control never takes focus from the tree item.
  const keepFocus = useCallback((event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    element.current?.focus();
  }, []);
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLElement>) => {
      if (event.target === event.currentTarget) {
        commands.keyDown(rowKey, event);
      }
    },
    [commands, rowKey]
  );
  const handleContextMenu = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      event.preventDefault();
      commands.openStackMenu(stack, anchorFromPoint(event.clientX, event.clientY));
    },
    [commands, stack]
  );
  const hintId = STACK_HINTS[stack];
  const descriptionId = `layer-stack-hint-${stack}`;
  return (
    // Attach hints to tree-item focus/hover with persistent described-by text; exclude action buttons with their
    // own tooltips.
    <FeatureHint hint={hintId}>
      <Box
        ref={element}
        aria-describedby={pinned ? undefined : descriptionId}
        aria-expanded={!collapsed}
        aria-hidden={pinned || undefined}
        aria-label={t(`widgets.layers.groups.${stack}`)}
        aria-level={1}
        aria-posinset={posInSet}
        aria-setsize={setSize}
        bg={pinned ? 'bg.panel' : 'transparent'}
        data-layer-row-id={pinned ? undefined : rowKey}
        h="full"
        role={pinned ? undefined : 'treeitem'}
        rounded="sm"
        tabIndex={pinned ? undefined : focused ? 0 : -1}
        _focusVisible={HEADER_FOCUS}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onFocus={pinned ? undefined : handleFocus}
        onKeyDown={pinned ? undefined : handleKeyDown}
      >
        {pinned ? null : <VisuallyHidden id={descriptionId}>{t(`hints.${hintId}.paragraphs.0`)}</VisuallyHidden>}
        <HStack gap="1" h="full" pe="1.5" ps="0">
          <IconButton
            aria-label={t(collapsed ? 'widgets.layers.groupActions.expand' : 'widgets.layers.groupActions.collapse')}
            color="fg.muted"
            size="2xs"
            tabIndex={-1}
            variant="ghost"
            onClick={handleToggle}
            onMouseDown={keepFocus}
          >
            <Icon
              as={ChevronDownIcon}
              boxSize="3.5"
              transform={collapsed ? 'rotate(-90deg)' : undefined}
              transitionDuration="fast"
              transitionProperty="transform"
            />
          </IconButton>
          <Text
            color="fg.muted"
            flex="1"
            fontSize="2xs"
            fontWeight="700"
            textTransform="uppercase"
            truncate
            userSelect="none"
          >
            {t(`widgets.layers.groups.${stack}`)} ({leafCount})
          </Text>
          <FeatureHintExclusion>
            <HStack gap="0.5" onClick={stopPropagation} onMouseDown={keepFocus}>
              {actions.map((action) => (
                <Tooltip key={action.id} content={action.label}>
                  <IconButton
                    aria-label={action.label}
                    color="fg.muted"
                    disabled={action.disabled}
                    size="2xs"
                    tabIndex={-1}
                    variant="ghost"
                    onClick={action.run}
                  >
                    <Icon as={action.icon} boxSize="3.5" />
                  </IconButton>
                </Tooltip>
              ))}
            </HStack>
          </FeatureHintExclusion>
        </HStack>
      </Box>
    </FeatureHint>
  );
};

export const LayerStackHeader = memo(LayerStackHeaderComponent);
