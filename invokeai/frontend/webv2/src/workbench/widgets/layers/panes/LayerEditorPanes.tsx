import type { ReactNode } from 'react';

import { Box, Flex, Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { ResizeHandle } from '@platform/ui/ResizeHandle';
import { SEGMENT_TABS_HEIGHT_PX, SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { Tooltip } from '@platform/ui/Tooltip';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import { useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import type {
  LayerColorPaneId,
  LayerColorPaneLayout,
  LayerEditorPaneId,
  LayerEditorPaneLayout,
  PaneBlockLayout,
} from './editorPaneLayout';

import { ColorPane } from './ColorPane';
import {
  clampColorPaneSize,
  clampLayerEditorPaneSize,
  COLOR_PANE_MAX_SIZE_PX,
  COLOR_PANE_MIN_SIZE_PX,
  LAYER_EDITOR_PANE_MAX_SIZE_PX,
  LAYER_EDITOR_PANE_MIN_SIZE_PX,
} from './editorPaneLayout';
import { OverviewPane } from './OverviewPane';
import { PropertiesPane } from './PropertiesPane';
import { SwatchesPane } from './SwatchesPane';
import { TransformPane } from './TransformPane';

/** Parity with the shell panels: releasing at the floor stops there; collapse asks for a real push past it. */
const COLLAPSE_OVERSHOOT_PX = 80;

interface PaneBlockLabels {
  collapse: string;
  expand: string;
  resize: string;
  tabs: string;
}

/**
 * Dock fixed pane blocks at either panel edge with preferred height, resize, and collapse-to-tabs; they are not
 * movable widgets.
 */
const LayerPaneBlock = ({
  activePane,
  blockId,
  children,
  clampSize,
  edge,
  labels,
  layout,
  maxSizePx,
  minSizePx,
  onLayoutChange,
  onSelectPane,
  panes,
}: {
  activePane: string;
  blockId: string;
  children: ReactNode;
  clampSize: (next: number) => number;
  edge: 'top' | 'bottom';
  labels: PaneBlockLabels;
  layout: PaneBlockLayout;
  maxSizePx: number;
  minSizePx: number;
  onLayoutChange: (next: PaneBlockLayout) => void;
  onSelectPane: (pane: string) => void;
  panes: ReadonlyArray<{ id: string; label: string }>;
}) => {
  const { isCollapsed, sizePx } = layout;
  const blockRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const panelId = segmentTabsPanelId(blockId);

  const patch = useCallback(
    (next: Partial<PaneBlockLayout>) => onLayoutChange({ ...layout, ...next }),
    [layout, onLayoutChange]
  );
  const toggle = useCallback(() => patch({ isCollapsed: !isCollapsed }), [isCollapsed, patch]);
  const commitSize = useCallback((next: number) => patch({ sizePx: clampSize(next) }), [clampSize, patch]);
  const collapse = useMemo(
    () => ({
      at: minSizePx - COLLAPSE_OVERSHOOT_PX,
      onCollapse: (source: 'keyboard' | 'pointer') => {
        // The separator unmounts, so keyboard focus moves to the strip first.
        if (source === 'keyboard') {
          toggleRef.current?.focus();
        }
        patch({ isCollapsed: true });
      },
      preview: SEGMENT_TABS_HEIGHT_PX,
    }),
    [minSizePx, patch]
  );
  const collapseLabel = isCollapsed ? labels.expand : labels.collapse;
  const collapseIcon =
    edge === 'top' ? (isCollapsed ? ChevronDownIcon : ChevronUpIcon) : isCollapsed ? ChevronUpIcon : ChevronDownIcon;
  const separator = !isCollapsed ? (
    <ResizeHandle
      collapse={collapse}
      label={labels.resize}
      max={maxSizePx}
      min={minSizePx}
      orientation="horizontal"
      pane={edge === 'top' ? 'before' : 'after'}
      paneRef={blockRef}
      sizeProperty="flexBasis"
      value={sizePx}
      onCommit={commitSize}
    />
  ) : null;
  const collapseButton = useMemo(
    () => (
      <Tooltip content={collapseLabel}>
        <IconButton
          ref={toggleRef}
          aria-expanded={!isCollapsed}
          aria-label={collapseLabel}
          color="fg.muted"
          size="2xs"
          variant="ghost"
          onClick={toggle}
        >
          <Icon as={collapseIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    ),
    [collapseIcon, collapseLabel, isCollapsed, toggle]
  );
  const strip = (
    <SegmentTabs
      activeId={activePane}
      ariaLabel={labels.tabs}
      idBase={blockId}
      showActivePanel={!isCollapsed}
      tabs={panes}
      trailing={collapseButton}
      onSelect={onSelectPane}
    />
  );
  const panel = !isCollapsed ? (
    <Box
      aria-labelledby={segmentTabsTabId(blockId, activePane)}
      flex="1"
      id={panelId}
      minH="0"
      overflow="hidden"
      role="tabpanel"
    >
      {children}
    </Box>
  ) : null;

  // Expanded, the separator draws the dividing line; collapsed, the strip keeps a border.
  const block = (
    <Flex
      ref={blockRef}
      borderColor="border.subtle"
      data-layer-pane-block={blockId}
      data-pane-collapsed={isCollapsed ? '' : undefined}
      direction="column"
      flex={isCollapsed ? '0 0 auto' : `0 1 ${sizePx}px`}
      minH={`${SEGMENT_TABS_HEIGHT_PX}px`}
      overflow="hidden"
      {...(isCollapsed ? (edge === 'top' ? { borderBottomWidth: '1px' } : { borderTopWidth: '1px' }) : {})}
    >
      {strip}
      {panel}
    </Flex>
  );

  return edge === 'top' ? (
    <>
      {block}
      {separator}
    </>
  ) : (
    <>
      {separator}
      {block}
    </>
  );
};

const EDITOR_PANES: ReadonlyArray<{ id: LayerEditorPaneId; labelKey: string }> = [
  { id: 'properties', labelKey: 'widgets.labels.properties' },
  { id: 'transform', labelKey: 'widgets.labels.transform' },
  { id: 'overview', labelKey: 'widgets.labels.overview' },
];

export const LayerEditorPanes = ({
  layout,
  onLayoutChange,
}: {
  layout: LayerEditorPaneLayout;
  onLayoutChange: (next: LayerEditorPaneLayout) => void;
}) => {
  const { t } = useTranslation();
  const { activePane } = layout;
  const panes = useMemo(() => EDITOR_PANES.map(({ id, labelKey }) => ({ id, label: t(labelKey) })), [t]);
  const labels = useMemo<PaneBlockLabels>(
    () => ({
      collapse: t('widgets.layers.panes.collapse'),
      expand: t('widgets.layers.panes.expand'),
      resize: t('widgets.layers.panes.resize'),
      tabs: t('widgets.layers.panes.tabs'),
    }),
    [t]
  );
  const onSelectPane = useCallback(
    (pane: string) =>
      onLayoutChange(
        pane === layout.activePane
          ? { ...layout, isCollapsed: !layout.isCollapsed }
          : { ...layout, activePane: pane as LayerEditorPaneId, isCollapsed: false }
      ),
    [layout, onLayoutChange]
  );
  const onBlockLayoutChange = useCallback(
    (next: PaneBlockLayout) => onLayoutChange({ ...layout, ...next }),
    [layout, onLayoutChange]
  );

  return (
    <LayerPaneBlock
      activePane={activePane}
      blockId="layer-editor-pane"
      clampSize={clampLayerEditorPaneSize}
      edge="bottom"
      labels={labels}
      layout={layout}
      maxSizePx={LAYER_EDITOR_PANE_MAX_SIZE_PX}
      minSizePx={LAYER_EDITOR_PANE_MIN_SIZE_PX}
      onLayoutChange={onBlockLayoutChange}
      onSelectPane={onSelectPane}
      panes={panes}
    >
      {activePane === 'transform' ? (
        <TransformPane />
      ) : activePane === 'overview' ? (
        <OverviewPane />
      ) : (
        <PropertiesPane />
      )}
    </LayerPaneBlock>
  );
};

const COLOR_PANES: ReadonlyArray<{ id: LayerColorPaneId; labelKey: string }> = [
  { id: 'color', labelKey: 'widgets.labels.color' },
  { id: 'swatches', labelKey: 'widgets.labels.swatches' },
];

export const LayerColorPane = ({
  layout,
  onLayoutChange,
}: {
  layout: LayerColorPaneLayout;
  onLayoutChange: (next: LayerColorPaneLayout) => void;
}) => {
  const { t } = useTranslation();
  const { activePane } = layout;
  const panes = useMemo(() => COLOR_PANES.map(({ id, labelKey }) => ({ id, label: t(labelKey) })), [t]);
  const labels = useMemo<PaneBlockLabels>(
    () => ({
      collapse: t('widgets.layers.colorPane.collapse'),
      expand: t('widgets.layers.colorPane.expand'),
      resize: t('widgets.layers.colorPane.resize'),
      tabs: t('widgets.layers.colorPane.tabs'),
    }),
    [t]
  );
  const onSelectPane = useCallback(
    (pane: string) =>
      onLayoutChange(
        pane === layout.activePane
          ? { ...layout, isCollapsed: !layout.isCollapsed }
          : { ...layout, activePane: pane as LayerColorPaneId, isCollapsed: false }
      ),
    [layout, onLayoutChange]
  );
  const onBlockLayoutChange = useCallback(
    (next: PaneBlockLayout) => onLayoutChange({ ...layout, ...next }),
    [layout, onLayoutChange]
  );

  return (
    <LayerPaneBlock
      activePane={activePane}
      blockId="layer-color-pane"
      clampSize={clampColorPaneSize}
      edge="top"
      labels={labels}
      layout={layout}
      maxSizePx={COLOR_PANE_MAX_SIZE_PX}
      minSizePx={COLOR_PANE_MIN_SIZE_PX}
      onLayoutChange={onBlockLayoutChange}
      onSelectPane={onSelectPane}
      panes={panes}
    >
      {activePane === 'swatches' ? <SwatchesPane /> : <ColorPane />}
    </LayerPaneBlock>
  );
};
