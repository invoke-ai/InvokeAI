import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { LucideIcon } from 'lucide-react';

import { HStack, Input, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { CopyIcon, FolderPlusIcon, Trash2Icon } from 'lucide-react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import {
  deleteLayers,
  duplicateLayers,
  groupLayers,
  useLayerCommandAvailability,
  useLayerCommandDeps,
} from './layerCommands';

const ACTION_TOOLTIP_POSITIONING = { placement: 'top' } as const;

interface LayersPanelFooterProps {
  degraded: boolean;
  editingLocked: boolean;
  engine: CanvasEngineHandle | null;
  filter: string;
  /** The panel's selection, which the footer's commands act on. */
  selectedIds: readonly string[];
  onFilterChange: (filter: string) => void;
}

const LayersPanelFooterComponent = ({
  degraded,
  editingLocked,
  engine,
  filter,
  selectedIds,
  onFilterChange,
}: LayersPanelFooterProps) => {
  const { t } = useTranslation();
  const deps = useLayerCommandDeps(engine);
  const available = useLayerCommandAvailability(engine, selectedIds);
  const handleFilter = useCallback(
    (event: { target: { value: string } }) => onFilterChange(event.target.value),
    [onFilterChange]
  );
  const handleDuplicate = useCallback(() => void duplicateLayers(deps, selectedIds), [deps, selectedIds]);
  const handleGroup = useCallback(() => groupLayers(deps, selectedIds), [deps, selectedIds]);
  const handleDelete = useCallback(() => deleteLayers(deps, selectedIds), [deps, selectedIds]);
  return (
    <HStack borderColor="border.subtle" borderTopWidth="1px" gap="2" minH="10" px="2" py="1">
      <Input
        aria-label={t('widgets.layers.footer.filter')}
        flex="1"
        minW="0"
        placeholder={t('widgets.layers.footer.filter')}
        value={filter}
        onChange={handleFilter}
      />
      {degraded ? (
        <Text color="fg.muted" flexShrink={0} fontSize="xs" whiteSpace="nowrap">
          {t('widgets.layers.footer.degraded')}
        </Text>
      ) : null}
      <HStack gap="0.5">
        <FooterAction
          disabled={editingLocked || !available.duplicate}
          icon={CopyIcon}
          label={t('widgets.layers.actions.duplicateSelected')}
          onRun={handleDuplicate}
        />
        <FooterAction
          disabled={editingLocked || !available.group}
          icon={FolderPlusIcon}
          label={t('widgets.layers.actions.groupSelected')}
          onRun={handleGroup}
        />
        <FooterAction
          colorPalette="red"
          disabled={editingLocked || !available.delete}
          icon={Trash2Icon}
          label={t('widgets.layers.actions.deleteSelected')}
          onRun={handleDelete}
        />
      </HStack>
    </HStack>
  );
};

const FooterAction = ({
  colorPalette,
  disabled,
  icon: ActionIcon,
  label,
  onRun,
}: {
  colorPalette?: string;
  disabled: boolean;
  icon: LucideIcon;
  label: string;
  onRun: () => void;
}) => (
  <Tooltip content={label} positioning={ACTION_TOOLTIP_POSITIONING}>
    <IconButton
      aria-label={label}
      colorPalette={colorPalette}
      disabled={disabled}
      size="sm"
      variant="ghost"
      onClick={onRun}
    >
      <ActionIcon />
    </IconButton>
  </Tooltip>
);

export const LayersPanelFooter = memo(LayersPanelFooterComponent);
