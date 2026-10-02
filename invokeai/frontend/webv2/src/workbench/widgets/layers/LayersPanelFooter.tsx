import type { LucideIcon } from 'lucide-react';

import { HStack, Input, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { CopyIcon, FolderPlusIcon, Trash2Icon } from 'lucide-react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayerSelectionCommands } from './useLayerSelectionCommands';

const ACTION_TOOLTIP_POSITIONING = { placement: 'top' } as const;

interface LayersPanelFooterProps {
  commands: LayerSelectionCommands;
  degraded: boolean;
  filter: string;
  onFilterChange: (filter: string) => void;
}

const LayersPanelFooterComponent = ({ commands, degraded, filter, onFilterChange }: LayersPanelFooterProps) => {
  const { t } = useTranslation();
  const handleFilter = useCallback(
    (event: { target: { value: string } }) => onFilterChange(event.target.value),
    [onFilterChange]
  );
  return (
    <HStack borderColor="border.subtle" borderTopWidth="1px" gap="2" minH="10" px="2" py="1">
      <Input
        aria-label={t('widgets.layers.footer.filter')}
        flex="1"
        minW="0"
        placeholder={t('widgets.layers.footer.filter')}
        size="2xs"
        value={filter}
        onChange={handleFilter}
      />
      {degraded ? (
        <Text color="fg.muted" flexShrink={0} fontSize="2xs" whiteSpace="nowrap">
          {t('widgets.layers.footer.degraded')}
        </Text>
      ) : null}
      <HStack gap="0.5">
        <FooterAction
          disabled={!commands.canDuplicate}
          icon={CopyIcon}
          label={t('widgets.layers.actions.duplicateSelected')}
          onRun={commands.duplicateSelected}
        />
        <FooterAction
          disabled={!commands.canGroup}
          icon={FolderPlusIcon}
          label={t('widgets.layers.actions.groupSelected')}
          onRun={commands.groupSelected}
        />
        <FooterAction
          colorPalette="red"
          disabled={!commands.canDelete}
          icon={Trash2Icon}
          label={t('widgets.layers.actions.deleteSelected')}
          onRun={commands.deleteSelected}
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
      size="2xs"
      variant="ghost"
      onClick={onRun}
    >
      <ActionIcon />
    </IconButton>
  </Tooltip>
);

export const LayersPanelFooter = memo(LayersPanelFooterComponent);
