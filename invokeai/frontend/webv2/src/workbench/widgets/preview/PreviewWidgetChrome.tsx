import type { WidgetLabelProps, WorkbenchRegion } from '@workbench/widgetContracts';

import { HStack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { isWidgetRegion } from '@workbench/layoutContracts';
import { useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { XIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { usePreviewHeaderContext } from './previewHeaderStore';

/**
 * Show board/image context from previewHeaderStore, falling back to Preview except where the center trigger
 * already names it. A named item carries a control that clears the gallery selection it shows.
 */
export const PreviewWidgetLabel = ({ region }: WidgetLabelProps) => {
  const { t } = useTranslation();
  const { boardName, itemName } = usePreviewHeaderContext();

  if (!itemName || !boardName) {
    if (region === 'center') {
      return null;
    }

    return (
      <Text fontSize="md" fontWeight="700">
        {t('widgets.labels.preview')}
      </Text>
    );
  }

  // Pad the truncated label away from the first header action in a panel header. In the center the label ends its
  // chrome island, whose own padding sets the clear button's inset like the island's other edge icon buttons.
  return (
    <HStack flex="1" gap="1" minW="0" pe={region === 'center' ? undefined : '2'}>
      <Text flexShrink={0} fontSize="md" fontWeight="700">
        {boardName}
      </Text>
      <Text color="fg.muted" flexShrink={0} fontSize="md">
        /
      </Text>
      <MiddleTruncate color="fg.muted" fontSize="md" minW="0" text={itemName} />
      <ClearSelectionButton region={region} />
    </HStack>
  );
};

const CLEAR_SELECTION_HOVER = { color: 'fg' } as const;

const ClearSelectionButton = ({ region }: { region: WorkbenchRegion }) => {
  const { t } = useTranslation();
  const { gallery } = useWorkbenchCommands();
  const { focusRegion } = useWorkbenchFocus();
  const label = t('widgets.gallery.commands.clearSelection');
  // The gallery's clear-selection command. This button leaves with the selection, so focus moves to the Preview's
  // region instead of falling to the document body.
  const clearSelection = useCallback(() => {
    if (isWidgetRegion(region)) {
      focusRegion(region, 'preview');
    }
    gallery.clearSelection();
  }, [focusRegion, gallery, region]);

  return (
    <IconButton
      _hover={CLEAR_SELECTION_HOVER}
      aria-label={label}
      color="fg.muted"
      flexShrink={0}
      size="sm"
      tooltip={label}
      variant="ghost"
      onClick={clearSelection}
    >
      <XIcon />
    </IconButton>
  );
};
