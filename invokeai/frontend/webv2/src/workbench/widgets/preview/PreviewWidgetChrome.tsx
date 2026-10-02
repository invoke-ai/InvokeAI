import type { WidgetLabelProps } from '@workbench/widgetContracts';

import { HStack, Text } from '@chakra-ui/react';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { useTranslation } from 'react-i18next';

import { usePreviewHeaderContext } from './previewHeaderStore';

/**
 * Show board/image context from previewHeaderStore, falling back to Preview except where the center trigger
 * already names it.
 */
export const PreviewWidgetLabel = ({ region }: WidgetLabelProps) => {
  const { t } = useTranslation();
  const { boardName, itemName } = usePreviewHeaderContext();

  if (!itemName || !boardName) {
    if (region === 'center') {
      return null;
    }

    return (
      <Text fontSize="xs" fontWeight="700">
        {t('widgets.labels.preview')}
      </Text>
    );
  }

  // Pad the truncated label away from the first header action.
  return (
    <HStack flex="1" gap="1" minW="0" pe="2">
      <Text flexShrink={0} fontSize="xs" fontWeight="700">
        {boardName}
      </Text>
      <Text color="fg.muted" flexShrink={0} fontSize="xs">
        /
      </Text>
      <MiddleTruncate color="fg.muted" fontSize="xs" minW="0" text={itemName} />
    </HStack>
  );
};
