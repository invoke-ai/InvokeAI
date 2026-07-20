import type { WidgetLabelProps } from '@workbench/widgetContracts';

import { HStack, Text } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';

import { usePreviewHeaderContext } from './previewHeaderStore';

/**
 * The preview widget's header label: `[board] / [image]` for the current
 * selection (published by the view via `previewHeaderStore`), falling back to
 * the static "Preview" title when nothing is selected.
 */
export const PreviewWidgetLabel = (_props: WidgetLabelProps) => {
  const { t } = useTranslation();
  const { boardName, imageName } = usePreviewHeaderContext();

  if (!imageName || !boardName) {
    return (
      <Text fontSize="xs" fontWeight="700">
        {t('widgets.labels.preview')}
      </Text>
    );
  }

  return (
    <HStack flex="1" gap="1" minW="0">
      <Text flexShrink={0} fontSize="xs" fontWeight="700">
        {boardName}
      </Text>
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        /
      </Text>
      <Text color="fg.subtle" fontSize="xs" truncate>
        {imageName}
      </Text>
    </HStack>
  );
};
