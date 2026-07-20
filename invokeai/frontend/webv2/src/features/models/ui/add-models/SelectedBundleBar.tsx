import type { StarterModelBundle } from '@features/models/core/types';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { DownloadIcon } from 'lucide-react';

/** Summary + install action for the currently selected bundle. */
export const SelectedBundleBar = ({
  bundle,
  isInstalling,
  onInstall,
}: {
  bundle: StarterModelBundle;
  isInstalling: boolean;
  onInstall: () => void;
}) => {
  const missingCount = bundle.models.filter((model) => !model.is_installed).length;

  return (
    <HStack gap="2" justify="space-between" px="3">
      <Text color="fg.subtle" fontSize="2xs" truncate>
        {bundle.name} · {bundle.models.length} models ·{' '}
        {missingCount === 0 ? 'all installed' : `${missingCount} to install`}
      </Text>
      {missingCount > 0 ? (
        <Button flexShrink={0} loading={isInstalling} size="2xs" variant="outline" onClick={onInstall}>
          <Icon as={DownloadIcon} boxSize="3" />
          Install bundle
        </Button>
      ) : null}
    </HStack>
  );
};
