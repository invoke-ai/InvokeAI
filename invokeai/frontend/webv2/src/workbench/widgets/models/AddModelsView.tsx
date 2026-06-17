import { Box, Icon, Stack } from '@chakra-ui/react';
import { Scrollable, Tabs } from '@workbench/components/ui';
import { updateModelsUi, useModelsUi, type AddModelsTab } from '@workbench/models/uiStore';
import { FolderSearchIcon, KeyIcon, LinkIcon, StarIcon } from 'lucide-react';
import { SiHuggingface } from 'react-icons/si';

import { ApiKeysSection } from './ApiKeysSection';
import { HuggingFaceTab } from './HuggingFaceTab';
import { ScanFolderTab } from './ScanFolderTab';
import { StarterModelsTab } from './StarterModelsTab';
import { UrlInstallTab } from './UrlInstallTab';

/**
 * Every way to acquire a model, one tab per source kind, plus credentials.
 * The active tab lives in the models UI store so it survives navigation.
 */
export const AddModelsView = () => {
  const { addTab } = useModelsUi();

  return (
    <Stack gap="3" h="full" minH="0">
      <Tabs.Root
        size="sm"
        value={addTab}
        onValueChange={(event) => updateModelsUi({ addTab: event.value as AddModelsTab })}
      >
        <Tabs.List>
          <Tabs.Trigger fontSize="xs" value="starter">
            <Icon as={StarIcon} boxSize="3" />
            Starter Models
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="url">
            <Icon as={LinkIcon} boxSize="3" />
            URL / Local Path
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="huggingface">
            <Icon as={SiHuggingface} boxSize="3" />
            HuggingFace
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="scan">
            <Icon as={FolderSearchIcon} boxSize="3" />
            Scan Folder
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="keys">
            <Icon as={KeyIcon} boxSize="3" />
            API Keys
          </Tabs.Trigger>
        </Tabs.List>
      </Tabs.Root>
      {addTab === 'starter' ? (
        // Two-column layout (bundle sidebar + list) with its own scrolling.
        <Box flex="1" minH="0">
          <StarterModelsTab />
        </Box>
      ) : (
        <Scrollable flex="1" label="Add models" minH="0" pr="1">
          {addTab === 'url' ? <UrlInstallTab /> : null}
          {addTab === 'huggingface' ? <HuggingFaceTab /> : null}
          {addTab === 'scan' ? <ScanFolderTab /> : null}
          {addTab === 'keys' ? <ApiKeysSection /> : null}
        </Scrollable>
      )}
    </Stack>
  );
};
