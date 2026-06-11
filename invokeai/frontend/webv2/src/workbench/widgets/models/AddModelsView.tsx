import { Stack } from '@chakra-ui/react';

import { Scrollable } from '../../components/ui/Scrollable';
import { Tabs } from '../../components/ui/Tabs';
import { updateModelsUi, useModelsUi, type AddModelsTab } from '../../models/uiStore';
import { ApiKeysSection } from './ApiKeysSection';
import { HuggingFaceTab } from './HuggingFaceTab';
import { ScanFolderTab } from './ScanFolderTab';
import { StarterModelsTab } from './StarterModelsTab';
import { UrlInstallTab } from './UrlInstallTab';
import { FolderSearchIcon, KeyIcon, LinkIcon, StarIcon } from 'lucide-react';
import { SiHuggingface } from 'react-icons/si';

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
            <StarIcon size={12} />
            Starter Models
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="url">
            <LinkIcon size={12} />
            URL / Local Path
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="huggingface">
            <SiHuggingface size={12} />
            HuggingFace
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="scan">
            <FolderSearchIcon size={12} />
            Scan Folder
          </Tabs.Trigger>
          <Tabs.Trigger fontSize="xs" value="keys">
            <KeyIcon size={12} />
            API Keys
          </Tabs.Trigger>
        </Tabs.List>
      </Tabs.Root>
      <Scrollable flex="1" label="Add models" minH="0" pr="1">
        {addTab === 'starter' ? <StarterModelsTab /> : null}
        {addTab === 'url' ? <UrlInstallTab /> : null}
        {addTab === 'huggingface' ? <HuggingFaceTab /> : null}
        {addTab === 'scan' ? <ScanFolderTab /> : null}
        {addTab === 'keys' ? <ApiKeysSection /> : null}
      </Scrollable>
    </Stack>
  );
};
