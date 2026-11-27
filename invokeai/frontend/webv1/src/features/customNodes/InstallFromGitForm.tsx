import {
  Alert,
  AlertDescription,
  AlertIcon,
  Button,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Input,
} from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useInstallCustomNodePackMutation } from 'services/api/endpoints/customNodes';

import { useCustomNodesInstallLog } from './useCustomNodesInstallLog';

export const InstallFromGitForm = memo(() => {
  const { t } = useTranslation();
  const [source, setSource] = useState('');
  const [installPack, { isLoading }] = useInstallCustomNodePackMutation();
  const { addLogEntry } = useCustomNodesInstallLog();

  const handleInstall = useCallback(async () => {
    if (!source.trim()) {
      return;
    }

    const trimmedSource = source.trim();
    addLogEntry({ name: trimmedSource, status: 'installing' });

    try {
      const result = await installPack({ source: trimmedSource }).unwrap();
      if (result.success) {
        addLogEntry({ name: result.name, status: 'completed', message: result.message });
        setSource('');
        if (result.requires_dependencies) {
          toast({
            id: `custom-nodes-deps-${result.name}`,
            title: t('customNodes.dependenciesRequiredTitle'),
            description: t('customNodes.dependenciesRequiredDescription', {
              name: result.name,
              file: result.dependency_file ?? 'requirements.txt',
            }),
            status: 'warning',
            duration: null,
            isClosable: true,
          });
        }
      } else {
        addLogEntry({ name: result.name, status: 'error', message: result.message });
      }
    } catch {
      addLogEntry({ name: trimmedSource, status: 'error', message: t('customNodes.installError') });
    }
  }, [source, installPack, addLogEntry, t]);

  const handleSourceChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSource(e.target.value);
  }, []);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        handleInstall();
      }
    },
    [handleInstall]
  );

  return (
    <Flex flexDir="column" gap={4} pt={4}>
      <Alert status="warning" borderRadius="base">
        <AlertIcon />
        <AlertDescription fontSize="sm">{t('customNodes.securityWarning')}</AlertDescription>
      </Alert>

      <FormControl orientation="vertical">
        <FormLabel>{t('customNodes.gitUrlLabel')}</FormLabel>
        <Flex alignItems="center" gap={3} w="full">
          <Input
            placeholder={t('customNodes.gitUrlPlaceholder')}
            value={source}
            onChange={handleSourceChange}
            onKeyDown={handleKeyDown}
          />
          <Button onClick={handleInstall} isDisabled={!source.trim()} isLoading={isLoading} size="sm">
            {t('customNodes.install')}
          </Button>
        </Flex>
        <FormHelperText>{t('customNodes.installDescription')}</FormHelperText>
      </FormControl>
    </Flex>
  );
});

InstallFromGitForm.displayName = 'InstallFromGitForm';
