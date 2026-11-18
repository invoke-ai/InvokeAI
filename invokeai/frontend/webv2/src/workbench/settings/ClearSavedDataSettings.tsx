import { chakra, HStack, Icon, Input, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useNavigate } from '@tanstack/react-router';
import { refreshProjectLibrary, useProjectLibrarySelector } from '@workbench/projects/library';
import { clearAllWorkbenchData } from '@workbench/projects/syncedPersistence';
import { useOptionalWorkbenchCommands, useOptionalWorkbenchPersistenceService } from '@workbench/WorkbenchContext';
import { ArrowRightIcon, Trash2Icon, TriangleAlertIcon } from 'lucide-react';
import { useCallback, useId, useRef, useState, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { clearWorkspaceData, describeWorkspaceClearFailure, rememberWorkspaceClearFailure } from './clearWorkspaceData';
import { clearWorkbenchSettings, useWorkbenchSettingsSelector } from './store';

const DANGER_BUTTON_HOVER_STYLES = { bg: 'fg.error', color: 'bg.subtle' };

/**
 * The one catastrophic action in Settings, in its own warning-toned area: it deletes every project the account has on
 * the server and every local setting, then reloads. Its confirmation names how many projects go, what survives (media
 * and boards), where to export first, and asks for a typed word.
 */
export const ClearSavedDataSettings = ({ description, label }: { description: string; label: string }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const commands = useOptionalWorkbenchCommands();
  const mountedPersistence = useOptionalWorkbenchPersistenceService();
  const scope = useWorkbenchSettingsSelector((snapshot) => snapshot.scope);
  const libraryStatus = useProjectLibrarySelector((snapshot) => snapshot.status);
  const projectCount = useProjectLibrarySelector((snapshot) => snapshot.summaries.length);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [typed, setTyped] = useState('');
  const confirmInputId = useId();
  // In the active language: the word is shown to be typed, so it must read as one.
  const confirmWord = t('settings.clearData.confirmWord');

  const [isClearing, setIsClearing] = useState(false);
  const confirmInputRef = useRef<HTMLInputElement>(null);
  const getConfirmInput = useCallback(() => confirmInputRef.current, []);

  const openConfirm = useCallback(() => {
    setTyped('');
    setIsConfirmOpen(true);
    // The count must be the server's now, not the one listed when the library last loaded (another tab may have
    // created or deleted projects since); the listing on hand shows until the fresh one lands.
    void refreshProjectLibrary();
  }, []);
  const closeConfirm = useCallback(() => setIsConfirmOpen(false), []);
  const handleTypedChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setTyped(event.currentTarget.value),
    []
  );
  const openProjects = useCallback(() => {
    setIsConfirmOpen(false);
    void navigate({ to: '/projects' });
  }, [navigate]);

  const clearSavedData = useCallback(async () => {
    setIsClearing(true);
    const failures = await clearWorkspaceData(
      () => (mountedPersistence ? mountedPersistence.clearWorkbench() : clearAllWorkbenchData()),
      clearWorkbenchSettings
    );
    if (failures.length === 0) {
      window.location.reload();
      return;
    }
    const message = t(describeWorkspaceClearFailure(failures));
    rememberWorkspaceClearFailure(message, window.sessionStorage);
    commands?.notifications.reportError({ area: 'workspace-clear', message, namespace: 'system' });
    window.location.reload();
  }, [commands, mountedPersistence, t]);

  const bodyKey =
    libraryStatus === 'ready' && projectCount === 0
      ? scope === 'user'
        ? 'settings.clearData.bodyUserNoProjects'
        : 'settings.clearData.bodyInstallNoProjects'
      : scope === 'user'
        ? 'settings.clearData.bodyUser'
        : 'settings.clearData.bodyInstall';
  const projects =
    libraryStatus === 'ready'
      ? t('settings.clearData.projectCount', { count: projectCount })
      : t('settings.clearData.projectCountUnknown');

  return (
    <Stack borderColor="border.error" borderWidth="1px" gap="3" p="4" rounded="lg">
      <Stack gap="1">
        <HStack gap="2">
          <Icon as={TriangleAlertIcon} boxSize="4" color="fg.error" />
          <Text as="h3" color="fg" fontSize="lg" fontWeight="500">
            {label}
          </Text>
        </HStack>
        <Text color="fg.muted" fontSize="md">
          {description}
        </Text>
      </Stack>
      <Button
        alignSelf="start"
        borderColor="border.emphasized"
        color="fg.error"
        size="lg"
        variant="outline"
        _hover={DANGER_BUTTON_HOVER_STYLES}
        onClick={openConfirm}
      >
        <Trash2Icon />
        {t('settings.clearData.action')}
      </Button>
      <ConfirmDialog
        body={
          <>
            <Text fontSize="md">{t(bodyKey, { projects })}</Text>
            <Text color="fg.muted" fontSize="md">
              {t('settings.clearData.mediaKept')}
            </Text>
            <HStack align="center" flexWrap="wrap" gap="2" justify="space-between">
              <Text color="fg.muted" fontSize="md">
                {t('settings.clearData.exportFirst')}
              </Text>
              <Button disabled={isClearing} size="sm" variant="outline" onClick={openProjects}>
                {t('settings.clearData.openProjects')}
                <ArrowRightIcon />
              </Button>
            </HStack>
            <Stack gap="1">
              <chakra.label fontSize="md" htmlFor={confirmInputId}>
                {t('settings.clearData.typeToConfirm', { word: confirmWord })}
              </chakra.label>
              <Input
                autoComplete="off"
                disabled={isClearing}
                id={confirmInputId}
                ref={confirmInputRef}
                spellCheck={false}
                value={typed}
                onChange={handleTypedChange}
              />
            </Stack>
          </>
        }
        confirmLabel={t('settings.clearData.confirm')}
        // The typed word guards the confirm, so the field can take focus; Enter confirms once it matches.
        initialFocusEl={getConfirmInput}
        isConfirmDisabled={typed.trim() !== confirmWord}
        isOpen={isConfirmOpen}
        title={t('settings.clearData.title')}
        onClose={closeConfirm}
        onConfirm={clearSavedData}
      />
    </Stack>
  );
};
