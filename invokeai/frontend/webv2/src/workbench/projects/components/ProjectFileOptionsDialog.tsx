import { Checkbox, Portal, Stack, Text } from '@chakra-ui/react';
import { Button, CloseButton } from '@platform/ui';
import { Dialog } from '@platform/ui/Dialog';
import { patchWorkbenchPreferences, useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useCallback, useId, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectFileOptionsRequest } from './ProjectFileOptionsProvider';

/** Each option's description starts under its label, past the checkbox and the gap after it. */
const OPTION_DESCRIPTION_INSET = '6';

export const ProjectFileOptionsDialog = ({
  isOpen,
  onExitComplete,
  request,
}: {
  isOpen: boolean;
  onExitComplete: () => void;
  request: ProjectFileOptionsRequest;
}) => {
  const { t } = useTranslation();
  const [includeFonts, setIncludeFonts] = useState(false);
  const workspaceDescriptionId = useId();
  const fontsDescriptionId = useId();
  const rememberedIncludeWorkspace = useWorkbenchPreferenceSelector(
    (preferences) => preferences.exportIncludesWorkspace
  );
  const [includeWorkspace, setIncludeWorkspace] = useState(rememberedIncludeWorkspace);
  const close = useCallback(() => request.settle(null), [request]);
  const isExport = request.kind === 'export';
  const returnFocus = useCallback(
    () => (request.returnFocus?.isConnected ? request.returnFocus : null),
    [request.returnFocus]
  );
  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        close();
      }
    },
    [close]
  );
  const handleFontsCheckedChange = useCallback(
    ({ checked }: { checked: boolean | 'indeterminate' }) => setIncludeFonts(checked === true),
    []
  );
  const handleWorkspaceCheckedChange = useCallback(
    ({ checked }: { checked: boolean | 'indeterminate' }) => setIncludeWorkspace(checked === true),
    []
  );
  const confirm = useCallback(() => {
    if (isExport && includeWorkspace !== rememberedIncludeWorkspace) {
      void patchWorkbenchPreferences({ exportIncludesWorkspace: includeWorkspace });
    }
    request.settle({ includeFonts, includeWorkspace });
  }, [includeFonts, includeWorkspace, isExport, rememberedIncludeWorkspace, request]);

  return (
    <Dialog.Root
      open={isOpen}
      placement="center"
      size="sm"
      finalFocusEl={returnFocus}
      onExitComplete={onExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>
                {t(isExport ? 'projects.fonts.exportTitle' : 'projects.fonts.importTitle', { name: request.name })}
              </Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="4">
                {isExport ? (
                  <>
                    <Stack gap="1">
                      <Checkbox.Root checked={includeWorkspace} onCheckedChange={handleWorkspaceCheckedChange}>
                        <Checkbox.HiddenInput aria-describedby={workspaceDescriptionId} />
                        <Checkbox.Control>
                          <Checkbox.Indicator />
                        </Checkbox.Control>
                        <Checkbox.Label>{t('projects.exportOptions.includeWorkspace')}</Checkbox.Label>
                      </Checkbox.Root>
                      <Text color="fg.muted" fontSize="lg" id={workspaceDescriptionId} ps={OPTION_DESCRIPTION_INSET}>
                        {t('projects.exportOptions.workspaceDescription')}
                      </Text>
                    </Stack>
                    <Stack gap="1">
                      <Checkbox.Root checked={includeFonts} onCheckedChange={handleFontsCheckedChange}>
                        <Checkbox.HiddenInput aria-describedby={fontsDescriptionId} />
                        <Checkbox.Control>
                          <Checkbox.Indicator />
                        </Checkbox.Control>
                        <Checkbox.Label>{t('projects.fonts.includeFiles')}</Checkbox.Label>
                      </Checkbox.Root>
                      <Text color="fg.muted" fontSize="lg" id={fontsDescriptionId} ps={OPTION_DESCRIPTION_INSET}>
                        {t('projects.fonts.exportDescription')}
                      </Text>
                      {includeFonts ? (
                        <Text fontSize="lg" ps={OPTION_DESCRIPTION_INSET}>
                          {t('projects.fonts.rightsReminder')}
                        </Text>
                      ) : null}
                    </Stack>
                  </>
                ) : (
                  <Text fontSize="lg">{t('projects.fonts.quotaDescription')}</Text>
                )}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button color="fg" variant="ghost" onClick={close}>
                {t('common.cancel')}
              </Button>
              <Button colorPalette="accent" onClick={confirm}>
                {t(isExport ? 'projects.fonts.exportAction' : 'projects.fonts.importReferences')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton aria-label={t('common.close')} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
