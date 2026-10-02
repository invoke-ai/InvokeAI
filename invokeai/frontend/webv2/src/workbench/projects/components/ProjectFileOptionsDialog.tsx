import { Checkbox, Dialog, Portal, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, CloseButton } from '@platform/ui';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ProjectFileOptionsRequest } from './ProjectFileOptionsProvider';

export const ProjectFileOptionsDialog = ({ request }: { request: ProjectFileOptionsRequest }) => {
  const { t } = useTranslation();
  const [includeFonts, setIncludeFonts] = useState(false);
  useMountEffect(() => registerHotkeyModalLayer('project-file-options'));
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
  const handleCheckedChange = useCallback(
    ({ checked }: { checked: boolean | 'indeterminate' }) => setIncludeFonts(checked === true),
    []
  );
  const confirm = useCallback(() => request.settle({ includeFonts }), [includeFonts, request]);

  return (
    <Dialog.Root open placement="center" size="sm" finalFocusEl={returnFocus} onOpenChange={handleOpenChange}>
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
              <Stack gap="3">
                {isExport ? (
                  <>
                    <Checkbox.Root checked={includeFonts} onCheckedChange={handleCheckedChange}>
                      <Checkbox.HiddenInput />
                      <Checkbox.Control>
                        <Checkbox.Indicator />
                      </Checkbox.Control>
                      <Checkbox.Label>{t('projects.fonts.includeFiles')}</Checkbox.Label>
                    </Checkbox.Root>
                    <Text color="fg.muted" fontSize="sm">
                      {t('projects.fonts.exportDescription')}
                    </Text>
                    {includeFonts ? <Text fontSize="sm">{t('projects.fonts.rightsReminder')}</Text> : null}
                  </>
                ) : (
                  <Text fontSize="sm">{t('projects.fonts.quotaDescription')}</Text>
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
