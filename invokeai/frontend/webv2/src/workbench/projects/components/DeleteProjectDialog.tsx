import type { DeleteProjectBoards } from '@workbench/projects/api';

import { RadioGroup, Stack, Text } from '@chakra-ui/react';
import { galleryBoardsOptions } from '@features/gallery/queries';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Deleting a project also decides what happens to the boards it holds. The inbox always goes with it; the choice
 * is offered only when there are other boards, with their names and what they hold, so the safe default — keeping
 * them in the Library — is never a surprise and the destructive one says what it costs.
 */
export const DeleteProjectDialog = ({
  body,
  finalFocusEl,
  isOpen,
  projectId,
  onClose,
  onConfirm,
}: {
  /** The host's own sentence about the project; the dialog adds what happens to its boards. */
  body: string;
  finalFocusEl?: () => HTMLElement | null;
  isOpen: boolean;
  projectId: string | null;
  onClose: () => void;
  onConfirm: (boards: DeleteProjectBoards) => Promise<void> | void;
}) => {
  const { t } = useTranslation();
  const [boards, setBoards] = useState<DeleteProjectBoards>('release');
  // Each opening starts from the safe choice again, set as it opens rather than once the last one finished closing:
  // a dialog reopened during that exit, for this project or another, must not inherit a destructive choice.
  const [choiceOpenFor, setChoiceOpenFor] = useState<string | null>(null);
  const openFor = isOpen ? projectId : null;
  if (openFor !== choiceOpenFor) {
    setChoiceOpenFor(openFor);
    if (openFor !== null) {
      setBoards('release');
    }
  }
  // Archived members go with the project just the same, so the list must include them.
  const { data, isError, isLoading } = useQuery({
    ...galleryBoardsOptions({ includeArchived: true }),
    enabled: isOpen && projectId !== null,
  });
  const members = useMemo(
    () => (data ?? []).filter((board) => board.projectId === projectId && !board.isInbox),
    [data, projectId]
  );
  const itemCount = members.reduce((total, board) => total + board.imageCount + board.assetCount + board.videoCount, 0);

  const handleValueChange = useCallback((event: { value: string | null }) => {
    setBoards(event.value === 'delete' ? 'delete' : 'release');
  }, []);
  const handleConfirm = useCallback(() => onConfirm(boards), [boards, onConfirm]);

  return (
    <ConfirmDialog
      body={
        <>
          <Text fontSize="md">{body}</Text>
          {isLoading ? (
            <Text color="fg.muted" fontSize="xs">
              {t('projects.deleteProjectCheckingBoards')}
            </Text>
          ) : null}
          {isError ? (
            <Text color="fg.muted" fontSize="xs">
              {t('projects.deleteProjectBoardsUnknown')}
            </Text>
          ) : null}
          {members.length > 0 ? (
            <RadioGroup.Root size="sm" value={boards} onValueChange={handleValueChange}>
              <Stack gap="2">
                <RadioGroup.Item alignItems="start" value="release">
                  <RadioGroup.ItemHiddenInput />
                  <RadioGroup.ItemIndicator mt="0.5" />
                  <Stack gap="0">
                    <RadioGroup.ItemText>{t('projects.deleteProjectKeepBoards')}</RadioGroup.ItemText>
                    <Text color="fg.muted" fontSize="xs">
                      {t('projects.deleteProjectKeepBoardsHint', {
                        boards: members.map((board) => board.name).join(', '),
                      })}
                    </Text>
                  </Stack>
                </RadioGroup.Item>
                <RadioGroup.Item alignItems="start" value="delete">
                  <RadioGroup.ItemHiddenInput />
                  <RadioGroup.ItemIndicator mt="0.5" />
                  <Stack gap="0">
                    <RadioGroup.ItemText>{t('projects.deleteProjectDeleteBoards')}</RadioGroup.ItemText>
                    <Text color="fg.muted" fontSize="xs">
                      {t('projects.deleteProjectDeleteBoardsHint', {
                        items: t('projects.deleteProjectItemCount', { count: itemCount }),
                      })}
                    </Text>
                  </Stack>
                </RadioGroup.Item>
              </Stack>
            </RadioGroup.Root>
          ) : null}
          <Text color="fg.muted" fontSize="xs">
            {t('projects.deleteProjectInboxNote')}
          </Text>
        </>
      }
      confirmLabel={t('projects.deleteProject')}
      finalFocusEl={finalFocusEl}
      // Not before the boards are known: the choice could otherwise appear after the user has already confirmed.
      isConfirmDisabled={isLoading}
      isOpen={isOpen}
      title={t('projects.deleteProjectQuestion')}
      onClose={onClose}
      onConfirm={handleConfirm}
    />
  );
};
