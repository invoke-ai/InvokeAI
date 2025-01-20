import { Box, Flex, IconButton, Popover, PopoverContent, PopoverTrigger, Text } from '@invoke-ai/ui-library';
import { useInputFieldNotes } from 'features/nodes/hooks/useInputFieldNotes';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiNoteBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldNotesIconButtonReadonly = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const notes = useInputFieldNotes(nodeId, fieldName);

  if (!notes?.trim()) {
    return null;
  }

  return (
    <Popover>
      <PopoverTrigger>
        <IconButton
          variant="ghost"
          tooltip={t('nodes.notes')}
          aria-label={t('nodes.notes')}
          icon={<PiNoteBold />}
          size="xs"
        />
      </PopoverTrigger>
      <PopoverContent p={2} w={256}>
        <Flex flexDir="column" gap={2}>
          <Text color="base.300" fontWeight="semibold" fontSize="sm">
            {t('nodes.notes')}
          </Text>
          <Box borderWidth={1} borderRadius="base" p={2} w="full" h="full">
            <Text fontSize="sm">{notes}</Text>
          </Box>
        </Flex>
      </PopoverContent>
    </Popover>
  );
});

InputFieldNotesIconButtonReadonly.displayName = 'InputFieldNotesIconButtonReadonly';
