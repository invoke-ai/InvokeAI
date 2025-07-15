import { Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { upperFirst } from 'es-toolkit/compat';
import { useTranslation } from 'react-i18next';

export const RefImageWarningTooltipContent = ({ warnings }: { warnings: string[] }) => {
  const { t } = useTranslation();

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">Invalid Reference Image:</Text>
      <UnorderedList>
        {warnings.map((tKey) => (
          <ListItem key={tKey}>{upperFirst(t(tKey))}</ListItem>
        ))}
      </UnorderedList>
    </Flex>
  );
};
