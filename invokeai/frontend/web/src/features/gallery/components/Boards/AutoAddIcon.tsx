import { Badge, Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const AutoAddIcon = () => {
  const { t } = useTranslation();
  return (
    <Flex position="absolute" insetInlineEnd={0} top={0} p={1}>
      <Badge variant="solid" bg="invokeBlue.400">
        {t('common.auto')}
      </Badge>
    </Flex>
  );
};

export default memo(AutoAddIcon);
