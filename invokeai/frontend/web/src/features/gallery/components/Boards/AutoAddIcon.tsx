import { Badge, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const AutoAddIcon = () => {
  const { t } = useTranslation();
  return (
    <Flex
      sx={{
        position: 'absolute',
        insetInlineEnd: 0,
        top: 0,
        p: 1,
      }}
    >
      <Badge
        variant="solid"
        sx={{ bg: 'accent.400', _dark: { bg: 'accent.500' } }}
      >
        {t('common.auto')}
      </Badge>
    </Flex>
  );
};

export default memo(AutoAddIcon);
