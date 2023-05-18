import { Flex, Grid } from '@chakra-ui/react';
import { memo, useState } from 'react';
import StatusIndicator from './StatusIndicator';

import InvokeAILogoComponent from './InvokeAILogoComponent';
import SiteHeaderMenu from './SiteHeaderMenu';
import useResolution from 'common/hooks/useResolution';
import { FaBars } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';
import IAIIconButton from 'common/components/IAIIconButton';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const [menuOpened, setMenuOpened] = useState(false);
  const resolution = useResolution();
  const { t } = useTranslation();

  return (
    <Grid
      gridTemplateColumns={{ base: 'auto', sm: 'auto max-content' }}
      paddingRight={{ base: 10, xl: 0 }}
      gap={2}
    >
      <Flex justifyContent={{ base: 'center', sm: 'start' }}>
        <InvokeAILogoComponent />
      </Flex>
      <Flex
        alignItems="center"
        gap={2}
        justifyContent={{ base: 'center', sm: 'start' }}
      >
        <StatusIndicator />

        {resolution === 'desktop' ? (
          <SiteHeaderMenu />
        ) : (
          <IAIIconButton
            icon={<FaBars />}
            aria-label={t('accessibility.menu')}
            background={menuOpened ? 'base.800' : 'none'}
            _hover={{ background: menuOpened ? 'base.800' : 'none' }}
            onClick={() => setMenuOpened(!menuOpened)}
            p={0}
          ></IAIIconButton>
        )}
      </Flex>

      {resolution !== 'desktop' && menuOpened && (
        <Flex
          position="absolute"
          right={6}
          top={{ base: 28, sm: 16 }}
          backgroundColor="base.800"
          padding={4}
          borderRadius={4}
          zIndex={{ base: 99, xl: 0 }}
        >
          <SiteHeaderMenu />
        </Flex>
      )}
    </Grid>
  );
};

export default memo(SiteHeader);
