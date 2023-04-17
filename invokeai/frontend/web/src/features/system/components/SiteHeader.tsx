import { Flex, Grid } from '@chakra-ui/react';
import { useState } from 'react';
import ModelSelect from './ModelSelect';
import StatusIndicator from './StatusIndicator';

import InvokeAILogoComponent from './InvokeAILogoComponent';
import SiteHeaderMenu from './SiteHeaderMenu';
import useResolution from 'common/hooks/useResolution';
import { FaBars } from 'react-icons/fa';

/**
 * Header, includes color mode toggle, settings button, status message.
 */
const SiteHeader = () => {
  const [menuOpened, setMenuOpened] = useState(false);
  const resolution = useResolution();

  const isMobile = ['mobile', 'tablet'].includes(resolution);

  return (
    <Grid gridTemplateColumns="auto max-content">
      <InvokeAILogoComponent />

      <Flex alignItems="center" gap={2}>
        <StatusIndicator />

        <ModelSelect />

        {!isMobile && <SiteHeaderMenu />}
        <Flex>
          {isMobile && <FaBars onClick={() => setMenuOpened(!menuOpened)} />}
        </Flex>
      </Flex>
      <Flex>{isMobile && menuOpened && <SiteHeaderMenu />}</Flex>
    </Grid>
  );
};

SiteHeader.displayName = 'SiteHeader';
export default SiteHeader;
