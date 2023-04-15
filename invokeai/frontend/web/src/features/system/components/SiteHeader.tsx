import { Box, Flex, Grid } from '@chakra-ui/react';
import { useState } from 'react';
import ModelSelect from './ModelSelect';
import StatusIndicator from './StatusIndicator';

import InvokeAILogoComponent from './InvokeAILogoComponent';
import MediaQuery from 'react-responsive';
import SiteHeaderMenu from './SiteHeaderMenu';
import { FaBars } from 'react-icons/fa';

/**
 * Header, includes logo, model select and status message.
 */
const SiteHeader = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const handleMenuToggle = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <>
      <MediaQuery minDeviceWidth={768}>
        <Grid gridTemplateColumns="auto max-content">
          <InvokeAILogoComponent />

          <Flex alignItems="center" gap={2}>
            <StatusIndicator />

            <ModelSelect />

            <SiteHeaderMenu />
          </Flex>
        </Grid>
      </MediaQuery>
      <MediaQuery maxDeviceWidth={768}>
        <Flex>
          <InvokeAILogoComponent />

          <Flex alignItems="center" gap={3} marginLeft="2rem">
            <StatusIndicator />

            <ModelSelect />
          </Flex>
          <Box
            onClick={handleMenuToggle}
            sx={{
              marginLeft: '20px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <FaBars />
          </Box>
        </Flex>
        {isMenuOpen && <SiteHeaderMenu />}
      </MediaQuery>
    </>
  );
};

SiteHeader.displayName = 'SiteHeader';
export default SiteHeader;
