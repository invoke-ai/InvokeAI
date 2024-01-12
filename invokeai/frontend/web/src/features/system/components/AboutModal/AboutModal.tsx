import { ExternalLinkIcon } from '@chakra-ui/icons';
import {
  Flex,
  Grid,
  GridItem,
  Image,
  Link,
  Table,
  TableContainer,
  Tbody,
  Td,
  Th,
  Thead,
  Tr,
  useDisclosure,
} from '@chakra-ui/react';
import { InvHeading } from 'common/components/InvHeading/wrapper';
import {
  InvModal,
  InvModalBody,
  InvModalCloseButton,
  InvModalContent,
  InvModalFooter,
  InvModalHeader,
  InvModalOverlay,
} from 'common/components/InvModal/wrapper';
import { InvText } from 'common/components/InvText/wrapper';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { fromPairs, map, toPairs } from 'lodash-es';
import InvokeLogoYellow from 'public/assets/images/invoke-symbol-ylw-lrg.svg';
import type { ReactElement } from 'react';
import { cloneElement, memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useGetAppDepsQuery,
  useGetAppVersionQuery,
} from 'services/api/endpoints/appInfo';

type AboutModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

const AboutModal = ({ children }: AboutModalProps) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { t } = useTranslation();
  const { data: appDeps } = useGetAppDepsQuery();
  const { data: appVersion } = useGetAppVersionQuery();
  const appDeps_arr = map(toPairs(appDeps), (d) => fromPairs([d]));
  const ref = useRef(null);
  const githubLink = 'http://github.com/invoke-ai/InvokeAI';
  const discordLink = 'https://discord.gg/ZmtBAhwWhy';
  const websiteLink = 'https://www.invoke.com/';

  return (
    <>
      {cloneElement(children, {
        onClick: onOpen,
      })}
      <InvModal isOpen={isOpen} onClose={onClose} isCentered size="2xl">
        <InvModalOverlay />
        <InvModalContent maxH="80vh" h="80vh">
          <InvModalHeader>{t('accessibility.about')}</InvModalHeader>
          <InvModalCloseButton />
          <InvModalBody display="flex" flexDir="column" gap={4}>
            <ScrollableContent>
              <Grid templateColumns="repeat(2, 1fr)">
                <GridItem background="black" borderRadius="xl">
                  <TableContainer>
                    <Table variant="simple" w="50%">
                      <Thead>
                        <Tr>
                          <Th fontSize="medium" color="white">
                            Local System
                          </Th>
                        </Tr>
                      </Thead>
                      {appDeps_arr.map((deps, index) => (
                        <Flex flexDir="column" key={index} gap={4}>
                          <Tbody key={index}>
                            {Object.entries(deps).map(([key, value]) => (
                              <Tr key={key} fontSize="sm" color="white">
                                <Td>{key}</Td>
                                <Td>{value ? value : 'Not Installed'}</Td>
                              </Tr>
                            ))}
                          </Tbody>
                        </Flex>
                      ))}
                    </Table>
                  </TableContainer>
                </GridItem>
                <GridItem border="2px solid black">
                  <Image
                    ref={ref}
                    src={InvokeLogoYellow}
                    alt="invoke-logo"
                    w="24px"
                    h="24px"
                    minW="24px"
                    minH="24px"
                    userSelect="none"
                  />
                  <InvText>v{appVersion?.version}</InvText>
                  <Grid templateColumns="repeat(2, 1fr)" gap="3">
                    <GridItem>
                      <Link fontSize="sm" href={githubLink} isExternal>
                        Github
                        <ExternalLinkIcon mx="2px" />
                      </Link>
                    </GridItem>
                    <GridItem>
                      <Link fontSize="sm" href={discordLink} isExternal>
                        Discord
                        <ExternalLinkIcon mx="2px" />
                      </Link>
                    </GridItem>
                  </Grid>
                  <InvHeading fontSize="md">Own Your Creative Power</InvHeading>
                  <InvText fontSize="sm">
                    Using Invoke for work? Check out:
                  </InvText>
                  <Link href={websiteLink} fontSize="sm">
                    www.invoke.com
                  </Link>
                </GridItem>
              </Grid>
            </ScrollableContent>
          </InvModalBody>
          <InvModalFooter />
        </InvModalContent>
      </InvModal>
    </>
  );
};

export default memo(AboutModal);
